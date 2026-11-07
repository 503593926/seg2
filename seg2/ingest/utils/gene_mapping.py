"""Resolve gene identities before counts normalization or gene selection."""

import csv
import gzip
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import cast

import h5py
import numpy as np
import pandas as pd
from scipy.io import mmread
from scipy.sparse import csc_matrix
from threadpoolctl import threadpool_limits


def _h5_values(group: h5py.Group, key: str, *, text: bool = False) -> np.ndarray:
    dataset = cast(h5py.Dataset, group[key])
    return np.asarray(dataset.asstr()[()] if text else dataset[()])


def open_count_text(path: Path):
    """Read either gzip or plain text, including incorrectly suffixed .gz files."""
    with path.open(mode="rb") as handle:
        compressed = handle.read(2) == b"\x1f\x8b"
    return gzip.open(filename=path, mode="rt") if compressed else path.open(mode="r")


def read_mtx_table(path: Path) -> pd.DataFrame:
    with open_count_text(path=path) as handle:
        return pd.read_csv(filepath_or_buffer=handle, sep="\t", header=None, dtype=str)


def _mtx_features(path: Path):
    features = read_mtx_table(path=path / "features.tsv.gz")
    if features.shape[1] == 1:
        selected = np.ones(len(features), dtype=bool)
    elif features.shape[1] == 3:
        selected = features[2].eq("Gene Expression").to_numpy()
    else:
        raise ValueError(
            "Matrix Market features require one gene column or three 10x columns"
        )
    return features, selected


def gene_header(path: Path) -> list[str]:
    if path.is_dir():
        features, selected = _mtx_features(path=path)
        return features.loc[selected, 0].tolist()
    if path.suffix == ".h5":
        with h5py.File(name=path, mode="r") as handle:
            features = cast(h5py.Group, handle["matrix/features"])
            selected = (
                _h5_values(group=features, key="feature_type", text=True)
                == "Gene Expression"
            )
            return _h5_values(group=features, key="id", text=True)[selected].tolist()
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as handle:
        return next(csv.reader(handle, delimiter="\t"))[1:]


def read_count_matrix(path: Path) -> pd.DataFrame:
    if path.is_dir():
        features, selected = _mtx_features(path=path)
        barcodes = read_mtx_table(path=path / "barcodes.tsv.gz")[0].to_numpy(dtype=str)
        with (
            open_count_text(path=path / "matrix.mtx.gz") as handle,
            threadpool_limits(limits=1),
        ):
            matrix = csc_matrix(mmread(source=handle))
        if matrix.shape != (len(features), len(barcodes)):
            raise ValueError(
                "Matrix Market dimensions do not match features and barcodes"
            )
        if (
            not np.issubdtype(matrix.dtype, np.integer)
            or (matrix.data < 0).any()
            or matrix.data.max(initial=0) > np.iinfo(np.int32).max
        ):
            raise ValueError(
                "Matrix Market input must contain nonnegative int32 raw counts"
            )
        return pd.DataFrame(
            data=matrix[selected].T.astype(dtype=np.int32).toarray(),
            index=barcodes,
            columns=features.loc[selected, 0].tolist(),
        )
    if path.suffix != ".h5":
        return pd.read_csv(filepath_or_buffer=path, sep="\t", index_col=0)
    with h5py.File(name=path, mode="r") as handle:
        matrix = cast(h5py.Group, handle["matrix"])
        features = cast(h5py.Group, matrix["features"])
        selected = (
            _h5_values(group=features, key="feature_type", text=True)
            == "Gene Expression"
        )
        counts = csc_matrix(
            (
                _h5_values(group=matrix, key="data"),
                _h5_values(group=matrix, key="indices"),
                _h5_values(group=matrix, key="indptr"),
            ),
            shape=tuple(_h5_values(group=matrix, key="shape")),
        )
        return pd.DataFrame(
            data=counts[selected].T.toarray(),
            index=_h5_values(group=matrix, key="barcodes", text=True),
            columns=_h5_values(group=features, key="id", text=True)[selected],
        )


def build_gene_mapping(
    genes: Iterable[str],
    reference: Path,
) -> pd.DataFrame:

    table = pd.read_csv(reference, sep="\t", dtype=str).fillna("")

    lookup: dict[str, dict[str, set[str]]] = {
        column: defaultdict(set)
        for column in (
            "symbol",
            "prev_symbol",
            "alias_symbol",
            "ensembl_gene_id",
            "entrez_id",
        )
    }

    symbols: dict[str, str] = {}
    for values in table.loc[table["status"].eq("Approved")].to_dict("records"):
        row = {str(key): str(value) for key, value in values.items()}
        hgnc_id, symbol = row["hgnc_id"], row["symbol"]
        symbols[hgnc_id] = symbol
        for column in lookup:
            for key in row.get(column, "").split("|"):
                if key:
                    lookup[column][key].add(hgnc_id)

    records = []
    for source in sorted(set(genes)):
        gene = source.strip()
        if gene.startswith("__"):
            candidates = set()
        elif re.fullmatch(r"ENSG\d+(?:\.\d+)?", gene):
            candidates = lookup["ensembl_gene_id"].get(gene.split(".", 1)[0], set())
        elif re.fullmatch(r"(?:NCBIGene:|GeneID:)?\d+", gene):
            candidates = lookup["entrez_id"].get(gene.rsplit(":", 1)[-1], set())
        else:
            candidates = lookup["symbol"].get(gene, set()) or (
                lookup["prev_symbol"].get(gene, set())
                | lookup["alias_symbol"].get(gene, set())
            )

        resolved = len(candidates) == 1
        hgnc_id = next(iter(candidates)) if resolved else ""

        records.append(
            {
                "source_gene": source,
                "approved_symbol": symbols[hgnc_id] if resolved else "",
                "hgnc_id": hgnc_id,
            }
        )
    return pd.DataFrame(records)


def standardize_counts(counts: pd.DataFrame, mapping: pd.DataFrame) -> pd.DataFrame:
    values = counts.to_numpy()
    if not np.issubdtype(values.dtype, np.integer):
        counts = counts.astype(np.int64)
    selected = mapping.set_index("source_gene").loc[counts.columns]
    keep = selected["hgnc_id"].ne("").to_numpy()
    result = counts.iloc[:, keep].copy()
    result.columns = pd.Index(selected.loc[keep, "approved_symbol"].tolist())
    return result.T.groupby(level=0, sort=True).sum().T


def invalidate_task_expression(task_root: Path, dataset_name: str) -> None:
    if not task_root.is_dir():
        return
    for task_dir in task_root.iterdir():
        if not task_dir.is_dir():
            continue
        related = (
            task_dir.name == f"{dataset_name}_patient_loocv"
            or ("_to_" in task_dir.name and dataset_name in task_dir.name.split("_to_"))
            or (task_dir / f"{dataset_name}.h5ad").is_file()
        )
        if related:
            for output in task_dir.glob("*.h5ad"):
                output.unlink()
            (task_dir / "gene_spec.json").unlink(missing_ok=True)


def prepare_counts(
    sources: Mapping[str, Path],
    data_dir: Path,
    reference: Path,
    task_root: Path,
    dataset_name: str,
) -> None:

    data_dir = data_dir.expanduser().resolve()
    reference = reference.expanduser().resolve()

    root = data_dir.parent
    root.mkdir(parents=True, exist_ok=True)

    headers = {name: gene_header(path=path) for name, path in sources.items()}
    mapping = build_gene_mapping(
        genes=set().union(*headers.values()),
        reference=reference,
    )
    mapping.to_csv(path_or_buf=root / "gene_mapping.tsv", sep="\t", index=False)

    for name, source in sources.items():
        output = data_dir / "samples" / name / "gene_counts.tsv.gz"

        counts = read_count_matrix(path=source)
        counts.columns = pd.Index(headers[name])
        counts.index = counts.index.astype(str)

        standardized = standardize_counts(counts=counts, mapping=mapping)
        sample_mapping = mapping.set_index("source_gene").loc[counts.columns]
        kept = sample_mapping["hgnc_id"].ne("").to_numpy()
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(".gz.tmp")
        standardized.to_csv(
            temporary, sep="\t", compression={"method": "gzip", "compresslevel": 1}
        )
        temporary.replace(output)
        print(
            f"[{name}] HGNC counts={standardized.shape}, excluded_columns={int((~kept).sum())}",
            flush=True,
        )
    invalidate_task_expression(
        task_root=task_root.expanduser().resolve(), dataset_name=dataset_name
    )


def counts_gene_info(data_dir: Path) -> pd.DataFrame:
    """Read confirmed gene identities directly from the mapping table."""
    root = data_dir.parent
    mapping_path = root / "gene_mapping.tsv"
    mapping = pd.read_csv(mapping_path, sep="\t", dtype=str).fillna("")
    info = mapping.loc[
        mapping["hgnc_id"].ne(""), ["approved_symbol", "hgnc_id"]
    ].drop_duplicates()
    if not info["approved_symbol"].is_unique or not info["hgnc_id"].is_unique:
        raise ValueError("Approved symbols and HGNC IDs must be one-to-one")
    return info.set_index("approved_symbol")
