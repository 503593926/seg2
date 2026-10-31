"""Resolve gene identities before counts normalization or gene selection."""

import csv
import gzip
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path

import numpy as np
import pandas as pd


def gene_header(path: Path) -> list[str]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as handle:
        return next(csv.reader(handle, delimiter="\t"))[1:]


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

        counts = pd.read_csv(source, sep="\t", index_col=0)
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
