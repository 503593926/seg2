from pathlib import Path


def sample_names(dataset):
    """Enumerate prepared counts, optionally restricted to patient prefixes."""
    root = Path(str(dataset.data_dir)) / "samples"
    prefixes = list(dataset.patient_names) or [""]
    return sorted(
        {
            path.parent.name
            for prefix in prefixes
            for path in root.glob(f"{prefix}*/gene_counts.tsv.gz")
        }
    )
