from pathlib import Path
import numpy as np
import pandas as pd


def load_embedding_conditions(
    data_dir,
    prefix,
    severities=(10, 30, 60)
):
    """
    Load frozen clean and defective embedding arrays.

    Expected filenames:
        {prefix}_embeddings_clean.npy
        {prefix}_embeddings_10pct.npy
        {prefix}_embeddings_30pct.npy
        {prefix}_embeddings_60pct.npy

    Returns
    -------
    dict
        {
            "clean": array,
            0.10: array,
            0.30: array,
            0.60: array
        }
    """

    data_dir = Path(data_dir)

    arrays = {
        "clean": np.load(
            data_dir
            / f"{prefix}_embeddings_clean.npy"
        )
    }

    for pct in severities:

        arrays[
            pct / 100
        ] = np.load(
            data_dir
            / f"{prefix}_embeddings_{pct}pct.npy"
        )

    return arrays


def load_id_file(
    data_dir,
    filename,
    id_column
):
    """
    Load and validate an ID file.
    """

    path = (
        Path(data_dir)
        / filename
    )

    df = pd.read_csv(path)

    if id_column not in df.columns:
        raise ValueError(
            f"{id_column} not found in {filename}"
        )

    if df[id_column].duplicated().any():
        raise ValueError(
            f"Duplicate IDs found in {filename}"
        )

    return df[id_column].to_numpy()


def save_embedding_ids(
    ids,
    output_path,
    id_column
):
    """
    Save the row-to-ID mapping associated
    with an embedding matrix.
    """

    import pandas as pd
    from pathlib import Path

    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    id_df = pd.DataFrame({
        id_column: ids
    })

    if id_df[
        id_column
    ].duplicated().any():

        raise ValueError(
            "Duplicate IDs detected."
        )

    id_df.to_csv(
        output_path,
        index=False
    )

    return id_df

def save_embedding_conditions(
    embedding_arrays,
    data_dir,
    prefix,
    conditions=("clean", 0.10, 0.30, 0.60)
):
    """
    Save clean and defective embedding arrays
    using the project's standard naming scheme.
    """

    from pathlib import Path
    import numpy as np

    data_dir = Path(data_dir)

    data_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    saved_paths = {}

    for condition in conditions:

        condition_name = (
            "clean"
            if condition == "clean"
            else f"{int(condition * 100)}pct"
        )

        path = (
            data_dir
            / (
                f"{prefix}_embeddings_"
                f"{condition_name}.npy"
            )
        )

        np.save(
            path,
            embedding_arrays[
                condition
            ]
        )

        saved_paths[
            condition
        ] = path

    return saved_paths


def save_affected_ids_by_severity(
    affected_ids_by_severity,
    output_path,
    id_column
):
    """
    Save ground-truth affected IDs for each
    severity level.

    Parameters
    ----------
    affected_ids_by_severity : dict
        Example:
        {
            0.10: array_of_ids,
            0.30: array_of_ids,
            0.60: array_of_ids
        }

    output_path : str or Path
        Destination CSV.

    id_column : str
        Name of the experimental-unit ID column.
    """

    from pathlib import Path
    import pandas as pd

    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    records = []

    for severity, ids in (
        affected_ids_by_severity.items()
    ):

        for unit_id in ids:

            records.append({
                "severity":
                    float(severity),

                id_column:
                    unit_id
            })

    affected_df = pd.DataFrame(
        records
    )

    affected_df.to_csv(
        output_path,
        index=False
    )

    return affected_df


def load_affected_ids_by_severity(
    input_path,
    id_column
):
    """
    Load ground-truth affected IDs and return
    one set of IDs per severity.
    """

    import pandas as pd
    from pathlib import Path

    input_path = Path(
        input_path
    )

    df = pd.read_csv(
        input_path
    )

    required = {
        "severity",
        id_column
    }

    missing = (
        required
        - set(df.columns)
    )

    if missing:

        raise ValueError(
            f"Missing columns: {missing}"
        )

    return {
        float(severity):
            set(
                group[
                    id_column
                ]
            )

        for severity, group
        in df.groupby(
            "severity"
        )
    }

import numpy as np


def stratified_disjoint_pair_indices(
    labels,
    n_pairs,
    seed=42
):
    """
    Create two disjoint clean samples with matched class composition.

    P and Q contain different observations, but every pair belongs
    to the same class.
    """
    labels = np.asarray(labels)
    rng = np.random.default_rng(seed)

    classes, counts = np.unique(
        labels,
        return_counts=True
    )

    proportions = counts / counts.sum()

    pair_counts = np.floor(
        proportions * n_pairs
    ).astype(int)

    remainder = n_pairs - pair_counts.sum()

    fractions = (
        proportions * n_pairs -
        pair_counts
    )

    order = np.argsort(
        fractions
    )[::-1]

    for i in order[:remainder]:
        pair_counts[i] += 1

    p_indices = []
    q_indices = []

    for cls, n_cls_pairs in zip(
        classes,
        pair_counts
    ):
        cls_idx = np.flatnonzero(
            labels == cls
        )

        if 2 * n_cls_pairs > len(cls_idx):
            raise ValueError(
                f"Not enough observations for class {cls}."
            )

        selected = rng.choice(
            cls_idx,
            size=2 * n_cls_pairs,
            replace=False
        )

        p_indices.extend(
            selected[:n_cls_pairs]
        )

        q_indices.extend(
            selected[n_cls_pairs:]
        )

    p_indices = np.asarray(p_indices)
    q_indices = np.asarray(q_indices)

    shuffle = rng.permutation(
        len(p_indices)
    )

    return (
        p_indices[shuffle],
        q_indices[shuffle]
    )