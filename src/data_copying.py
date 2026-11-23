import numpy as np
import pandas as pd


def make_exact_copy_plan(
    q_meta,
    training_meta,
    severities=(0.10, 0.30, 0.60),
    seed=42,
    label_column="label",
    id_column="message_id"
):
    """
    Create a nested exact-copy plan for a candidate sample Q.

    Selected Q slots are replaced by training observations
    with the same class label.
    """

    rng = np.random.default_rng(seed)

    q_order = {}
    train_order = {}

    for label in sorted(q_meta[label_column].unique()):
        q_ids = q_meta.loc[
            q_meta[label_column] == label, id_column
        ].to_numpy().copy()

        train_ids = training_meta.loc[
            training_meta[label_column] == label, id_column
        ].to_numpy().copy()

        rng.shuffle(q_ids)
        rng.shuffle(train_ids)

        q_order[label] = q_ids
        train_order[label] = train_ids

    records = []

    for severity in severities:
        for label in sorted(q_order):
            n_label = len(q_order[label])
            n_copy = int(round(n_label * severity))

            for slot_id, source_id in zip(
                q_order[label][:n_copy],
                train_order[label][:n_copy]
            ):
                records.append({
                    "severity": float(severity),
                    "q_message_id": slot_id,
                    "source_message_id": source_id,
                    "label": label
                })

    return pd.DataFrame(records)

def inject_exact_training_copies(
    q_embeddings,
    q_meta,
    training_embeddings,
    training_meta,
    copy_plan,
    severity,
    id_column="message_id"
):
    """
    Replace selected Q embeddings with exact training embeddings.
    """

    Q = np.asarray(q_embeddings).copy()

    q_lookup = {
        message_id: i
        for i, message_id in enumerate(q_meta[id_column])
    }

    train_lookup = {
        message_id: i
        for i, message_id in enumerate(training_meta[id_column])
    }

    plan = copy_plan.loc[
        copy_plan["severity"] == severity
    ]

    for _, row in plan.iterrows():
        q_pos = q_lookup[row["q_message_id"]]
        t_pos = train_lookup[row["source_message_id"]]

        Q[q_pos] = training_embeddings[t_pos]

    return Q

from sklearn.metrics import pairwise_distances


def nearest_training_distances(X, T):
    """Distance from each observation in X to its nearest training example."""
    D = pairwise_distances(
        np.asarray(X, dtype=np.float64),
        np.asarray(T, dtype=np.float64),
        metric="euclidean"
    )
    return D.min(axis=1)


def copying_statistic(d_p, d_q):
    """
    Positive values indicate that Q is closer to training than P.
    """
    return float(np.mean(d_p) - np.mean(d_q))


def data_copying_permutation_test(
    T, P, Q, p_labels, q_labels,
    n_permutations=100, seed=42, alpha=0.05
):
    """
    One-sided stratified permutation test for data copying.

    H0: P and Q have the same proximity to training T.
    H1: Q is systematically closer to T than P.
    """
    d_p = nearest_training_distances(P, T)
    d_q = nearest_training_distances(Q, T)

    observed = copying_statistic(d_p, d_q)

    distances = np.concatenate([d_p, d_q])
    labels = np.concatenate([
        np.asarray(p_labels),
        np.asarray(q_labels)
    ])

    n_p = len(d_p)
    group = np.array(["P"] * len(d_p) + ["Q"] * len(d_q))
    rng = np.random.default_rng(seed)

    null_stats = np.empty(n_permutations)

    for b in range(n_permutations):
        perm_group = group.copy()

        for label in np.unique(labels):
            idx = np.flatnonzero(labels == label)
            shuffled = rng.permutation(idx)

            n_p_label = np.sum(
                (labels == label) & (group == "P")
            )

            perm_group[shuffled[:n_p_label]] = "P"
            perm_group[shuffled[n_p_label:]] = "Q"

        null_stats[b] = copying_statistic(
            distances[perm_group == "P"],
            distances[perm_group == "Q"]
        )

    p_value = (
        1 + np.sum(null_stats >= observed)
    ) / (n_permutations + 1)

    return {
        "statistic": observed,
        "p_value": float(p_value),
        "reject": bool(p_value < alpha),
        "mean_distance_P": float(d_p.mean()),
        "mean_distance_Q": float(d_q.mean()),
        "distance_difference": observed,
        "null_mean": float(null_stats.mean()),
        "null_std": float(null_stats.std())
    }

def data_copying_test_from_distances(
    d_p, d_q, p_labels, q_labels,
    n_permutations=999, seed=42, alpha=0.05
):
    d_p = np.asarray(d_p)
    d_q = np.asarray(d_q)

    observed = copying_statistic(d_p, d_q)

    distances = np.concatenate([d_p, d_q])
    labels = np.concatenate([p_labels, q_labels])
    group = np.array(["P"] * len(d_p) + ["Q"] * len(d_q))

    rng = np.random.default_rng(seed)
    null_stats = np.empty(n_permutations)

    for b in range(n_permutations):
        perm_group = group.copy()

        for label in np.unique(labels):
            idx = np.flatnonzero(labels == label)
            shuffled = rng.permutation(idx)

            n_p_label = np.sum(
                (labels == label) & (group == "P")
            )

            perm_group[shuffled[:n_p_label]] = "P"
            perm_group[shuffled[n_p_label:]] = "Q"

        null_stats[b] = copying_statistic(
            distances[perm_group == "P"],
            distances[perm_group == "Q"]
        )

    p_value = (
        1 + np.sum(null_stats >= observed)
    ) / (n_permutations + 1)

    return {
        "statistic": float(observed),
        "p_value": float(p_value),
        "reject": bool(p_value < alpha)
    }

def apply_exact_copy_plan_to_distances(
    q_distances, q_meta, copy_plan, severity,
    id_column="message_id"
):
    """
    Apply an exact-copy plan directly in nearest-training-distance space.

    Exact copies from T have nearest-training distance equal to zero.
    """
    q_distances = np.asarray(q_distances, dtype=float).copy()

    if severity == 0:
        return q_distances, np.zeros(len(q_distances), dtype=bool)

    plan = copy_plan.loc[
        np.isclose(copy_plan["severity"], severity)
    ]

    copied_ids = set(plan["q_message_id"])
    copied_mask = q_meta[id_column].isin(copied_ids).to_numpy()

    q_distances[copied_mask] = 0.0

    return q_distances, copied_mask