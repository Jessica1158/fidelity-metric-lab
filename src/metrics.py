
import numpy as np
from scipy import sparse
from scipy.stats import binomtest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.model_selection import train_test_split


# ============================================================
# Input helpers
# ============================================================

def _as_matrix(X):
    """
    Convert input to a 2D numerical matrix.

    Accepts:
        - 1D NumPy arrays
        - 2D NumPy arrays
        - scipy sparse matrices
    """
    if sparse.issparse(X):
        X = X.tocsr().astype(np.float64)

        if not np.isfinite(X.data).all():
            raise ValueError("Non-finite values in input.")

    else:
        X = np.asarray(X, dtype=np.float64)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if X.ndim != 2 or not np.isfinite(X).all():
            raise ValueError("Input must be a finite 2D matrix.")

    if X.shape[0] < 2 or X.shape[1] < 1:
        raise ValueError(
            "At least two rows and one feature are required."
        )

    return X


def _prepare(P, Q):
    """
    Validate that P and Q share the same feature space.
    """
    P = _as_matrix(P)
    Q = _as_matrix(Q)

    if P.shape[1] != Q.shape[1]:
        raise ValueError(
            "P and Q must share the same feature space."
        )

    if sparse.issparse(P) or sparse.issparse(Q):
        P = sparse.csr_matrix(P)
        Q = sparse.csr_matrix(Q)

    return P, Q


def _pool(P, Q):
    """
    Combine two samples while preserving sparse format.
    """
    if sparse.issparse(P):
        return sparse.vstack([P, Q], format="csr")

    return np.vstack([P, Q])


# ============================================================
# C2ST
# ============================================================

def c2st(
    P,
    Q,
    test_size=0.5,
    seed=42,
    alpha=0.05
):
    """
    Classifier Two-Sample Test using Logistic Regression.

    H0: P = Q
    H1: P != Q

    Uses held-out classification accuracy and a one-sided
    binomial test against chance accuracy = 0.5.
    """
    P, Q = _prepare(P, Q)

    X = _pool(P, Q)

    y = np.r_[
        np.zeros(P.shape[0], dtype=int),
        np.ones(Q.shape[0], dtype=int)
    ]

    Xtr, Xte, ytr, yte = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=y
    )

    clf = LogisticRegression(
        random_state=seed,
        max_iter=1000
    )

    clf.fit(Xtr, ytr)

    pred = clf.predict(Xte)

    correct = int(np.sum(pred == yte))
    n_test = len(yte)

    accuracy = correct / n_test

    p = float(
        binomtest(
            correct,
            n=n_test,
            p=0.5,
            alternative="greater"
        ).pvalue
    )

    return {
        "accuracy": accuracy,
        "p_value": p,
        "reject_H0": p < alpha,
        "correct": correct,
        "n_test": n_test,
        "seed": seed
    }


# ============================================================
# Shared distance / kernel helpers
# ============================================================

def _matrix(P, Q, kind, sigma=1.0):
    """
    Build a pairwise kernel or distance matrix.

    For MMD:
        Gaussian RBF kernel.

    For Energy Distance:
        Euclidean distances.

    The input may be sparse, but the resulting pairwise
    matrix is dense.
    """
    P, Q = _prepare(P, Q)

    X = _pool(P, Q)

    D2 = euclidean_distances(
        X,
        squared=True
    )

    # Correct small negative values caused by numerical error.
    np.maximum(D2, 0, out=D2)

    if kind == "kernel":

        if not np.isfinite(sigma) or sigma <= 0:
            raise ValueError(
                "sigma must be positive and finite."
            )

        np.exp(
            -D2 / (2 * sigma**2),
            out=D2
        )

    elif kind == "distance":

        np.sqrt(D2, out=D2)

    else:
        raise ValueError("Unknown matrix kind.")

    return D2, P.shape[0], Q.shape[0]


def _statistic(M, m, n, kind):
    """
    Compute the unbiased MMD² or Energy statistic
    from a pairwise matrix.
    """
    A = M[:m, :m]
    B = M[m:, m:]

    # Exclude diagonal terms from within-sample sums.
    within = (
        (A.sum() - np.trace(A)) / (m * (m - 1))
        +
        (B.sum() - np.trace(B)) / (n * (n - 1))
    )

    between = 2 * M[:m, m:].mean()

    if kind == "kernel":
        return float(within - between)

    return float(between - within)


# ============================================================
# MMD
# ============================================================

def mmd2_unbiased(
    P,
    Q,
    sigma=1.0
):
    """
    Unbiased quadratic MMD² with Gaussian RBF kernel.

    Accepts multidimensional dense or sparse matrices.
    """
    M, m, n = _matrix(
        P,
        Q,
        kind="kernel",
        sigma=sigma
    )

    return _statistic(
        M,
        m,
        n,
        kind="kernel"
    )


# ============================================================
# Energy Distance
# ============================================================

def energy_distance_statistic(P, Q):
    """
    Unbiased Energy Distance statistic using Euclidean
    distances.

    Accepts multidimensional dense or sparse matrices.
    """
    M, m, n = _matrix(
        P,
        Q,
        kind="distance"
    )

    return _statistic(
        M,
        m,
        n,
        kind="distance"
    )


# ============================================================
# Shared permutation test
# ============================================================

def _permutation_test(
    P,
    Q,
    kind,
    sigma=1.0,
    n_permutations=200,
    seed=42,
    alpha=0.05
):
    """
    Permutation test for MMD or Energy Distance.

    The pairwise matrix is calculated once and reused
    across permutations.
    """
    if (
        not isinstance(n_permutations, (int, np.integer))
        or n_permutations < 1
    ):
        raise ValueError(
            "n_permutations must be a positive integer."
        )

    M, m, n = _matrix(
        P,
        Q,
        kind=kind,
        sigma=sigma
    )

    observed = _statistic(
        M,
        m,
        n,
        kind
    )

    rng = np.random.default_rng(seed)

    null = np.empty(n_permutations)

    for b in range(n_permutations):

        order = rng.permutation(m + n)

        Mp = M[np.ix_(order, order)]

        null[b] = _statistic(
            Mp,
            m,
            n,
            kind
        )

    p = float(
        (
            1 + np.count_nonzero(null >= observed)
        ) / (n_permutations + 1)
    )

    return {
        "statistic": observed,
        "p_value": p,
        "reject_H0": p < alpha,
        "n_permutations": n_permutations,
        "seed": seed
    }


# ============================================================
# MMD permutation test
# ============================================================

def mmd_permutation_test(
    P,
    Q,
    sigma=1.0,
    n_permutations=200,
    seed=42,
    alpha=0.05
):
    """
    Permutation test for unbiased MMD².
    """
    return _permutation_test(
        P,
        Q,
        kind="kernel",
        sigma=sigma,
        n_permutations=n_permutations,
        seed=seed,
        alpha=alpha
    )


# ============================================================
# Energy Distance permutation test
# ============================================================

def energy_permutation_test(
    P,
    Q,
    n_permutations=200,
    seed=42,
    alpha=0.05
):
    """
    Permutation test for the unbiased Energy statistic.
    """
    return _permutation_test(
        P,
        Q,
        kind="distance",
        n_permutations=n_permutations,
        seed=seed,
        alpha=alpha
    )

def _prepare_paired(P, Q):
    """
    Validate paired samples.

    P[i] and Q[i] must correspond to the same
    observational unit.
    """

    P, Q = _prepare(P, Q)

    if P.shape[0] != Q.shape[0]:
        raise ValueError(
            "Paired samples must contain the same "
            "number of observations."
        )

    return P, Q

def c2st_paired(
    P,
    Q,
    test_size=0.5,
    n_permutations=100,
    seed=42,
    alpha=0.05
):
    """
    Paired Classifier Two-Sample Test.

    Pair identities are split before constructing the
    classifier training and test sets, preventing members
    of the same pair from appearing in different splits.

    Statistical significance is evaluated by randomly
    swapping clean/defective labels within test pairs.
    """

    if (
        not isinstance(n_permutations, (int, np.integer))
        or n_permutations < 1
    ):
        raise ValueError(
            "n_permutations must be a positive integer."
        )

    P, Q = _prepare_paired(P, Q)

    n_pairs = P.shape[0]

    pair_indices = np.arange(
        n_pairs
    )

    train_idx, test_idx = train_test_split(
        pair_indices,
        test_size=test_size,
        random_state=seed,
        shuffle=True
    )

    # ---------------------------------------------
    # Training data
    # ---------------------------------------------

    P_train = P[train_idx]
    Q_train = Q[train_idx]

    X_train = _pool(
        P_train,
        Q_train
    )

    y_train = np.r_[
        np.zeros(
            len(train_idx),
            dtype=int
        ),
        np.ones(
            len(train_idx),
            dtype=int
        )
    ]

    # ---------------------------------------------
    # Test data
    # ---------------------------------------------

    P_test = P[test_idx]
    Q_test = Q[test_idx]

    X_test = _pool(
        P_test,
        Q_test
    )

    y_test = np.r_[
        np.zeros(
            len(test_idx),
            dtype=int
        ),
        np.ones(
            len(test_idx),
            dtype=int
        )
    ]

    # ---------------------------------------------
    # Classifier
    # ---------------------------------------------

    clf = LogisticRegression(
        random_state=seed,
        max_iter=5000
    )

    clf.fit(
        X_train,
        y_train
    )

    predictions = clf.predict(
        X_test
    )

    observed_accuracy = (
        predictions == y_test
    ).mean()

    # Predictions corresponding to each side
    # of the test pairs.
    n_test_pairs = len(test_idx)

    pred_P = predictions[
        :n_test_pairs
    ]

    pred_Q = predictions[
        n_test_pairs:
    ]

    # ---------------------------------------------
    # Paired permutation test
    # ---------------------------------------------

    rng = np.random.default_rng(
        seed
    )

    null_accuracies = np.empty(
        n_permutations
    )

    for b in range(
        n_permutations
    ):

        # True = swap labels inside this pair.
        swap = rng.integers(
            0,
            2,
            size=n_test_pairs
        ).astype(bool)

        labels_P = swap.astype(int)

        labels_Q = (
            1 - labels_P
        )

        correct = (
            np.sum(
                pred_P == labels_P
            )
            +
            np.sum(
                pred_Q == labels_Q
            )
        )

        null_accuracies[b] = (
            correct
            /
            (2 * n_test_pairs)
        )

    p_value = float(
        (
            1
            +
            np.count_nonzero(
                null_accuracies
                >= observed_accuracy
            )
        )
        /
        (n_permutations + 1)
    )

    return {
        "accuracy":
            float(observed_accuracy),

        "p_value":
            p_value,

        "reject_H0":
            p_value < alpha,

        "n_pairs":
            n_pairs,

        "n_train_pairs":
            len(train_idx),

        "n_test_pairs":
            n_test_pairs,

        "n_test":
            2 * n_test_pairs,

        "n_permutations":
            n_permutations,

        "seed":
            seed
    }

def _statistic_from_indices(
    M,
    P_indices,
    Q_indices,
    kind
):
    """
    Calculate MMD or Energy statistic using arbitrary
    group assignments from a precomputed pairwise matrix.
    """

    m = len(P_indices)
    n = len(Q_indices)

    A = M[
        np.ix_(
            P_indices,
            P_indices
        )
    ]

    B = M[
        np.ix_(
            Q_indices,
            Q_indices
        )
    ]

    cross = M[
        np.ix_(
            P_indices,
            Q_indices
        )
    ]

    within = (
        (
            A.sum()
            - np.trace(A)
        )
        /
        (m * (m - 1))
        +
        (
            B.sum()
            - np.trace(B)
        )
        /
        (n * (n - 1))
    )

    between = (
        2 * cross.mean()
    )

    if kind == "kernel":
        return float(
            within - between
        )

    elif kind == "distance":
        return float(
            between - within
        )

    else:
        raise ValueError(
            "Unknown statistic kind."
        )

def _paired_permutation_test(
    P,
    Q,
    kind,
    sigma=1.0,
    n_permutations=100,
    seed=42,
    alpha=0.05
):
    """
    Paired permutation test for MMD or Energy Distance.

    Under the null hypothesis, P[i] and Q[i] are
    exchangeable within each pair.

    Pair membership is preserved during every permutation.
    """

    if (
        not isinstance(
            n_permutations,
            (int, np.integer)
        )
        or n_permutations < 1
    ):
        raise ValueError(
            "n_permutations must be "
            "a positive integer."
        )

    P, Q = _prepare_paired(
        P,
        Q
    )

    n_pairs = P.shape[0]

    # Pairwise matrix is calculated only once.
    M, m, n = _matrix(
        P,
        Q,
        kind=kind,
        sigma=sigma
    )

    if m != n:
        raise ValueError(
            "Paired tests require equal "
            "sample sizes."
        )

    # Original organization:
    #
    # 0 ... n-1       = P
    # n ... 2n-1      = Q

    P_indices = np.arange(
        n_pairs
    )

    Q_indices = (
        np.arange(n_pairs)
        + n_pairs
    )

    observed = (
        _statistic_from_indices(
            M,
            P_indices,
            Q_indices,
            kind
        )
    )

    rng = np.random.default_rng(
        seed
    )

    null_statistics = np.empty(
        n_permutations
    )

    base = np.arange(
        n_pairs
    )

    for b in range(
        n_permutations
    ):

        swap = rng.integers(
            0,
            2,
            size=n_pairs
        ).astype(bool)

        # If swap=False:
        #   P gets P_i
        #   Q gets Q_i
        #
        # If swap=True:
        #   P gets Q_i
        #   Q gets P_i

        perm_P = np.where(
            swap,
            base + n_pairs,
            base
        )

        perm_Q = np.where(
            swap,
            base,
            base + n_pairs
        )

        null_statistics[b] = (
            _statistic_from_indices(
                M,
                perm_P,
                perm_Q,
                kind
            )
        )

    p_value = float(
        (
            1
            +
            np.count_nonzero(
                null_statistics
                >= observed
            )
        )
        /
        (n_permutations + 1)
    )

    return {
        "statistic":
            observed,

        "p_value":
            p_value,

        "reject_H0":
            p_value < alpha,

        "n_pairs":
            n_pairs,

        "n_permutations":
            n_permutations,

        "seed":
            seed
    }
def mmd_paired_permutation_test(
    P,
    Q,
    sigma=1.0,
    n_permutations=100,
    seed=42,
    alpha=0.05
):
    """
    Paired permutation test using the MMD statistic.
    """

    return _paired_permutation_test(
        P,
        Q,
        kind="kernel",
        sigma=sigma,
        n_permutations=n_permutations,
        seed=seed,
        alpha=alpha
    )

def energy_paired_permutation_test(
    P,
    Q,
    n_permutations=100,
    seed=42,
    alpha=0.05
):
    """
    Paired permutation test using the
    Energy Distance statistic.
    """

    return _paired_permutation_test(
        P,
        Q,
        kind="distance",
        n_permutations=n_permutations,
        seed=seed,
        alpha=alpha
    )



from sklearn.metrics import pairwise_distances


def median_heuristic_sigma(X):
    """
    Estimate the RBF bandwidth using the median
    pairwise Euclidean distance.

    The input should contain only clean/reference data.
    """

    X = np.asarray(X)

    if X.ndim != 2:
        raise ValueError(
            "X must be a 2D array."
        )

    if not np.isfinite(X).all():
        raise ValueError(
            "X contains non-finite values."
        )

    distances = pairwise_distances(
        X,
        metric="euclidean"
    )

    upper_triangle = distances[
        np.triu_indices_from(
            distances,
            k=1
        )
    ]

    positive_distances = (
        upper_triangle[
            upper_triangle > 0
        ]
    )

    if len(positive_distances) == 0:
        raise ValueError(
            "No positive pairwise distances found."
        )

    sigma = np.median(
        positive_distances
    )

    return float(sigma)

def get_paired_samples(
    clean_df,
    defective_df,
    unit_ids,
    features,
    id_column="campaign_id"
):
    """
    Extract aligned clean/defective paired samples.

    Parameters
    ----------
    clean_df : pd.DataFrame
    defective_df : pd.DataFrame
    unit_ids : iterable
        IDs to sample in the desired order.
    features : list[str]
        Feature columns.
    id_column : str
        Identifier column. Defaults to 'campaign_id'
        for backward compatibility.

    Returns
    -------
    P, Q : np.ndarray
        Aligned clean and defective samples.
    """

    if id_column not in clean_df.columns:
        raise ValueError(
            f"{id_column} not found in clean_df."
        )

    if id_column not in defective_df.columns:
        raise ValueError(
            f"{id_column} not found in defective_df."
        )

    clean_indexed = (
        clean_df
        .set_index(id_column)
    )

    defective_indexed = (
        defective_df
        .set_index(id_column)
    )

    unit_ids = list(unit_ids)

    missing_clean = (
        set(unit_ids)
        - set(clean_indexed.index)
    )

    missing_defective = (
        set(unit_ids)
        - set(defective_indexed.index)
    )

    if missing_clean:
        raise ValueError(
            f"Missing clean IDs: "
            f"{sorted(missing_clean)}"
        )

    if missing_defective:
        raise ValueError(
            f"Missing defective IDs: "
            f"{sorted(missing_defective)}"
        )

    P = (
        clean_indexed
        .loc[
            unit_ids,
            features
        ]
        .to_numpy()
    )

    Q = (
        defective_indexed
        .loc[
            unit_ids,
            features
        ]
        .to_numpy()
    )

    return P, Q

def extract_metric_result(
    result,
    metric_name,
    alpha=0.05
):
    """
    Normalize outputs from C2ST, MMD,
    and Energy test functions.
    """

    import numpy as np

    if "p" in result:
        p_value = result["p"]

    elif "p_value" in result:
        p_value = result["p_value"]

    else:
        raise KeyError(
            "No p-value found in metric result."
        )

    if metric_name == "C2ST":

        statistic = result.get(
            "accuracy",
            result.get(
                "statistic",
                np.nan
            )
        )

    elif metric_name == "MMD":

        statistic = result.get(
            "statistic",
            result.get(
                "mmd2",
                result.get(
                    "mmd",
                    np.nan
                )
            )
        )

    elif metric_name == "Energy":

        statistic = result.get(
            "statistic",
            result.get(
                "energy",
                result.get(
                    "energy_distance",
                    np.nan
                )
            )
        )

    else:
        raise ValueError(
            f"Unknown metric: {metric_name}"
        )

    reject = result.get(
        "reject",
        p_value < alpha
    )

    return (
        float(statistic),
        float(p_value),
        bool(reject)
    )

def median_pairwise_euclidean_scale(X):
    import numpy as np
    from sklearn.metrics import pairwise_distances

    X = np.asarray(X, dtype=float)

    if X.ndim != 2 or len(X) < 2:
        raise ValueError("X must contain at least two observations.")

    D = pairwise_distances(X, metric="euclidean")
    upper = D[np.triu_indices(len(X), k=1)]

    return float(np.median(upper))

def _make_sinkhorn_loss(blur, p=2, backend="tensorized"):
    from geomloss import SamplesLoss

    if blur <= 0:
        raise ValueError("blur must be positive.")

    return SamplesLoss(
        loss="sinkhorn",
        p=p,
        blur=float(blur),
        debias=True,
        backend=backend
    )


def _sinkhorn_value(P, Q, criterion):
    import numpy as np
    import torch

    P = np.asarray(P)
    Q = np.asarray(Q)

    if P.ndim != 2 or Q.ndim != 2:
        raise ValueError("P and Q must be 2D arrays.")

    if P.shape[1] != Q.shape[1]:
        raise ValueError("P and Q must have the same feature dimension.")

    x = torch.as_tensor(P, dtype=torch.float32, device="cpu")
    y = torch.as_tensor(Q, dtype=torch.float32, device="cpu")

    with torch.no_grad():
        value = criterion(x, y)

    return float(value.detach().cpu().item())


def sinkhorn_divergence(P, Q, blur, p=2, backend="tensorized"):
    criterion = _make_sinkhorn_loss(
        blur=blur,
        p=p,
        backend=backend
    )

    return _sinkhorn_value(P, Q, criterion)


def sinkhorn_permutation_test(
    P, Q, blur,
    n_permutations=100,
    seed=42,
    alpha=0.05,
    p=2,
    backend="tensorized"
):
    import numpy as np

    P = np.asarray(P)
    Q = np.asarray(Q)

    criterion = _make_sinkhorn_loss(blur, p=p, backend=backend)
    observed = _sinkhorn_value(P, Q, criterion)

    Z = np.vstack([P, Q])
    n_p = len(P)

    rng = np.random.default_rng(seed)
    null_stats = np.empty(n_permutations)

    for b in range(n_permutations):
        perm = rng.permutation(len(Z))
        P_b = Z[perm[:n_p]]
        Q_b = Z[perm[n_p:]]
        null_stats[b] = _sinkhorn_value(P_b, Q_b, criterion)

    p_value = (
        1 + np.sum(null_stats >= observed)
    ) / (n_permutations + 1)

    return {
        "statistic": float(observed),
        "p_value": float(p_value),
        "reject": bool(p_value < alpha),
        "null_mean": float(null_stats.mean()),
        "null_std": float(null_stats.std())
    }


def sinkhorn_paired_permutation_test(
    P, Q, blur,
    n_permutations=100,
    seed=42,
    alpha=0.05,
    p=2,
    backend="tensorized"
):
    import numpy as np

    P = np.asarray(P)
    Q = np.asarray(Q)

    if P.shape != Q.shape:
        raise ValueError("Paired P and Q must have identical shape.")

    criterion = _make_sinkhorn_loss(blur, p=p, backend=backend)
    observed = _sinkhorn_value(P, Q, criterion)

    rng = np.random.default_rng(seed)
    null_stats = np.empty(n_permutations)

    for b in range(n_permutations):
        swap = rng.random(len(P)) < 0.5

        P_b = P.copy()
        Q_b = Q.copy()

        P_b[swap] = Q[swap]
        Q_b[swap] = P[swap]

        null_stats[b] = _sinkhorn_value(P_b, Q_b, criterion)

    p_value = (
        1 + np.sum(null_stats >= observed)
    ) / (n_permutations + 1)

    return {
        "statistic": float(observed),
        "p_value": float(p_value),
        "reject": bool(p_value < alpha),
        "null_mean": float(null_stats.mean()),
        "null_std": float(null_stats.std())
    }


import numpy as np
from scipy import sparse
from scipy.stats import binomtest
from sklearn.model_selection import GroupShuffleSplit
from sklearn.svm import SVC

def c2st_rbf_paired(
    P,
    Q,
    test_size=0.5,
    n_permutations=100,
    seed=42,
    alpha=0.05,
    C=1.0,
    gamma="scale"
):
    """
    Paired C2ST using an RBF-SVM.

    Uses the same pair split and paired permutation
    significance test as c2st_paired().
    """

    if not isinstance(n_permutations,(int,np.integer)) or n_permutations<1:
        raise ValueError("n_permutations must be a positive integer.")

    if P.shape!=Q.shape:
        raise ValueError("P and Q must have the same shape.")

    n_pairs=P.shape[0]
    pair_indices=np.arange(n_pairs)

    train_idx,test_idx=train_test_split(
        pair_indices,
        test_size=test_size,
        random_state=seed,
        shuffle=True
    )

    P_train=P[train_idx]
    Q_train=Q[train_idx]
    P_test=P[test_idx]
    Q_test=Q[test_idx]

    if sparse.issparse(P) or sparse.issparse(Q):
        X_train=sparse.vstack([P_train,Q_train],format="csr")
        X_test=sparse.vstack([P_test,Q_test],format="csr")
    else:
        P_train=np.asarray(P_train,dtype=float)
        Q_train=np.asarray(Q_train,dtype=float)
        P_test=np.asarray(P_test,dtype=float)
        Q_test=np.asarray(Q_test,dtype=float)
        X_train=np.vstack([P_train,Q_train])
        X_test=np.vstack([P_test,Q_test])

    y_train=np.r_[
        np.zeros(len(train_idx),dtype=int),
        np.ones(len(train_idx),dtype=int)
    ]

    y_test=np.r_[
        np.zeros(len(test_idx),dtype=int),
        np.ones(len(test_idx),dtype=int)
    ]

    clf=SVC(kernel="rbf",C=C,gamma=gamma)
    clf.fit(X_train,y_train)

    predictions=clf.predict(X_test)
    observed_accuracy=float((predictions==y_test).mean())

    n_test_pairs=len(test_idx)
    pred_P=predictions[:n_test_pairs]
    pred_Q=predictions[n_test_pairs:]

    rng=np.random.default_rng(seed)
    null_accuracies=np.empty(n_permutations)

    for b in range(n_permutations):
        swap=rng.integers(0,2,size=n_test_pairs).astype(bool)
        labels_P=swap.astype(int)
        labels_Q=1-labels_P

        correct=(
            np.sum(pred_P==labels_P)
            +np.sum(pred_Q==labels_Q)
        )

        null_accuracies[b]=correct/(2*n_test_pairs)

    p_value=float(
        (1+np.count_nonzero(null_accuracies>=observed_accuracy))
        /(n_permutations+1)
    )

    return {
        "accuracy":observed_accuracy,
        "p_value":p_value,
        "reject_H0":p_value<alpha,
        "n_pairs":n_pairs,
        "n_train_pairs":len(train_idx),
        "n_test_pairs":n_test_pairs,
        "n_test":2*n_test_pairs,
        "n_permutations":n_permutations,
        "seed":seed,
        "C":C,
        "gamma":gamma
    }