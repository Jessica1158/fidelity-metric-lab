from pathlib import Path
import numpy as np
from sklearn.metrics import pairwise_distances

def knn_radii(X,k=3):
    X=np.asarray(X,dtype=float)
    if X.ndim!=2: raise ValueError("X must be a 2D array.")
    if k<1 or k>=len(X): raise ValueError("k must satisfy 1 <= k < n.")
    D=pairwise_distances(X,metric="euclidean")
    np.fill_diagonal(D,np.inf)
    return np.partition(D,k-1,axis=1)[:,k-1]

def manifold_membership(query,reference,k=3):
    query=np.asarray(query,dtype=float)
    reference=np.asarray(reference,dtype=float)
    radii=knn_radii(reference,k=k)
    D=pairwise_distances(query,reference,metric="euclidean")
    return np.any(D<=radii[None,:],axis=1)

def improved_precision_recall(real,generated,k=3,require_equal_size=True):
    real=np.asarray(real,dtype=float)
    generated=np.asarray(generated,dtype=float)
    if real.ndim!=2 or generated.ndim!=2: raise ValueError("Inputs must be 2D arrays.")
    if real.shape[1]!=generated.shape[1]: raise ValueError("Feature dimensions must match.")
    if require_equal_size and len(real)!=len(generated): raise ValueError("Real and generated sets must have equal size.")
    precision_mask=manifold_membership(generated,real,k=k)
    recall_mask=manifold_membership(real,generated,k=k)
    return {"precision":float(precision_mask.mean()),"recall":float(recall_mask.mean()),"n_real":len(real),"n_generated":len(generated),"k":k}

def precision_recall_membership(real,generated,k=3):
    return {"generated_in_real":manifold_membership(generated,real,k=k),"real_in_generated":manifold_membership(real,generated,k=k)}

def repeated_matched_pr(X,n=100,reps=200,k=3,seed=42000):
    X=np.asarray(X,dtype=float)
    if 2*n>len(X): raise ValueError("Need at least 2*n observations.")
    rows=[]
    for rep in range(reps):
        rng=np.random.default_rng(seed+rep)
        idx=rng.choice(len(X),size=2*n,replace=False)
        r=improved_precision_recall(X[idx[:n]],X[idx[n:]],k=k)
        rows.append({"repetition":rep,"precision":r["precision"],"recall":r["recall"]})
    return rows

def repeated_defect_pr(clean,conditions,ids=None,affected_ids=None,n=100,reps=200,k=3,seed=42):
    clean=np.asarray(clean,dtype=float)
    ids=None if ids is None else np.asarray(ids).astype(str)
    rows=[]
    for rep in range(reps):
        rng=np.random.default_rng(seed+rep)
        idx=rng.choice(len(clean),size=2*n,replace=False)
        p_idx,q_idx=idx[:n],idx[n:]
        P=clean[p_idx]
        for severity,Qall in conditions.items():
            Q=np.asarray(Qall,dtype=float)[q_idx]
            r=improved_precision_recall(P,Q,k=k)
            row={"repetition":rep,"severity":float(severity),"precision":r["precision"],"recall":r["recall"]}
            if ids is not None and affected_ids is not None:
                affected=set(map(str,affected_ids[severity]))
                row["n_affected"]=sum(x in affected for x in ids[q_idx])
                row["sample_affected_rate"]=row["n_affected"]/n
            rows.append(row)
    return rows

def fit_knn_manifold(reference,k=3):
    reference=np.asarray(reference,dtype=float)
    return {"reference":reference,"radii":knn_radii(reference,k=k),"k":k}

def membership_from_manifold(query,manifold):
    query=np.asarray(query,dtype=float)
    D=pairwise_distances(query,manifold["reference"],metric="euclidean")
    return np.any(D<=manifold["radii"][None,:],axis=1)