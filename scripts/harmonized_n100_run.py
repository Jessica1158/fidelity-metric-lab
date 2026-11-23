import argparse,json,sqlite3,time,warnings,hashlib,inspect
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

import src.paths as paths
import src.metrics as metrics
import src.defects as defects
import src.representations as representations
import src.io_utils as io_utils

from src.paths import get_project_paths
from src.io_utils import load_embedding_conditions,load_id_file
from src.defects import make_broken_timing_plan,apply_broken_timing_plan
from src.representations import (
    structural_campaign_summary,
    serialize_campaign_dataset,
    encode_serialized_campaigns
)

# ============================================================
# POST-PREREGISTRATION HARMONIZED N=100 SENSITIVITY RUN
#
# This script is a copy of the final confirmatory runner adapted
# to rerun the complete n=100 grid under the same operational
# sampling / seed / test procedures used by the final n=50/n=200
# confirmatory extension.
#
# It DOES NOT replace the historical n=100 anchors used for
# preregistered pruning.
# ============================================================

# ============================================================
# FROZEN CONFIG
# ============================================================

P=get_project_paths()
DATA=P["processed"]
RESULTS=P["results"]

PREREG=RESULTS/"week8"/"preregistration"
OUT=RESULTS/"week12"/"harmonized_n100"
CACHE=OUT/"frozen_reference_cache"

OUT.mkdir(parents=True,exist_ok=True)
CACHE.mkdir(parents=True,exist_ok=True)

REPS=200
N_PERM=100
ALPHA=.05
BASE_SEED=42
SEVERITIES=[.10,.30,.60]
TESTS=["C2ST-Linear","C2ST-RBF","MMD","Energy","Sinkhorn"]

MANIFEST=OUT/"harmonized_n100_execution_manifest.csv"
SAMPLES=OUT/"harmonized_n100_sampling_plan.csv"
CONFIG=OUT/"harmonized_n100_run_config.json"
DB_PATH=OUT/"harmonized_n100_checkpoint.sqlite"
ERROR_LOG=OUT/"harmonized_n100_errors.log"
MANIFEST_HASH=OUT/"harmonized_n100_manifest.sha256"
SAMPLES_HASH=OUT/"harmonized_n100_sampling_plan.sha256"

BT_FEATURES=[
    "n_messages","campaign_duration","mean_time_gap","median_time_gap",
    "std_time_gap","n_channels","channel_switch_rate","channel_entropy",
    "dominant_channel_share","mean_message_length","std_message_length",
    "mean_token_count","std_token_count"
]

SCENARIO_REP={
    "Template Repetition":"Template TF-IDF",
    "Broken Timing":"Broken Timing Structural",
    "Entity Inconsistency":"Campaign MiniLM",
    "Missing Cross-Channel Reference":"Campaign MiniLM",
    "Over-Polished Language":"Message MiniLM"
}

NULL_REPS=[
    "Template TF-IDF",
    "Broken Timing Structural",
    "Campaign MiniLM",
    "Message MiniLM"
]

SCALES={
    "Template TF-IDF":{
        "sigma":1.4142135623730951,
        "blur":0.141421
    },
    "Broken Timing Structural":{
        "sigma":4.829537929442184,
        "blur":0.482954
    },
    "Campaign MiniLM":{
        "sigma":0.4312044032354748,
        "blur":0.043720
    },
    "Message MiniLM":{
        "sigma":1.3281240463256836,
        "blur":0.132812
    }
}


# ============================================================
# HELPERS
# ============================================================

def sha256(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""):
            h.update(chunk)
    return h.hexdigest()


def canon_scenario(x):
    x=str(x).lower()
    if "template" in x:return "Template Repetition"
    if "broken" in x:return "Broken Timing"
    if "entity" in x:return "Entity Inconsistency"
    if "missing" in x:return "Missing Cross-Channel Reference"
    if "polish" in x:return "Over-Polished Language"
    raise ValueError(x)


def canon_test(x):
    x=str(x).lower().replace("_","-")
    if "rbf" in x:return "C2ST-RBF"
    if "c2st" in x or "linear" in x:return "C2ST-Linear"
    if "mmd" in x:return "MMD"
    if "energy" in x:return "Energy"
    if "sink" in x:return "Sinkhorn"
    raise ValueError(x)


def call_filtered(fn,P,Q,**kwargs):
    sig=inspect.signature(fn)
    return fn(P,Q,**{k:v for k,v in kwargs.items() if k in sig.parameters})


def extract_result(r):
    p=r.get("p_value",r.get("p"))
    if p is None:
        raise KeyError("Metric result has no p-value.")

    stat=r.get(
        "accuracy",
        r.get("statistic",r.get("mmd2",r.get("energy",np.nan)))
    )

    reject=r.get("reject_H0",r.get("reject",float(p)<ALPHA))

    return {
        "statistic":float(stat),
        "p_value":float(p),
        "reject":int(bool(reject)),
        "null_mean":float(r.get("null_mean",np.nan)),
        "null_std":float(r.get("null_std",np.nan))
    }


def run_metric(test,P,Q,representation,seed):
    common={
        "n_permutations":N_PERM,
        "seed":int(seed),
        "alpha":ALPHA
    }

    if test=="C2ST-Linear":
        r=call_filtered(
            metrics.c2st_paired,
            P,Q,test_size=.5,**common
        )

    elif test=="C2ST-RBF":
        r=call_filtered(
            metrics.c2st_rbf_paired,
            P,Q,test_size=.5,C=1.0,gamma="scale",**common
        )

    elif test=="MMD":
        r=call_filtered(
            metrics.mmd_paired_permutation_test,
            P,Q,sigma=SCALES[representation]["sigma"],**common
        )

    elif test=="Energy":
        r=call_filtered(
            metrics.energy_paired_permutation_test,
            P,Q,**common
        )

    elif test=="Sinkhorn":
        r=call_filtered(
            metrics.sinkhorn_paired_permutation_test,
            P,Q,blur=SCALES[representation]["blur"],**common
        )

    else:
        raise ValueError(test)

    return extract_result(r)


# ============================================================
# LOAD / RECONSTRUCT FROZEN EVALUATION DATA
# ============================================================

def load_evaluation_data():
    entity=load_embedding_conditions(DATA,"entity",(10,30,60))
    missing=load_embedding_conditions(DATA,"missing_reference",(10,30,60))
    op=load_embedding_conditions(DATA,"over_polished",(10,30,60))

    entity_ids=np.asarray(
        load_id_file(
            DATA,
            "entity_embedding_campaign_ids.csv",
            "campaign_id"
        )
    ).astype(str)

    missing_ids=np.asarray(
        load_id_file(
            DATA,
            "missing_reference_embedding_campaign_ids.csv",
            "campaign_id"
        )
    ).astype(str)

    op_ids=np.asarray(
        load_id_file(
            DATA,
            "over_polished_embedding_message_ids.csv",
            "message_id"
        )
    ).astype(str)

    tpl={"clean":load_npz(DATA/"template_tfidf_clean.npz").toarray()}
    for s in SEVERITIES:
        tpl[s]=load_npz(
            DATA/f"template_tfidf_{int(s*100)}pct.npz"
        ).toarray()

    tpl_ids=pd.read_csv(
        DATA/"template_repetition_formal_eval_ids.csv"
    )["message_id"].astype(str).to_numpy()

    # ---------------- Broken Timing exact reconstruction ----------------

    clean_campaigns=pd.read_csv(
        DATA/"clean_structured_campaigns.csv",
        parse_dates=["timestamp"]
    )

    split=pd.read_csv(DATA/"structural_campaign_split.csv")

    train_ids=set(
        split.loc[split["split"]=="train","campaign_id"].astype(str)
    )
    eval_ids=set(
        split.loc[split["split"]=="evaluation","campaign_id"].astype(str)
    )

    cc=clean_campaigns.copy()
    cc["campaign_id"]=cc["campaign_id"].astype(str)

    clean_train=cc[cc["campaign_id"].isin(train_ids)].copy()
    clean_eval=cc[cc["campaign_id"].isin(eval_ids)].copy()

    assert clean_train["campaign_id"].nunique()==700
    assert clean_eval["campaign_id"].nunique()==300

    plan=make_broken_timing_plan(
        clean_eval,
        severities=(.10,.30,.60),
        target_gap_minutes=-5.0,
        seed=42
    )

    conditions={"clean":clean_eval}

    for s in SEVERITIES:
        x=apply_broken_timing_plan(clean_eval,plan,s)
        if isinstance(x,tuple):x=x[0]
        if isinstance(x,dict):x=x.get("defective",x.get("data",x))
        conditions[s]=x

    train_raw=structural_campaign_summary(clean_train)
    raw={k:structural_campaign_summary(v) for k,v in conditions.items()}

    scaler=StandardScaler().fit(train_raw[BT_FEATURES])

    bt_ids=raw["clean"]["campaign_id"].astype(str).to_numpy()
    bt={}

    for k,d in raw.items():
        aligned=d.set_index(
            d["campaign_id"].astype(str)
        )
        aligned.index=aligned.index.astype(str)
        bt[k]=scaler.transform(aligned.loc[bt_ids,BT_FEATURES])

    # ---------------- validations ----------------

    D={
        "Template Repetition":{"ids":tpl_ids,**tpl},
        "Broken Timing":{"ids":bt_ids,**bt},
        "Entity Inconsistency":{"ids":entity_ids,**entity},
        "Missing Cross-Channel Reference":{"ids":missing_ids,**missing},
        "Over-Polished Language":{"ids":op_ids,**op}
    }

    for scenario,x in D.items():
        assert len(x["ids"])==300
        assert len(set(x["ids"]))==300
        for k in ["clean",*SEVERITIES]:
            A=np.asarray(x[k])
            assert A.shape[0]==300
            assert np.isfinite(A).all()

    # exact Broken Timing validation already established in pilot
    expected={.10:1.507846,.30:1.614294,.60:1.624370}

    for s,target in expected.items():
        dist=np.linalg.norm(bt[s]-bt["clean"],axis=1)
        obs=dist[dist>1e-12].mean()
        assert np.isclose(obs,target,atol=1e-5)

    sigma=metrics.median_heuristic_sigma(
        scaler.transform(train_raw[BT_FEATURES])
    )
    assert np.isclose(sigma,4.829537929442184,atol=1e-8)

    return D,clean_train,train_raw,scaler


# ============================================================
# BUILD / LOAD 700-UNIT CLEAN REFERENCE REPRESENTATIONS
# ============================================================

def prepare_reference_cache(clean_train,train_raw,scaler):
    from sentence_transformers import SentenceTransformer

    print("Preparing frozen 700-unit clean reference representations...")

    # --------------------------------------------------------
    # 1. Structural 13D
    # --------------------------------------------------------

    struct_ids=train_raw["campaign_id"].astype(str).to_numpy()
    struct_X=scaler.transform(train_raw[BT_FEATURES])

    assert struct_X.shape==(700,13)
    assert np.isfinite(struct_X).all()

    np.save(CACHE/"structural_reference.npy",struct_X)
    pd.DataFrame({"id":struct_ids}).to_csv(
        CACHE/"structural_reference_ids.csv",index=False
    )

    # --------------------------------------------------------
    # 2. Campaign MiniLM
    # --------------------------------------------------------

    with open(RESULTS/"semantic_embedding_configuration.json") as f:
        sem_cfg=json.load(f)

    model=SentenceTransformer(
        sem_cfg["model"],
        revision=sem_cfg["revision"]
    )

    serialized=serialize_campaign_dataset(clean_train)

    enc=encode_serialized_campaigns(
        serialized,
        model,
        batch_size=64
    )

    campaign_X=enc[0] if isinstance(enc,tuple) else enc
    campaign_ids=serialized["campaign_id"].astype(str).to_numpy()

    assert campaign_X.shape==(700,384)
    assert np.isfinite(campaign_X).all()

    sigma=metrics.median_heuristic_sigma(campaign_X)
    assert np.isclose(
        sigma,
        0.4312044032354748,
        atol=1e-8
    )

    np.save(CACHE/"campaign_minilm_reference.npy",campaign_X)
    pd.DataFrame({"id":campaign_ids}).to_csv(
        CACHE/"campaign_minilm_reference_ids.csv",index=False
    )

    # --------------------------------------------------------
    # 3. SMS split used for Message MiniLM + Template TF-IDF
    # --------------------------------------------------------

    sms=pd.read_csv(DATA/"over_polished_sms_split.csv")
    sms["message_id"]=sms["message_id"].astype(str)

    sms_ref=sms[sms["split"]=="reference"].copy()
    sms_eval=sms[sms["split"]=="evaluation"].copy()

    assert len(sms_ref)==700
    assert len(sms_eval)==300
    assert not set(sms_ref.message_id)&set(sms_eval.message_id)

    # --------------------------------------------------------
    # 4. Message MiniLM
    # --------------------------------------------------------

    op_X=model.encode(
        sms_ref["text"].astype(str).tolist(),
        batch_size=64,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    op_X=np.asarray(op_X)

    assert op_X.shape==(700,384)
    assert np.isfinite(op_X).all()

    sigma=metrics.median_heuristic_sigma(op_X)
    assert np.isclose(
        sigma,
        1.3281240463256836,
        atol=1e-8
    )

    np.save(CACHE/"message_minilm_reference.npy",op_X)
    pd.DataFrame({
        "id":sms_ref["message_id"].astype(str)
    }).to_csv(
        CACHE/"message_minilm_reference_ids.csv",index=False
    )

    # --------------------------------------------------------
    # 5. Template TF-IDF
    # Reconstruct from frozen SMS reference, then REQUIRE
    # reproduction of frozen evaluation representation.
    # --------------------------------------------------------

    tpl_ids=pd.read_csv(
        DATA/"template_repetition_formal_eval_ids.csv"
    )["message_id"].astype(str).to_numpy()

    if set(tpl_ids)!=set(sms_eval["message_id"].astype(str)):
        raise RuntimeError(
            "Template evaluation IDs do not match the frozen SMS split. "
            "Do not proceed until the exact Template reference split is recovered."
        )

    vectorizer=TfidfVectorizer(
        max_features=5000,
        ngram_range=(1,2),
        norm="l2"
    )

    tpl_ref=vectorizer.fit_transform(
        sms_ref["text"].astype(str)
    )

    aligned_eval=(
        sms_eval.set_index("message_id")
        .loc[tpl_ids]
    )

    reconstructed_eval=vectorizer.transform(
        aligned_eval["text"].astype(str)
    ).toarray()

    frozen_eval=load_npz(
        DATA/"template_tfidf_clean.npz"
    ).toarray()

    if reconstructed_eval.shape!=frozen_eval.shape:
        raise RuntimeError(
            "Template TF-IDF reconstruction shape differs from frozen artifact."
        )

    max_diff=np.max(np.abs(reconstructed_eval-frozen_eval))

    print("Template reconstruction max abs diff:",max_diff)

    if not np.allclose(
        reconstructed_eval,
        frozen_eval,
        rtol=1e-7,
        atol=1e-10
    ):
        raise RuntimeError(
            "Template TF-IDF reconstruction did not reproduce the frozen "
            "evaluation artifact. Recover the exact original vectorizer/split "
            "before the formal run."
        )

    tpl_ref=tpl_ref.toarray()

    assert tpl_ref.shape==(700,5000)
    assert np.isfinite(tpl_ref).all()

    np.save(CACHE/"template_tfidf_reference.npy",tpl_ref)

    pd.DataFrame({
        "id":sms_ref["message_id"].astype(str)
    }).to_csv(
        CACHE/"template_tfidf_reference_ids.csv",index=False
    )

    print("Frozen reference cache prepared successfully.")


def load_reference_cache():
    files={
        "Template TF-IDF":(
            "template_tfidf_reference.npy",
            "template_tfidf_reference_ids.csv"
        ),
        "Broken Timing Structural":(
            "structural_reference.npy",
            "structural_reference_ids.csv"
        ),
        "Campaign MiniLM":(
            "campaign_minilm_reference.npy",
            "campaign_minilm_reference_ids.csv"
        ),
        "Message MiniLM":(
            "message_minilm_reference.npy",
            "message_minilm_reference_ids.csv"
        )
    }

    R={}

    for name,(xf,idf) in files.items():
        xp=CACHE/xf
        ip=CACHE/idf

        if not xp.exists() or not ip.exists():
            raise FileNotFoundError(
                f"Missing frozen reference cache for {name}. "
                "Run: python scripts/harmonized_n100_run.py prepare"
            )

        X=np.load(xp)
        ids=pd.read_csv(ip)["id"].astype(str).to_numpy()

        assert len(X)==len(ids)==700
        assert len(set(ids))==700
        assert np.isfinite(X).all()

        R[name]={"X":X,"ids":ids}

    return R


# ============================================================
# HARMONIZED N=100 FULL DESIGN
# ============================================================

def load_n100_design():
    """
    Build the complete post-preregistration harmonized n=100 grid.

    This is a sensitivity / comparability analysis, not a replacement
    for the historical n=100 anchors used for preregistered pruning.

    Grid:
        5 scenarios x 3 severities x 5 tests = 75 power cells
    """
    rows=[]

    for scenario in SCENARIO_REP:
        for severity in SEVERITIES:
            for test in TESTS:
                rows.append({
                    "scenario":scenario,
                    "severity":float(severity),
                    "test":test,
                    "N":100
                })

    d=pd.DataFrame(rows)

    assert len(d)==75
    assert set(d["scenario"])==set(SCENARIO_REP.keys())
    assert set(d["severity"])==set(SEVERITIES)
    assert set(d["test"])==set(TESTS)
    assert set(d["N"])=={100}
    assert not d.duplicated(
        ["scenario","severity","test","N"]
    ).any()

    return d


# ============================================================
# MANIFEST + FROZEN SAMPLING PLAN
# ============================================================

def build_manifests(D,R):
    if MANIFEST.exists() or SAMPLES.exists():
        raise RuntimeError(
            "Manifest already exists. It is treated as frozen. "
            "Do not overwrite it."
        )

    design=load_n100_design()

    manifest=[]

    # ---------------- POWER ----------------

    for _,c in design.iterrows():
        rep_name=SCENARIO_REP[c.scenario]

        for rep in range(REPS):
            sample_key=f"P|{c.scenario}|N{int(c.N)}|R{rep:03d}"

            run_id=(
                f"POWER|{c.scenario}|S{c.severity:.2f}|"
                f"N{int(c.N)}|{c.test}|R{rep:03d}"
            )

            manifest.append({
                "run_id":run_id,
                "kind":"power",
                "scenario":c.scenario,
                "representation":rep_name,
                "severity":c.severity,
                "N":int(c.N),
                "test":c.test,
                "rep":rep,
                "sampling_seed":BASE_SEED,
                "test_seed":BASE_SEED+rep,
                "sample_key":sample_key
            })

    # ---------------- NULL ----------------

    for rep_name in NULL_REPS:
        for n in [100]:
            for test in TESTS:
                for rep in range(REPS):
                    sample_key=f"N|{rep_name}|N{n}|R{rep:03d}"

                    run_id=(
                        f"NULL|{rep_name}|N{n}|{test}|R{rep:03d}"
                    )

                    manifest.append({
                        "run_id":run_id,
                        "kind":"null",
                        "scenario":"",
                        "representation":rep_name,
                        "severity":np.nan,
                        "N":n,
                        "test":test,
                        "rep":rep,
                        "sampling_seed":BASE_SEED,
                        "test_seed":BASE_SEED+rep,
                        "sample_key":sample_key
                    })

    M=pd.DataFrame(manifest)

    # ---------------- exact manifest validation ----------------

    power=M[M.kind=="power"]
    null=M[M.kind=="null"]

    assert len(power)==75*REPS
    assert len(null)==20*REPS
    assert len(M)==19000

    assert power.groupby(
        ["scenario","severity","test","N"]
    ).ngroups==75

    assert null.groupby(
        ["representation","test","N"]
    ).ngroups==20

    assert M.run_id.is_unique

    # ---------------- sampling plans ----------------

    plans=[]

    # Power: one sampled ID set per Scenario x N x repetition.
    # Same IDs reused across severities and test configurations.

    for scenario in design.scenario.unique():
        ids=np.asarray(D[scenario]["ids"]).astype(str)

        for n in sorted(
            design.loc[design.scenario==scenario,"N"].unique()
        ):
            rng=np.random.default_rng(BASE_SEED)

            for rep in range(REPS):
                sampled=rng.choice(
                    ids,
                    size=int(n),
                    replace=False
                )

                plans.append({
                    "sample_key":f"P|{scenario}|N{int(n)}|R{rep:03d}",
                    "kind":"power",
                    "scenario":scenario,
                    "representation":SCENARIO_REP[scenario],
                    "N":int(n),
                    "rep":rep,
                    "sampling_seed":BASE_SEED,
                    "P_ids":json.dumps(sampled.tolist()),
                    "Q_ids":json.dumps(sampled.tolist())
                })

    # Null: revised formal rule.
    #
    # 700 clean reference units
    # select 2*N DISTINCT units without replacement
    # first N = P
    # second N = Q
    #
    # This makes n=200 feasible because 400 distinct units
    # are available from the frozen 700-unit reference pool.

    for rep_name in NULL_REPS:
        ids=np.asarray(R[rep_name]["ids"]).astype(str)

        assert len(ids)==700

        for n in [100]:
            rng=np.random.default_rng(BASE_SEED)

            for rep in range(REPS):
                selected=rng.choice(
                    ids,
                    size=2*n,
                    replace=False
                )

                A=selected[:n]
                B=selected[n:]

                assert len(set(A)&set(B))==0

                plans.append({
                    "sample_key":f"N|{rep_name}|N{n}|R{rep:03d}",
                    "kind":"null",
                    "scenario":"",
                    "representation":rep_name,
                    "N":n,
                    "rep":rep,
                    "sampling_seed":BASE_SEED,
                    "P_ids":json.dumps(A.tolist()),
                    "Q_ids":json.dumps(B.tolist())
                })

    S=pd.DataFrame(plans)

    assert S.sample_key.is_unique
    assert set(M.sample_key)<=set(S.sample_key)

    # Validate every stored sample
    for _,r in S.iterrows():
        A=json.loads(r.P_ids)
        B=json.loads(r.Q_ids)

        assert len(A)==r.N
        assert len(B)==r.N

        if r.kind=="power":
            assert A==B
        else:
            assert not set(A)&set(B)

    M.to_csv(MANIFEST,index=False)
    S.to_csv(SAMPLES,index=False)

    cfg={
        "analysis_type":"posthoc_harmonized_n100_sensitivity",
        "status":"post-preregistration harmonization analysis",
        "purpose":(
            "Assess comparability of historical n=100 anchors with the "
            "final n=50/n=200 confirmatory procedure."
        ),
        "repetitions":REPS,
        "n_permutations":N_PERM,
        "alpha":ALPHA,
        "base_seed":BASE_SEED,
        "sample_size":100,
        "power_cells":75,
        "null_cells":20,
        "power_runs":15000,
        "null_runs":4000,
        "total_runs":19000,
        "null_sampling_rule":{
            "pool_size":700,
            "rule":(
                "sample 2*N distinct clean reference units without replacement; "
                "first N=P, second N=Q"
            )
        },
        "scales":SCALES
    }

    with open(CONFIG,"w") as f:
        json.dump(cfg,f,indent=2)

    with open(MANIFEST_HASH,"w") as f:
        f.write(sha256(MANIFEST))

    with open(SAMPLES_HASH,"w") as f:
        f.write(sha256(SAMPLES))

    print("Manifest frozen:")
    print("  Power runs:",len(power))
    print("  Null runs :",len(null))
    print("  Total     :",len(M))
    print("  Power cells:",power.groupby(
        ["scenario","severity","test","N"]
    ).ngroups)
    print("  Null cells :",null.groupby(
        ["representation","test","N"]
    ).ngroups)


def verify_frozen_files():
    if not MANIFEST.exists() or not SAMPLES.exists():
        raise FileNotFoundError(
            "Frozen manifest not found. Run prepare first."
        )

    expected=MANIFEST_HASH.read_text().strip()
    expected_s=SAMPLES_HASH.read_text().strip()

    if sha256(MANIFEST)!=expected:
        raise RuntimeError("Execution manifest changed after freezing.")

    if sha256(SAMPLES)!=expected_s:
        raise RuntimeError("Sampling plan changed after freezing.")


# ============================================================
# DATA ACCESS BY STORED IDS
# ============================================================

def rows_by_ids(X,ids,wanted):
    ids=np.asarray(ids).astype(str)
    pos={x:i for i,x in enumerate(ids)}

    try:
        idx=[pos[str(x)] for x in wanted]
    except KeyError as e:
        raise KeyError(f"Sample ID absent from frozen representation: {e}")

    return np.asarray(X)[idx]


def get_PQ(row,D,R,S_lookup):
    p=S_lookup[row.sample_key]

    P_ids=json.loads(p["P_ids"])
    Q_ids=json.loads(p["Q_ids"])

    if row.kind=="power":
        d=D[row.scenario]

        P=rows_by_ids(
            d["clean"],
            d["ids"],
            P_ids
        )

        Q=rows_by_ids(
            d[float(row.severity)],
            d["ids"],
            Q_ids
        )

    else:
        r=R[row.representation]

        P=rows_by_ids(
            r["X"],
            r["ids"],
            P_ids
        )

        Q=rows_by_ids(
            r["X"],
            r["ids"],
            Q_ids
        )

        # formal clean-null disjointness
        assert not set(P_ids)&set(Q_ids)

    assert P.shape[0]==Q.shape[0]==int(row.N)
    assert np.isfinite(P).all()
    assert np.isfinite(Q).all()

    return P,Q


# ============================================================
# SQLITE CHECKPOINT / RESUME
# ============================================================

def connect_db():
    con=sqlite3.connect(DB_PATH)

    con.execute("""
    CREATE TABLE IF NOT EXISTS results(
        run_id TEXT PRIMARY KEY,
        kind TEXT,
        scenario TEXT,
        representation TEXT,
        severity REAL,
        N INTEGER,
        test TEXT,
        rep INTEGER,
        sampling_seed INTEGER,
        test_seed INTEGER,
        statistic REAL,
        p_value REAL,
        reject INTEGER,
        null_mean REAL,
        null_std REAL,
        runtime REAL,
        warning_text TEXT,
        completed_at TEXT
    )
    """)

    con.execute("""
    CREATE TABLE IF NOT EXISTS errors(
        run_id TEXT,
        error TEXT,
        created_at TEXT
    )
    """)

    con.commit()
    return con


def completed_ids(con):
    return {
        x[0]
        for x in con.execute("SELECT run_id FROM results").fetchall()
    }


def save_result(con,row,r,runtime,warning_text):
    con.execute("""
    INSERT INTO results VALUES(
        ?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,datetime('now')
    )
    """,(
        row.run_id,
        row.kind,
        row.scenario,
        row.representation,
        None if pd.isna(row.severity) else float(row.severity),
        int(row.N),
        row.test,
        int(row.rep),
        int(row.sampling_seed),
        int(row.test_seed),
        r["statistic"],
        r["p_value"],
        r["reject"],
        r["null_mean"],
        r["null_std"],
        float(runtime),
        warning_text
    ))

    con.commit()


def save_error(con,row,e):
    con.execute(
        "INSERT INTO errors VALUES(?,?,datetime('now'))",
        (row.run_id,repr(e))
    )
    con.commit()

    with open(ERROR_LOG,"a") as f:
        f.write(f"{row.run_id}\t{repr(e)}\n")


def export_results(con):
    df=pd.read_sql_query(
        "SELECT * FROM results ORDER BY kind,N,scenario,severity,test,rep",
        con
    )

    df[df.kind=="power"].to_csv(
        OUT/"harmonized_n100_power_runs.csv",
        index=False
    )

    df[df.kind=="null"].to_csv(
        OUT/"harmonized_n100_clean_null_runs.csv",
        index=False
    )


# ============================================================
# SMOKE TEST — NO FORMAL RESULTS SAVED
# ============================================================

def smoke():
    verify_frozen_files()

    D,_,_,_=load_evaluation_data()
    R=load_reference_cache()

    M=pd.read_csv(
    MANIFEST,
    keep_default_na=False,
    na_values=[""]
    )
    S=pd.read_csv(
    SAMPLES,
    keep_default_na=False,
    na_values=[""]
    )

    S_lookup=S.set_index("sample_key").to_dict("index")

    checks=[]

    # one route for every representation/test
    for rep_name in NULL_REPS:
        for test in TESTS:
            row=M[
                (M.kind=="null")&
                (M.representation==rep_name)&
                (M.N==100)&
                (M.test==test)&
                (M.rep==0)
            ].iloc[0]

            P0,Q0=get_PQ(row,D,R,S_lookup)

            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                r=run_metric(
                    row.test,
                    P0,Q0,
                    row.representation,
                    row.test_seed
                )

            assert np.isfinite(r["p_value"])
            assert 0<=r["p_value"]<=1
            assert len(w)==0

            checks.append((rep_name,test))

    # one power route per scenario/test
    for scenario in D:
        sub=M[
            (M.kind=="power")&
            (M.scenario==scenario)&
            (M.rep==0)
        ]

        for test in sorted(sub.test.unique()):
            row=sub[sub.test==test].iloc[0]
            P0,Q0=get_PQ(row,D,R,S_lookup)

            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                r=run_metric(
                    row.test,
                    P0,Q0,
                    row.representation,
                    row.test_seed
                )

            assert np.isfinite(r["p_value"])
            assert 0<=r["p_value"]<=1
            assert len(w)==0

            checks.append((scenario,test))

    print("FINAL SMOKE CHECK PASSED")
    print("Routes checked:",len(checks))
    print("No formal results were saved.")


# ============================================================
# FORMAL EXECUTION
# ============================================================

def select_phase(M,phase):
    if phase=="all":
        return M

    if phase=="power100":
        return M[(M.kind=="power")&(M.N==100)]

    if phase=="null100":
        return M[(M.kind=="null")&(M.N==100)]

    raise ValueError(phase)


def run_formal(phase):
    verify_frozen_files()

    D,_,_,_=load_evaluation_data()
    R=load_reference_cache()

    M=pd.read_csv(
    MANIFEST,
    keep_default_na=False,
    na_values=[""]
    )
    S=pd.read_csv(
    SAMPLES,
    keep_default_na=False,
    na_values=[""]
    )

    S_lookup=S.set_index("sample_key").to_dict("index")

    # deterministic execution order
    kind_order={"power":0,"null":1}

    M["_kind_order"]=M.kind.map(kind_order)

    M=M.sort_values(
        ["_kind_order","N","scenario","severity","test","rep"],
        na_position="last"
    ).drop(columns="_kind_order")

    M=select_phase(M,phase).reset_index(drop=True)

    con=connect_db()
    done=completed_ids(con)

    pending=M[~M.run_id.isin(done)].copy()

    print("Phase:",phase)
    print("Manifest rows:",len(M))
    print("Already complete:",len(M)-len(pending))
    print("Pending:",len(pending))

    if len(pending)==0:
        print("Nothing to run.")
        export_results(con)
        con.close()
        return

    start_all=time.perf_counter()

    for i,row in pending.iterrows():
        try:
            P0,Q0=get_PQ(row,D,R,S_lookup)

            t=time.perf_counter()

            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")

                result=run_metric(
                    row.test,
                    P0,Q0,
                    row.representation,
                    row.test_seed
                )

            runtime=time.perf_counter()-t
            warning_text=" | ".join(str(x.message) for x in w)

            if not np.isfinite(result["p_value"]):
                raise RuntimeError("Non-finite p-value.")

            if not 0<=result["p_value"]<=1:
                raise RuntimeError(
                    f"Invalid p-value: {result['p_value']}"
                )

            save_result(
                con,row,result,runtime,warning_text
            )

        except Exception as e:
            save_error(con,row,e)
            export_results(con)
            con.close()

            print("\nFORMAL RUN STOPPED")
            print("Failed run:",row.run_id)
            print("Error:",repr(e))
            print("Checkpoint preserved.")
            raise

        completed=i+1

        if completed%100==0 or completed==len(pending):
            elapsed=(time.perf_counter()-start_all)/60

            print(
                f"{completed}/{len(pending)} pending runs completed "
                f"| elapsed {elapsed:.2f} min"
            )

    export_results(con)

    n_results=con.execute(
        "SELECT COUNT(*) FROM results"
    ).fetchone()[0]

    n_errors=con.execute(
        "SELECT COUNT(*) FROM errors"
    ).fetchone()[0]

    n_warnings=con.execute(
        "SELECT COUNT(*) FROM results "
        "WHERE warning_text IS NOT NULL AND warning_text!=''"
    ).fetchone()[0]

    con.close()

    print("\nPHASE COMPLETE")
    print("Total checkpointed results:",n_results)
    print("Logged errors:",n_errors)
    print("Runs with warnings:",n_warnings)
    print("No power or Type I summaries were calculated.")


# ============================================================
# STATUS — COUNTS ONLY, NO SCIENTIFIC RESULTS
# ============================================================

def status():
    verify_frozen_files()

    M=pd.read_csv(
    MANIFEST,
    keep_default_na=False,
    na_values=[""]
    )

    if not DB_PATH.exists():
        print("Formal checkpoint database does not exist yet.")
        print("Expected total runs:",len(M))
        return

    con=connect_db()

    x=pd.read_sql_query("""
        SELECT kind,N,COUNT(*) AS completed
        FROM results
        GROUP BY kind,N
        ORDER BY kind,N
    """,con)

    errors=con.execute(
        "SELECT COUNT(*) FROM errors"
    ).fetchone()[0]

    warnings_n=con.execute(
        "SELECT COUNT(*) FROM results "
        "WHERE warning_text IS NOT NULL AND warning_text!=''"
    ).fetchone()[0]

    total=con.execute(
        "SELECT COUNT(*) FROM results"
    ).fetchone()[0]

    con.close()

    print(x.to_string(index=False))
    print("\nCompleted:",total,"/",len(M))
    print("Errors:",errors)
    print("Runs with warnings:",warnings_n)
    print("Scientific outcomes intentionally not summarized.")


# ============================================================
# PREPARE
# ============================================================

def prepare():
    if MANIFEST.exists() or SAMPLES.exists():
        raise RuntimeError(
            "Frozen harmonized n=100 manifest already exists. "
            "Preparation will not overwrite it."
        )

    D,clean_train,train_raw,scaler=load_evaluation_data()

    prepare_reference_cache(
        clean_train,
        train_raw,
        scaler
    )

    R=load_reference_cache()

    build_manifests(D,R)

    print("\nPREPARATION COMPLETE")
    print("Reference cache: validated")
    print("Execution manifest: frozen")
    print("Sampling plan: frozen")
    print("Ready for final smoke check.")


# ============================================================
# CLI
# ============================================================

def main():
    parser=argparse.ArgumentParser()

    sub=parser.add_subparsers(
        dest="command",
        required=True
    )

    sub.add_parser("prepare")
    sub.add_parser("smoke")
    sub.add_parser("status")

    run_parser=sub.add_parser("run")
    run_parser.add_argument(
        "--phase",
        choices=[
            "all",
            "power100",
            "null100"
        ],
        default="all"
    )

    args=parser.parse_args()

    if args.command=="prepare":
        prepare()

    elif args.command=="smoke":
        smoke()

    elif args.command=="run":
        run_formal(args.phase)

    elif args.command=="status":
        status()


if __name__=="__main__":
    main()