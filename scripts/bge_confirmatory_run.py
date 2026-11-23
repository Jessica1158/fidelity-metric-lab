# imports + cpnfiguration 
import argparse,json,sqlite3,time,warnings
from pathlib import Path
import numpy as np,pandas as pd

import confirmatory_run as core
import src.metrics as metrics

from src.paths import get_project_paths
from src.representations import serialize_campaign_dataset,encode_serialized_campaigns
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import pdist

P=get_project_paths()
DATA=P["processed"]
RESULTS=P["results"]

BGE=DATA/"robustness_bge"
OUT=RESULTS/"week13"/"bge_confirmatory"
OUT.mkdir(parents=True,exist_ok=True)

MANIFEST=OUT/"bge_execution_manifest.csv"
SAMPLES=OUT/"bge_sampling_plan.csv"
DB=OUT/"bge_checkpoint.sqlite"

N=100
REPS=200
PERMS=100
ALPHA=.05
BASE_SEED=42

SEV=[.10,.30,.60]
TESTS=["C2ST-Linear","C2ST-RBF","MMD","Energy","Sinkhorn"]
SCENARIOS=["Entity Inconsistency","Missing Cross-Channel Reference"]
REP="BGE Campaign"

SIGMA=0.3873605130049743
MEDIAN=0.3873605130049747
BLUR=0.038736051300497475

core.SCALES[REP]={"sigma":SIGMA,"blur":BLUR}

# load and validate BGE + reference 700
def load_eval():
    specs={
        "Entity Inconsistency":("entity_bge","entity_bge_campaign_ids.csv"),
        "Missing Cross-Channel Reference":(
            "missing_reference_bge",
            "missing_reference_bge_campaign_ids.csv"
        )
    }

    D={}

    for scenario,(prefix,idfile) in specs.items():
        ids=pd.read_csv(BGE/idfile)["campaign_id"].astype(str).to_numpy()

        x={"ids":ids}
        x["clean"]=np.load(BGE/f"{prefix}_clean.npy")

        for s in SEV:
            x[s]=np.load(BGE/f"{prefix}_{int(s*100)}pct.npy")

        assert len(ids)==300 and len(set(ids))==300

        for k in ["clean",*SEV]:
            assert x[k].shape==(300,768)
            assert np.isfinite(x[k]).all()
            assert np.allclose(np.linalg.norm(x[k],axis=1),1,atol=1e-5)

        D[scenario]=x

    A=D[SCENARIOS[0]]
    B=D[SCENARIOS[1]]

    assert np.array_equal(A["ids"],B["ids"])
    assert np.allclose(A["clean"],B["clean"],atol=1e-7)

    print("BGE evaluation artifacts validated.")
    return D


def load_or_build_reference():
    xp=OUT/"bge_reference_700.npy"
    ip=OUT/"bge_reference_700_ids.csv"

    if xp.exists() and ip.exists():
        return np.load(xp),pd.read_csv(ip)["id"].astype(str).to_numpy()

    cfg=json.load(open(BGE/"entity_bge_configuration.json"))

    model=SentenceTransformer(
        cfg["model"],
        revision=cfg["revision"]
    )

    campaigns=pd.read_csv(
        DATA/"clean_structured_campaigns.csv",
        parse_dates=["timestamp"]
    )

    split=pd.read_csv(DATA/"structural_campaign_split.csv")

    train_ids=set(
        split.loc[split["split"]=="train","campaign_id"].astype(str)
    )

    campaigns["campaign_id"]=campaigns["campaign_id"].astype(str)

    ref=campaigns[
        campaigns["campaign_id"].isin(train_ids)
    ].copy()

    assert ref["campaign_id"].nunique()==700

    serialized=serialize_campaign_dataset(ref)

    out=encode_serialized_campaigns(
        serialized,
        model,
        batch_size=cfg["batch_size"]
    )

    X=out[0] if isinstance(out,tuple) else out
    ids=serialized["campaign_id"].astype(str).to_numpy()

    assert X.shape==(700,768)
    assert np.isfinite(X).all()
    assert np.allclose(np.linalg.norm(X,axis=1),1,atol=1e-5)

    sigma=metrics.median_heuristic_sigma(X)
    median=float(np.median(pdist(X)))

    print("Recovered sigma :",sigma)
    print("Recovered median:",median)

    assert np.isclose(sigma,SIGMA,rtol=1e-5,atol=1e-6)
    assert np.isclose(median,MEDIAN,rtol=1e-5,atol=1e-6)

    np.save(xp,X)
    pd.DataFrame({"id":ids}).to_csv(ip,index=False)

    print("Frozen BGE 700-reference validated.")
    return X,ids

# manifest and sampling plan 
def prepare():
    if MANIFEST.exists() or SAMPLES.exists():
        raise RuntimeError("BGE manifest already frozen.")

    D=load_eval()
    R,Rids=load_or_build_reference()

    eval_ids=D[SCENARIOS[0]]["ids"]

    rng=np.random.default_rng(BASE_SEED)
    power_samples=[
        rng.choice(eval_ids,N,replace=False)
        for _ in range(REPS)
    ]

    rng=np.random.default_rng(BASE_SEED)
    null_samples=[
        rng.choice(Rids,2*N,replace=False)
        for _ in range(REPS)
    ]

    S=[]

    for rep,ids in enumerate(power_samples):
        S.append({
            "sample_key":f"P|R{rep:03d}",
            "kind":"power",
            "rep":rep,
            "P_ids":json.dumps(ids.tolist()),
            "Q_ids":json.dumps(ids.tolist())
        })

    for rep,ids in enumerate(null_samples):
        A,B=ids[:N],ids[N:]
        assert not set(A)&set(B)

        S.append({
            "sample_key":f"N|R{rep:03d}",
            "kind":"null",
            "rep":rep,
            "P_ids":json.dumps(A.tolist()),
            "Q_ids":json.dumps(B.tolist())
        })

    S=pd.DataFrame(S)

    M=[]

    for scenario in SCENARIOS:
        for severity in SEV:
            for test in TESTS:
                for rep in range(REPS):
                    M.append({
                        "run_id":f"POWER|{scenario}|S{severity}|{test}|R{rep:03d}",
                        "kind":"power",
                        "scenario":scenario,
                        "severity":severity,
                        "test":test,
                        "rep":rep,
                        "test_seed":BASE_SEED+rep,
                        "sample_key":f"P|R{rep:03d}"
                    })

    for test in TESTS:
        for rep in range(REPS):
            M.append({
                "run_id":f"NULL|{test}|R{rep:03d}",
                "kind":"null",
                "scenario":"",
                "severity":np.nan,
                "test":test,
                "rep":rep,
                "test_seed":BASE_SEED+rep,
                "sample_key":f"N|R{rep:03d}"
            })

    M=pd.DataFrame(M)

    assert len(M[M.kind=="power"])==6000
    assert len(M[M.kind=="null"])==1000
    assert len(M)==7000
    assert M.run_id.is_unique

    M.to_csv(MANIFEST,index=False)
    S.to_csv(SAMPLES,index=False)

    (OUT/"manifest.sha256").write_text(core.sha256(MANIFEST))
    (OUT/"sampling.sha256").write_text(core.sha256(SAMPLES))

    print("BGE PREPARE COMPLETE")
    print("Power:",6000)
    print("Null :",1000)
    print("Total:",7000)

# execution + checkpoint
def read_csv(p):
    return pd.read_csv(p,keep_default_na=False,na_values=[""])


def verify():
    assert core.sha256(MANIFEST)==(OUT/"manifest.sha256").read_text().strip()
    assert core.sha256(SAMPLES)==(OUT/"sampling.sha256").read_text().strip()


def rows(X,ids,wanted):
    pos={str(x):i for i,x in enumerate(ids)}
    return X[[pos[str(x)] for x in wanted]]


def get_pq(row,D,R,Rids,S):
    s=S.loc[row.sample_key]
    A=json.loads(s.P_ids)
    B=json.loads(s.Q_ids)

    if row.kind=="power":
        d=D[row.scenario]
        P0=rows(d["clean"],d["ids"],A)
        Q0=rows(d[float(row.severity)],d["ids"],B)
    else:
        P0=rows(R,Rids,A)
        Q0=rows(R,Rids,B)
        assert not set(A)&set(B)

    assert P0.shape==Q0.shape==(N,768)
    return P0,Q0


def db():
    con=sqlite3.connect(DB)

    con.execute("""
    CREATE TABLE IF NOT EXISTS results(
        run_id TEXT PRIMARY KEY,kind TEXT,scenario TEXT,severity REAL,
        test TEXT,rep INTEGER,statistic REAL,p_value REAL,reject INTEGER,
        null_mean REAL,null_std REAL,runtime REAL,warning_text TEXT,
        completed_at TEXT
    )
    """)

    con.commit()
    return con


def export(con):
    x=pd.read_sql_query("SELECT * FROM results",con)
    x[x.kind=="power"].to_csv(OUT/"bge_power_runs.csv",index=False)
    x[x.kind=="null"].to_csv(OUT/"bge_null_runs.csv",index=False)


def execute(phase):
    verify()

    D=load_eval()
    R,Rids=load_or_build_reference()

    M=read_csv(MANIFEST)
    S=read_csv(SAMPLES).set_index("sample_key")

    if phase!="all":
        M=M[M.kind==phase]

    con=db()
    done={x[0] for x in con.execute("SELECT run_id FROM results")}
    M=M[~M.run_id.isin(done)].reset_index(drop=True)

    print("Phase:",phase,"| Pending:",len(M))

    for i,row in M.iterrows():
        P0,Q0=get_pq(row,D,R,Rids,S)
        t=time.perf_counter()

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            r=core.run_metric(
                row.test,P0,Q0,REP,int(row.test_seed)
            )

        warning=" | ".join(str(x.message) for x in w)

        if warning:
            raise RuntimeError(
                f"Warning in {row.run_id}: {warning}"
            )

        con.execute("""
        INSERT INTO results VALUES(
        ?,?,?,?,?,?,?,?,?,?,?,?,?,datetime('now'))
        """,(
            row.run_id,row.kind,row.scenario,
            None if pd.isna(row.severity) else float(row.severity),
            row.test,int(row.rep),
            r["statistic"],r["p_value"],r["reject"],
            r["null_mean"],r["null_std"],
            time.perf_counter()-t,warning
        ))

        con.commit()

        if (i+1)%100==0:
            print(f"{i+1}/{len(M)} completed")

    export(con)

    total=con.execute("SELECT COUNT(*) FROM results").fetchone()[0]
    con.close()

    print("PHASE COMPLETE")
    print("Total checkpointed:",total)


def smoke():
    verify()
    D=load_eval()
    R,Rids=load_or_build_reference()
    M=read_csv(MANIFEST)
    S=read_csv(SAMPLES).set_index("sample_key")

    checks=[]

    for scenario in SCENARIOS:
        for test in TESTS:
            row=M[
                (M.kind=="power")&
                (M.scenario==scenario)&
                (M.test==test)&
                (M.rep==0)
            ].iloc[0]

            P0,Q0=get_pq(row,D,R,Rids,S)
            core.run_metric(test,P0,Q0,REP,int(row.test_seed))
            checks.append((scenario,test))

    for test in TESTS:
        row=M[
            (M.kind=="null")&
            (M.test==test)&
            (M.rep==0)
        ].iloc[0]

        P0,Q0=get_pq(row,D,R,Rids,S)
        core.run_metric(test,P0,Q0,REP,int(row.test_seed))
        checks.append(("null",test))

    print("BGE SMOKE PASSED")
    print("Routes:",len(checks))
    print("No formal results saved.")


def status():
    if not DB.exists():
        print("0 / 7000 complete")
        return

    con=db()

    print(pd.read_sql_query("""
        SELECT kind,COUNT(*) AS completed
        FROM results GROUP BY kind
    """,con))

    print("Total:",
          con.execute("SELECT COUNT(*) FROM results").fetchone()[0],
          "/ 7000")

    con.close()

# CLI 
def main():
    p=argparse.ArgumentParser()
    sub=p.add_subparsers(dest="cmd",required=True)

    sub.add_parser("prepare")
    sub.add_parser("smoke")
    sub.add_parser("status")

    r=sub.add_parser("run")
    r.add_argument(
        "--phase",
        choices=["power","null","all"],
        default="all"
    )

    a=p.parse_args()

    if a.cmd=="prepare":prepare()
    elif a.cmd=="smoke":smoke()
    elif a.cmd=="status":status()
    elif a.cmd=="run":execute(a.phase)


if __name__=="__main__":
    main()