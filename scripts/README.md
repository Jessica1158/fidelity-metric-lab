# How to Run `confirmatory_run.py`

The script `confirmatory_run.py` controls the final confirmatory run of the experiment.
It is designed to execute the frozen experimental design, save checkpoints, and resume
automatically if execution is interrupted.

## 1. Activate the environment

From the project root:

```bash
conda activate /Users/jessica/fidelity-metric-lab/.venv312
```

Verify:

```bash
python --version
```

mmust use Python 3.12.

---

## 2. `prepare`

```bash
python scripts/confirmatory_run.py prepare
```

It has to be executed **only once before the formal run**.

This phase:
- validates the frozen representations;
- prepares the 700-unit clean reference representations;
- creates the execution manifest;
- creates the sampling plan;
- freezes both using hashes.

**Do not rerun prepare after the experiment has been frozen**


---

## 3. `smoke`

```bash
python scripts/confirmatory_run.py smoke
```

It has to be executed after `prepare` and before any formal run.

It executes a small test across all main pathways:
• Representations
• C2ST Linear
• C2ST RBF
• MMD
• Energy
• Sinkhorn
• Power
• Clean-null

It verifies that data loads correctly, sizes are valid, and tests produce correct p-values. 
**It does not estimate power or Type I error, and no formal results are saved.**

Expected output:

```text
FINAL SMOKE CHECK PASSED
No formal results were saved.
```

---

## 4. `status`

```bash
python scripts/confirmatory_run.py status
```

Displays only the operational status of the run:

- Completed executions
- Pending executions
- Errors
- Warnings

It does not calculate or show scientific results such as power or Type I error. It can be run at any
time.

---

# 5. Formal Runs

The complete run contains:

- 110 power cells × 200 repetitions = 22,000 executions.
- 40 clean-null cells × 200 repetitions = 8,000 executions.
- Total = 30,000 executions.

It is operationally divided into four phases.

## `power50`

```bash
python scripts/confirmatory_run.py run --phase power50
```
Executes all retained confirmatory power cells with:

`N = 50`

Compares clean samples against their corresponding defective conditions.

## `power200`

```bash
python scripts/confirmatory_run.py run --phase power200
```
Executes retained power cells with:

`N = 200``

Uses the same frozen design, metrics, defects, severities, and configurations.

## `null50`

```Bash
python scripts/confirmatory_run.py run --phase null50
```

Executes clean-vs-clean calibrations with:

`N = 50``

Used later to estimate the Type I error for each test and representation.

## `null200`
```bash
python scripts/confirmatory_run.py run --phase null200
```
Executes clean-vs-clean calibrations with:

`N = 200``

For each repetition, it selects 400 distinct units from the frozen clean reference pool of 700:


200 → P
200 → Q
overlap = 0

This fixes the previous issue of attempting to obtain two independent groups of 200 from an evaluation pool of only 300 units.

## Checkpoint / Resume
Every completed execution is saved immediately to SQLite.

If a run is interrupted, simply re-run the exact same command. For example:

```Bash
python scripts/confirmatory_run.py run --phase power200
```
The script automatically detects already completed executions and continues only with the pending ones.

There is no need to start over from scratch.

# BGE Confirmatory Run



1. 
```Bash
python scripts/bge_confirmatory_run.py prepare
```
   We expect:
   BGE PREPARE COMPLETE
   Power: 6000
   Null : 1000
   Total: 7000

2. 
```Bash
python scripts/bge_confirmatory_run.py smoke
```
   We expect:
   BGE SMOKE PASSED
   Routes: 15
   No formal results saved.


3. Then, the formal run:
```Bash
   python scripts/bge_confirmatory_run.py run --phase power
```
4. Afterward:
```Bash
   python scripts/bge_confirmatory_run.py run --phase null
```
5. And at any time:
```Bash
   python scripts/bge_confirmatory_run.py status
```
If the 'power' phase is interrupted, simply run the exact same command again; the SQLite database will skip the runs that have already been completed.

An important note before launching the 7,000 runs:
When running `prepare`

Recovered sigma : ...
Recovered median: ...

should be approximately:
0.3873605130049743
0.3873605130049747

If those assertions pass and the following appears:
BGE PREPARE COMPLETE

then proceed with the `smoke` test.