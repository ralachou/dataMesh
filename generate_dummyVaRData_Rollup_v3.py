"""
Reconciled synthetic VaR99_PnL dataset generator (business tree + risk-factor explainability + rf hierarchy drilldowns)

What this script produces (all perfectly reconciled):
A) business_leaf_var
   - One row per business leaf (b_level7 path)
   - Contains leaf_var99_pnl for as_of_date=2020-03-09

B) leaf_rf_contrib  (BASE FACT TABLE)
   - One row per (business leaf, riskFactor)
   - value_usd contributions per riskFactor
   - CRITICAL: For every leaf: SUM(value_usd over riskFactors) == leaf_var99_pnl

C) agg_var_business
   - Business tree rollups by b_level1..b_level7 (pure SUM of leaf_rf_contrib)

D) agg_var_explain
   - Explainability rollups by (business_agg_level, rf_agg_level) (pure SUM of leaf_rf_contrib)
   - Lets you click on a business node and explain by rf_level1, then expand to rf_level2..rf_level5

If the base fact table (leaf_rf_contrib) is reconciled at the leaf level, every rollup is automatically reconciled.
"""

#from __future__ import annotations

import random
import numpy as np
import pandas as pd


# =========================
# CONFIG
# =========================
BUSINESS_FILE = "CIB_Markets_Business_Hierarchy.xlsx"
RISK_FILE = "risk_factor_hierarchy_186k.xlsx"

AS_OF_DATE = "2020-03-09"
METRIC = "VaR99_PnL"

# Optional: fix firm-level total (remove if you want purely random firm total)
FIRM_TARGET = -600_000_000.0

# Risk factors per leaf (keep total rows under 1,000,000)
MIN_RF_PER_LEAF = 20
MAX_RF_PER_LEAF = 120

# Percent of positive (hedging) contributions inside a leaf
DEFAULT_POS_PROB = 0.06

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)


# =========================
# LOAD INPUTS
# =========================
biz = pd.read_excel(
    BUSINESS_FILE,
    sheet_name="business_hierarchy",
    usecols=["b_level1","b_level2","b_level3","b_level4","b_level5","b_level6","b_level7"],
)

rf = pd.read_excel(
    RISK_FILE,
    sheet_name="risk_factor_hierarchy",
    usecols=["riskFactor","rf_level1","rf_level2","rf_level3","rf_level4","rf_level5"],
)

# Normalize nulls + types
for c in biz.columns:
    biz[c] = biz[c].fillna("").astype(str)
for c in rf.columns:
    rf[c] = rf[c].fillna("").astype(str)


# =========================
# STEP 1) DEFINE BUSINESS LEAVES + ASSIGN LEAF VaR
# =========================
# Every row in biz is a leaf (b_level7 node)
leaves = biz.copy()
leaves["as_of_date"] = AS_OF_DATE
leaves["metric"] = METRIC

# A simple, realistic way to generate leaf VaR:
# - heavy-tailed magnitudes (lognormal)
# - mostly negative contributions (loss), with some positives possible
# - then scale to firm target if desired
rng = np.random.default_rng(SEED)

n_leaves = len(leaves)
magnitudes = rng.lognormal(mean=0.0, sigma=1.0, size=n_leaves)

# make most leaves negative, some positive (hedges / offsets)
leaf_signs = np.where(rng.random(n_leaves) < 0.12, 1.0, -1.0)  # ~12% positive leaves
leaf_var = leaf_signs * magnitudes

# scale to something like hundreds of millions
leaf_var *= 2_000_000.0  # rough scale per leaf (adjust if needed)

# optional: scale exactly to firm target
if FIRM_TARGET is not None:
    scale = FIRM_TARGET / leaf_var.sum()
    leaf_var = leaf_var * scale

leaves["leaf_var99_pnl"] = leaf_var.astype("float64")

business_leaf_var = leaves[
    ["as_of_date","metric",
     "b_level1","b_level2","b_level3","b_level4","b_level5","b_level6","b_level7",
     "leaf_var99_pnl"]
].copy()


# =========================
# STEP 2) MAP EACH LEAF TO RISK FACTORS + GENERATE CONTRIBUTIONS
# =========================
# We will create the BASE FACT TABLE:
# leaf_rf_contrib: one row per (leaf, riskFactor) with value_usd.
#
# Reconciliation rule enforced per leaf:
#   SUM(value_usd over riskFactors for that leaf) == leaf_var99_pnl
#
# The ONLY non-trivial part is choosing "plausible" rf pools.
# For simplicity, we use b_level4/b_level5 to bias which rf_level1 pools are used.

def rf_pool(expr: str) -> pd.DataFrame:
    return rf.query(expr).reset_index(drop=True)

POOL = {
    "CP_RATE": rf_pool("rf_level1=='CP_RATE'"),
    "CP_VOL":  rf_pool("rf_level1=='CP_VOL'"),
    "IR_RATE": rf_pool("rf_level1=='IR_RATE'"),
    "IR_VOL":  rf_pool("rf_level1=='IR_VOL'"),
    "FX_RATE": rf_pool("rf_level1=='FX_RATE'"),
    "FX_VOL":  rf_pool("rf_level1=='FX_VOL'"),
    "EQ_PRICE":rf_pool("rf_level1=='EQ_PRICE'"),
    "EQ_VOL":  rf_pool("rf_level1=='EQ_VOL'"),
    "COMM_PRICE": rf_pool("rf_level1=='COMM_PRICE'"),
    "COMM_VOL":   rf_pool("rf_level1=='COMM_VOL'"),
}

def sample_rfs(df_pool: pd.DataFrame, n: int) -> list[str]:
    """Sample n unique risk factors from a pool."""
    if len(df_pool) == 0 or n <= 0:
        return []
    n = min(n, len(df_pool))
    return df_pool.sample(n=n, replace=False, random_state=random.randint(0, 1_000_000))["riskFactor"].tolist()

def choose_risk_factors_for_leaf(row: pd.Series, n: int) -> list[str]:
    """
    Choose risk factors consistent with the business leaf.
    This is only for realism; reconciliation does not depend on it.
    """
    b4 = row["b_level4"]
    b5 = row["b_level5"]
    b6 = row["b_level6"].upper()
    b7 = row["b_level7"].upper()

    # Equities platform: EQ_PRICE + EQ_VOL
    if b4 == "Equities":
        n_price = int(n * 0.6)
        return (sample_rfs(POOL["EQ_PRICE"], n_price) +
                sample_rfs(POOL["EQ_VOL"],   n - n_price))[:n]

    # Interest Rates: IR_RATE + IR_VOL
    if b5 == "Interest Rates":
        n_rate = int(n * 0.7)
        return (sample_rfs(POOL["IR_RATE"], n_rate) +
                sample_rfs(POOL["IR_VOL"],  n - n_rate))[:n]

    # FX: FX_RATE + FX_VOL
    if b5 == "FX":
        n_rate = int(n * 0.7)
        return (sample_rfs(POOL["FX_RATE"], n_rate) +
                sample_rfs(POOL["FX_VOL"],  n - n_rate))[:n]

    # Commodities: COMM_PRICE + COMM_VOL
    if b5 == "Commodities":
        n_price = int(n * 0.7)
        return (sample_rfs(POOL["COMM_PRICE"], n_price) +
                sample_rfs(POOL["COMM_VOL"],   n - n_price))[:n]

    # Credit / Securitized: CP_RATE + CP_VOL (optionally refine by product using rf_level3)
    if b5 in ["Credit", "Securitized Products"]:
        # desk-aware product refinement (still only for realism)
        if b5 == "Securitized Products":
            if "CMBS" in b6 or "CMBX" in b7:
                cp_rate = rf_pool("rf_level1=='CP_RATE' and rf_level3=='CMBS'")
                cp_vol  = rf_pool("rf_level1=='CP_VOL'  and rf_level3=='CMBS'")
            elif "ABS" in b6:
                cp_rate = rf_pool("rf_level1=='CP_RATE' and rf_level3=='ABS'")
                cp_vol  = rf_pool("rf_level1=='CP_VOL'  and rf_level3=='ABS'")
            else:
                cp_rate = rf_pool("rf_level1=='CP_RATE' and rf_level3=='RMBS'")
                cp_vol  = rf_pool("rf_level1=='CP_VOL'  and rf_level3=='RMBS'")
        else:
            # Credit
            if row["b_level6"] == "Municipal Trading":
                cp_rate = rf_pool("rf_level1=='CP_RATE' and rf_level3=='Muni'")
                cp_vol  = rf_pool("rf_level1=='CP_VOL'  and rf_level3=='Muni'")
            elif "CDS" in b7:
                cp_rate = rf_pool("rf_level1=='CP_RATE' and rf_level3=='CDS'")
                cp_vol  = rf_pool("rf_level1=='CP_VOL'  and rf_level3=='CDS'")
            elif ("CDX" in b7) or ("INDEX" in b7) or ("CMBX" in b7):
                cp_rate = rf_pool("rf_level1=='CP_RATE' and rf_level3=='CDX'")
                cp_vol  = rf_pool("rf_level1=='CP_VOL'  and rf_level3=='CDX'")
            else:
                cp_rate = rf_pool("rf_level1=='CP_RATE' and rf_level3=='CORP'")
                cp_vol  = rf_pool("rf_level1=='CP_VOL'  and rf_level3=='CORP'")

        n_rate = int(n * 0.7)
        return (sample_rfs(cp_rate, n_rate) +
                sample_rfs(cp_vol,  n - n_rate))[:n]

    # Financing & XVA / other: mixed exposures
    per = max(1, n // 5)
    mix = (
        sample_rfs(POOL["IR_RATE"], per) +
        sample_rfs(POOL["FX_RATE"], per) +
        sample_rfs(POOL["CP_RATE"], per) +
        sample_rfs(POOL["EQ_PRICE"], per) +
        sample_rfs(POOL["COMM_PRICE"], n - 4 * per)
    )
    return mix[:n]

def generate_leaf_contrib(leaf_target: float, rfs_list: list[str], pos_prob: float) -> np.ndarray:
    """
    Generate contributions per riskFactor for one leaf, then enforce exact reconciliation:
      sum(contribs) == leaf_target

    We use:
    - heavy-tailed weights (lognormal)
    - mostly negative signs with a small positive probability (pos_prob)
    - then the 'offset trick' to force exact sum.
    """
    n = len(rfs_list)
    if n == 0:
        return np.array([], dtype="float64")

    # weights
    w = rng.lognormal(mean=0.0, sigma=1.0, size=n)
    w = w / w.sum()

    # signs
    signs = np.where(rng.random(n) < pos_prob, 1.0, -1.0)

    # raw magnitude baseline (arbitrary but stable)
    raw = signs * w * (abs(leaf_target) * 1.5 + 1_000_000.0)

    # EXACT reconciliation (offset trick)
    raw += (leaf_target - raw.sum()) / n
    return raw.astype("float64")

# Build leaf_rf_contrib (base fact)
fact_parts = []
for _, row in leaves.iterrows():
    leaf_target = float(row["leaf_var99_pnl"])

    n_rf = random.randint(MIN_RF_PER_LEAF, MAX_RF_PER_LEAF)
    rfs_list = choose_risk_factors_for_leaf(row, n_rf)

    # Slightly more positives for Equities / CDS-style nodes (optional realism)
    pos_prob = DEFAULT_POS_PROB
    if row["b_level4"] == "Equities":
        pos_prob = 0.12
    if row["b_level5"] == "Credit" and "CDS" in row["b_level7"].upper():
        pos_prob = 0.18

    contribs = generate_leaf_contrib(leaf_target, rfs_list, pos_prob=pos_prob)

    tmp = pd.DataFrame({
        "as_of_date": AS_OF_DATE,
        "metric": METRIC,
        "b_level1": row["b_level1"],
        "b_level2": row["b_level2"],
        "b_level3": row["b_level3"],
        "b_level4": row["b_level4"],
        "b_level5": row["b_level5"],
        "b_level6": row["b_level6"],
        "b_level7": row["b_level7"],
        "riskFactor": rfs_list,
        "value_usd": contribs,
    })
    fact_parts.append(tmp)

leaf_rf_contrib = pd.concat(fact_parts, ignore_index=True)

# Attach rf hierarchy levels for drilldowns
leaf_rf_contrib = leaf_rf_contrib.merge(rf, on="riskFactor", how="left")

# Keep a clean column order
leaf_rf_contrib = leaf_rf_contrib[
    ["as_of_date","metric",
     "b_level1","b_level2","b_level3","b_level4","b_level5","b_level6","b_level7",
     "riskFactor","rf_level1","rf_level2","rf_level3","rf_level4","rf_level5",
     "value_usd"]
].copy()


# =========================
# STEP 3) BUSINESS TREE ROLLUPS (pure SUM)
# =========================
def rollup_business(df: pd.DataFrame, level_n: int) -> pd.DataFrame:
    """
    Sum contributions for a chosen business depth.
    Example: level_n=6 gives VaR by b_level1..b_level6 (desk-level), blanking b_level7.
    """
    keep = [f"b_level{i}" for i in range(1, level_n + 1)]
    grp = df.groupby(["as_of_date","metric"] + keep, as_index=False)["value_usd"].sum()
    grp["agg_level"] = f"b_level{level_n}"
    for j in range(level_n + 1, 8):
        grp[f"b_level{j}"] = ""
    return grp[["as_of_date","metric","agg_level"] + [f"b_level{i}" for i in range(1,8)] + ["value_usd"]]

agg_var_business = pd.concat(
    [rollup_business(leaf_rf_contrib, n) for n in range(7, 0, -1)],
    ignore_index=True
)


# =========================
# STEP 4) RISK-FACTOR HIERARCHY DRILLDOWNS (pure SUM)
# =========================
def rollup_explain(df: pd.DataFrame, business_level_n: int, rf_level_k: int) -> pd.DataFrame:
    """
    Sum contributions for a chosen business aggregation depth AND rf hierarchy depth.
    This powers the drilldown: click business node → show rf_level1 → expand to rf_level2 → ... → rf_level5.

    business_level_n ∈ {1..7}
    rf_level_k       ∈ {1..5}
    """
    b_keep = [f"b_level{i}" for i in range(1, business_level_n + 1)]
    r_keep = [f"rf_level{i}" for i in range(1, rf_level_k + 1)]
    grp = df.groupby(["as_of_date","metric"] + b_keep + r_keep, as_index=False)["value_usd"].sum()

    grp["business_agg_level"] = f"b_level{business_level_n}"
    grp["rf_agg_level"] = f"rf_level{rf_level_k}"

    # blank deeper business levels
    for j in range(business_level_n + 1, 8):
        grp[f"b_level{j}"] = ""

    # blank deeper rf levels
    for j in range(rf_level_k + 1, 6):
        grp[f"rf_level{j}"] = ""

    return grp[
        ["as_of_date","metric","business_agg_level"] +
        [f"b_level{i}" for i in range(1,8)] +
        ["rf_agg_level"] +
        [f"rf_level{i}" for i in range(1,6)] +
        ["value_usd"]
    ]

agg_var_explain = pd.concat(
    [rollup_explain(leaf_rf_contrib, bn, rk) for bn in range(1, 8) for rk in range(1, 6)],
    ignore_index=True
)


# =========================
# STEP 5) RECONCILIATION CHECKS (must be zero mismatch)
# =========================
# (A) Firm total from business rollup
firm_total = float(agg_var_business.loc[agg_var_business["agg_level"]=="b_level1","value_usd"].sum())

# (B) Firm total from rf-level1 explainability (Top_of_House by rf_level1)
firm_rf1_total = float(
    agg_var_explain.loc[
        (agg_var_explain["business_agg_level"]=="b_level1") &
        (agg_var_explain["rf_agg_level"]=="rf_level1"),
        "value_usd"
    ].sum()
)

# (C) Leaf reconciliation: compare business_leaf_var vs sum of leaf_rf_contrib by leaf
leaf_sums = (
    leaf_rf_contrib.groupby(
        ["b_level1","b_level2","b_level3","b_level4","b_level5","b_level6","b_level7"],
        as_index=False
    )["value_usd"].sum()
    .rename(columns={"value_usd":"leaf_sum_from_rfs"})
)

leaf_check = business_leaf_var.merge(
    leaf_sums,
    on=["b_level1","b_level2","b_level3","b_level4","b_level5","b_level6","b_level7"],
    how="left"
)
leaf_check["diff"] = leaf_check["leaf_var99_pnl"] - leaf_check["leaf_sum_from_rfs"]
max_abs_leaf_diff = float(leaf_check["diff"].abs().max())

recon_report = pd.DataFrame([{
    "fact_rows": len(leaf_rf_contrib),
    "business_leaf_rows": len(business_leaf_var),
    "agg_var_business_rows": len(agg_var_business),
    "agg_var_explain_rows": len(agg_var_explain),
    "firm_total": firm_total,
    "firm_total_minus_target": firm_total - (FIRM_TARGET if FIRM_TARGET is not None else firm_total),
    "firm_rf_level1_total": firm_rf1_total,
    "rf1_minus_firm": firm_rf1_total - firm_total,
    "max_abs_leaf_diff": max_abs_leaf_diff
}])

print("\n=== RECON REPORT (all should be ~0 mismatch) ===")
print(recon_report.to_string(index=False))

# Optional: show top contributors at b_level5
top_b5 = (
    agg_var_business[agg_var_business["agg_level"]=="b_level5"]
    .groupby("b_level5", as_index=False)["value_usd"].sum()
    .sort_values("value_usd")
)
print("\n=== Top contributors at b_level5 ===")
print(top_b5.to_string(index=False))


# =========================
# STEP 6) WRITE OUTPUTS
# =========================
OUT_EXCEL = "reconciled_var_demo_2020-03-09.xlsx"
with pd.ExcelWriter(OUT_EXCEL, engine="openpyxl") as writer:
    business_leaf_var.to_excel(writer, sheet_name="business_leaf_var", index=False)
    leaf_rf_contrib.to_excel(writer, sheet_name="leaf_rf_contrib", index=False)
    agg_var_business.to_excel(writer, sheet_name="agg_var_business", index=False)
    agg_var_explain.to_excel(writer, sheet_name="agg_var_explain", index=False)
    recon_report.to_excel(writer, sheet_name="RECON_REPORT", index=False)
    top_b5.to_excel(writer, sheet_name="TOP_b_level5", index=False)
    pd.DataFrame({"README":[
        "business_leaf_var: one VaR99_PnL per business leaf (b_level7 row).",
        "leaf_rf_contrib: base table (leaf, riskFactor) contributions. Per-leaf sums equal leaf VaR exactly.",
        "agg_var_business: business rollups (pure sum from leaf_rf_contrib).",
        "agg_var_explain: drilldowns by business depth and rf hierarchy depth (pure sum from leaf_rf_contrib).",
        "If leaf_rf_contrib is reconciled, all drilldowns reconcile automatically."
    ]}).to_excel(writer, sheet_name="README", index=False)

print(f"\nWrote: {OUT_EXCEL}")
