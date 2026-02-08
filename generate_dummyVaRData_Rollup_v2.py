import pandas as pd
import numpy as np
import random

# =========================
# INPUTS / CONFIG
# =========================
BUSINESS_FILE = "CIB_Markets_Business_Hierarchy.xlsx"
RISK_FILE     = "risk_factor_hierarchy_186k.xlsx"

AS_OF_DATE = "2020-03-09"
METRIC = "VaR99_PnL"
FIRM_TARGET = -600_000_000.0

# Narrative targets using ONLY existing b_level columns (no extra business bucketing columns)
TARGETS = {
    "Securitized Products":  -300_000_000.0,
    "Credit":               -200_000_000.0,
    "Interest Rates":       -100_000_000.0,
    "FX":                   -100_000_000.0,
    "__EQUITIES_PLATFORM__": +100_000_000.0,  # all b_level4 == Equities
}

# Sub-shaping (still only b_level columns)
SEC_SUBTARGETS = {
    "SEC_CMO_IOPO": -150_000_000.0,  # b_level7 in {"CMO","IO/PO"}
    "SEC_TBA":      -100_000_000.0,  # b_level7 startswith "TBA"
    "SEC_OTHER":     -50_000_000.0,
}

CREDIT_SUBTARGETS = {
    "CREDIT_CORP":  -180_000_000.0,
    "CREDIT_MUNI":   -20_000_000.0,
    "CREDIT_CDS":    +10_000_000.0,
    "CREDIT_INDEX":  -10_000_000.0,  # CDX/CMBX/Index-type
}

# Keep base table under 1M rows
RF_PER_LEAF_MIN = 60
RF_PER_LEAF_MAX = 140

random.seed(42)
np.random.seed(42)

# =========================
# LOAD INPUT FILES
# =========================
biz = pd.read_excel(BUSINESS_FILE, sheet_name="business_hierarchy")
rf  = pd.read_excel(
    RISK_FILE,
    sheet_name="risk_factor_hierarchy",
    usecols=["riskFactor", "rf_level1", "rf_level2", "rf_level3", "rf_level4", "rf_level5"]
)

# Normalize to strings
for c in biz.columns:
    biz[c] = biz[c].fillna("").astype(str)
for c in rf.columns:
    rf[c] = rf[c].fillna("").astype(str)

# =========================
# HELPERS
# =========================
def allocate(total: float, idxs: list[int], seed: int = 0, sigma: float = 0.7) -> pd.Series:
    """Allocate a total across idxs using heavy-tailed weights (lognormal)."""
    if len(idxs) == 0:
        return pd.Series([], dtype="float64")
    rng = np.random.default_rng(seed)
    w = rng.lognormal(mean=0.0, sigma=sigma, size=len(idxs))
    w = w / w.sum()
    return pd.Series(total * w, index=idxs)

def rf_pool(expr: str) -> pd.DataFrame:
    return rf.query(expr).reset_index(drop=True)

def sample_rfs(df_pool: pd.DataFrame, n: int) -> list[str]:
    if len(df_pool) == 0 or n <= 0:
        return []
    n = min(n, len(df_pool))
    return df_pool.sample(n=n, replace=False, random_state=random.randint(0, 1_000_000))["riskFactor"].tolist()

# Risk factor pools
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

def choose_risk_factors(row: pd.Series, n: int) -> list[str]:
    """Choose risk factors consistent with the desk using ONLY b_level columns."""
    b_level5 = row["b_level5"]
    desk6 = row["b_level6"].upper()
    strat7 = row["b_level7"].upper()

    # Equities platform
    if row["b_level4"] == "Equities":
        n_p = int(n * 0.6)
        return (sample_rfs(POOL["EQ_PRICE"], n_p) + sample_rfs(POOL["EQ_VOL"], n - n_p))[:n]

    # Rates
    if b_level5 == "Interest Rates":
        n_r = int(n * 0.7)
        return (sample_rfs(POOL["IR_RATE"], n_r) + sample_rfs(POOL["IR_VOL"], n - n_r))[:n]

    # FX
    if b_level5 == "FX":
        n_r = int(n * 0.7)
        return (sample_rfs(POOL["FX_RATE"], n_r) + sample_rfs(POOL["FX_VOL"], n - n_r))[:n]

    # Commodities
    if b_level5 == "Commodities":
        n_p = int(n * 0.7)
        return (sample_rfs(POOL["COMM_PRICE"], n_p) + sample_rfs(POOL["COMM_VOL"], n - n_p))[:n]

    # Credit & Securitized map to CP pools (filter via rf_level3)
    if b_level5 in ["Credit", "Securitized Products"]:
        if b_level5 == "Securitized Products":
            if "CMBS" in desk6 or "CMBX" in strat7:
                cp_rate = rf_pool("rf_level1=='CP_RATE' and rf_level3=='CMBS'")
                cp_vol  = rf_pool("rf_level1=='CP_VOL'  and rf_level3=='CMBS'")
            elif "ABS" in desk6:
                cp_rate = rf_pool("rf_level1=='CP_RATE' and rf_level3=='ABS'")
                cp_vol  = rf_pool("rf_level1=='CP_VOL'  and rf_level3=='ABS'")
            else:
                cp_rate = rf_pool("rf_level1=='CP_RATE' and rf_level3=='RMBS'")
                cp_vol  = rf_pool("rf_level1=='CP_VOL'  and rf_level3=='RMBS'")
        else:
            # Credit: Municipal desk explicit; else infer strategy from b_level7 labels
            if row["b_level6"] == "Municipal Trading":
                cp_rate = rf_pool("rf_level1=='CP_RATE' and rf_level3=='Muni'")
                cp_vol  = rf_pool("rf_level1=='CP_VOL'  and rf_level3=='Muni'")
            elif "CDS" in strat7:
                cp_rate = rf_pool("rf_level1=='CP_RATE' and rf_level3=='CDS'")
                cp_vol  = rf_pool("rf_level1=='CP_VOL'  and rf_level3=='CDS'")
            elif ("CDX" in strat7) or ("INDEX" in strat7) or ("CMBX" in strat7):
                cp_rate = rf_pool("rf_level1=='CP_RATE' and rf_level3=='CDX'")
                cp_vol  = rf_pool("rf_level1=='CP_VOL'  and rf_level3=='CDX'")
            else:
                cp_rate = rf_pool("rf_level1=='CP_RATE' and rf_level3=='CORP'")
                cp_vol  = rf_pool("rf_level1=='CP_VOL'  and rf_level3=='CORP'")

        n_r = int(n * 0.7)
        return (sample_rfs(cp_rate, n_r) + sample_rfs(cp_vol, n - n_r))[:n]

    # Financing & XVA / rest: mixed exposures across major classes
    per = max(1, n // 5)
    mix = (
        sample_rfs(POOL["IR_RATE"], per) +
        sample_rfs(POOL["FX_RATE"], per) +
        sample_rfs(POOL["CP_RATE"], per) +
        sample_rfs(POOL["EQ_PRICE"], per) +
        sample_rfs(POOL["COMM_PRICE"], n - 4 * per)
    )
    return mix[:n]

# =========================
# LEAF TARGETS (each row is a VaR leaf)
# =========================
leaves = biz.copy()
leaves["leaf_target"] = 0.0

b5 = leaves["b_level5"]
b4 = leaves["b_level4"]
b7u = leaves["b_level7"].str.upper()

# --- Securitized subtargets (ONLY existing levels) ---
sec_mask = (b5 == "Securitized Products")
sec_cmo_iopo_mask = sec_mask & (b7u.isin(["CMO", "IO/PO"]))
sec_tba_mask      = sec_mask & (b7u.str.startswith("TBA"))
sec_other_mask    = sec_mask & ~(sec_cmo_iopo_mask | sec_tba_mask)

idx_sec_cmo_iopo = leaves.index[sec_cmo_iopo_mask].tolist()
idx_sec_tba      = leaves.index[sec_tba_mask].tolist()
idx_sec_other    = leaves.index[sec_other_mask].tolist()

leaves.loc[idx_sec_cmo_iopo, "leaf_target"] = allocate(SEC_SUBTARGETS["SEC_CMO_IOPO"], idx_sec_cmo_iopo, seed=101).values
leaves.loc[idx_sec_tba,      "leaf_target"] = allocate(SEC_SUBTARGETS["SEC_TBA"],      idx_sec_tba,      seed=102).values
leaves.loc[idx_sec_other,    "leaf_target"] = allocate(SEC_SUBTARGETS["SEC_OTHER"],    idx_sec_other,    seed=103).values

# --- Credit subtargets (ONLY existing levels) ---
credit_mask = (b5 == "Credit")
credit_muni_mask  = credit_mask & (leaves["b_level6"] == "Municipal Trading")
credit_cds_mask   = credit_mask & (b7u.str.contains("CDS", na=False))
credit_index_mask = credit_mask & (
    b7u.str.contains("CDX", na=False) | b7u.str.contains("CMBX", na=False) | b7u.str.contains("INDEX", na=False)
)
credit_corp_mask  = credit_mask & ~(credit_muni_mask | credit_cds_mask | credit_index_mask)

idx_corp  = leaves.index[credit_corp_mask].tolist()
idx_muni  = leaves.index[credit_muni_mask].tolist()
idx_cds   = leaves.index[credit_cds_mask].tolist()
idx_index = leaves.index[credit_index_mask].tolist()

leaves.loc[idx_corp,  "leaf_target"] = allocate(CREDIT_SUBTARGETS["CREDIT_CORP"],  idx_corp,  seed=201).values
leaves.loc[idx_muni,  "leaf_target"] = allocate(CREDIT_SUBTARGETS["CREDIT_MUNI"],  idx_muni,  seed=202).values
leaves.loc[idx_cds,   "leaf_target"] = allocate(CREDIT_SUBTARGETS["CREDIT_CDS"],   idx_cds,   seed=203).values
leaves.loc[idx_index, "leaf_target"] = allocate(CREDIT_SUBTARGETS["CREDIT_INDEX"], idx_index, seed=204).values

# --- IR and FX totals ---
for bucket, tgt, seed in [
    ("Interest Rates", TARGETS["Interest Rates"], 301),
    ("FX",             TARGETS["FX"],             302),
]:
    idxs = leaves.index[b5 == bucket].tolist()
    leaves.loc[idxs, "leaf_target"] = allocate(tgt, idxs, seed=seed).values

# --- Equities platform total across b_level4 == Equities ---
idx_eq = leaves.index[b4 == "Equities"].tolist()
leaves.loc[idx_eq, "leaf_target"] = allocate(TARGETS["__EQUITIES_PLATFORM__"], idx_eq, seed=401, sigma=0.5).values

# Final exact firm scaling
total = leaves["leaf_target"].sum()
if abs(total) < 1e-9:
    raise RuntimeError("Leaf targets sum to ~0; check your hierarchy masks.")
leaves["leaf_target"] *= (FIRM_TARGET / total)

# =========================
# BUILD fact_var_contrib (exact per-leaf reconciliation)
# =========================
rng = np.random.default_rng(777)
fact_parts = []

for _, row in leaves.iterrows():
    leaf_target = float(row["leaf_target"])
    n_rf = random.randint(RF_PER_LEAF_MIN, RF_PER_LEAF_MAX)
    rfs = choose_risk_factors(row, n_rf)
    if not rfs:
        continue

    n = len(rfs)
    w = rng.lognormal(mean=0.0, sigma=1.0, size=n)
    w = w / w.sum()

    # Allow a few positive contributions (hedges); more for Equities and CDS
    pos_prob = 0.06
    if row["b_level4"] == "Equities":
        pos_prob = 0.12
    if (row["b_level5"] == "Credit") and ("CDS" in row["b_level7"].upper()):
        pos_prob = 0.18

    signs = np.where(rng.random(n) < pos_prob, 1.0, -1.0)

    # heavy-tailed magnitude baseline
    raw = signs * w * (abs(leaf_target) * 1.6 + 5_000_000.0)

    # Exact reconcile at leaf
    raw += (leaf_target - raw.sum()) / n

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
        "riskFactor": rfs,
        "value_usd": raw.astype("float64"),
    })
    fact_parts.append(tmp)

fact = pd.concat(fact_parts, ignore_index=True).merge(rf, on="riskFactor", how="left")
fact = fact[
    ["as_of_date","metric",
     "b_level1","b_level2","b_level3","b_level4","b_level5","b_level6","b_level7",
     "riskFactor","rf_level1","rf_level2","rf_level3","rf_level4","rf_level5",
     "value_usd"]
]

# =========================
# agg_var_business
# =========================
def business_rollup(df: pd.DataFrame, level_n: int) -> pd.DataFrame:
    keep = [f"b_level{i}" for i in range(1, level_n + 1)]
    grp = df.groupby(["as_of_date","metric"] + keep, as_index=False)["value_usd"].sum()
    grp["agg_level"] = f"b_level{level_n}"
    for j in range(level_n + 1, 8):
        grp[f"b_level{j}"] = ""
    return grp[["as_of_date","metric","agg_level"] + [f"b_level{i}" for i in range(1,8)] + ["value_usd"]]

agg_business = pd.concat([business_rollup(fact, n) for n in range(7, 0, -1)], ignore_index=True)

# =========================
# agg_var_explain
# =========================
def explain_rollup(df: pd.DataFrame, business_level_n: int, rf_level_k: int) -> pd.DataFrame:
    b_keep = [f"b_level{i}" for i in range(1, business_level_n + 1)]
    r_keep = [f"rf_level{i}" for i in range(1, rf_level_k + 1)]
    grp = df.groupby(["as_of_date","metric"] + b_keep + r_keep, as_index=False)["value_usd"].sum()
    grp["business_agg_level"] = f"b_level{business_level_n}"
    grp["rf_agg_level"] = f"rf_level{rf_level_k}"
    for j in range(business_level_n + 1, 8):
        grp[f"b_level{j}"] = ""
    for j in range(rf_level_k + 1, 6):
        grp[f"rf_level{j}"] = ""
    return grp[
        ["as_of_date","metric","business_agg_level"] +
        [f"b_level{i}" for i in range(1,8)] +
        ["rf_agg_level"] +
        [f"rf_level{i}" for i in range(1,6)] +
        ["value_usd"]
    ]

agg_explain = pd.concat(
    [explain_rollup(fact, bn, rk) for bn in range(1, 8) for rk in range(1, 6)],
    ignore_index=True
)

# =========================
# RECON CHECKS + OUTPUT
# =========================
firm_total = float(agg_business.loc[agg_business["agg_level"]=="b_level1","value_usd"].sum())
firm_rf1_total = float(
    agg_explain.loc[
        (agg_explain["business_agg_level"]=="b_level1") & (agg_explain["rf_agg_level"]=="rf_level1"),
        "value_usd"
    ].sum()
)

top_b5 = (
    agg_business[agg_business["agg_level"]=="b_level5"]
    .groupby("b_level5", as_index=False)["value_usd"].sum()
    .sort_values("value_usd")
)

recon_report = pd.DataFrame([{
    "fact_rows": len(fact),
    "agg_var_business_rows": len(agg_business),
    "agg_var_explain_rows": len(agg_explain),
    "firm_total": firm_total,
    "delta_to_target": firm_total - FIRM_TARGET,
    "firm_rf_level1_total": firm_rf1_total,
    "rf1_minus_firm": firm_rf1_total - firm_total
}])

OUT_EXCEL = "reconciled_var_explainability_2020-03-09.xlsx"
with pd.ExcelWriter(OUT_EXCEL, engine="openpyxl") as writer:
    fact.to_excel(writer, sheet_name="fact_var_contrib", index=False)
    agg_business.to_excel(writer, sheet_name="agg_var_business", index=False)
    agg_explain.to_excel(writer, sheet_name="agg_var_explain", index=False)
    recon_report.to_excel(writer, sheet_name="RECON_REPORT", index=False)
    top_b5.to_excel(writer, sheet_name="TOP_b_level5", index=False)
    pd.DataFrame({"notes":[
        "No derived business bucketing columns were used. Only b_level1..b_level7 from the hierarchy input.",
        "Leaf reconciliation: for each (b_level1..b_level7) leaf, sum over riskFactor contributions == leaf VaR99_PnL.",
        "All rollups are computed by pure summation from fact_var_contrib."
    ]}).to_excel(writer, sheet_name="README", index=False)

print("Wrote:", OUT_EXCEL)
print(recon_report.to_string(index=False))
