import pandas as pd
import numpy as np
import random
import gzip
from pathlib import Path

# =========================
# USER CONFIG
# =========================
BUSINESS_FILE = "CIB_Markets_Business_Hierarchy.xlsx"
RISK_FILE     = "risk_factor_hierarchy_186k.xlsx"

AS_OF_DATE = "2020-03-09"
METRIC = "VaR99_PnL"
FIRM_TARGET = -600_000_000.0

# Keep base rows small (<< 1,000,000)
RF_PER_LEAF_MIN = 50
RF_PER_LEAF_MAX = 120

OUT_EXCEL = "reconciled_var_fact_and_business_2020-03-09.xlsx"
OUT_EXPLAIN_CSVGZ = "reconciled_var_agg_explain_2020-03-09.csv.gz"

random.seed(42)
np.random.seed(42)

# =========================
# SHAPE TARGETS (your narrative)
# =========================
B5_TARGETS = {
    "Securitized Products": -300_000_000.0,
    "Credit":              -200_000_000.0,
    "Interest Rates":      -100_000_000.0,
    "FX":                  -100_000_000.0,
    "__EQUITIES__":        +100_000_000.0,   # all Equities platform
    "__OTHER__":            0.0,
}

SEC_SUBTARGETS = {
    "RMBS_CMO":   -150_000_000.0,
    "RMBS_TBA":   -100_000_000.0,
    "SEC_OTHER":   -50_000_000.0,
}

CREDIT_SUBTARGETS = {
    "CORP":   -180_000_000.0,
    "MUNI":    -20_000_000.0,
    "CDS":     +10_000_000.0,
    "AGENCY":  -10_000_000.0,  # interpret as index/agency-like (e.g., CDX)
}

# =========================
# LOAD INPUTS
# =========================
biz = pd.read_excel(BUSINESS_FILE, sheet_name="business_hierarchy" , usecols=["b_level1","b_level2","b_level3","b_level4","b_level5","b_level6","b_level7"])
rf  = pd.read_excel(RISK_FILE, sheet_name="risk_factor_hierarchy",usecols=["riskFactor","rf_level1","rf_level2","rf_level3","rf_level4","rf_level5"])

for c in biz.columns:
    biz[c] = biz[c].fillna("").astype(str)
for c in rf.columns:
    rf[c] = rf[c].fillna("").astype(str)

def pool(expr: str) -> pd.DataFrame:
    return rf.query(expr).reset_index(drop=True)

# CP pools by product
cp_rate_rmbs = pool("rf_level1=='CP_RATE' and rf_level3=='RMBS'")
cp_rate_cmbs = pool("rf_level1=='CP_RATE' and rf_level3=='CMBS'")
cp_rate_abs  = pool("rf_level1=='CP_RATE' and rf_level3=='ABS'")
cp_rate_corp = pool("rf_level1=='CP_RATE' and rf_level3=='CORP'")
cp_rate_cds  = pool("rf_level1=='CP_RATE' and rf_level3=='CDS'")
cp_rate_cdx  = pool("rf_level1=='CP_RATE' and rf_level3=='CDX'")
cp_rate_muni = pool("rf_level1=='CP_RATE' and rf_level3=='Muni'")

cp_vol_rmbs  = pool("rf_level1=='CP_VOL' and rf_level3=='RMBS'")
cp_vol_cmbs  = pool("rf_level1=='CP_VOL' and rf_level3=='CMBS'")
cp_vol_abs   = pool("rf_level1=='CP_VOL' and rf_level3=='ABS'")
cp_vol_corp  = pool("rf_level1=='CP_VOL' and rf_level3=='CORP'")
cp_vol_cds   = pool("rf_level1=='CP_VOL' and rf_level3=='CDS'")
cp_vol_cdx   = pool("rf_level1=='CP_VOL' and rf_level3=='CDX'")
cp_vol_muni  = pool("rf_level1=='CP_VOL' and rf_level3=='Muni'")

# Non-CP pools
ir_rate = pool("rf_level1=='IR_RATE'")
ir_vol  = pool("rf_level1=='IR_VOL'")
fx_rate = pool("rf_level1=='FX_RATE'")
fx_vol  = pool("rf_level1=='FX_VOL'")
eq_price= pool("rf_level1=='EQ_PRICE'")
eq_vol  = pool("rf_level1=='EQ_VOL'")
co_price= pool("rf_level1=='COMM_PRICE'")
co_vol  = pool("rf_level1=='COMM_VOL'")

# =========================
# BUSINESS BUCKETING
# =========================
def biz_bucket(row):
    if row["b_level4"] == "Equities":
        return "__EQUITIES__"
    if row["b_level5"] in ["Securitized Products","Credit","Interest Rates","FX"]:
        return row["b_level5"]
    return "__OTHER__"

def securitized_subbucket(row):
    # Example logic: RMBS desk then use b_level7 tag
    if "RMBS" in row["b_level6"].upper():
        if row["b_level7"].strip().upper() == "CMO":
            return "RMBS_CMO"
        if row["b_level7"].strip().upper().startswith("TBA"):
            return "RMBS_TBA"
    return "SEC_OTHER"

def credit_subbucket(row):
    l6 = row["b_level6"].upper()
    l7 = row["b_level7"].upper()
    if "MUNICIPAL" in l6:
        return "MUNI"
    if "CDS" in l7 or "CDS" in l6:
        return "CDS"
    if "CORP" in l7 or "CORP" in l6:
        return "CORP"
    if "CDX" in l7 or "INDEX CDS" in l7 or "TRANCHE" in l7:
        return "AGENCY"
    return "CREDIT_OTHER"

biz = biz.copy()
biz["b5_bucket"] = biz.apply(biz_bucket, axis=1)
biz["sec_subbucket"] = np.where(biz["b5_bucket"]=="Securitized Products", biz.apply(securitized_subbucket, axis=1), "")
biz["credit_subbucket"] = np.where(biz["b5_bucket"]=="Credit", biz.apply(credit_subbucket, axis=1), "")
biz["leaf_target"] = 0.0

def allocate(total, idxs, seed=0):
    rng = np.random.default_rng(seed)
    w = rng.lognormal(mean=0.0, sigma=0.6, size=len(idxs))
    w = w / w.sum()
    return pd.Series(total * w, index=idxs)

# =========================
# LEAF TARGET ALLOCATION (exactly reconciled at firm)
# =========================
# Securitized by sub-bucket
sec_cmo = biz.index[(biz["b5_bucket"]=="Securitized Products") & (biz["sec_subbucket"]=="RMBS_CMO")].tolist()
sec_tba = biz.index[(biz["b5_bucket"]=="Securitized Products") & (biz["sec_subbucket"]=="RMBS_TBA")].tolist()
sec_oth = biz.index[(biz["b5_bucket"]=="Securitized Products") & (biz["sec_subbucket"]=="SEC_OTHER")].tolist()

if sec_cmo: biz.loc[sec_cmo,"leaf_target"] = allocate(SEC_SUBTARGETS["RMBS_CMO"], sec_cmo, seed=1).values
if sec_tba: biz.loc[sec_tba,"leaf_target"] = allocate(SEC_SUBTARGETS["RMBS_TBA"], sec_tba, seed=2).values
if sec_oth: biz.loc[sec_oth,"leaf_target"] = allocate(SEC_SUBTARGETS["SEC_OTHER"], sec_oth, seed=3).values

# Credit by sub-bucket
for sub, tgt, seed in [
    ("CORP",   CREDIT_SUBTARGETS["CORP"],   11),
    ("MUNI",   CREDIT_SUBTARGETS["MUNI"],   12),
    ("CDS",    CREDIT_SUBTARGETS["CDS"],    13),
    ("AGENCY", CREDIT_SUBTARGETS["AGENCY"], 14),
]:
    idxs = biz.index[(biz["b5_bucket"]=="Credit") & (biz["credit_subbucket"]==sub)].tolist()
    if idxs:
        biz.loc[idxs,"leaf_target"] = allocate(tgt, idxs, seed=seed).values

# IR/FX/Equities
for bucket, tgt, seed in [
    ("Interest Rates", B5_TARGETS["Interest Rates"], 21),
    ("FX",             B5_TARGETS["FX"],             22),
    ("__EQUITIES__",   B5_TARGETS["__EQUITIES__"],   23),
]:
    idxs = biz.index[biz["b5_bucket"]==bucket].tolist()
    if idxs:
        biz.loc[idxs,"leaf_target"] = allocate(tgt, idxs, seed=seed).values

# Firm exact scale
biz["leaf_target"] *= (FIRM_TARGET / biz["leaf_target"].sum())

# =========================
# RISK FACTOR ASSIGNMENT
# =========================
def sample_from(df_pool, n):
    if len(df_pool) == 0:
        return []
    n = min(n, len(df_pool))
    return df_pool.sample(n=n, replace=False, random_state=random.randint(0, 10_000))["riskFactor"].tolist()

def assign_risk_factors(row, n):
    bucket = row["b5_bucket"]
    if bucket == "Securitized Products":
        n_rate = int(n*0.7); n_vol = n-n_rate
        l6u = row["b_level6"].upper()
        if "CMBS" in l6u:
            return (sample_from(cp_rate_cmbs,n_rate) + sample_from(cp_vol_cmbs,n_vol))[:n]
        if "ABS" in l6u:
            return (sample_from(cp_rate_abs,n_rate)  + sample_from(cp_vol_abs,n_vol))[:n]
        return (sample_from(cp_rate_rmbs,n_rate) + sample_from(cp_vol_rmbs,n_vol))[:n]

    if bucket == "Credit":
        n_rate = int(n*0.7); n_vol = n-n_rate
        sub = row["credit_subbucket"]
        if sub == "MUNI":
            return (sample_from(cp_rate_muni,n_rate) + sample_from(cp_vol_muni,n_vol))[:n]
        if sub == "CDS":
            return (sample_from(cp_rate_cds,n_rate)  + sample_from(cp_vol_cds,n_vol))[:n]
        if sub == "AGENCY":
            return (sample_from(cp_rate_cdx,n_rate)  + sample_from(cp_vol_cdx,n_vol))[:n]
        return (sample_from(cp_rate_corp,n_rate) + sample_from(cp_vol_corp,n_vol))[:n]

    if bucket == "Interest Rates":
        n_rate = int(n*0.7); n_vol = n-n_rate
        return (sample_from(ir_rate,n_rate) + sample_from(ir_vol,n_vol))[:n]

    if bucket == "FX":
        n_rate = int(n*0.7); n_vol = n-n_rate
        return (sample_from(fx_rate,n_rate) + sample_from(fx_vol,n_vol))[:n]

    if bucket == "__EQUITIES__":
        n_p = int(n*0.6); n_v = n-n_p
        return (sample_from(eq_price,n_p) + sample_from(eq_vol,n_v))[:n]

    # OTHER
    mix = [co_price, co_vol, ir_rate, fx_rate, eq_price, cp_rate_corp]
    picks = []
    per = max(1, n//len(mix))
    for p in mix:
        picks += sample_from(p, per)
    if len(picks) < n:
        picks += sample_from(ir_rate, n-len(picks))
    return picks[:n]

# =========================
# BUILD fact_var_contrib (exact leaf reconciliation)
# =========================
rng = np.random.default_rng(7)
fact_parts = []

for _, row in biz.iterrows():
    n_rf = random.randint(RF_PER_LEAF_MIN, RF_PER_LEAF_MAX)
    rfs = assign_risk_factors(row, n_rf)
    n = len(rfs)
    if n == 0:
        continue

    # heavy-tailed weights
    mag = rng.lognormal(mean=0.0, sigma=1.0, size=n)
    mag = mag / mag.sum()

    # allow hedges
    if row["b5_bucket"]=="Credit" and row["credit_subbucket"]=="CDS":
        signs = np.where(rng.random(n) < 0.18, 1.0, -1.0)
    else:
        signs = np.where(rng.random(n) < 0.06, 1.0, -1.0)

    contrib = signs * mag

    # Stress scaling: securitized/credit larger
    base_notional = 1.8 if row["b5_bucket"] in ["Securitized Products","Credit"] else (1.2 if row["b5_bucket"] in ["Interest Rates","FX"] else 1.1)

    raw_scale = abs(row["leaf_target"]) * base_notional
    raw = contrib * raw_scale

    # FORCE exact sum to leaf_target by constant offset
    offset = (row["leaf_target"] - raw.sum()) / n
    val = raw + offset

    tmp = pd.DataFrame({
        "as_of_date": AS_OF_DATE,
        "metric": METRIC,
        "riskFactor": rfs,
        "value_usd": val.astype("float64"),
    })
    for lvl in [f"b_level{i}" for i in range(1,8)]:
        tmp[lvl] = row[lvl]
    fact_parts.append(tmp)

fact = pd.concat(fact_parts, ignore_index=True)
fact = fact.merge(rf, on="riskFactor", how="left")

fact = fact[["as_of_date","metric"] +
            [f"b_level{i}" for i in range(1,8)] +
            ["riskFactor"] +
            [f"rf_level{i}" for i in range(1,6)] +
            ["value_usd"]]

# =========================
# agg_var_business
# =========================
def business_rollup(fact_df, level_n):
    keep = [f"b_level{i}" for i in range(1, level_n+1)]
    grp = fact_df.groupby(["as_of_date","metric"] + keep, as_index=False)["value_usd"].sum()
    grp["agg_level"] = f"b_level{level_n}"
    for j in range(level_n+1, 8):
        grp[f"b_level{j}"] = ""
    return grp[["as_of_date","metric","agg_level"] + [f"b_level{i}" for i in range(1,8)] + ["value_usd"]]

agg_business = pd.concat([business_rollup(fact, n) for n in range(7,0,-1)], ignore_index=True)

# =========================
# agg_var_explain (stream to gzip CSV)
# =========================
def explain_rollup(fact_df, business_level_n, rf_level_k):
    b_keep = [f"b_level{i}" for i in range(1, business_level_n+1)]
    r_keep = [f"rf_level{i}" for i in range(1, rf_level_k+1)]
    grp = fact_df.groupby(["as_of_date","metric"] + b_keep + r_keep, as_index=False)["value_usd"].sum()
    grp["business_agg_level"] = f"b_level{business_level_n}"
    grp["rf_agg_level"] = f"rf_level{rf_level_k}"
    for j in range(business_level_n+1, 8):
        grp[f"b_level{j}"] = ""
    for j in range(rf_level_k+1, 6):
        grp[f"rf_level{j}"] = ""
    return grp[["as_of_date","metric","business_agg_level"] +
               [f"b_level{i}" for i in range(1,8)] +
               ["rf_agg_level"] +
               [f"rf_level{i}" for i in range(1,6)] +
               ["value_usd"]]

header_written = False
with gzip.open(OUT_EXPLAIN_CSVGZ, "wt", newline="") as gz:
    for bn in range(1,8):
        for rk in range(1,6):
            chunk = explain_rollup(fact, bn, rk)
            chunk.to_csv(gz, index=False, header=not header_written)
            header_written = True

# =========================
# RECON REPORT
# =========================
firm_total = float(agg_business.loc[agg_business["agg_level"]=="b_level1","value_usd"].sum())
firm_rf1_total = float(explain_rollup(fact, 1, 1)["value_usd"].sum())

recon = pd.DataFrame([{
    "fact_rows": len(fact),
    "agg_business_rows": len(agg_business),
    "firm_total": firm_total,
    "delta_to_target": firm_total - FIRM_TARGET,
    "firm_rf_level1_total": firm_rf1_total,
    "rf1_minus_firm": firm_rf1_total - firm_total,
}])

top_b5 = (agg_business[agg_business["agg_level"]=="b_level5"]
          .groupby("b_level5", as_index=False)["value_usd"].sum()
          .sort_values("value_usd"))

# =========================
# WRITE EXCEL (fact + business + reports)
# =========================
with pd.ExcelWriter(OUT_EXCEL, engine="openpyxl") as writer:
    fact.to_excel(writer, sheet_name="fact_var_contrib", index=False)
    agg_business.to_excel(writer, sheet_name="agg_var_business", index=False)
    recon.to_excel(writer, sheet_name="RECON_REPORT", index=False)
    top_b5.to_excel(writer, sheet_name="TOP_b_level5", index=False)
    pd.DataFrame({"note":[
        f"Explainability rollups are in {OUT_EXPLAIN_CSVGZ} (compressed CSV).",
        "All totals reconcile: aggregates are computed from fact_var_contrib via summation."
    ]}).to_excel(writer, sheet_name="README", index=False)

print("DONE")
print("Wrote:", OUT_EXCEL)
print("Wrote:", OUT_EXPLAIN_CSVGZ)
print(recon.to_string(index=False))
