import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---------- Load ----------
excel_path = "/mnt/data/data_rf_pnl.xlsx"
sheet_name = "TopRiskFactor_Attribution"

df = pd.read_excel(excel_path, sheet_name=sheet_name)
df.columns = [str(c).strip() for c in df.columns]

# Parse dates
for c in ["businessDate","closeDate"]:
    if not pd.api.types.is_datetime64_any_dtype(df[c]):
        df[c] = pd.to_datetime(df[c], errors="coerce")

# Combine factor label
df["factor"] = df["assetClass"].astype(str).str.strip() + " | " + df["driverGroup"].astype(str).str.strip()
df = df.sort_values(["deskName","factor","businessDate"])

# ---------- Output dir ----------
out_dir = Path("/mnt/data/rf_pnl_charts")
out_dir.mkdir(parents=True, exist_ok=True)

# ---------- Plot function ----------
def render_plots_for_desk_with_bar(desk_name: str, top_n: int = 8, bins: int = 30, save_png: bool = True):
    d = df[df["deskName"] == desk_name].copy()
    if d.empty:
        print(f"No data for desk '{desk_name}'")
        return None

    # Choose top factors by absolute PnL
    factor_abs = d.groupby("factor")["sumPnL"].apply(lambda s: s.abs().sum()).sort_values(ascending=False)
    top_factors = factor_abs.head(top_n).index.tolist()

    # ----- LINE CHART -----
    piv = d.pivot_table(index="businessDate", columns="factor", values="sumPnL", aggfunc="sum").sort_index()
    cols = [c for c in piv.columns if c in top_factors]

    plt.figure(figsize=(10,5))
    for col in cols:
        plt.plot(piv.index, piv[col], label=col)
    plt.title(f"{desk_name} — SumPnL by Factor (Line)")
    plt.xlabel("Business Date")
    plt.ylabel("Sum PnL")
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    line_path = out_dir / f"{desk_name}_line.png"
    if save_png:
        plt.savefig(line_path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()

    # ----- HISTOGRAM -----
    plt.figure(figsize=(10,5))
    for f in top_factors:
        vals = d.loc[d["factor"] == f, "sumPnL"].dropna().values
        if len(vals) > 0:
            plt.hist(vals, bins=bins, alpha=0.5, label=f, histtype="stepfilled")
    plt.title(f"{desk_name} — SumPnL Distribution by Factor (Histogram)")
    plt.xlabel("Sum PnL")
    plt.ylabel("Frequency")
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    hist_path = out_dir / f"{desk_name}_hist.png"
    if save_png:
        plt.savefig(hist_path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()

    # ----- BAR CHART (Hedging-Down / Loss-Up) -----
    agg = (d.groupby("factor", as_index=False)["sumPnL"]
           .sum()
           .sort_values("sumPnL", key=lambda s: s.abs(), ascending=False))
    agg_top = agg[agg["factor"].isin(top_factors)].copy()
    agg_top["plot_value"] = -agg_top["sumPnL"]  # flip sign

    plt.figure(figsize=(10,5))
    x = np.arange(len(agg_top))
    plt.bar(x, agg_top["plot_value"])
    plt.axhline(0, linewidth=1)
    plt.xticks(x, agg_top["factor"], rotation=30, ha="right")
    plt.title(f"{desk_name} — Total SumPnL by Factor (Bar: Hedging Down, Loss Up)")
    plt.ylabel("Plot Value (= -Sum PnL)")
    plt.tight_layout()
    bar_path = out_dir / f"{desk_name}_bar_attribution.png"
    if save_png:
        plt.savefig(bar_path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()

    return str(line_path), str(hist_path), str(bar_path)

# ---------- Run for all desks ----------
desks = sorted(df["deskName"].dropna().unique().tolist())
for dname in desks:
    render_plots_for_desk_with_bar(dname, top_n=8, bins=30, save_png=True)
