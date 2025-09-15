# SVaR / PnL Attribution — Interactive Charts

import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display

# Matplotlib defaults
plt.rcParams.update({
    "figure.figsize": (10, 5),
    "axes.grid": True
})

# ---------------- Load Data ----------------
excel_path = "data_rf_pnl.xlsx"   # <-- put your file path here
sheet_name = "TopRiskFactor_Attribution"

if not os.path.exists(excel_path):
    raise FileNotFoundError(f"Excel not found: {excel_path}")

df = pd.read_excel(excel_path, sheet_name=sheet_name)
df.columns = [str(c).strip() for c in df.columns]

expected_cols = ["deskID","deskName","businessDate","closeDate","assetClass","driverGroup","sumPnL"]
missing = [c for c in expected_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}\nFound: {list(df.columns)}")

for c in ["businessDate","closeDate"]:
    if not pd.api.types.is_datetime64_any_dtype(df[c]):
        df[c] = pd.to_datetime(df[c], errors="coerce")

# Combine factor label
df["factor"] = df["assetClass"].astype(str).str.strip() + " | " + df["driverGroup"].astype(str).str.strip()
df = df.sort_values(["deskName","factor","businessDate"]).reset_index(drop=True)

# Output directory for saving PNGs
out_dir = Path("rf_pnl_charts")
out_dir.mkdir(parents=True, exist_ok=True)

# ---------------- Summary ----------------
summary = (
    df.groupby(["deskName","factor"], as_index=False)
      .agg(first_date=("businessDate","min"),
           last_date=("businessDate","max"),
           obs=("sumPnL","count"),
           mean_pnl=("sumPnL","mean"),
           std_pnl=("sumPnL","std"),
           total_pnl=("sumPnL","sum"),
           total_abs_pnl=("sumPnL", lambda s: s.abs().sum()))
      .sort_values(["deskName","total_abs_pnl"], ascending=[True, False])
)
display(summary)

# ---------------- Plot Function ----------------
def render_plots_for_desk(desk_name: str, top_n: int = 8, bins: int = 30, save_png: bool = True):
    d = df[df["deskName"] == desk_name].copy()
    if d.empty:
        print(f"No data for desk '{desk_name}'")
        return None

    # Top factors by absolute PnL
    factor_abs = d.groupby("factor")["sumPnL"].apply(lambda s: s.abs().sum()).sort_values(ascending=False)
    top_factors = factor_abs.head(top_n).index.tolist()

    # ---- Line chart ----
    piv = d.pivot_table(index="businessDate", columns="factor", values="sumPnL", aggfunc="sum").sort_index()
    cols = [c for c in piv.columns if c in top_factors]

    plt.figure()
    for col in cols:
        plt.plot(piv.index, piv[col], label=col)
    plt.title(f"{desk_name} — SumPnL by Factor (Line)")
    plt.xlabel("Business Date")
    plt.ylabel("Sum PnL")
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    if save_png:
        plt.savefig(out_dir / f"{desk_name}_line.png", dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()

    # ---- Histogram ----
    plt.figure()
    for f in top_factors:
        vals = d.loc[d["factor"] == f, "sumPnL"].dropna().values
        if len(vals) > 0:
            plt.hist(vals, bins=bins, alpha=0.5, label=f, histtype="stepfilled")
    plt.title(f"{desk_name} — SumPnL Distribution by Factor (Histogram)")
    plt.xlabel("Sum PnL")
    plt.ylabel("Frequency")
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    if save_png:
        plt.savefig(out_dir / f"{desk_name}_hist.png", dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()

    # ---- Bar chart (hedging-down / loss-up) ----
    agg = (d.groupby("factor", as_index=False)["sumPnL"]
             .sum()
             .sort_values("sumPnL", key=lambda s: s.abs(), ascending=False))
    agg_top = agg[agg["factor"].isin(top_factors)].copy()
    agg_top["plot_value"] = -agg_top["sumPnL"]  # flip: +PnL down, -PnL up

    plt.figure()
    x = np.arange(len(agg_top))
    plt.bar(x, agg_top["plot_value"])
    plt.axhline(0, linewidth=1)
    plt.xticks(x, agg_top["factor"], rotation=30, ha="right")
    plt.title(f"{desk_name} — Total SumPnL by Factor (Bar: Hedging Down, Loss Up)")
    plt.ylabel("Plot Value (= -Sum PnL)")
    plt.tight_layout()
    if save_png:
        plt.savefig(out_dir / f"{desk_name}_bar_attribution.png", dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()

# ---------------- Widgets ----------------
desks = sorted(df['deskName'].dropna().unique().tolist())
desk_dd = widgets.Dropdown(options=desks, description="Desk:", value=desks[0] if desks else None)
topn_slider = widgets.IntSlider(value=8, min=1, max=20, step=1, description='Top N factors:')
bins_slider = widgets.IntSlider(value=30, min=10, max=100, step=5, description='Histogram bins:')
run_btn = widgets.Button(description="Render (Line + Hist + Bar)")

out = widgets.Output()

def on_click(_):
    with out:
        out.clear_output(wait=True)
        render_plots_for_desk(desk_dd.value, topn_slider.value, bins_slider.value, save_png=True)

run_btn.on_click(on_click)
display(widgets.VBox([desk_dd, topn_slider, bins_slider, run_btn, out]))
