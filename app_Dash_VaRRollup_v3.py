#!/usr/bin/env python3
"""
VaR Explainability Dashboard (Dash + Dash AG Grid + Plotly)
- Loads: reconciled_var_demo_2020-03-09.xlsx
- Business Tree / Row Grouping
- Risk Factor Tree / Row Grouping
- Explainability panel (cards + tables + charts) with reconciliation delta
- Clickable breadcrumbs for business + risk-factor drill paths
- Uses Polars for filtering/aggregation where feasible
"""

from __future__ import annotations

import math
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import polars as pl
import pandas as pd  # Excel reader fallback; converted to Polars immediately
from dash import Dash, dcc, html, Input, Output, State, callback_context, no_update, ALL
import dash_ag_grid as dag
import plotly.graph_objects as go


# =========================
# CONFIG
# =========================
FILE_PATH = Path("reconciled_var_demo_2020-03-09.xlsx")  # place next to app.py
PORT = 8050

BUSINESS_LEVELS = [f"b_level{i}" for i in range(1, 8)]
RF_LEVELS = [f"rf_level{i}" for i in range(1, 6)]

BUSINESS_DEPTH_OPTIONS = [{"label": f"b_level{i}", "value": i} for i in range(1, 8)]
RF_DEPTH_OPTIONS = [{"label": f"rf_level{i}", "value": i} for i in range(1, 6)]

DEFAULT_BUSINESS_DEPTH = 7
DEFAULT_RF_DEPTH = 2
DEFAULT_TOPN = 15


# =========================
# UTIL: formatting
# =========================
def fmt_mm(x: float) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return ""
    return f"{x/1e6:,.1f}"

def normalize_path(levels: List[str]) -> List[str]:
    """Trim blanks and whitespace; keep order."""
    out = []
    for s in levels:
        s = (s or "").strip()
        if s:
            out.append(s)
    return out

def business_path_from_row(row: Dict[str, Any], depth: int) -> List[str]:
    return normalize_path([row.get(f"b_level{i}", "") for i in range(1, depth + 1)])

def rf_path_from_row(row: Dict[str, Any], depth: int) -> List[str]:
    return normalize_path([row.get(f"rf_level{i}", "") for i in range(1, depth + 1)])

def path_str(path: List[str]) -> str:
    return " / ".join(path) if path else ""

def sign_label(v: float) -> str:
    return "Gain" if v > 0 else "Loss"


# =========================
# DATA LOADING (Excel -> Polars)
# =========================
def read_excel_to_polars(xlsx_path: Path, sheet: str, usecols: Optional[List[str]] = None) -> pl.DataFrame:
    """
    Robust Excel load: pandas/openpyxl -> Polars.
    (Polars read_excel availability varies by version; this is stable.)
    """
    df = pd.read_excel(xlsx_path, sheet_name=sheet, usecols=usecols, engine="openpyxl")
    pl_df = pl.from_pandas(df)
    del df
    return pl_df

def load_data(xlsx_path: Path) -> Dict[str, pl.DataFrame]:
    if not xlsx_path.exists():
        raise FileNotFoundError(f"Could not find {xlsx_path.resolve()}")

    agg_bus = read_excel_to_polars(
        xlsx_path, "agg_var_business",
        usecols=["as_of_date","metric","agg_level"] + BUSINESS_LEVELS + ["value_usd"]
    )
    agg_exp = read_excel_to_polars(
        xlsx_path, "agg_var_explain",
        usecols=["as_of_date","metric","business_agg_level"] + BUSINESS_LEVELS + ["rf_agg_level"] + RF_LEVELS + ["value_usd"]
    )
    fact = read_excel_to_polars(
        xlsx_path, "leaf_rf_contrib",
        usecols=["as_of_date","metric"] + BUSINESS_LEVELS + ["riskFactor"] + RF_LEVELS + ["value_usd"]
    )

    def clean_strings(df: pl.DataFrame, cols: List[str]) -> pl.DataFrame:
        for c in cols:
            df = df.with_columns(
                pl.when(pl.col(c).is_null())
                .then(pl.lit(""))
                .otherwise(pl.col(c).cast(pl.Utf8))
                .alias(c)
            )
        return df

    agg_bus = clean_strings(agg_bus, ["agg_level"] + BUSINESS_LEVELS)
    agg_exp = clean_strings(agg_exp, ["business_agg_level","rf_agg_level"] + BUSINESS_LEVELS + RF_LEVELS)
    fact    = clean_strings(fact, BUSINESS_LEVELS + ["riskFactor"] + RF_LEVELS)

    return {"agg_bus": agg_bus, "agg_exp": agg_exp, "fact": fact}


DATA = load_data(FILE_PATH)
AGG_BUS = DATA["agg_bus"]
AGG_EXP = DATA["agg_exp"]
FACT    = DATA["fact"]

# A convenient firm node label (usually Top_of_House)
firm_nodes = (
    AGG_BUS.filter(pl.col("agg_level") == "b_level1")
    .select("b_level1").unique().to_series().to_list()
)
FIRM_NODE = firm_nodes[0] if firm_nodes else "Top_of_House"


# =========================
# FILTER HELPERS (Polars)
# =========================
def filter_by_business_path(df: pl.DataFrame, path: List[str]) -> pl.DataFrame:
    """Filter rows that match the prefix business path."""
    if not path:
        return df
    exprs = [(pl.col(f"b_level{i}") == val) for i, val in enumerate(path, start=1)]
    return df.filter(pl.all_horizontal(exprs))

def filter_by_rf_path(df: pl.DataFrame, path: List[str]) -> pl.DataFrame:
    if not path:
        return df
    exprs = [(pl.col(f"rf_level{i}") == val) for i, val in enumerate(path, start=1)]
    return df.filter(pl.all_horizontal(exprs))

def get_business_node_total(biz_path: List[str]) -> float:
    """Get business node total from agg_var_business by agg_level=len(path)."""
    level_n = max(1, len(biz_path))
    df = AGG_BUS.filter(pl.col("agg_level") == f"b_level{level_n}")
    df = filter_by_business_path(df, biz_path)
    out = df.select(pl.col("value_usd").sum()).item()
    return float(out) if out is not None else 0.0

def get_rf_breakdown_for_business(biz_path: List[str], rf_depth: int) -> pl.DataFrame:
    """Get agg_var_explain slice for business node and rf_depth."""
    biz_level_n = max(1, len(biz_path))
    df = AGG_EXP.filter(
        (pl.col("business_agg_level") == f"b_level{biz_level_n}") &
        (pl.col("rf_agg_level") == f"rf_level{rf_depth}")
    )
    df = filter_by_business_path(df, biz_path)
    return df

def rf_contrib_total_within_business(biz_path: List[str], rf_depth: int) -> float:
    df = get_rf_breakdown_for_business(biz_path, rf_depth)
    out = df.select(pl.col("value_usd").sum()).item()
    return float(out) if out is not None else 0.0

def rf_selected_contribution(biz_path: List[str], rf_path: List[str], rf_depth: int) -> float:
    df = get_rf_breakdown_for_business(biz_path, rf_depth)
    df = filter_by_rf_path(df, rf_path)
    out = df.select(pl.col("value_usd").sum()).item()
    return float(out) if out is not None else 0.0

def top_risk_factors(biz_path: List[str], rf_path: List[str], topn: int, only_negatives: bool) -> pl.DataFrame:
    """Top individual riskFactor contributions from FACT (never shipped wholesale)."""
    df = FACT
    df = filter_by_business_path(df, biz_path)
    df = filter_by_rf_path(df, rf_path)
    if only_negatives:
        df = df.filter(pl.col("value_usd") < 0)

    total = df.select(pl.col("value_usd").sum()).item()
    total = float(total) if total is not None else 0.0

    grp = (
        df.group_by("riskFactor")
        .agg(pl.col("value_usd").sum().alias("value_usd"))
        .with_columns(pl.col("value_usd").abs().alias("_abs"))
        .sort("_abs", descending=True)
        .head(max(1, int(topn)))
        .drop("_abs")
    )

    if abs(total) < 1e-12:
        grp = grp.with_columns(pl.lit(0.0).alias("share_pct"))
    else:
        grp = grp.with_columns((pl.col("value_usd") / total * 100.0).alias("share_pct"))
    return grp


# =========================
# BUILD GRID ROWDATA (small slices only)
# =========================
def business_grid_rows(depth: int, biz_path: List[str]) -> List[Dict[str, Any]]:
    df = AGG_BUS.filter(pl.col("agg_level") == f"b_level{depth}")
    df = filter_by_business_path(df, biz_path)

    node_total = get_business_node_total(biz_path)
    if abs(node_total) < 1e-12:
        node_total = float(df.select(pl.col("value_usd").sum()).item() or 0.0)

    rows: List[Dict[str, Any]] = []
    for r in df.to_dicts():
        p = business_path_from_row(r, depth)
        v = float(r.get("value_usd") or 0.0)
        share = (v / node_total * 100.0) if abs(node_total) > 1e-12 else 0.0

        rows.append({
            "path": p,
            "node": p[-1] if p else "",
            "path_str": path_str(p),
            "value_usd": v,
            "value_mm": v / 1e6,
            "share_pct": share,
            "sign": sign_label(v),
            **{k: r.get(k, "") for k in BUSINESS_LEVELS},
        })
    return rows

def rf_grid_rows(biz_path: List[str], rf_depth: int, rf_path: List[str]) -> Tuple[List[Dict[str, Any]], float]:
    df = get_rf_breakdown_for_business(biz_path, rf_depth)
    df = filter_by_rf_path(df, rf_path)

    biz_total = get_business_node_total(biz_path)
    if abs(biz_total) < 1e-12:
        biz_total = float(df.select(pl.col("value_usd").sum()).item() or 0.0)

    rows: List[Dict[str, Any]] = []
    for r in df.to_dicts():
        p = rf_path_from_row(r, rf_depth)
        v = float(r.get("value_usd") or 0.0)
        share = (v / biz_total * 100.0) if abs(biz_total) > 1e-12 else 0.0

        rows.append({
            "rf_path": p,
            "rf_node": p[-1] if p else "",
            "rf_path_str": path_str(p),
            "value_usd": v,
            "value_mm": v / 1e6,
            "share_pct": share,
            "asset_class": r.get("rf_level1", ""),
            "sign": sign_label(v),
            **{k: r.get(k, "") for k in RF_LEVELS},
        })
    return rows, biz_total


# =========================
# "STAIRS" HIERARCHY VIEW (flat outline with indentation)
# =========================
def build_stairs_from_paths(
    rows: List[Dict[str, Any]],
    path_key: str,      # "path" or "rf_path"
    node_key: str,      # "node" or "rf_node"
    total_value_key: str = "value_usd",
    total_denominator: Optional[float] = None,  # for share_pct recompute (optional)
) -> List[Dict[str, Any]]:
    """
    Build an outline-style hierarchy ("stairs") without TreeData or Row Grouping.

    Input rows are assumed to represent the most granular nodes for the current view (e.g., b_level7 rows,
    or rf_levelK rows). We expand each row's path into all prefixes and sum values across those prefixes.

    Output rows:
      - level: indentation depth (1..N)
      - max_depth: overall max depth for styling
      - share_pct: recomputed vs total_denominator if provided
    """
    agg: Dict[Tuple[str, ...], float] = {}

    for r in rows:
        p = r.get(path_key) or []
        v = float(r.get(total_value_key) or 0.0)
        for k in range(1, len(p) + 1):
            prefix = tuple(p[:k])
            agg[prefix] = agg.get(prefix, 0.0) + v

    max_depth = max((len(k) for k in agg.keys()), default=1)
    denom = total_denominator
    if denom is None:
        denom = float(sum(agg.values())) if agg else 0.0

    out: List[Dict[str, Any]] = []
    for prefix in sorted(agg.keys()):
        v = float(agg[prefix])
        p_list = list(prefix)
        share = (v / denom * 100.0) if abs(denom) > 1e-12 else 0.0
        out.append({
            path_key: p_list,
            node_key: p_list[-1] if p_list else "",
            "path_str": path_str(p_list),
            "value_usd": v,
            "value_mm": v / 1e6,
            "share_pct": share,
            "level": len(p_list),
            "max_depth": max_depth,
            "sign": sign_label(v),
        })
    return out


# =========================
# AG GRID CONFIG
# =========================
JS_GET_DATA_PATH_BIZ = {"function": "return data.path;"}
JS_GET_DATA_PATH_RF  = {"function": "return data.rf_path;"}

VALUE_FORMATTER_MM = {"function": "return (params.value === null || params.value === undefined) ? '' : params.value.toLocaleString(undefined,{minimumFractionDigits:1, maximumFractionDigits:1});"}
VALUE_FORMATTER_PCT = {"function": "return (params.value === null || params.value === undefined) ? '' : params.value.toLocaleString(undefined,{minimumFractionDigits:1, maximumFractionDigits:1});"}

CELL_CLASS_RULES_VALUE = {
    "neg-cell": "x < 0",
    "pos-cell": "x > 0",
}


# =========================
# STAIRS COLUMN DEFS (indentation)
# =========================
STAIRS_NODE_CELL_STYLE = {
    "function": """
        const lvl = (params.data && params.data.level) ? params.data.level : 1;
        const isGroup = (params.data && params.data.max_depth) ? (lvl < params.data.max_depth) : false;
        return {
          paddingLeft: (10 + (lvl-1)*18) + 'px',
          fontWeight: isGroup ? '900' : '700',
        };
    """
}

BIZ_STAIRS_COL_DEFS = [
    {"headerName": "Node", "field": "node", "minWidth": 240, "cellStyle": STAIRS_NODE_CELL_STYLE},
    {"headerName": "Path", "field": "path_str", "minWidth": 320, "flex": 1},
    {"headerName": "VaR ($MM)", "field": "value_mm", "type": "numericColumn", "valueFormatter": VALUE_FORMATTER_MM,
     "cellClassRules": CELL_CLASS_RULES_VALUE},
    {"headerName": "Share %", "field": "share_pct", "type": "numericColumn", "valueFormatter": VALUE_FORMATTER_PCT},
    {"headerName": "Sign", "field": "sign", "minWidth": 120},
]

RF_STAIRS_COL_DEFS = [
    {"headerName": "RF Node", "field": "rf_node", "minWidth": 240, "cellStyle": STAIRS_NODE_CELL_STYLE},
    {"headerName": "RF Path", "field": "rf_path_str", "minWidth": 320, "flex": 1},
    {"headerName": "VaR ($MM)", "field": "value_mm", "type": "numericColumn", "valueFormatter": VALUE_FORMATTER_MM,
     "cellClassRules": CELL_CLASS_RULES_VALUE},
    {"headerName": "Share % (of business)", "field": "share_pct", "type": "numericColumn", "valueFormatter": VALUE_FORMATTER_PCT},
    {"headerName": "Asset Class", "field": "asset_class", "minWidth": 140},
]


DEFAULT_COL_DEF = {"resizable": True, "sortable": True, "filter": True, "minWidth": 110}

BIZ_COL_DEFS = [
    {"headerName": "Node", "field": "node", "minWidth": 190, "cellRenderer": "agGroupCellRenderer"},
    {"headerName": "Path", "field": "path_str", "minWidth": 320, "flex": 1},
    {"headerName": "VaR ($MM)", "field": "value_mm", "type": "numericColumn", "valueFormatter": VALUE_FORMATTER_MM,
     "cellClassRules": CELL_CLASS_RULES_VALUE},
    {"headerName": "Share %", "field": "share_pct", "type": "numericColumn", "valueFormatter": VALUE_FORMATTER_PCT},
    {"headerName": "Sign", "field": "sign", "minWidth": 120},
]

RF_COL_DEFS = [
    {"headerName": "RF Node", "field": "rf_node", "minWidth": 190, "cellRenderer": "agGroupCellRenderer"},
    {"headerName": "RF Path", "field": "rf_path_str", "minWidth": 320, "flex": 1},
    {"headerName": "VaR ($MM)", "field": "value_mm", "type": "numericColumn", "valueFormatter": VALUE_FORMATTER_MM,
     "cellClassRules": CELL_CLASS_RULES_VALUE},
    {"headerName": "Share % (of business)", "field": "share_pct", "type": "numericColumn", "valueFormatter": VALUE_FORMATTER_PCT},
    {"headerName": "Asset Class", "field": "asset_class", "minWidth": 140},
]

def biz_group_col_defs(depth: int) -> List[Dict[str, Any]]:
    defs: List[Dict[str, Any]] = []
    for i in range(1, depth + 1):
        defs.append({"field": f"b_level{i}", "rowGroup": True, "hide": True})
    defs.extend([
        {"headerName": "VaR ($MM)", "field": "value_mm", "aggFunc": "sum", "type": "numericColumn",
         "valueFormatter": VALUE_FORMATTER_MM, "cellClassRules": CELL_CLASS_RULES_VALUE},
        {"headerName": "Sign", "field": "sign", "minWidth": 120},
    ])
    return defs

def rf_group_col_defs(depth: int) -> List[Dict[str, Any]]:
    defs: List[Dict[str, Any]] = []
    for i in range(1, depth + 1):
        defs.append({"field": f"rf_level{i}", "rowGroup": True, "hide": True})
    defs.extend([
        {"headerName": "VaR ($MM)", "field": "value_mm", "aggFunc": "sum", "type": "numericColumn",
         "valueFormatter": VALUE_FORMATTER_MM, "cellClassRules": CELL_CLASS_RULES_VALUE},
        {"headerName": "Asset Class", "field": "asset_class", "minWidth": 140},
    ])
    return defs


# =========================
# UI HELPERS
# =========================
def card(title: str, value: str, subtitle: str = "", tone: str = "neutral") -> html.Div:
    tone_class = {"neutral": "card", "good": "card card-good", "bad": "card card-bad"}.get(tone, "card")
    return html.Div(
        className=tone_class,
        children=[
            html.Div(title, className="card-title"),
            html.Div(value, className="card-value"),
            html.Div(subtitle, className="card-subtitle") if subtitle else html.Div(),
        ],
    )

def make_badge(text: str, ok: bool) -> html.Span:
    return html.Span(text, className="badge badge-ok" if ok else "badge badge-bad")

def build_bar_chart(items: List[Dict[str, Any]], title: str) -> go.Figure:
    labels = [it["label"] for it in items]
    values = [it["value_mm"] for it in items]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=values,
        y=labels,
        orientation="h",
        text=[f"{v:,.1f}MM" for v in values],
        textposition="auto",
    ))
    fig.update_layout(
        title=title,
        height=380,
        margin=dict(l=10, r=10, t=50, b=10),
        yaxis=dict(autorange="reversed"),
        xaxis_title="VaR ($MM)",
        template="plotly_white",
    )
    return fig

def html_chip(text: str, idx: int, chip_type: str, title: str) -> html.Button:
    return html.Button(text, id={"type": chip_type, "index": idx}, n_clicks=0, className="chip", title=title)


# =========================
# APP
# =========================
app = Dash(__name__)
app.title = "VaR Explainability — Drillable"

INIT_BIZ_PATH = [FIRM_NODE]
INIT_RF_PATH: List[str] = []

init_biz_rows = business_grid_rows(DEFAULT_BUSINESS_DEPTH, INIT_BIZ_PATH)
init_rf_rows, init_biz_total = rf_grid_rows(INIT_BIZ_PATH, DEFAULT_RF_DEPTH, INIT_RF_PATH)

app.layout = html.Div(
    className="page",
    children=[
        html.Style("""
            :root{
              --bg:#0b1220; --panel:#0f1b2d; --card:#13233b; --text:#e7eefc; --muted:#a9b8d6;
              --accent:#4aa3ff; --good:#38d996; --bad:#ff5b6e; --border:rgba(255,255,255,0.08);
              --shadow:0 10px 25px rgba(0,0,0,0.35);
            }
            body{background:var(--bg); margin:0; font-family: ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,Arial; color:var(--text);}
            .page{padding:18px 18px 26px 18px;}
            .header{display:flex; align-items:center; justify-content:space-between; margin-bottom:12px;}
            .title{font-size:20px; font-weight:800; letter-spacing:0.2px;}
            .subtitle{color:var(--muted); font-size:12px; margin-top:4px;}
            .toolbar{display:flex; gap:10px; align-items:center; flex-wrap:wrap; background:var(--panel);
                     border:1px solid var(--border); border-radius:16px; padding:12px; box-shadow:var(--shadow);}
            .control{min-width:170px;}
            .pill{display:inline-flex; gap:6px; align-items:center; background:rgba(255,255,255,0.06);
                  border:1px solid var(--border); border-radius:999px; padding:6px 10px;}
            .badge{padding:6px 10px; border-radius:999px; font-weight:700; font-size:12px; border:1px solid var(--border);}
            .badge-ok{background:rgba(56,217,150,0.15); color:var(--good);}
            .badge-bad{background:rgba(255,91,110,0.15); color:var(--bad);}
            .btn{background:linear-gradient(135deg, rgba(74,163,255,0.25), rgba(74,163,255,0.08));
                 border:1px solid rgba(74,163,255,0.35); color:var(--text); padding:8px 12px; border-radius:12px;
                 font-weight:800; cursor:pointer;}
            .btn:active{transform:translateY(1px);}
            .crumb-row{margin-top:12px; display:grid; grid-template-columns: 1fr 1fr; gap:12px;}
            .crumb-panel{background:var(--panel); border:1px solid var(--border); border-radius:16px; padding:10px 12px; box-shadow:var(--shadow);}
            .crumb-title{font-weight:800; font-size:12px; color:var(--muted); margin-bottom:8px;}
            .chips{display:flex; gap:8px; flex-wrap:wrap; align-items:center;}
            .chip{background:rgba(255,255,255,0.06); border:1px solid var(--border); color:var(--text);
                  padding:6px 10px; border-radius:999px; cursor:pointer; font-weight:700; font-size:12px;}
            .chip:hover{border-color:rgba(74,163,255,0.55);}
            .chip-clear{background:rgba(255,91,110,0.12); border-color:rgba(255,91,110,0.35);}
            .grid{margin-top:14px; display:grid; grid-template-columns: 1.05fr 1.05fr 1.4fr; gap:12px;}
            .panel{background:var(--panel); border:1px solid var(--border); border-radius:18px; padding:12px;
                   box-shadow:var(--shadow); min-height:520px; display:flex; flex-direction:column;}
            .panel h3{margin:0 0 10px 0; font-size:14px; letter-spacing:0.2px;}
            .panel .hint{color:var(--muted); font-size:12px; margin-bottom:10px;}
            .ag-theme-alpine{
              --ag-background-color: rgba(255,255,255,0.02);
              --ag-foreground-color: var(--text);
              --ag-header-foreground-color: var(--text);
              --ag-header-background-color: rgba(255,255,255,0.03);
              --ag-row-hover-color: rgba(74,163,255,0.10);
              --ag-border-color: rgba(255,255,255,0.08);
              --ag-font-size: 12px;
              --ag-font-family: ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,Arial;
            }
            .neg-cell{color:var(--bad); background:rgba(255,91,110,0.09); font-weight:800;}
            .pos-cell{color:var(--good); background:rgba(56,217,150,0.09); font-weight:800;}
            .cards{display:grid; grid-template-columns: 1fr 1fr 1fr; gap:10px; margin-bottom:10px;}
            .card{background:var(--card); border:1px solid var(--border); border-radius:16px; padding:12px;}
            .card-title{color:var(--muted); font-size:12px; font-weight:800;}
            .card-value{font-size:22px; font-weight:900; margin-top:6px;}
            .card-subtitle{color:var(--muted); font-size:12px; margin-top:4px;}
            .card-good{border-color:rgba(56,217,150,0.35);}
            .card-bad{border-color:rgba(255,91,110,0.35);}
            .tables{display:grid; grid-template-columns: 1fr 1fr; gap:10px; margin-top:10px;}
            .mini-table{border:1px solid var(--border); border-radius:16px; overflow:hidden;}
            .mini-head{padding:10px 12px; background:rgba(255,255,255,0.03); font-weight:900;}
            .mini-body{padding:10px 12px;}
            .row{display:flex; justify-content:space-between; gap:12px; padding:6px 0;
                 border-bottom:1px solid rgba(255,255,255,0.06); font-size:12px;}
            .row:last-child{border-bottom:none;}
            .muted{color:var(--muted);}
        """),

        html.Div(className="header", children=[
            html.Div([
                html.Div("VaR Explainability — Business × Risk Factors", className="title"),
                html.Div("Drill Top of House → desks → strategies, then explain by RF hierarchy. All totals reconcile.", className="subtitle"),
            ]),
            html.Div(id="recon-badge", className="pill"),
        ]),

        html.Div(className="toolbar", children=[
            html.Div(className="control", children=[
                html.Div("Business depth", className="muted", style={"fontSize":"12px","fontWeight":"800","marginBottom":"6px"}),
                dcc.Dropdown(id="business-depth", options=BUSINESS_DEPTH_OPTIONS, value=DEFAULT_BUSINESS_DEPTH, clearable=False),
            ]),
            html.Div(className="control", children=[
                html.Div("RF depth", className="muted", style={"fontSize":"12px","fontWeight":"800","marginBottom":"6px"}),
                dcc.Dropdown(id="rf-depth", options=RF_DEPTH_OPTIONS, value=DEFAULT_RF_DEPTH, clearable=False),
            ]),
            html.Div(className="control", children=[
                html.Div("Business view", className="muted", style={"fontSize":"12px","fontWeight":"800","marginBottom":"6px"}),
                dcc.RadioItems(
                    id="business-view-mode",
                    options=[{"label":"Tree", "value":"tree"}, {"label":"Row Grouping", "value":"group"}],
                    value="tree", inline=True, inputStyle={"marginRight":"6px"},
                    labelStyle={"marginRight":"10px", "fontWeight":"800"},
                ),
            ]),
            html.Div(className="control", children=[
                html.Div("RF view", className="muted", style={"fontSize":"12px","fontWeight":"800","marginBottom":"6px"}),
                dcc.RadioItems(
                    id="rf-view-mode",
                    options=[{"label":"Tree", "value":"tree"}, {"label":"Row Grouping", "value":"group"}],
                    value="tree", inline=True, inputStyle={"marginRight":"6px"},
                    labelStyle={"marginRight":"10px", "fontWeight":"800"},
                ),
            ]),
            html.Div(className="control", style={"minWidth":"220px"}, children=[
                html.Div("Top N", className="muted", style={"fontSize":"12px","fontWeight":"800","marginBottom":"6px"}),
                dcc.Slider(
                    id="topn", min=5, max=50, step=1, value=DEFAULT_TOPN,
                    marks={5:"5", 15:"15", 30:"30", 50:"50"},
                    tooltip={"placement":"bottom", "always_visible":False},
                ),
            ]),
            html.Div(className="control", children=[
                dcc.Checklist(
                    id="only-negatives",
                    options=[{"label":"Only negatives", "value":"neg"}],
                    value=[],
                    style={"fontWeight":"900"},
                )
            ]),
            html.Button("Reset to Firm", id="reset-firm", className="btn"),
        ]),

        html.Div(className="crumb-row", children=[
            html.Div(className="crumb-panel", children=[
                html.Div("Business Breadcrumbs", className="crumb-title"),
                html.Div(id="biz-crumbs", className="chips"),
            ]),
            html.Div(className="crumb-panel", children=[
                html.Div("Risk Factor Breadcrumbs", className="crumb-title"),
                html.Div(id="rf-crumbs", className="chips"),
            ]),
        ]),

        html.Div(className="grid", children=[
            html.Div(className="panel", children=[
                html.H3("Business VaR"),
                html.Div("Click a node to drill. Use breadcrumbs to jump back up.", className="hint"),
                dcc.Input(
                    id="biz-quick-filter", placeholder="Search…", type="text",
                    style={"width":"100%","padding":"10px","borderRadius":"12px","border":"1px solid rgba(255,255,255,0.10)",
                           "background":"rgba(255,255,255,0.03)","color":"#e7eefc","marginBottom":"10px"},
                ),
                dag.AgGrid(
                    id="biz-grid", className="ag-theme-alpine",
                    columnDefs=BIZ_COL_DEFS, defaultColDef=DEFAULT_COL_DEF,
                    rowData=init_biz_rows,
                    dashGridOptions={
                        "treeData": True,
                        "getDataPath": JS_GET_DATA_PATH_BIZ,
                        "animateRows": True,
                        "rowSelection": "single",
                        "groupDefaultExpanded": 1,
                    },
                    columnSize="sizeToFit",
                    style={"height":"430px", "width":"100%"},
                ),
            ]),

            html.Div(className="panel", children=[
                html.H3("Risk Factor Explainability"),
                html.Div("RF breakdown updates as you navigate the business tree.", className="hint"),
                dcc.Input(
                    id="rf-quick-filter", placeholder="Search…", type="text",
                    style={"width":"100%","padding":"10px","borderRadius":"12px","border":"1px solid rgba(255,255,255,0.10)",
                           "background":"rgba(255,255,255,0.03)","color":"#e7eefc","marginBottom":"10px"},
                ),
                dcc.Loading(
                    type="circle",
                    children=[
                        dag.AgGrid(
                            id="rf-grid", className="ag-theme-alpine",
                            columnDefs=RF_COL_DEFS, defaultColDef=DEFAULT_COL_DEF,
                            rowData=init_rf_rows,
                            dashGridOptions={
                                "treeData": True,
                                "getDataPath": JS_GET_DATA_PATH_RF,
                                "animateRows": True,
                                "rowSelection": "single",
                                "groupDefaultExpanded": 0,
                            },
                            columnSize="sizeToFit",
                            style={"height":"430px", "width":"100%"},
                        ),
                    ],
                ),
            ]),

            html.Div(className="panel", children=[
                html.H3("Explainability"),
                html.Div("All views reconcile. Δ should be 0.", className="hint"),
                dcc.Loading(
                    type="circle",
                    children=[
                        html.Div(id="cards", className="cards"),
                        dcc.Graph(id="bar-chart", figure=build_bar_chart([], "Top contributors"), config={"displayModeBar": False}),
                        html.Div(className="tables", children=[
                            html.Div(className="mini-table", children=[
                                html.Div("RF Breakdown (current depth)", className="mini-head"),
                                html.Div(id="rf-breakdown-table", className="mini-body"),
                            ]),
                            html.Div(className="mini-table", children=[
                                html.Div("Top riskFactors (granular)", className="mini-head"),
                                html.Div(id="top-rf-table", className="mini-body"),
                            ]),
                        ]),
                    ],
                ),
            ]),
        ]),

        dcc.Store(id="store-biz-path", data=INIT_BIZ_PATH),
        dcc.Store(id="store-rf-path", data=INIT_RF_PATH),
    ],
)


# =========================
# CALLBACK 1: Controls -> Business grid rebuild + optional reset
# =========================
@app.callback(
    Output("biz-grid", "rowData"),
    Output("biz-grid", "columnDefs"),
    Output("biz-grid", "dashGridOptions"),
    Output("store-biz-path", "data"),
    Output("store-rf-path", "data"),
    Input("business-depth", "value"),
    Input("business-view-mode", "value"),
    Input("reset-firm", "n_clicks"),
    State("store-biz-path", "data"),
    prevent_initial_call=False,
)
def on_business_controls(depth: int, view_mode: str, reset_clicks: int, current_path: List[str]):
    trig = (callback_context.triggered[0]["prop_id"] if callback_context.triggered else "")
    if trig.startswith("reset-firm"):
        biz_path = [FIRM_NODE]
        rf_path = []
    else:
        biz_path = current_path or [FIRM_NODE]
        rf_path = no_update

    rows = business_grid_rows(int(depth), biz_path)

    if view_mode == "tree":
        col_defs = BIZ_COL_DEFS
        grid_opts = {
            "treeData": True,
            "getDataPath": JS_GET_DATA_PATH_BIZ,
            "animateRows": True,
            "rowSelection": "single",
            "groupDefaultExpanded": 1,
        }
        final_rows = rows
    elif view_mode == "group":
        col_defs = biz_group_col_defs(int(depth))
        grid_opts = {
            "treeData": False,
            "animateRows": True,
            "rowSelection": "single",
            "groupDefaultExpanded": 1,
            "groupDisplayType": "multipleColumns",
            "groupIncludeFooter": True,
        }
        final_rows = rows
    else:  # "stairs"
        col_defs = BIZ_STAIRS_COL_DEFS
        grid_opts = {
            "treeData": False,
            "animateRows": True,
            "rowSelection": "single",
        }
        node_total = get_business_node_total(biz_path)
        final_rows = build_stairs_from_paths(rows, path_key="path", node_key="node", total_denominator=node_total)

    return final_rows, col_defs, grid_opts, biz_path, rf_path


# Quick filters
@app.callback(
    Output("biz-grid", "dashGridOptions", allow_duplicate=True),
    Input("biz-quick-filter", "value"),
    State("biz-grid", "dashGridOptions"),
    prevent_initial_call=True,
)
def biz_quick_filter(q: str, opts: Dict[str, Any]):
    opts = dict(opts or {})
    opts["quickFilterText"] = q or ""
    return opts

@app.callback(
    Output("rf-grid", "dashGridOptions", allow_duplicate=True),
    Input("rf-quick-filter", "value"),
    State("rf-grid", "dashGridOptions"),
    prevent_initial_call=True,
)
def rf_quick_filter(q: str, opts: Dict[str, Any]):
    opts = dict(opts or {})
    opts["quickFilterText"] = q or ""
    return opts


# =========================
# CALLBACK 2: Business selection (grid OR breadcrumbs) -> update business store + clear rf store + rebuild rf grid + explain (cards/tables/chart)
# =========================
@app.callback(
    Output("store-biz-path", "data", allow_duplicate=True),
    Output("store-rf-path", "data", allow_duplicate=True),
    Output("rf-grid", "rowData"),
    Output("rf-grid", "columnDefs"),
    Output("rf-grid", "dashGridOptions"),
    Output("cards", "children"),
    Output("rf-breakdown-table", "children"),
    Output("top-rf-table", "children"),
    Output("bar-chart", "figure"),
    Input("biz-grid", "selectedRows"),
    Input({"type":"biz-crumb","index": ALL}, "n_clicks"),
    Input({"type":"biz-crumb-clear","index": 0}, "n_clicks"),
    Input("rf-depth", "value"),
    Input("rf-view-mode", "value"),
    Input("topn", "value"),
    Input("only-negatives", "value"),
    State("store-biz-path", "data"),
    prevent_initial_call=True,
)
def on_business_selected(
    selected_rows: Optional[List[Dict[str, Any]]],
    crumb_clicks: List[int],
    crumb_clear: Optional[int],
    rf_depth: int,
    rf_view_mode: str,
    topn: int,
    only_negs: List[str],
    current_biz_path: List[str],
):
    trig = (callback_context.triggered[0]["prop_id"] if callback_context.triggered else "")
    biz_path = current_biz_path or [FIRM_NODE]

    if "biz-crumb-clear" in trig:
        biz_path = [FIRM_NODE]
    elif "biz-crumb" in trig:
        try:
            trig_id = json.loads(trig.split(".")[0])
            idx = int(trig_id.get("index", -1))
            if idx >= 0:
                biz_path = biz_path[: idx + 1]
        except Exception:
            pass
    elif trig.startswith("biz-grid") and selected_rows:
        row = selected_rows[0]
        p = row.get("path")
        if isinstance(p, list) and p:
            biz_path = normalize_path([str(x) for x in p])
        else:
            inferred = normalize_path([row.get(k, "") for k in BUSINESS_LEVELS])
            if inferred:
                biz_path = inferred

    # Business changed => clear RF
    rf_path: List[str] = []

    # RF grid
    rf_rows, biz_total = rf_grid_rows(biz_path, int(rf_depth), rf_path)

    if rf_view_mode == "tree":
        rf_col_defs = RF_COL_DEFS
        rf_grid_opts = {
            "treeData": True,
            "getDataPath": JS_GET_DATA_PATH_RF,
            "animateRows": True,
            "rowSelection": "single",
            "groupDefaultExpanded": 0,
        }
        rf_final_rows = rf_rows
    elif rf_view_mode == "group":
        rf_col_defs = rf_group_col_defs(int(rf_depth))
        rf_grid_opts = {
            "treeData": False,
            "animateRows": True,
            "rowSelection": "single",
            "groupDefaultExpanded": 0,
            "groupDisplayType": "multipleColumns",
            "groupIncludeFooter": True,
        }
        rf_final_rows = rf_rows
    else:  # "stairs"
        rf_col_defs = RF_STAIRS_COL_DEFS
        rf_grid_opts = {
            "treeData": False,
            "animateRows": True,
            "rowSelection": "single",
        }
        rf_final_rows = build_stairs_from_paths(rf_rows, path_key="rf_path", node_key="rf_node", total_denominator=biz_total)

    rf_sum = rf_contrib_total_within_business(biz_path, int(rf_depth))
    delta = biz_total - rf_sum
    rf_selected = biz_total  # none selected yet => all RFs

    # RF breakdown (top 12 groups)
    bd = (
        get_rf_breakdown_for_business(biz_path, int(rf_depth))
        .group_by([f"rf_level{i}" for i in range(1, int(rf_depth) + 1)])
        .agg(pl.col("value_usd").sum().alias("value_usd"))
        .with_columns(pl.col("value_usd").abs().alias("_abs"))
        .sort("_abs", descending=True)
        .head(12)
        .drop("_abs")
    )

    bd_children = []
    for r in bd.to_dicts():
        label = normalize_path([r.get(f"rf_level{i}", "") for i in range(1, int(rf_depth) + 1)])
        v = float(r.get("value_usd") or 0.0)
        share = (v / biz_total * 100.0) if abs(biz_total) > 1e-12 else 0.0
        bd_children.append(
            html.Div(className="row", children=[
                html.Div(" / ".join(label) if label else "(blank)",
                         style={"maxWidth":"68%","overflow":"hidden","textOverflow":"ellipsis","whiteSpace":"nowrap"}),
                html.Div([
                    html.Span(f"{v/1e6:,.1f}MM", className="neg-cell" if v < 0 else "pos-cell",
                              style={"padding":"2px 6px","borderRadius":"10px"}),
                    html.Span(f"{share:,.1f}%", className="muted", style={"marginLeft":"8px"}),
                ]),
            ])
        )
    if not bd_children:
        bd_children = [html.Div("No data", className="muted")]

    # Top riskFactors (granular)
    only_neg = "neg" in (only_negs or [])
    top_df = top_risk_factors(biz_path, [], int(topn or DEFAULT_TOPN), only_neg)

    top_children = []
    chart_items = []
    for r in top_df.to_dicts():
        name = r["riskFactor"]
        v = float(r["value_usd"] or 0.0)
        share = float(r["share_pct"] or 0.0)
        top_children.append(
            html.Div(className="row", children=[
                html.Div(name, style={"maxWidth":"70%","overflow":"hidden","textOverflow":"ellipsis","whiteSpace":"nowrap"}),
                html.Div([
                    html.Span(f"{v/1e6:,.1f}MM", className="neg-cell" if v < 0 else "pos-cell",
                              style={"padding":"2px 6px","borderRadius":"10px"}),
                    html.Span(f"{share:,.1f}%", className="muted", style={"marginLeft":"8px"}),
                ]),
            ])
        )
        chart_items.append({"label": name, "value_mm": v/1e6})
    if not top_children:
        top_children = [html.Div("No data", className="muted")]

    fig = build_bar_chart(chart_items[: int(topn or DEFAULT_TOPN)], "Top riskFactor contributors")

    biz_label = biz_path[-1] if biz_path else FIRM_NODE
    delta_tone = "good" if abs(delta) < 1e-6 else "bad"
    cards = [
        card("Business VaR", f"{fmt_mm(biz_total)}MM", biz_label, tone="neutral"),
        card("RF Selection", f"{fmt_mm(rf_selected)}MM", "All RFs (click RF tree to narrow)", tone="neutral"),
        card("Reconciliation Δ", f"{fmt_mm(delta)}MM", "business_total - rf_sum_at_depth", tone=delta_tone),
    ]

    return biz_path, rf_path, rf_final_rows, rf_col_defs, rf_grid_opts, cards, bd_children, top_children, fig


# =========================
# CALLBACK 3: RF selection (grid OR breadcrumbs) -> update rf store + refresh cards + top RF table + chart
# =========================
@app.callback(
    Output("store-rf-path", "data", allow_duplicate=True),
    Output("cards", "children", allow_duplicate=True),
    Output("top-rf-table", "children", allow_duplicate=True),
    Output("bar-chart", "figure", allow_duplicate=True),
    Input("rf-grid", "selectedRows"),
    Input({"type":"rf-crumb","index": ALL}, "n_clicks"),
    Input({"type":"rf-crumb-clear","index": 0}, "n_clicks"),
    State("store-biz-path", "data"),
    State("store-rf-path", "data"),
    State("rf-depth", "value"),
    State("topn", "value"),
    State("only-negatives", "value"),
    prevent_initial_call=True,
)
def on_rf_selected(
    selected_rows: Optional[List[Dict[str, Any]]],
    rf_crumb_clicks: List[int],
    rf_clear: Optional[int],
    biz_path: List[str],
    current_rf_path: List[str],
    rf_depth: int,
    topn: int,
    only_negs: List[str],
):
    trig = (callback_context.triggered[0]["prop_id"] if callback_context.triggered else "")
    rf_path = current_rf_path or []

    if "rf-crumb-clear" in trig:
        rf_path = []
    elif "rf-crumb" in trig:
        try:
            trig_id = json.loads(trig.split(".")[0])
            idx = int(trig_id.get("index", -1))
            if idx >= 0:
                rf_path = rf_path[: idx + 1]
        except Exception:
            pass
    elif trig.startswith("rf-grid") and selected_rows:
        row = selected_rows[0]
        p = row.get("rf_path")
        if isinstance(p, list) and p:
            rf_path = normalize_path([str(x) for x in p])
        else:
            inferred = normalize_path([row.get(k, "") for k in RF_LEVELS])
            if inferred:
                rf_path = inferred

    biz_total = get_business_node_total(biz_path or [FIRM_NODE])
    rf_contrib = rf_selected_contribution(biz_path or [FIRM_NODE], rf_path, int(rf_depth))

    rf_label = " / ".join(rf_path) if rf_path else "All RFs"
    delta = biz_total - rf_contrib_total_within_business(biz_path or [FIRM_NODE], int(rf_depth))

    only_neg = "neg" in (only_negs or [])
    top_df = top_risk_factors(biz_path or [FIRM_NODE], rf_path, int(topn or DEFAULT_TOPN), only_neg)

    top_children = []
    chart_items = []
    for r in top_df.to_dicts():
        name = r["riskFactor"]
        v = float(r["value_usd"] or 0.0)
        share = float(r["share_pct"] or 0.0)
        top_children.append(
            html.Div(className="row", children=[
                html.Div(name, style={"maxWidth":"70%","overflow":"hidden","textOverflow":"ellipsis","whiteSpace":"nowrap"}),
                html.Div([
                    html.Span(f"{v/1e6:,.1f}MM", className="neg-cell" if v < 0 else "pos-cell",
                              style={"padding":"2px 6px","borderRadius":"10px"}),
                    html.Span(f"{share:,.1f}%", className="muted", style={"marginLeft":"8px"}),
                ]),
            ])
        )
        chart_items.append({"label": name, "value_mm": v/1e6})
    if not top_children:
        top_children = [html.Div("No data", className="muted")]

    fig = build_bar_chart(chart_items[: int(topn or DEFAULT_TOPN)], f"Top riskFactors within: {rf_label}")

    biz_label = (biz_path[-1] if biz_path else FIRM_NODE)
    delta_tone = "good" if abs(delta) < 1e-6 else "bad"
    cards = [
        card("Business VaR", f"{fmt_mm(biz_total)}MM", biz_label, tone="neutral"),
        card("RF Selection", f"{fmt_mm(rf_contrib)}MM", rf_label, tone="neutral"),
        card("Reconciliation Δ", f"{fmt_mm(delta)}MM", "business_total - rf_sum_at_depth", tone=delta_tone),
    ]

    return rf_path, cards, top_children, fig


# =========================
# CALLBACK 4: Breadcrumb rendering + status badge
# =========================
@app.callback(
    Output("biz-crumbs", "children"),
    Output("rf-crumbs", "children"),
    Output("recon-badge", "children"),
    Input("store-biz-path", "data"),
    Input("store-rf-path", "data"),
    Input("rf-depth", "value"),
)
def render_breadcrumbs(biz_path: List[str], rf_path: List[str], rf_depth: int):
    biz_path = biz_path or [FIRM_NODE]
    rf_path = rf_path or []

    biz_chips = [
        html.Button("✕", id={"type":"biz-crumb-clear","index":0}, n_clicks=0, className="chip chip-clear",
                    title="Reset business selection to firm")
    ]
    for i, name in enumerate(biz_path):
        biz_chips.append(html_chip(name, i, "biz-crumb", title="Click to jump to this business level"))

    rf_chips = [
        html.Button("✕", id={"type":"rf-crumb-clear","index":0}, n_clicks=0, className="chip chip-clear",
                    title="Clear RF selection (show all RFs for current business)")
    ]
    for i, name in enumerate(rf_path):
        rf_chips.append(html_chip(name, i, "rf-crumb", title="Click to jump to this RF level"))

    biz_total = get_business_node_total(biz_path)
    rf_sum = rf_contrib_total_within_business(biz_path, int(rf_depth))
    ok = abs(biz_total - rf_sum) < 1e-6
    badge = make_badge("Reconciled ✅" if ok else "Mismatch ⚠️", ok)

    return biz_chips, rf_chips, [html.Span("Status:", className="muted", style={"fontWeight":"900"}), badge]


if __name__ == "__main__":
    app.run_server(debug=True, port=PORT)
