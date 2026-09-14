"""
Customer Segmentation & Churn Pattern Analytics - European Banking
Unified Mentor Data Analyst Internship Project
Author: Sanjoy Das

Design: Formal audit / regulatory report identity.
Deliberately distinct from the Thales predictive maintenance dashboard
(industrial dark theme) and the ECB Retention dashboard (navy/gold),
so this project reads as its own artifact, not a re-skin.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

# =========================================================
# CHART SIZE CONTROLS — edit these two lines to resize every
# chart + breakdown panel pair across all 8 tabs at once.
# CHART_HEIGHT is applied as an exact pixel height to BOTH the
# Plotly figure and the breakdown panel's inline style, so the
# two are mathematically forced to match rather than relying on
# CSS percentage-height auto-stretching (which Streamlit's column
# layout does not reliably guarantee).
# CHART_COL_RATIO controls the chart-vs-panel width split, e.g.
# [6, 4] = 60% chart / 40% panel. [7, 3] would widen the chart.
# =========================================================
CHART_HEIGHT = 420
CHART_COL_RATIO = [6, 4]

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="European Bank | Churn Segmentation Report",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================================================
# CUSTOM CSS — FORMAL AUDIT / REGULATORY REPORT THEME
# Palette: charcoal-slate (#22303C) + deep burgundy (#7A1F2B) + muted bronze (#B08D57)
# on a warm paper background (#F7F5F0). Serif headers throughout.
# NOTE: no f-strings inside this CSS block, to avoid brace conflicts.
# =========================================================
st.markdown(
    """
    <style>
    .main { background-color: #F7F5F0; }

    h1, h2, h3 { font-family: Georgia, 'Times New Roman', serif !important; }

    /* ---- Letterhead banner ---- */
    .letterhead {
        background-color: #22303C;
        padding: 22px 28px;
        border-top: 4px solid #B08D57;
        border-bottom: 4px solid #B08D57;
        margin-bottom: 4px;
        width: 100% !important;
    }

    .block-container,
    div[data-testid="stAppViewContainer"] .main .block-container {
        max-width: 95% !important;
        width: 95% !important;
        padding-left: 0.5rem !important;
        padding-right: 0.5rem !important;
    }

        /* Reduce gap between Streamlit top bar / Deploy area and header */
    .block-container,
    div[data-testid="stAppViewContainer"] .main .block-container {
        padding-top: 0.2rem !important;
    }

    .letterhead-title {
        font-family: Georgia, 'Times New Roman', serif;
        color: #F7F5F0;
        font-size: 24px;
        font-weight: 700;
        letter-spacing: 1px;
        text-transform: uppercase;
        margin: 0;
    }
    .letterhead-subtitle {
        font-family: Georgia, 'Times New Roman', serif;
        color: #C9BBA8;
        font-size: 13.5px;
        font-style: italic;
        margin-top: 4px;
    }
    .meta-strip {
        background-color: #EDE8DE;
        border-bottom: 1px solid #B08D57;
        padding: 8px 28px;
        font-size: 12px;
        color: #4A4A4A;
        letter-spacing: 0.3px;
        display: flex;
        justify-content: space-between;
        margin-bottom: 18px;
    }
    .classification-tag {
        display: inline-block;
        background-color: #F4E6E9;
        color: #7A1F2B;
        border: 1px solid #7A1F2B;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 1px;
        padding: 2px 10px;
        text-transform: uppercase;
    }

    /* ---- Section headers ---- */
    .section-header {
        color: #22303C;
        font-family: Georgia, 'Times New Roman', serif;
        border-bottom: 2px solid #7A1F2B;
        padding-bottom: 6px;
        margin-top: 6px;
        letter-spacing: 0.3px;
    }

    /* ---- KPI report cards: white card, burgundy left rail, serif figure ---- */
    .kpi-report-card {
        background-color: #FFFFFF;
        border: 1px solid #D8D2C4;
        border-left: 6px solid #7A1F2B;
        padding: 14px 16px;
        margin-bottom: 6px;
        box-shadow: 1px 2px 4px rgba(0,0,0,0.06);
        height: 170px;
        min-height: 170px;
        box-sizing: border-box;
        display: flex;
        flex-direction: column;
        justify-content: center;
        overflow: hidden;
    }
    .kpi-report-name {
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 0.8px;
        text-transform: uppercase;
        color: #7A1F2B;
        margin-bottom: 2px;
    }
    .kpi-report-desc {
        font-size: 11.5px;
        font-style: italic;
        color: #6B6B6B;
        margin-bottom: 10px;
    }
    .kpi-report-value {
        font-family:  'Times New Roman', serif;
        font-size: 20px;
        font-weight: 700;
        color: #22303C;
        line-height: 1.1;
    }
    .kpi-report-sub {
        font-size: 11.5px;
        color: #4A4A4A;
        margin-top: 4px;
    }

        

    
    /* =====================================================
    MAIN FIVE KPI CARDS — PERFECT ALIGNMENT
    ===================================================== */

    .kpi-report-card {
        background-color: #FFFFFF;
        border: 1px solid #D8D2C4;
        border-left: 6px solid #7A1F2B;

        height: 150px;
        min-height: 170px;

        padding: 12px 14px;
        margin-bottom: 6px;

        box-sizing: border-box;
        overflow: hidden;

        display: grid;
        grid-template-rows: 18px 32px 38px 1fr;

        align-items: start;

        box-shadow: 1px 2px 4px rgba(0,0,0,0.06);
    }


    /* KPI NAME */
    .kpi-report-name {
        height: 18px;

        font-size: 10.5px;
        font-weight: 700;
        letter-spacing: 0.6px;
        text-transform: uppercase;

        color: #7A1F2B;

        margin: 0;
        padding: 0;

        line-height: 18px;

        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    


    .kpi-report-name {
        height: 28px;
        font-size: 10px;
        font-weight: 700;
        letter-spacing: 0.5px;
        text-transform: uppercase;
        color: #7A1F2B;
        margin: 0;
        padding: 0;
        line-height: 14px;

        white-space: normal;
        overflow-wrap: break-word;
        word-break: normal;
    }






    /* KPI DESCRIPTION */
    .kpi-report-desc {
        height: 32px;

        font-size: 11px;
        font-style: italic;

        color: #6B6B6B;

        margin: 0;
        padding: 2px 0 0 0;

        line-height: 15px;

        overflow: hidden;
    }


    /* KPI VALUE */
    .kpi-report-value {
        height: 38px;

        font-family: 'Times New Roman', serif;

        font-size: 24px;
        font-weight: 700;

        color: #22303C;

        margin: 0;
        padding: 0;

        line-height: 38px;

        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }


    /* KPI SUBTITLE */
    .kpi-report-sub {
        align-self: start;

        font-size: 10.5px;

        color: #4A4A4A;

        margin: 2px 0 0 0;
        padding: 0;

        line-height: 14px;

        overflow: hidden;
    }
    




    





    /* ---- Simple KPI cards (used in other tabs) ---- */
    .kpi-card {
        background-color: #22303C;
        padding: 16px 14px;
        border-top: 3px solid #B08D57;
        text-align: center;
        color: #F7F5F0;
        height: 110px;
        min-height: 110px;
        box-sizing: border-box;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    .kpi-value {
        font-family: Georgia, 'Times New Roman', serif;
        font-size: 26px;
        font-weight: 700;
        color: #EDE0C8;
    }
    .kpi-label {
        font-size: 11.5px;
        color: #C9BBA8;
        text-transform: uppercase;
        letter-spacing: 0.6px;
        margin-top: 4px;
    }

    /* ---- Footer ---- */
    .report-footer {
        border-top: 1px solid #B08D57;
        margin-top: 30px;
        padding-top: 10px;
        font-size: 11px;
        color: #7A7A7A;
        font-style: italic;
    }

    /* =====================================================
       SIDEBAR — "Filter Panel" formal styling
       ===================================================== */
    section[data-testid="stSidebar"] {
        background-color: #1C2733;
        border-right: 3px solid #B08D57;
    }
    section[data-testid="stSidebar"] * { color: #EDE8DE; }

    .sb-header {
        padding: 4px 2px 14px 2px;
        border-bottom: 2px solid #B08D57;
        margin-bottom: 14px;
    }
    .sb-header-title {
        font-family: Georgia, 'Times New Roman', serif;
        color: #EDE0C8;
        font-size: 19px;
        font-weight: 700;
        letter-spacing: 1.2px;
        text-transform: uppercase;
        margin: 0;
    }
    .sb-header-sub {
        font-size: 11.5px;
        font-style: italic;
        color: #9AA5B1;
        margin-top: 3px;
    }

    .sb-group-label {
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 1px;
        text-transform: uppercase;
        color: #B08D57;
        border-bottom: 1px solid #3B4A5A;
        padding-bottom: 4px;
        margin-top: 18px;
        margin-bottom: 8px;
    }

    section[data-testid="stSidebar"] div[data-testid="stExpander"] {
        background-color: #24313F;
        border: 1px solid #3B4A5A;
        border-left: 3px solid #B08D57;
        border-radius: 2px;
        margin-bottom: 10px;
    }
    section[data-testid="stSidebar"] div[data-testid="stExpander"] summary {
        font-size: 12.5px;
        font-weight: 700;
        letter-spacing: 0.4px;
        text-transform: uppercase;
        color: #EDE0C8 !important;
    }

    section[data-testid="stSidebar"] .stMultiSelect [data-baseweb="tag"] {
        background-color: #B08D57 !important;
        border-radius: 2px;
    }
    section[data-testid="stSidebar"] .stMultiSelect [data-baseweb="tag"] span {
        color: #22303C !important;
        font-weight: 600;
    }
    section[data-testid="stSidebar"] .stMultiSelect > div {
        background-color: #24313F;
        border: 1px solid #3B4A5A !important;
        border-radius: 2px;
    }


        /* Sidebar Expander Title */
    section[data-testid="stSidebar"] div[data-testid="stExpander"] summary,
    section[data-testid="stSidebar"] div[data-testid="stExpander"] summary p,
    section[data-testid="stSidebar"] div[data-testid="stExpander"] summary span {
        color: #000000 !important;
        font-weight: 700 !important;
    }

    /* Hover color */
    section[data-testid="stSidebar"] div[data-testid="stExpander"] summary:hover,
    section[data-testid="stSidebar"] div[data-testid="stExpander"] summary:hover p,
    section[data-testid="stSidebar"] div[data-testid="stExpander"] summary:hover span {
        color: #FFFFFF !important;
    }




    section[data-testid="stSidebar"] label {
        font-size: 12px !important;
        font-weight: 600;
        letter-spacing: 0.3px;
        color: #C9BBA8 !important;
        text-transform: uppercase;
    }

    section[data-testid="stSidebar"] .stButton > button {
        background-color: #7A1F2B;
        color: #F7F5F0;
        border: 1px solid #9A3B47;
        border-radius: 2px;
        font-size: 12.5px;
        font-weight: 700;
        letter-spacing: 0.6px;
        text-transform: uppercase;
        width: 100%;
        padding: 8px 0;
        margin-top: 6px;
    }
    section[data-testid="stSidebar"] .stButton > button:hover {
        background-color: #92202E;
        border: 1px solid #B08D57;
        color: #EDE0C8;
    }

    .sb-snapshot {
        background-color: #24313F;
        border: 1px solid #3B4A5A;
        border-top: 3px solid #B08D57;
        padding: 12px 14px;
        margin-top: 18px;
    }
    .sb-snapshot-label {
        font-size: 10.5px;
        font-weight: 700;
        letter-spacing: 1px;
        text-transform: uppercase;
        color: #B08D57;
        margin-bottom: 6px;
    }
    .sb-snapshot-value {
        font-family: Georgia, 'Times New Roman', serif;
        font-size: 21px;
        font-weight: 700;
        color: #EDE0C8;
        line-height: 1.2;
    }
    .sb-snapshot-sub {
        font-size: 11px;
        color: #9AA5B1;
        margin-top: 2px;
    }
    .sb-bar-track {
        background-color: #3B4A5A;
        height: 6px;
        border-radius: 3px;
        margin-top: 8px;
        overflow: hidden;
    }
    .sb-bar-fill {
        background-color: #B08D57;
        height: 6px;
    }

    /* =====================================================
       FINDING CARDS — formal Observation / Interpretation /
       Recommendation blocks, used once per tab
       ===================================================== */
    .finding-box {
        background-color: #FFFFFF;
        border: 1px solid #D8D2C4;
        border-left: 7px solid #22303C;
        padding: 16px 20px;
        margin: 20px 0 24px 0;
        box-shadow: 1px 2px 5px rgba(0,0,0,0.06);
    }
    .finding-label {
        font-size: 10.5px;
        font-weight: 700;
        letter-spacing: 1.6px;

        text-transform: uppercase;
        color: #7A1F2B;
        margin-bottom: 4px;
    }
    .finding-title {
        font-family: Georgia, 'Times New Roman', serif;
        font-size: 17px;
        font-weight: 700;
        color: #22303C;
        margin-bottom: 10px;
        padding-bottom: 8px;
        border-bottom: 1px solid #EDE8DE;
    }
    .finding-row {
        font-size: 13px;
        color: #333333;
        margin-bottom: 7px;
        line-height: 1.5;
    }
    .finding-row b {
        color: #22303C;
        font-weight: 700;
    }
    .finding-row.rec {
        background-color: #F7F5F0;
        border-left: 3px solid #B08D57;
        padding: 8px 10px;
        margin-top: 10px;
    }

    /* =====================================================
       BREAKDOWN PANELS — sit beside every chart (6/4 split)
       ===================================================== */
    .breakdown-panel {
        background-color: #FFFFFF;
        border: 1px solid #D8D2C4;
        border-top: 3px solid #B08D57;
        padding: 14px 16px;
        box-sizing: border-box;
    }
    .breakdown-panel-title {
        font-size: 10.5px;
        font-weight: 700;
        letter-spacing: 1.2px;
        text-transform: uppercase;
        color: #7A1F2B;
        margin-bottom: 10px;
        padding-bottom: 7px;
        border-bottom: 1px solid #EDE8DE;
    }
    .breakdown-stat-row {
        display: flex;
        justify-content: space-between;
        align-items: baseline;
        font-size: 12px;
        padding: 6px 0;
        border-bottom: 1px dotted #E0DACB;
    }
    .breakdown-stat-row:last-child { border-bottom: none; }
    .breakdown-stat-label {
        color: #6B6B6B;
        padding-right: 8px;
    }
    .breakdown-stat-value {
        color: #22303C;
        font-weight: 700;
        font-family: Georgia, 'Times New Roman', serif;
        font-size: 13px;
        text-align: right;
        white-space: nowrap;
    }

    /* Finding text embedded inside the breakdown panel, below the stat rows */
    .panel-finding {
        margin-top: 10px;
        padding-top: 10px;
        border-top: 1px solid #EDE8DE;
    }
    .panel-finding-label {
        font-size: 9.5px;
        font-weight: 700;
        letter-spacing: 1.4px;
        text-transform: uppercase;
        color: #B08D57;
        margin-bottom: 5px;
    }
    .panel-finding-text {
        font-size: 12px;
        color: #333333;
        line-height: 1.55;
    }
    .panel-finding-text b {
        color: #7A1F2B;
        font-weight: 700;
    }


        /* =====================================================
    EXECUTIVE SUMMARY CARD
    ===================================================== */
    .exec-summary-card {
        background-color: #FFFFFF;
        border: 1px solid #D8D2C4;
        border-top: 4px solid #B08D57;
        border-radius: 8px;
        padding: 20px 28px;
        margin: 10px 0 24px 0;
        box-shadow: 0 6px 20px rgba(34, 48, 60, 0.08);
    }
    .exec-summary-card h3 {
        font-family: Georgia, 'Times New Roman', serif !important;
        font-size: 19px;
        font-weight: 700;
        color: #22303C !important;
        letter-spacing: 0.4px;
        margin: 0 0 10px 0 !important;
        padding-bottom: 8px;
        border-bottom: 2px solid #7A1F2B;
    }
    .exec-row {
        display: flex;
        align-items: baseline;
        gap: 6px;
        font-size: 13.5px;
        color: #5A5A5A;
        padding: 5px 2px;
        line-height: 1.35;
        border-bottom: 1px dotted #EDE8DE;
    }
    .exec-row:last-of-type {
        border-bottom: none;
    }
    .exec-row strong {
        font-family: Georgia, 'Times New Roman', serif;
        font-weight: 700;
        color: #22303C;
        font-size: 14.5px;
    }
    .exec-row.alert strong {
        color: #7A1F2B;
    }
    .exec-strategy {
        background-color: #22303C;
        border-left: 6px solid #B08D57;
        border-radius: 0 5px 5px 0;
        padding: 11px 16px;
        margin-top: 9px;
        font-size: 13px;
        color: #D8D2C4;
        line-height: 1.45;
    }
    .exec-strategy strong {
        font-family: Georgia, 'Times New Roman', serif;
        letter-spacing: 0.3px;
    }



        /* ---- ECB Logo in Main Header ---- */
    .letterhead-logo {
        display: block;
        height: 42px;
        max-width: 220px;
        width: auto;
        margin: 0 auto 10px auto;
    }

    # /* Center the complete header */
    # .letterhead {
    #     text-align: center;
    # }

        /* =====================================================
       PERSISTENT KPI RIBBON — sits above the tabs, so it is
       visible regardless of which tab is currently selected
       ===================================================== */
    .kpi-ribbon {
        background-color: #22303C;
        border-top: 3px solid #B08D57;
        border-bottom: 3px solid #B08D57;
        padding: 12px 6px 6px 6px;
        margin-bottom: 6px;
    }
    .ribbon-item {
        text-align: center;
        border-right: 1px solid #3B4A5A;
        padding: 2px 8px;
    }
    .ribbon-value {
        font-family: Georgia, 'Times New Roman', serif;
        font-size: 21px;
        font-weight: 700;
        color: #EDE0C8;
        line-height: 1.15;
    }
    .ribbon-label {
        font-size: 10px;
        font-weight: 600;
        letter-spacing: 0.5px;
        text-transform: uppercase;
        color: #9AA5B1;
        margin-top: 3px;
    }

    /* =====================================================
       KEY FINDING BANNER — one guaranteed per tab, at the end
       ===================================================== */
    .key-finding-banner {
        background-color: #22303C;
        border-left: 8px solid #B08D57;
        padding: 18px 24px;
        margin: 30px 0 10px 0;
    }
    .kfb-label {
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 1.8px;
        text-transform: uppercase;
        color: #B08D57;
        margin-bottom: 6px;
    }
    .kfb-text {
        font-family: Georgia, 'Times New Roman', serif;
        font-size: 15.5px;
        color: #F7F5F0;
        line-height: 1.6;
    }
    .kfb-text b { color: #EDE0C8; }

    .styled-divider {
        border-top: 2px solid #B08D57;
        margin: 32px 0 24px 0;
    }
    </style>
""",
    unsafe_allow_html=True,
)


# =========================================================
# DEFENSIVE HELPER FUNCTIONS
# =========================================================
def safe_get_value(dataframe, column, agg="mean", default=0):
    """Safely compute an aggregate on a possibly-empty dataframe/column."""
    try:
        if dataframe is None or len(dataframe) == 0 or column not in dataframe.columns:
            return default
        series = dataframe[column].dropna()
        if len(series) == 0:
            return default
        if agg == "mean":
            return series.mean()
        elif agg == "sum":
            return series.sum()
        elif agg == "count":
            return series.count()
        elif agg == "median":
            return series.median()
        return default
    except Exception:
        return default


def safe_crosstab_value(
    dataframe,
    row_col,
    col_col,
    row_val,
    col_val,
    agg_col="Exited",
    agg="mean",
    default=0,
):
    """Safely pull a single cell out of a groupby/crosstab that may not exist post-filtering."""
    try:
        subset = dataframe[
            (dataframe[row_col] == row_val) & (dataframe[col_col] == col_val)
        ]
        if len(subset) == 0:
            return default
        return safe_get_value(subset, agg_col, agg, default)
    except Exception:
        return default


def kpi_card(label, value, suffix=""):
    st.markdown(
        '<div class="kpi-card"><div class="kpi-value">'
        + str(value)
        + str(suffix)
        + '</div><div class="kpi-label">'
        + str(label)
        + "</div></div>",
        unsafe_allow_html=True,
    )


def kpi_report_card(name, description, value, subtitle=""):
    """Formal audit-style KPI card used on the Executive KPI Summary tab."""
    st.markdown(
        '<div class="kpi-report-card">'
        '<div class="kpi-report-name">' + str(name) + "</div>"
        '<div class="kpi-report-desc">' + str(description) + "</div>"
        '<div class="kpi-report-value">' + str(value) + "</div>"
        '<div class="kpi-report-sub">' + str(subtitle) + "</div>"
        "</div>",
        unsafe_allow_html=True,
    )


def finding_box(number, title, observation, interpretation="", recommendation=""):
    """Formal audit-style finding card: Observation -> Interpretation -> Recommendation.
        All three fields are passed in as already-computed, data-driven strings so the
    text can never contradict whatever the current sidebar filters are showing."""
    html = (
        '<div class="finding-box">'
        '<div class="finding-label">Finding ' + str(number) + "</div>"
        '<div class="finding-title">' + str(title) + "</div>"
        '<div class="finding-row"><b>Observation: </b>' + str(observation) + "</div>"
        '<div class="finding-row"><b></b>'
        + str(interpretation)
        + "</div>"
    )
    if recommendation:
        html += (
            '<div class="finding-row rec"><b>Recommendation: </b>'
            + str(recommendation)
            + "</div>"
        )
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)

def finding_box1(number, title, observation, interpretation="", recommendation=""):
    """Formal audit-style finding card: Observation -> Interpretation -> Recommendation.
        All three fields are passed in as already-computed, data-driven strings so the
    text can never contradict whatever the current sidebar filters are showing."""
    html = (
        '<div class="finding-box">'
        '<div class="finding-label">Finding ' + str(number) + "</div>"
        '<div class="finding-title">' + str(title) + "</div>"
        '<div class="finding-row"><b>Observation: </b>' + str(observation) + "</div>"
        '<div class="finding-row"><b></b>'
        + str(interpretation)
        + "</div>"
    )
    if recommendation:
        html += (
            '<div class="finding-row rec"><b>Recommendation: </b>'
            + str(recommendation)
            + "</div>"
        )
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)

def breakdown_panel(title, stats, finding_text=None, height=None):
    """Renders the stat-card that sits beside a chart. stats is a list of
    (label, value) tuples. finding_text, if given, is appended inside the same
    card below a divider as a formally-worded, data-driven observation.
    height, when given, is applied as an exact inline pixel height so the
    panel matches its neighbouring chart precisely rather than approximately —
    overflow-y:auto is set as a safety valve in case stats + finding text
    exceed that height at very small CHART_HEIGHT values."""
    style_attr = ""
    if height is not None:
        style_attr = ' style="height:{}px; overflow-y:auto;"'.format(int(height))
    html = (
        '<div class="breakdown-panel"'
        + style_attr
        + '><div class="breakdown-panel-title">'
        + str(title)
        + "</div>"
    )
    for label, value in stats:
        html += (
            '<div class="breakdown-stat-row">'
            '<span class="breakdown-stat-label">' + str(label) + "</span>"
            '<span class="breakdown-stat-value">' + str(value) + "</span>"
            "</div>"
        )
    if finding_text:
        html += (
            '<div class="panel-finding">'
            '<div class="panel-finding-label">Finding</div>'
            '<div class="panel-finding-text">' + str(finding_text) + "</div>"
            "</div>"
        )
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


def key_finding_banner(tab_label, text):
    """Renders the mandatory Key Finding block at the end of every tab —
    a single, high-level, formally-worded takeaway distinct from the more
    granular Finding Nn cards and the per-chart panel findings above it."""
    st.markdown(
        '<div class="key-finding-banner">'
        '<div class="kfb-label">Key Finding \u2014 ' + str(tab_label) + "</div>"
        '<div class="kfb-text">' + str(text) + "</div>"
        "</div>",
        unsafe_allow_html=True,
    )


def chart_row(
    fig,
    panel_title,
    panel_stats,
    insight_text,
    chart_key=None,
    height=None,
    col_ratio=None,
):
    """Standard chart layout used throughout the report: chart on the left,
    breakdown panel with an embedded, formally-worded finding on the right.
    height defaults to the global CHART_HEIGHT constant — pass a different
    number here only if one specific chart needs to differ from the rest.
    col_ratio defaults to the global CHART_COL_RATIO (chart-vs-panel width split).
    The SAME height value is applied to both the Plotly figure and the panel's
    inline CSS height, which is what forces them to match exactly regardless
    of chart type, content length, or CHART_HEIGHT changes made later."""
    h = height if height is not None else CHART_HEIGHT
    ratio = col_ratio if col_ratio is not None else CHART_COL_RATIO

    # Keep categorical bars visually lighter/less bulky across the dashboard.
    # This applies only to Plotly bar traces, so pie/donut charts are unaffected.
    fig.update_traces(width=0.55, selector=dict(type="bar"))

    # Donut/pie charts read better at a more compact height than the standard
    # analytical charts; the neighbouring breakdown panel is reduced with it.
    if any(getattr(trace, "type", None) in ("pie", "funnelarea") for trace in fig.data):
        h = min(h, 270)

    fig.update_layout(height=h)
    col_chart, col_panel = st.columns(ratio)
    with col_chart:
        st.plotly_chart(fig, use_container_width=True, key=chart_key)
    with col_panel:
        breakdown_panel(panel_title, panel_stats, finding_text=insight_text, height=h)
    st.markdown("<div style='margin-bottom:24px'></div>", unsafe_allow_html=True)


def rate_breakdown_stats(table, cat_col, rate_col, count_col=None, overall=None):
    """Generic breakdown-panel stats for any 'rate by category' bar chart —
    reused across most tabs so every such chart reports the same shape of facts."""
    if len(table) == 0:
        return [("Status", "No data for current filters")]
    sorted_t = table.sort_values(rate_col, ascending=False)
    top, bottom = sorted_t.iloc[0], sorted_t.iloc[-1]
    stats = [
        ("Highest", f"{top[cat_col]} \u2014 {top[rate_col]:.1f}%"),
        ("Lowest", f"{bottom[cat_col]} \u2014 {bottom[rate_col]:.1f}%"),
        ("Spread", f"{(top[rate_col]-bottom[rate_col]):.1f} pts"),
        ("Categories shown", f"{len(table)}"),
    ]
    if overall is not None:
        stats.append(("Portfolio average", f"{overall:.1f}%"))
    if count_col and count_col in table.columns:
        stats.append(("Total customers", f"{int(table[count_col].sum()):,}"))
    return stats


def rate_insight_text(table, cat_col, rate_col, overall=None, noun="segment"):
    """Generic dynamic finding sentence to match rate_breakdown_stats above,
    phrased in formal audit-report register for use inside the breakdown panel."""
    if len(table) == 0:
        return "Insufficient data is available to support a finding under the current filter selection."
    sorted_t = table.sort_values(rate_col, ascending=False)
    top, bottom = sorted_t.iloc[0], sorted_t.iloc[-1]
    spread = top[rate_col] - bottom[rate_col]
    txt = (
        f"The <b>{top[cat_col]}</b> segment exhibits the highest observed attrition rate at {top[rate_col]:.1f}%, "
        f"whereas <b>{bottom[cat_col]}</b> reflects the lowest at {bottom[rate_col]:.1f}%"
    )
    if overall is not None:
        txt += f", against a portfolio-wide baseline of {overall:.1f}%."
    else:
        txt += f" within this {noun}."
    if spread >= 10:
        txt += " The magnitude of this spread constitutes a materially significant risk differential warranting targeted attention."
    elif spread >= 5:
        txt += " This represents a moderate but discernible risk differential across the segmentation."
    return txt


# =========================================================
# DYNAMIC PHRASING HELPERS
# These exist so that finding_box() interpretation and
# recommendation text can genuinely change its WORDING based on
# the underlying data (magnitude bands, direction of effect),
# rather than reusing one fixed paragraph with only a number
# substituted in. Every finding across every tab draws its
# qualitative language from these functions so that a filter
# change which flips a relationship (e.g. active churn exceeding
# inactive churn in a narrow slice) also flips the sentence.
# =========================================================


def magnitude_word(
    value, thresholds=(3, 8, 15), words=("marginal", "moderate", "material", "severe")
):
    """Maps an absolute numeric gap to a severity adjective via threshold bands."""
    v = abs(value)
    if v < thresholds[0]:
        return words[0]
    elif v < thresholds[1]:
        return words[1]
    elif v < thresholds[2]:
        return words[2]
    return words[3]


def direction_word(value, pos="higher", neg="lower", zero="on par with"):
    """Maps the sign of a numeric difference to a direction word, so sentences
    never assert a direction the data doesn't actually support."""
    if value > 0.05:
        return pos
    elif value < -0.05:
        return neg
    return zero


def urgency_clause(value, thresholds=(3, 8, 15)):
    """Maps a gap magnitude to an urgency-graded action clause, used to close
    out recommendation text with language proportionate to the actual finding."""
    v = abs(value)
    if v < thresholds[0]:
        return "warrants monitoring but does not yet justify a dedicated intervention"
    elif v < thresholds[1]:
        return "justifies inclusion in the standard quarterly retention review cycle"
    elif v < thresholds[2]:
        return "warrants a dedicated action plan within the current planning cycle"
    return "requires immediate escalation as a priority retention risk"


def compute_top_segment(dataframe, dims, min_size=20):
    """Scan every segmentation dimension and return the single highest-churn
    segment value that meets a minimum sample size, so a 3-customer outlier
    segment can never masquerade as the headline finding."""
    rows = []
    for dim in dims:
        if dim not in dataframe.columns:
            continue
        g = (
            dataframe.groupby(dim, observed=True)
            .agg(Customers=("Exited", "count"), Churned=("Exited", "sum"))
            .reset_index()
        )
        if len(g) == 0:
            continue
        g["ChurnRate"] = g["Churned"] / g["Customers"] * 100
        g["Dimension"] = dim
        g = g.rename(columns={dim: "Value"})
        rows.append(g[["Dimension", "Value", "Customers", "Churned", "ChurnRate"]])
    if not rows:
        return None
    combined = pd.concat(rows, ignore_index=True)
    eligible = combined[combined["Customers"] >= min_size]
    if len(eligible) == 0:
        eligible = combined
    if len(eligible) == 0:
        return None
    return eligible.sort_values("ChurnRate", ascending=False).iloc[0]


def compute_top_segments(dataframe, dims, min_size=20, top_n=5):
    """Same scan as compute_top_segment, but returns the top N highest-churn
    segments across all dimensions instead of just the single highest —
    used to build a prioritised marketing target list."""
    rows = []
    for dim in dims:
        if dim not in dataframe.columns:
            continue
        g = (
            dataframe.groupby(dim, observed=True)
            .agg(Customers=("Exited", "count"), Churned=("Exited", "sum"))
            .reset_index()
        )
        if len(g) == 0:
            continue
        g["ChurnRate"] = g["Churned"] / g["Customers"] * 100
        g["Dimension"] = dim
        g = g.rename(columns={dim: "Value"})
        rows.append(g[["Dimension", "Value", "Customers", "Churned", "ChurnRate"]])
    if not rows:
        return pd.DataFrame(
            columns=["Dimension", "Value", "Customers", "Churned", "ChurnRate"]
        )
    combined = pd.concat(rows, ignore_index=True)
    eligible = combined[combined["Customers"] >= min_size]
    if len(eligible) == 0:
        eligible = combined
    return (
        eligible.sort_values("ChurnRate", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )


# =========================================================
# DATA LOADING & FEATURE ENGINEERING
# =========================================================
@st.cache_data
def load_data():
    df = pd.read_csv("European_Bank.csv")

    drop_cols = [c for c in ["Year", "Surname"] if c in df.columns]
    df = df.drop(columns=drop_cols)

    df["AgeGroup"] = pd.cut(
        df["Age"], bins=[0, 30, 45, 60, 200], labels=["<30", "30-45", "46-60", "60+"]
    )
    df["CreditBand"] = pd.cut(
        df["CreditScore"], bins=[0, 580, 700, 1000], labels=["Low", "Medium", "High"]
    )
    df["TenureGroup"] = pd.cut(
        df["Tenure"],
        bins=[-1, 2, 6, 100],
        labels=["New (0-2yr)", "Mid-term (3-6yr)", "Long-term (7-10yr)"],
    )
    df["BalanceSegment"] = pd.cut(
        df["Balance"],
        bins=[-1, 0, 100000, np.inf],
        labels=["Zero-balance", "Low-balance", "High-balance"],
    )

    df["ActivityStatus"] = df["IsActiveMember"].map({1: "Active", 0: "Inactive"})
    df["ChurnStatus"] = df["Exited"].map({1: "Churned", 0: "Retained"})

    balance_q75 = df["Balance"].quantile(0.75)
    salary_q75 = df["EstimatedSalary"].quantile(0.75)
    df["HighValueCustomer"] = np.where(
        (df["Balance"] >= balance_q75) | (df["EstimatedSalary"] >= salary_q75),
        "High-Value",
        "Standard",
    )
    return df


df_raw = load_data()
SEGMENT_DIMS = [
    "Geography",
    "Gender",
    "AgeGroup",
    "CreditBand",
    "TenureGroup",
    "BalanceSegment",
    "ActivityStatus",
]

# =========================================================
# PREDICTIVE CHURN MODEL — trained once, cached as a resource
# (not @st.cache_data, because the return value is a model object,
# not a plain dataframe — st.cache_resource is the correct decorator
# for anything that shouldn't be re-pickled/re-hashed on every rerun)
# =========================================================
MODEL_FEATURES = [
    "CreditScore",
    "Geography_enc",
    "Gender_enc",
    "Age",
    "Tenure",
    "Balance",
    "NumOfProducts",
    "HasCrCard",
    "IsActiveMember",
    "EstimatedSalary",
]


@st.cache_resource
def train_churn_model(_df_raw):
    """Trains once on the full, unfiltered portfolio (not on whatever the sidebar
    currently shows) so that a customer's risk score reflects the whole bank's
    history and doesn't silently change definition every time a filter is touched."""
    data = _df_raw.copy()
    le_geo = LabelEncoder()
    le_gender = LabelEncoder()
    data["Geography_enc"] = le_geo.fit_transform(data["Geography"])
    data["Gender_enc"] = le_gender.fit_transform(data["Gender"])

    X = data[MODEL_FEATURES]
    y = data["Exited"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = GradientBoostingClassifier(n_estimators=200, max_depth=3, random_state=42)
    model.fit(X_train, y_train)

    rf_for_importance = RandomForestClassifier(
        n_estimators=300, max_depth=8, random_state=42, class_weight="balanced"
    )
    rf_for_importance.fit(X_train, y_train)

    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    metrics = {
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred),
        "Recall": recall_score(y_test, pred),
        "F1": f1_score(y_test, pred),
        "ROC-AUC": roc_auc_score(y_test, proba),
    }
    importances = pd.Series(
        rf_for_importance.feature_importances_, index=MODEL_FEATURES
    ).sort_values(ascending=False)

    return {
        "model": model,
        "le_geo": le_geo,
        "le_gender": le_gender,
        "metrics": metrics,
        "importances": importances,
        "test_size": len(X_test),
        "train_size": len(X_train),
    }


_churn_model_bundle = train_churn_model(df_raw)


def score_churn_risk(dataframe, bundle):
    """Applies the already-trained model to any (filtered) dataframe and
    returns risk scores + buckets. Never re-fits the model on the filtered
    subset, since predictions must stay comparable across filter changes."""
    if len(dataframe) == 0:
        return dataframe.assign(ChurnRiskScore=[], RiskBucket=[])
    d = dataframe.copy()
    try:
        d["Geography_enc"] = bundle["le_geo"].transform(d["Geography"])
        d["Gender_enc"] = bundle["le_gender"].transform(d["Gender"])
        d["ChurnRiskScore"] = (
            bundle["model"].predict_proba(d[MODEL_FEATURES])[:, 1] * 100
        )
    except Exception:
        d["ChurnRiskScore"] = 0.0
    d["RiskBucket"] = pd.cut(
        d["ChurnRiskScore"], bins=[-1, 30, 60, 101], labels=["Low", "Medium", "High"]
    )
    return d


# =========================================================
# SIDEBAR FILTERS
# =========================================================

st.markdown(
    """
<style>
section[data-testid="stSidebar"] div[data-testid="stSidebarContent"] {
    padding-top: 0rem;
}
</style>
""",
    unsafe_allow_html=True,
)
st.markdown(
    """
<style>

[data-testid="stSidebarContent"]{
    padding-top:-70rem;
    margin-top:-70px;
}

</style>
""",
    unsafe_allow_html=True,
)

st.sidebar.markdown('<div class="sidebar-logo">', unsafe_allow_html=True)

st.sidebar.image("logo.png", width=100)

st.sidebar.markdown("</div>", unsafe_allow_html=True)

# st.sidebar.markdown(
#     """
# <div style="padding: 0.8rem 0 0.4rem 0;">
#     <div style="text-align:center; margin-bottom:1rem; background:white;
#                 border-radius:8px; padding:0.6rem;">
#         <img src="https://www.ecb.europa.eu/shared/img/logo/logo_name.en.svg"
#              style="height:36px; max-width:100%;"
#              onerror="this.style.display='none';">
#     </div>
#     <div style="font-size:1rem; font-weight:700; color:#F1F5F9;
#                 letter-spacing:0.06em; text-transform:uppercase;
#                 border-bottom:1px solid #0040CC; padding-bottom:0.6rem;
#                 margin-bottom:1rem;">
#         ⚙ Dashboard Filters
#     </div>
# </div>
# """,
#     unsafe_allow_html=True,
# )

st.sidebar.markdown(
    '<div class="sb-header">'
    '<p class="sb-header-title">Filter Panel</p>'
    '<p class="sb-header-sub">Refine the customer population under review</p>'
    "</div>",
    unsafe_allow_html=True,
)




def reset_filters():
    st.session_state.geo_filter = list(df_raw["Geography"].unique())
    st.session_state.gender_filter = list(df_raw["Gender"].unique())
    st.session_state.age_filter = list(df_raw["AgeGroup"].cat.categories)
    st.session_state.credit_filter = list(df_raw["CreditBand"].cat.categories)
    st.session_state.tenure_filter = list(df_raw["TenureGroup"].cat.categories)
    st.session_state.balance_filter = list(df_raw["BalanceSegment"].cat.categories)
    st.session_state.activity_filter = list(df_raw["ActivityStatus"].unique())


if "geo_filter" not in st.session_state:
    reset_filters()

# ---- Group 1: Geographic & Demographic filters ----
with st.sidebar.expander("🌍  Geographic & Demographic", expanded=True):
    st.session_state.geo_filter = st.multiselect(
        "Geography",
        options=list(df_raw["Geography"].unique()),
        default=st.session_state.geo_filter,
    )
    st.session_state.gender_filter = st.multiselect(
        "Gender",
        options=list(df_raw["Gender"].unique()),
        default=st.session_state.gender_filter,
    )
    st.session_state.age_filter = st.multiselect(
        "Age Group",
        options=list(df_raw["AgeGroup"].cat.categories),
        default=st.session_state.age_filter,
    )

# ---- Group 2: Financial & Engagement filters ----
with st.sidebar.expander("💰  Financial & Engagement", expanded=True):
    st.session_state.credit_filter = st.multiselect(
        "Credit Score Band",
        options=list(df_raw["CreditBand"].cat.categories),
        default=st.session_state.credit_filter,
    )
    st.session_state.tenure_filter = st.multiselect(
        "Tenure Group",
        options=list(df_raw["TenureGroup"].cat.categories),
        default=st.session_state.tenure_filter,
    )
    st.session_state.balance_filter = st.multiselect(
        "Balance Segment",
        options=list(df_raw["BalanceSegment"].cat.categories),
        default=st.session_state.balance_filter,
    )
    st.session_state.activity_filter = st.multiselect(
        "Activity Status",
        options=list(df_raw["ActivityStatus"].unique()),
        default=st.session_state.activity_filter,
    )

st.sidebar.button("Reset All Filters", on_click=reset_filters)

df = df_raw[
    df_raw["Geography"].isin(st.session_state.geo_filter)
    & df_raw["Gender"].isin(st.session_state.gender_filter)
    & df_raw["AgeGroup"].isin(st.session_state.age_filter)
    & df_raw["CreditBand"].isin(st.session_state.credit_filter)
    & df_raw["TenureGroup"].isin(st.session_state.tenure_filter)
    & df_raw["BalanceSegment"].isin(st.session_state.balance_filter)
    & df_raw["ActivityStatus"].isin(st.session_state.activity_filter)
].copy()

# ---- Live Population Snapshot card ----
_pop_pct = (len(df) / len(df_raw) * 100) if len(df_raw) > 0 else 0
_snap_churn = (df["Exited"].mean() * 100) if len(df) > 0 else 0
st.sidebar.markdown(
    '<div class="sb-snapshot">'
    '<div class="sb-snapshot-label">Population Snapshot</div>'
    '<div class="sb-snapshot-value">'
    + f"{len(df):,}"
    + ' <span style="font-size:13px; color:#9AA5B1;">of '
    + f"{len(df_raw):,}"
    + " customers</span></div>"
    '<div class="sb-bar-track"><div class="sb-bar-fill" style="width:'
    + f"{_pop_pct:.1f}"
    + '%;"></div></div>'
    '<div class="sb-snapshot-sub">Selected churn rate: '
    + f"{_snap_churn:.1f}"
    + "%</div>"
    "</div>",
    unsafe_allow_html=True,
)

# =========================================================
# LETTERHEAD / REPORT HEADER
# =========================================================
report_id = "RPT-ECB-CSA-" + datetime.now().strftime("%Y%m%d")
generated_on = datetime.now().strftime("%d %B %Y, %H:%M")



# st.markdown(
#     """
#     <div class="letterhead">

#         <img
#             src="https://www.ecb.europa.eu/shared/img/logo/logo_name.en.svg"
#             class="letterhead-logo"
#         >

#         <p class="letterhead-title">
#             European Bank &mdash; Customer Segmentation &amp; Churn Intelligence Report
#         </p>

#         <p class="letterhead-subtitle">
#             Prepared under the Unified Mentor Data Analyst Internship Programme
#         </p>

#     </div>
#     """,
#     unsafe_allow_html=True,
# )








st.markdown('<div class="sidebar-logo">', unsafe_allow_html=True)

st.image("ecb.png", width=400)

st.markdown("</div>", unsafe_allow_html=True)

st.markdown(
    """
    <div class="letterhead">        
        <p class="letterhead-title">European Bank &mdash; Customer Segmentation &amp; Churn Intelligence Report</p>
        <p class="letterhead-subtitle">Prepared under the Unified Mentor Data Analyst Internship Programme</p>
    </div>
 """,
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="meta-strip">'
    "<span>Report ID: "
    + report_id
    + " &nbsp;|&nbsp; Generated: "
    + generated_on
    + " &nbsp;|&nbsp; Prepared by: Data Analytics Division (S. Das)</span>"
    '<span class="classification-tag">Confidential &mdash; Internal Use</span>'
    "</div>",
    unsafe_allow_html=True,
)

if len(df) == 0:
    st.warning(
        "No customers match the current filter combination. Please broaden your selection using the Filter Panel."
    )
    st.stop()

# =========================================================
# CORE KPI COMPUTATION — done once, above the tabs, so the
# same five figures back both the persistent ribbon below and
# the detailed Tab I cards, and can never disagree with each other
# =========================================================
overall_rate = safe_get_value(df, "Exited", "mean") * 100
top_seg = compute_top_segment(df, SEGMENT_DIMS, min_size=20)
hv_rate = (
    safe_get_value(df[df["HighValueCustomer"] == "High-Value"], "Exited", "mean") * 100
    if "High-Value" in df["HighValueCustomer"].values
    else 0
)
geo_tbl = df.groupby("Geography", observed=True)["Exited"].mean().mul(100).reset_index()
geo_tbl.columns = ["Geography", "ChurnRate"]
geo_tbl["RiskIndex"] = geo_tbl["ChurnRate"] / overall_rate if overall_rate > 0 else 0
top_geo = (
    geo_tbl.sort_values("RiskIndex", ascending=False).iloc[0]
    if len(geo_tbl) > 0
    else None
)
active_rate = (
    safe_get_value(df[df["ActivityStatus"] == "Active"], "Exited", "mean") * 100
    if "Active" in df["ActivityStatus"].values
    else 0
)
inactive_rate = (
    safe_get_value(df[df["ActivityStatus"] == "Inactive"], "Exited", "mean") * 100
    if "Inactive" in df["ActivityStatus"].values
    else 0
)
engagement_gap = inactive_rate - active_rate

# =========================================================
# PERSISTENT KPI RIBBON
# Rendered once, above st.tabs(), so it stays visible no matter
# which tab is currently selected — satisfies "KPI visible in all
# tabs" far more reliably than repeating the cards inside each tab.
# =========================================================
_ribbon_geo_text = (
    f"{top_geo['Geography']} ({top_geo['RiskIndex']:.2f}\u00d7)"
    if top_geo is not None
    else "N/A"
)
_ribbon_seg_text = f"{top_seg['ChurnRate']:.1f}%" if top_seg is not None else "N/A"
_ribbon_sign = "+" if engagement_gap >= 0 else ""

st.markdown("</div>", unsafe_allow_html=True)
r1c1, r1c2, r1c3, r1c4, r1c5 = st.columns(5)
with r1c1:
    kpi_report_card(
        "Overall Churn Rate",
        "% customers who exited",
        f"{overall_rate:.1f}%",
        f"{int(safe_get_value(df,'Exited','sum')):,} of {len(df):,} <br>customers exited",
    )
with r1c2:
    if top_seg is not None:
        kpi_report_card(
            "Segment Churn Rate",
            "Churn % by segment",
            f"{top_seg['ChurnRate']:.1f}%",
            f"Highest-risk segment: {top_seg['Dimension']} = {top_seg['Value']} (n={int(top_seg['Customers'])}). Full breakdown in Tab IV.",
        )
    else:
        kpi_report_card(
            "Segment Churn Rate",
            "Churn % by segment",
            "N/A",
            "Insufficient segment data",
        )
with r1c3:
    kpi_report_card(
        "High-Value Churn Ratio",
        "Churn among premium customers",
        f"{hv_rate:.1f}%",
        "Premium = top 25% by balance or salary, fixed on full portfolio",
    )

# r2c1, r2c2 = st.columns(2)
with r1c4:
    if top_geo is not None:
        kpi_report_card(
            "Geographic Risk Index",
            "Regional churn<br> exposure",
            f"{top_geo['RiskIndex']:.2f}\u00d7",
            f"{top_geo['Geography']} churns at {top_geo['RiskIndex']:.2f}\u00d7 the portfolio average ({top_geo['ChurnRate']:.1f}%)",
        )
    else:
        kpi_report_card("Geographic Risk Index", "Regional churn exposure", "N/A", "")
with r1c5:
    sign = "+" if engagement_gap >= 0 else ""
    kpi_report_card(
        "Engagement Drop <br>Indicator",
        "<br>Inactivity vs churn",
        f"{sign}{engagement_gap:.1f} pts",
        f"Inactive {inactive_rate:.1f}% vs Active {active_rate:.1f}% churn rate",
    )

# =========================================================
# TABS
# =========================================================
tab1, tab2, tab3, tab4, tab5, tab7 = st.tabs(
    [
        "I. Overall Churn Summary",
        "II. Geography Wise Churn Comparison",
        "III. Segmentation Explorer",
        "IV. High-Value Customer Churn EXplorer",
        "V. Age & Tenure Churn Comparison",
        "VI. Predictive Churn Risk Model",
    ]
)

# ---------------------------------------------------------
# TAB 0 — EXECUTIVE KPI SUMMARY
# ---------------------------------------------------------
# with tab0:
# st.markdown(
#   "<h3 class='section-header'>Key Performance Indicators</h3>",
#  unsafe_allow_html=True,
# )
# st.caption(
#   "The five indicators below are the primary metrics defined under the project's Key Performance Indicators specification."
# )


st.markdown("<br>", unsafe_allow_html=True)
# st.markdown(
#     "<h3 class='section-header'>KPI Reference Table</h3>", unsafe_allow_html=True
# )
kpi_ref = pd.DataFrame(
    {
        "KPI Name": [
            "Overall Churn Rate",
            "Segment Churn Rate",
            "High-Value Churn Ratio",
            "Geographic Risk Index",
            "Engagement Drop Indicator",
        ],
        "Description": [
            "% customers who exited",
            "Churn % by segment",
            "Churn among premium customers",
            "Regional churn exposure",
            "Inactivity vs churn",
        ],
        "Current Value": [
            f"{overall_rate:.1f}%",
            (
                f"{top_seg['ChurnRate']:.1f}% ({top_seg['Dimension']}: {top_seg['Value']})"
                if top_seg is not None
                else "N/A"
            ),
            f"{hv_rate:.1f}%",
            (
                f"{top_geo['RiskIndex']:.2f}\u00d7 ({top_geo['Geography']})"
                if top_geo is not None
                else "N/A"
            ),
            f"{sign}{engagement_gap:.1f} pts",
        ],
    }
)
# st.dataframe(kpi_ref, hide_index=True, use_container_width=True)

if top_geo is not None:
    _geo_clause = f"the {top_geo['Geography']} region at {top_geo['RiskIndex']:.2f}\u00d7 average risk"
else:
    _geo_clause = "not determinable with the current filter"

_engagement_excess = abs(engagement_gap)
_hv_excess = abs(hv_rate - overall_rate)
_geo_excess = ((top_geo["RiskIndex"] - 1) * overall_rate) if top_geo is not None else 0
_seg_excess = (top_seg["ChurnRate"] - overall_rate) if top_seg is not None else 0
_excess_map = {
    "geographic concentration": _geo_excess,
    "segment-level concentration": _seg_excess,
    "customer engagement": _engagement_excess,
    "high-value customer risk": _hv_excess,
}
_ranked_factors = sorted(_excess_map, key=_excess_map.get, reverse=True)
_dominant_factor, _second_factor = _ranked_factors[0], _ranked_factors[1]
_dominant_value = _excess_map[_dominant_factor]

# finding_box(
#     "01",
#     "Which KPI Deserves Leadership's Attention First",
#     f"Across the five KPIs above, the widest deviation from the {overall_rate:.1f}% portfolio baseline is "
#     f"{_geo_clause}, while High-Value customers churn at {hv_rate:.1f}% and the Inactive-vs-Active "
#     f"engagement gap stands at {engagement_gap:.1f} percentage points.",
#     (
#         f"A single KPI rarely tells the full story in isolation. Within the current selection, "
#         f"<b>{_dominant_factor}</b> represents the {magnitude_word(_dominant_value)} deviation from baseline "
#         f"among the five indicators, with <b>{_second_factor}</b> as the second-largest contributor. These two "
#         f"factors compound rather than operate independently — customers exposed to both simultaneously carry "
#         f"materially higher risk than either factor would suggest in isolation."
#     ),
#     (
#         f"Prioritise retention outreach on the intersection of {_dominant_factor} and {_second_factor} first, "
#         f"rather than treating each KPI as an independent workstream — see Tab IV for the exact segment breakdown. "
#         f"This {urgency_clause(_dominant_value, thresholds=(2, 5, 10))}."
#     ),
# )


# ---------------------------------------------------------
# TAB 1 — PORTFOLIO OVERVIEW
# ---------------------------------------------------------
with tab1:
    st.markdown(
        "<h3 class='section-header'>Overall Churn Summary</h3>", unsafe_allow_html=True
    )

    total_customers = len(df)
    churned_customers = int(safe_get_value(df, "Exited", "sum"))
    retained_customers = total_customers - churned_customers

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi_card("Overall Churn Rate", f"{overall_rate:.1f}", "%")
    with c2:
        kpi_card("Total Customers", f"{total_customers:,}")
    with c3:
        kpi_card("Churned Customers", f"{churned_customers:,}")
    with c4:
        kpi_card("Retained Customers", f"{retained_customers:,}")

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("**Churn vs Retained Split**")
    pie_df = df["ChurnStatus"].value_counts().reset_index()
    pie_df.columns = ["Status", "Count"]
    fig = px.pie(
        pie_df,
        names="Status",
        values="Count",
        color="Status",
        color_discrete_map={"Retained": "#22303C", "Churned": "#7A1F2B"},
        hole=0.45,
    )
    fig.update_layout(
        height=270,
        margin=dict(t=6, b=6, l=6, r=6),
    )
    _churned_n = (
        int(pie_df.loc[pie_df["Status"] == "Churned", "Count"].sum())
        if "Churned" in pie_df["Status"].values
        else 0
    )
    _retained_n = (
        int(pie_df.loc[pie_df["Status"] == "Retained", "Count"].sum())
        if "Retained" in pie_df["Status"].values
        else 0
    )
    _ratio = round(_retained_n / _churned_n, 1) if _churned_n > 0 else 0
    chart_row(
        fig,
        "Churn vs Retained Breakdown",
        [
            ("Retained customers", f"{_retained_n:,}"),
            ("Churned customers", f"{_churned_n:,}"),
            ("Retention-to-churn ratio", f"{_ratio}:1"),
            ("Overall churn rate", f"{overall_rate:.1f}%"),
        ],
        f"The portfolio-wide retention-to-attrition ratio stands at <b>{_ratio}:1</b>, corresponding to an overall churn rate of <b>{overall_rate:.1f}%</b> across the reviewed population.",
    )

    st.markdown("**Engagement Drop Indicator: Active vs Inactive Churn**")
    eng_df = pd.DataFrame(
        {
            "Activity Status": ["Active", "Inactive"],
            "Churn Rate (%)": [active_rate, inactive_rate],
        }
    )
    fig2 = px.bar(
        eng_df,
        x="Activity Status",
        y="Churn Rate (%)",
        color="Activity Status",
        color_discrete_map={"Active": "#22303C", "Inactive": "#7A1F2B"},
        text_auto=".1f",
    )
    fig2.update_layout(showlegend=False, margin=dict(t=10, b=10, l=10, r=10))




        # =========================================================
    # ENGAGEMENT DROP INDICATOR — FULLY DYNAMIC
    # =========================================================

    # Difference between inactive and active churn
    _engagement_gap_abs = abs(engagement_gap)

    # ---------------------------------------------------------
    # Dynamic direction
    # ---------------------------------------------------------

    if engagement_gap > 0:
        _higher_activity_group = "Inactive"
        _lower_activity_group = "Active"

    elif engagement_gap < 0:
        _higher_activity_group = "Active"
        _lower_activity_group = "Inactive"

    else:
        _higher_activity_group = "both"
        _lower_activity_group = "both"


    # ---------------------------------------------------------
    # Dynamic interpretation based on gap magnitude
    # ---------------------------------------------------------

    if _engagement_gap_abs >= 10:

        _engagement_interpretation = (
            f"The difference is substantial at "
            f"<b>{_engagement_gap_abs:.1f} percentage points</b>, "
            f"indicating a pronounced association between activity status "
            f"and observed churn within the current selection."
        )

    elif _engagement_gap_abs >= 5:

        _engagement_interpretation = (
            f"The difference is noticeable at "
            f"<b>{_engagement_gap_abs:.1f} percentage points</b>, "
            f"indicating a meaningful difference in observed churn "
            f"between the two activity groups."
        )

    elif _engagement_gap_abs >= 3:

        _engagement_interpretation = (
            f"The difference is moderate at "
            f"<b>{_engagement_gap_abs:.1f} percentage points</b>, "
            f"showing some separation in observed churn between "
            f"the two activity groups."
        )

    elif _engagement_gap_abs > 0:

        _engagement_interpretation = (
            f"The difference is relatively narrow at only "
            f"<b>{_engagement_gap_abs:.1f} percentage points</b>, "
            f"so activity status shows limited separation in observed "
            f"churn under the current selection."
        )

    else:

        _engagement_interpretation = (
            "Active and inactive customers show the same observed "
            "churn rate under the current selection."
        )


    # ---------------------------------------------------------
    # Dynamic finding
    # ---------------------------------------------------------

    if engagement_gap > 0:

        _engagement_finding = (
            f"Inactive customers exhibit a higher observed churn rate of "
            f"<b>{inactive_rate:.1f}%</b>, compared with "
            f"<b>{active_rate:.1f}%</b> for active customers — "
            f"a difference of <b>{_engagement_gap_abs:.1f} percentage points</b>. "
            f"{_engagement_interpretation}"
        )

    elif engagement_gap < 0:

        _engagement_finding = (
            f"Active customers exhibit a higher observed churn rate of "
            f"<b>{active_rate:.1f}%</b>, compared with "
            f"<b>{inactive_rate:.1f}%</b> for inactive customers — "
            f"a difference of <b>{_engagement_gap_abs:.1f} percentage points</b>. "
            f"{_engagement_interpretation}"
        )

    else:

        _engagement_finding = (
            f"Active and inactive customers both exhibit an observed churn "
            f"rate of <b>{active_rate:.1f}%</b>. "
            f"{_engagement_interpretation}"
        )


    chart_row(
        fig2,
        "Engagement Breakdown",
        [
            ("Active customer churn", f"{active_rate:.1f}%"),
            ("Inactive customer churn", f"{inactive_rate:.1f}%"),
            ("Engagement gap", f"{engagement_gap:+.1f} pts"),
            ("Portfolio average", f"{overall_rate:.1f}%"),
        ],
        _engagement_finding,
    )


        

    # chart_row(
    #     fig2,
    #     "Engagement Breakdown",
    #     [
    #         ("Active customer churn", f"{active_rate:.1f}%"),
    #         ("Inactive customer churn", f"{inactive_rate:.1f}%"),
    #         ("Engagement gap", f"{engagement_gap:.1f} pts"),
    #         ("Portfolio average", f"{overall_rate:.1f}%"),
    #     ],
    #     f"{'Inactive' if engagement_gap >= 0 else 'Active'} customers exhibit an attrition rate "
    #     f"<b>{abs(engagement_gap):.1f} percentage points</b> {direction_word(engagement_gap)} than "
    #     f"{'active' if engagement_gap >= 0 else 'inactive'} customers, "
    #     f"{'establishing engagement status as a leading indicator of churn risk rather than a coincidental correlate.' if abs(engagement_gap) >= 3 else 'though the gap under the current filter is too narrow to treat engagement status as a strong standalone signal.'}",
    # )

    if inactive_rate > active_rate:
        print("ggg")
        # finding_box(
        #     "02",
        #     "Engagement Is a Leading Indicator of Churn",
        #     (
        #         f"Inactive customers churn at {inactive_rate:.1f}%, compared to {active_rate:.1f}% for active customers "
        #         f"— a gap of {engagement_gap:.1f} percentage points within the currently selected population, "
        #         f"representing a {(inactive_rate/active_rate):.1f}\u00d7 elevation in relative risk."
        #         if active_rate > 0
        #         else f"Inactive customers churn at {inactive_rate:.1f}%, compared to {active_rate:.1f}% for active customers."
        #     ),
        #     (
        #         f"Account activity constitutes a {magnitude_word(engagement_gap)} predictor of churn within the current "
        #         f"selection. Customers who cease engaging with the bank's products exhibit measurable early warning "
        #         f"signs {('well before' if abs(engagement_gap) >= 8 else 'in advance of')} formal account closure, "
        #         f"which positions activity status as a leading rather than lagging indicator "
        #         f"{'of substantial predictive value' if abs(engagement_gap) >= 8 else 'of modest but non-trivial predictive value'}."
        #     ),
        #     (
        #         f"Route customers transitioning from Active to Inactive status into a retention workflow "
        #         f"{'immediately, given the scale of the observed gap' if abs(engagement_gap) >= 8 else 'as part of the standard outreach cadence, given the moderate scale of the observed gap'}, "
        #         f"rather than waiting for a churn event to trigger outreach. This {urgency_clause(engagement_gap)}."
        #     ),
        # )

    st.markdown(
        "<h3 class='section-header'>Comparison of Churned vs Retained Profiles</h3>",
        unsafe_allow_html=True,
    )
    st.caption(
        "Average profile of customers who left the bank versus those who stayed, across the core financial and relationship fields."
    )

    profile_cols = [
        "CreditScore",
        "Age",
        "Tenure",
        "Balance",
        "NumOfProducts",
        "EstimatedSalary",
    ]
    retained_profile = (
        df[df["Exited"] == 0][profile_cols].mean()
        if (df["Exited"] == 0).any()
        else pd.Series(0, index=profile_cols)
    )
    churned_profile = (
        df[df["Exited"] == 1][profile_cols].mean()
        if (df["Exited"] == 1).any()
        else pd.Series(0, index=profile_cols)
    )

    profile_table = pd.DataFrame(
        {
            "Metric": profile_cols,
            "Retained (avg)": retained_profile.values.round(1),
            "Churned (avg)": churned_profile.values.round(1),
        }
    )
    profile_table["Difference (%)"] = np.where(
        profile_table["Retained (avg)"] != 0,
        (
            (profile_table["Churned (avg)"] - profile_table["Retained (avg)"])
            / profile_table["Retained (avg)"]
            * 100
        ).round(1),
        0,
    )

    st.dataframe(profile_table, hide_index=True, use_container_width=True)

    st.markdown("**Churned Vs Retailned Customer Profile Gaps**")

    fig_profile = px.bar(
        profile_table,
        x="Difference (%)",
        y="Metric",
        orientation="h",
        color=profile_table["Difference (%)"] > 0,
        color_discrete_map={True: "#7A1F2B", False: "#22303C"},
        text="Difference (%)",
    )
    fig_profile.update_traces(texttemplate="%{text}%", textposition="outside")
    fig_profile.update_layout(
        showlegend=False,
        xaxis_title="Churned vs Retained — % Difference",
        yaxis_title="",
    )

    _widest = profile_table.reindex(
        profile_table["Difference (%)"].abs().sort_values(ascending=False).index
    ).iloc[0]
    _direction = "higher" if _widest["Difference (%)"] > 0 else "lower"
    _narrowest = profile_table.reindex(
        profile_table["Difference (%)"].abs().sort_values(ascending=True).index
    ).iloc[0]





        # =========================================================
    # CHURNED VS RETAINED PROFILE GAPS — FULLY DYNAMIC
    # =========================================================

    if len(profile_table) > 0:

        # -----------------------------------------------------
        # Identify widest and narrowest gaps
        # -----------------------------------------------------

        _widest = profile_table.loc[
            profile_table["Difference (%)"].abs().idxmax()
        ]

        _narrowest = profile_table.loc[
            profile_table["Difference (%)"].abs().idxmin()
        ]

        _widest_gap = _widest["Difference (%)"]
        _widest_gap_abs = abs(_widest_gap)

        _narrowest_gap = _narrowest["Difference (%)"]
        _narrowest_gap_abs = abs(_narrowest_gap)


        # -----------------------------------------------------
        # Dynamic direction of widest gap
        # -----------------------------------------------------

        if _widest_gap > 0:

            _widest_direction = "higher"

        elif _widest_gap < 0:

            _widest_direction = "lower"

        else:

            _widest_direction = "similar"


        # -----------------------------------------------------
        # Dynamic interpretation of widest gap
        # -----------------------------------------------------

        if _widest_gap_abs >= 30:

            _profile_interpretation = (
                f"<b>{_widest['Metric']}</b> shows a substantial difference of "
                f"<b>{_widest_gap_abs:.1f}%</b> between churned and retained "
                f"customers, making it the largest observed separation among "
                f"the profile fields examined."
            )

        elif _widest_gap_abs >= 15:

            _profile_interpretation = (
                f"<b>{_widest['Metric']}</b> shows a noticeable difference of "
                f"<b>{_widest_gap_abs:.1f}%</b> between churned and retained "
                f"customers and represents the largest observed separation "
                f"among the profile fields examined."
            )

        elif _widest_gap_abs >= 5:

            _profile_interpretation = (
                f"<b>{_widest['Metric']}</b> shows a moderate difference of "
                f"<b>{_widest_gap_abs:.1f}%</b> between churned and retained "
                f"customers, although the overall separation remains limited."
            )

        elif _widest_gap_abs > 0:

            _profile_interpretation = (
                f"The largest observed difference is only "
                f"<b>{_widest_gap_abs:.1f}%</b> for <b>{_widest['Metric']}</b>, "
                f"indicating relatively limited separation across the "
                f"profile fields examined."
            )

        else:

            _profile_interpretation = (
                "Churned and retained customers show no observed percentage "
                "difference across the profile fields examined."
            )


        # -----------------------------------------------------
        # Dynamic direction sentence
        # -----------------------------------------------------

        if _widest_gap > 0:

            _direction_sentence = (
                f"Churned customers have a higher average "
                f"<b>{_widest['Metric']}</b> than retained customers."
            )

        elif _widest_gap < 0:

            _direction_sentence = (
                f"Churned customers have a lower average "
                f"<b>{_widest['Metric']}</b> than retained customers."
            )

        else:

            _direction_sentence = (
                f"The average <b>{_widest['Metric']}</b> is the same for "
                f"churned and retained customers."
            )


        # -----------------------------------------------------
        # Dynamic finding
        # -----------------------------------------------------

        _profile_finding = (
            f"<b>{_widest['Metric']}</b> shows the widest observed "
            f"churned-versus-retained difference at "
            f"<b>{_widest_gap_abs:.1f}%</b> "
            f"({_widest_direction}). "
            f"{_direction_sentence} "
            f"{_profile_interpretation}"
        )


        chart_row(
            fig_profile,
            "Profile Gap Breakdown",
            [
                (
                    "Widest gap",
                    f"{_widest['Metric']} "
                    f"({_widest_gap:+.1f}%)"
                ),
                (
                    "Narrowest gap",
                    f"{_narrowest['Metric']} "
                    f"({_narrowest_gap:+.1f}%)"
                ),
                (
                    "Metrics compared",
                    f"{len(profile_table)}"
                ),
            ],
            _profile_finding,
        )

    else:

        chart_row(
            fig_profile,
            "Profile Gap Breakdown",
            [
                ("Status", "No profile data available"),
            ],
            "Insufficient data is available to compare churned and retained customer profiles under the current filter selection.",
        )




    # chart_row(
    #     fig_profile,
    #     "Profile Gap Breakdown",
    #     [
    #         ("Widest gap", f"{_widest['Metric']} ({_widest['Difference (%)']:+.1f}%)"),
    #         (
    #             "Narrowest gap",
    #             f"{_narrowest['Metric']} ({_narrowest['Difference (%)']:+.1f}%)",
    #         ),
    #         ("Metrics compared", f"{len(profile_table)}"),
    #     ],
    #     f"<b>{_widest['Metric']}</b> constitutes the most material point of divergence between churned and retained customers, at {abs(_widest['Difference (%)']):.1f}% {_direction}, and should be regarded as the primary differentiating factor among the profile fields examined.",
    # )

    _sorted_by_gap = profile_table.reindex(
        profile_table["Difference (%)"].abs().sort_values(ascending=False).index
    )
    _widest = _sorted_by_gap.iloc[0]
    _second_metric = _sorted_by_gap.iloc[1] if len(_sorted_by_gap) > 1 else None
    _direction = "higher" if _widest["Difference (%)"] > 0 else "lower"
    # finding_box(
    #     "03",
    #     "The Metric That Most Separates Churned From Retained Customers",
    #     f"{_widest['Metric']} shows the widest gap between the two groups: churned customers average "
    #     f"{_widest['Churned (avg)']:,.1f} versus {_widest['Retained (avg)']:,.1f} for retained customers "
    #     f"— {abs(_widest['Difference (%)']):.1f}% {_direction}.",
    #     (
    #         f"A {magnitude_word(_widest['Difference (%)'], thresholds=(5, 15, 30))} gap of "
    #         f"{abs(_widest['Difference (%)']):.1f}% in {_widest['Metric']} between churned and retained customers "
    #         f"indicates this field carries real predictive signal"
    #         + (
    #             f", well ahead of the next-largest gap of {abs(_second_metric['Difference (%)']):.1f}% observed in "
    #             f"{_second_metric['Metric']}, rather than merely reflecting correlation with geography or age already "
    #             f"captured elsewhere in this report."
    #             if _second_metric is not None
    #             else "."
    #         )
    #     ),
    #     (
    #         f"Treat {_widest['Metric']} as the priority feature if this analysis is extended into a predictive churn "
    #         f"model"
    #         + (
    #             f", with {_second_metric['Metric']} as a secondary candidate"
    #             if _second_metric is not None
    #             else ""
    #         )
    #         + f", and consider it as a screening criterion for proactive retention lists. This "
    #         f"{urgency_clause(_widest['Difference (%)'], thresholds=(5, 15, 30))}."
    #     ),
    # )

    # ---------------------------------------------------------
    # Geography x Gender Snapshot
    # ---------------------------------------------------------
    st.markdown("**Geography x Gender Snapshot**")

    snap = (
        df.groupby(["Geography", "Gender"], observed=True)["Exited"]
        .mean()
        .mul(100)
        .round(1)
        .reset_index()
    )

    snap.columns = ["Geography", "Gender", "Churn Rate (%)"]

    fig3 = px.bar(
        snap,
        x="Geography",
        y="Churn Rate (%)",
        color="Gender",
        barmode="group",
        color_discrete_map={"Male": "#22303C", "Female": "#B08D57"},
    )

    # ---------------------------------------------------------
    # Highest Geography × Gender combination
    # ---------------------------------------------------------
    _snap_top = (
        snap.sort_values("Churn Rate (%)", ascending=False).iloc[0]
        if len(snap) > 0
        else None
    )

    # ---------------------------------------------------------
    # Overall Female and Male churn rates
    # ---------------------------------------------------------
    _f_avg = (
        df.loc[df["Gender"] == "Female", "Exited"].mean() * 100
        if (df["Gender"] == "Female").any()
        else 0
    )

    _m_avg = (
        df.loc[df["Gender"] == "Male", "Exited"].mean() * 100
        if (df["Gender"] == "Male").any()
        else 0
    )

    # ---------------------------------------------------------
    # Dynamic gender comparison
    # ---------------------------------------------------------
    _gender_gap = _f_avg - _m_avg

    if abs(_gender_gap) < 0.1:
        _gender_finding = (
            "At the aggregate level, female and male customers show "
            "approximately the same churn rate across the reviewed population."
        )
    elif _gender_gap > 0:
        _gender_finding = (
            f"At the aggregate level, female customers show a higher churn rate "
            f"than male customers by <b>{abs(_gender_gap):.1f} percentage points</b> "
            f"across the reviewed population."
        )
    else:
        _gender_finding = (
            f"At the aggregate level, male customers show a higher churn rate "
            f"than female customers by <b>{abs(_gender_gap):.1f} percentage points</b> "
            f"across the reviewed population."
        )

    # ---------------------------------------------------------
    # Dynamic KPI statistics
    # ---------------------------------------------------------
    _snap_stats = [
        ("Combinations shown", f"{len(snap)}"),
    ]

    if _snap_top is not None:
        _snap_stats.insert(
            0,
            (
                "Highest cell",
                f"{_snap_top['Geography']} / "
                f"{_snap_top['Gender']} — "
                f"{_snap_top['Churn Rate (%)']:.1f}%",
            ),
        )

    _snap_stats.append(("Female avg", f"{_f_avg:.1f}%"))
    _snap_stats.append(("Male avg", f"{_m_avg:.1f}%"))

    # ---------------------------------------------------------
    # Final dynamic chart finding
    # ---------------------------------------------------------
    if _snap_top is not None:

        _chart_finding = (
            f"The <b>{_snap_top['Geography']} / "
            f"{_snap_top['Gender']}</b> combination reflects the "
            f"highest observed churn rate, at "
            f"<b>{_snap_top['Churn Rate (%)']:.1f}%</b>. "
            f"{_gender_finding}"
        )

    else:

        _chart_finding = (
            "Insufficient data is available to support a "
            "Geography × Gender churn finding."
        )

    # ---------------------------------------------------------
    # Render chart + findings
    # ---------------------------------------------------------
    chart_row(
        fig3,
        "Geography x Gender Breakdown",
        _snap_stats,
        _chart_finding,
    )


        # =========================================================
    # FINDING 01 — OVERALL CHURN SUMMARY
    # =========================================================

    _total_customers = len(df)

    _total_churned = int(df["Exited"].sum())

    _total_retained = int(_total_customers - _total_churned)

    _overall_churn_rate = (
        (_total_churned / _total_customers) * 100
        if _total_customers > 0
        else 0
    )

    _retention_rate = (
        (_total_retained / _total_customers) * 100
        if _total_customers > 0
        else 0
    )

    # ---------------------------------------------------------
    # Churn severity interpretation
    # ---------------------------------------------------------

    if _overall_churn_rate >= 30:
        _severity_text = (
            "a high level of customer attrition that requires strong "
            "retention attention"
        )
    elif _overall_churn_rate >= 20:
        _severity_text = (
            "a substantial level of customer attrition that warrants "
            "focused retention attention"
        )
    elif _overall_churn_rate >= 15:
        _severity_text = (
            "a meaningful level of customer attrition that should be "
            "actively monitored and addressed"
        )
    elif _overall_churn_rate >= 10:
        _severity_text = (
            "a moderate level of customer attrition that requires "
            "continued monitoring"
        )
    else:
        _severity_text = (
            "a relatively lower level of customer attrition, although "
            "continued retention monitoring remains important"
        )

    # ---------------------------------------------------------
    # Churn-to-retention relationship
    # ---------------------------------------------------------

    if _total_churned > 0 and _total_retained > 0:

        _retained_to_churned = _total_retained / _total_churned

        _relationship_text = (
            f"For every 1 customer who has churned, approximately "
            f"<b>{_retained_to_churned:.1f}</b> customers remain retained."
        )

    else:
        _relationship_text = (
            "There is insufficient churn or retention volume to calculate "
            "a meaningful retained-to-churned relationship."
        )

    # ---------------------------------------------------------
    # Dynamic business interpretation
    # ---------------------------------------------------------

    if _overall_churn_rate >= 20:

        _priority_text = (
            "The scale of observed attrition makes customer retention a "
            "<b>high-priority business area</b>. Further analysis should "
            "focus on identifying the customer segments, engagement "
            "patterns and financial characteristics associated with the "
            "highest churn exposure."
        )

    elif _overall_churn_rate >= 10:

        _priority_text = (
            "The observed attrition level is material enough to justify "
            "<b>targeted retention analysis</b>. The next step should be "
            "to identify which customer groups contribute most to churn "
            "and where retention interventions can have the greatest "
            "potential impact."
        )

    else:

        _priority_text = (
            "The overall attrition level is comparatively lower, but "
            "retention should still be monitored across customer segments "
            "to identify emerging areas of elevated churn."
        )

    # =========================================================
    # SINGLE FINDING FOR THE ENTIRE OVERALL CHURN SUMMARY TAB
    # =========================================================

    finding_box(
        "01",
        "Overall Customer Churn Summary",

        (
            f"<ol>"
            f"<li><b>Overall churn position:</b> "
            f"Out of <b>{_total_customers:,}</b> customers in the current "
            f"selection, <b>{_total_churned:,}</b> customers have churned, "
            f"resulting in an overall churn rate of "
            f"<b>{_overall_churn_rate:.1f}%</b>. "
            f"This represents {_severity_text}.</li>"

            f"<li><b>Customer retention position:</b> "
            f"<b>{_total_retained:,}</b> customers remain retained, "
            f"representing <b>{_retention_rate:.1f}%</b> of the current "
            f"customer base. "
            f"{_relationship_text}</li>"

            f"<li><b>Scale of the issue:</b> "
            f"The churn rate should be interpreted together with the "
            f"absolute number of churned customers. "
            f"A percentage alone shows the rate of attrition, while the "
            f"customer count shows the actual scale of customers that "
            f"have been lost.</li>"

            f"<li><b>Management implication:</b> "
            f"{_priority_text}</li>"
            f"</ol>"
        ),

        # (
        #     f"<b>Overall interpretation:</b> "
        #     f"The current customer base shows an observed churn rate of "
        #     f"<b>{_overall_churn_rate:.1f}%</b>, with "
        #     f"<b>{_total_churned:,}</b> customers having exited and "
        #     f"<b>{_total_retained:,}</b> customers remaining retained. "
        #     f"The overall result provides the baseline against which all "
        #     f"subsequent segment, geography, demographic and engagement "
        #     f"analyses should be compared. "
        #     f"In other words, the portfolio-wide churn rate establishes "
        #     f"the reference point for identifying groups with relatively "
        #     f"higher or lower churn exposure."
        # ),

        # (
        #     f"<b>Recommended action:</b> "
        #     f"Use the <b>{_overall_churn_rate:.1f}%</b> overall churn rate "
        #     f"as the dashboard's baseline metric. "
        #     f"Then use the remaining dashboard analyses to determine "
        #     f"<b>where churn is concentrated</b>, <b>which customer groups "
        #     f"are most exposed</b>, and <b>which areas should receive "
        #     f"retention attention first</b>. "
        #     f"The overall summary describes the scale of churn; it should "
        #     f"not by itself be interpreted as identifying the causes of churn."
        # )
    )

#
# ---------------------------------------------------------
# TAB 2 — GEOGRAPHY WISE CHURN COMPARISON
# ---------------------------------------------------------
with tab2:

    # =========================================================
    # 1. GEOGRAPHY-WISE CHURN VISUALIZATION
    # =========================================================
    st.markdown(
        "<h3 class='section-header'>Geography-Wise Churn Visualization</h3>",
        unsafe_allow_html=True,
    )

    geo_churn = (
        df.groupby("Geography", observed=True)
        .agg(
            Churn_Rate=("Exited", "mean"),
            Customers=("Exited", "count"),
            Churned=("Exited", "sum"),
        )
        .reset_index()
    )

    geo_churn["Churn_Rate"] = (geo_churn["Churn_Rate"] * 100).round(1)

    geo_churn["Geographic_Risk_Index"] = (
        geo_churn["Churn_Rate"] / overall_rate if overall_rate > 0 else 0
    )

    fig4 = px.bar(
        geo_churn,
        x="Geography",
        y="Churn_Rate",
        text="Churn_Rate",
        color="Geography",
        color_discrete_sequence=[
            "#22303C",
            "#7A1F2B",
            "#B08D57",
        ],
    )

    fig4.update_traces(
        texttemplate="%{text}%",
        textposition="outside",
    )

    fig4.update_layout(
        yaxis_title="Churn Rate (%)",
        showlegend=False,
    )

    # ---------------------------------------------------------
    # Dynamic geography ranking
    # ---------------------------------------------------------
    _geo_sorted = geo_churn.sort_values("Churn_Rate", ascending=False).reset_index(
        drop=True
    )

    _geo_high = _geo_sorted.iloc[0]
    _geo_low = _geo_sorted.iloc[-1]

    _geo_gap = _geo_high["Churn_Rate"] - _geo_low["Churn_Rate"]

    # ---------------------------------------------------------
    # Dynamic geography finding
    # ---------------------------------------------------------
    if len(_geo_sorted) == 1:

        _geo_finding = (
            f"<b>{_geo_high['Geography']}</b> records a churn rate of "
            f"<b>{_geo_high['Churn_Rate']:.1f}%</b> across "
            f"{int(_geo_high['Customers']):,} customers in the "
            f"current filtered population."
        )

    else:

        _geo_finding = (
            f"<b>{_geo_high['Geography']}</b> records the highest observed "
            f"churn rate at <b>{_geo_high['Churn_Rate']:.1f}%</b>, "
            f"while <b>{_geo_low['Geography']}</b> records the lowest at "
            f"<b>{_geo_low['Churn_Rate']:.1f}%</b>. "
            f"The difference between the highest- and lowest-churn "
            f"geographies is <b>{_geo_gap:.1f} percentage points</b>."
        )

    _geo_stats = [
        (
            f"{r['Geography']} — Risk Index",
            f"{r['Geographic_Risk_Index']:.2f}× " f"({r['Churn_Rate']:.1f}%)",
        )
        for _, r in _geo_sorted.iterrows()
    ]

    _geo_stats.append(("Portfolio average", f"{overall_rate:.1f}%"))

    chart_row(
        fig4,
        "Geographic Risk Index",
        _geo_stats,
        _geo_finding,
    )

    # =========================================================
    # 3. GEOGRAPHY × AGE INTERACTION ANALYSIS
    # =========================================================
    st.markdown(
        "<h3 class='section-header'>Geography x Age Interaction Analysis</h3>",
        unsafe_allow_html=True,
    )

    heat_data = (
        df.groupby(["Geography", "AgeGroup"], observed=True)["Exited"]
        .mean()
        .mul(100)
        .round(1)
        .reset_index()
    )

    heat_data.columns = [
        "Geography",
        "AgeGroup",
        "Churn Rate (%)",
    ]

    heat_pivot = heat_data.pivot(
        index="Geography",
        columns="AgeGroup",
        values="Churn Rate (%)",
    )

    fig7 = px.imshow(
        heat_pivot,
        text_auto=True,
        color_continuous_scale=[
            "#F7F5F0",
            "#7A1F2B",
        ],
        labels=dict(color="Churn Rate (%)"),
    )

    _heat_flat = heat_data.sort_values("Churn Rate (%)", ascending=False).reset_index(
        drop=True
    )

    if len(_heat_flat) > 0:

        # -----------------------------------------------------
        # Highest and lowest Geography × Age cells
        # -----------------------------------------------------
        _hot = _heat_flat.iloc[0]
        _cold = _heat_flat.iloc[-1]

        # -----------------------------------------------------
        # Geography-only churn for highest-risk geography
        # -----------------------------------------------------
        _hot_geo_rate = (
            df.loc[df["Geography"] == _hot["Geography"], "Exited"].mean() * 100
        )

        # -----------------------------------------------------
        # Age-only churn for highest-risk age group
        # -----------------------------------------------------
        _hot_age_rate = (
            df.loc[df["AgeGroup"] == _hot["AgeGroup"], "Exited"].mean() * 100
        )

        # -----------------------------------------------------
        # Compare combined cell with both individual dimensions
        # -----------------------------------------------------
        _component_max = max(_hot_geo_rate, _hot_age_rate)

        _combined_excess = _hot["Churn Rate (%)"] - _component_max

        # -----------------------------------------------------
        # Dynamic interpretation
        # -----------------------------------------------------
        if _combined_excess > 0:

            _interaction_text = (
                f"The combined segment records "
                f"<b>{_hot['Churn Rate (%)']:.1f}%</b> churn, "
                f"which is <b>{_combined_excess:.1f} percentage points</b> "
                f"above the higher of its geography-only rate "
                f"({_hot_geo_rate:.1f}%) and age-group-only rate "
                f"({_hot_age_rate:.1f}%). This identifies a "
                f"particularly elevated observed churn concentration "
                f"for this specific Geography × Age segment."
            )

        elif _combined_excess < 0:

            _interaction_text = (
                f"The combined segment records "
                f"<b>{_hot['Churn Rate (%)']:.1f}%</b> churn, "
                f"which is <b>{abs(_combined_excess):.1f} percentage points</b> "
                f"below the higher of its geography-only rate "
                f"({_hot_geo_rate:.1f}%) and age-group-only rate "
                f"({_hot_age_rate:.1f}%). The combined segment therefore "
                f"does not exceed the individual dimension rates."
            )

        else:

            _interaction_text = (
                f"The combined segment records "
                f"<b>{_hot['Churn Rate (%)']:.1f}%</b> churn, "
                f"matching the higher of its geography-only rate "
                f"({_hot_geo_rate:.1f}%) and age-group-only rate "
                f"({_hot_age_rate:.1f}%)."
            )

        # -----------------------------------------------------
        # Fully dynamic main finding
        # -----------------------------------------------------
        _heat_finding = (
            f"<b>{_hot['Geography']} / {_hot['AgeGroup']}</b> "
            f"records the highest observed Geography × Age churn rate "
            f"at <b>{_hot['Churn Rate (%)']:.1f}%</b>, compared with "
            f"a portfolio-wide churn rate of <b>{overall_rate:.1f}%</b>. "
            f"{_interaction_text}"
        )

        chart_row(
            fig7,
            "Interaction Breakdown",
            [
                (
                    "Highest-risk cell",
                    f"{_hot['Geography']} / "
                    f"{_hot['AgeGroup']} — "
                    f"{_hot['Churn Rate (%)']:.1f}%",
                ),
                (
                    "Lowest-risk cell",
                    f"{_cold['Geography']} / "
                    f"{_cold['AgeGroup']} — "
                    f"{_cold['Churn Rate (%)']:.1f}%",
                ),
                (
                    "Cells shown",
                    f"{len(_heat_flat)}",
                ),
                (
                    "Portfolio average",
                    f"{overall_rate:.1f}%",
                ),
            ],
            _heat_finding,
        )

        # -----------------------------------------------------
        # Dynamic Finding Box 04
        # -----------------------------------------------------
        # finding_box(
        #     "04",
        #     "Highest-Risk Geography x Age Combination",
        #     (
        #         f"The highest observed Geography × Age segment is "
        #         f"<b>{_hot['Geography']}</b> / "
        #         f"<b>{_hot['AgeGroup']}</b>, with a churn rate of "
        #         f"<b>{_hot['Churn Rate (%)']:.1f}%</b> versus the "
        #         f"<b>{overall_rate:.1f}%</b> portfolio average."
        #     ),
        #     _interaction_text,
        #     (
        #         f"Prioritise retention analysis for "
        #         f"<b>{_hot['Geography']}</b> customers in the "
        #         f"<b>{_hot['AgeGroup']}</b> age group, while using the "
        #         f"current filtered population to monitor whether this "
        #         f"segment remains the highest observed-risk combination."
        #     ),
        # )

    # =========================================================
    # 4. GENDER-BASED CHURN DIFFERENCES
    # =========================================================
    st.markdown(
        "<h3 class='section-header'>Gender-Based Churn Differences</h3>",
        unsafe_allow_html=True,
    )

    gender_churn = (
        df.groupby("Gender", observed=True)["Exited"]
        .mean()
        .mul(100)
        .round(1)
        .reset_index()
    )

    gender_churn.columns = ["Gender", "Churn Rate (%)"]

    fig8 = px.bar(
        gender_churn,
        x="Gender",
        y="Churn Rate (%)",
        text="Churn Rate (%)",
        color="Gender",
        color_discrete_map={
            "Male": "#22303C",
            "Female": "#B08D57",
        },
    )

    fig8.update_traces(
        texttemplate="%{text}%",
        textposition="outside",
    )

    # ---------------------------------------------------------
    # Extract gender rates dynamically
    # ---------------------------------------------------------
    _female_rate = (
        df.loc[df["Gender"] == "Female", "Exited"].mean() * 100
        if (df["Gender"] == "Female").any()
        else None
    )

    _male_rate = (
        df.loc[df["Gender"] == "Male", "Exited"].mean() * 100
        if (df["Gender"] == "Male").any()
        else None
    )

    # ---------------------------------------------------------
    # Dynamic gender finding
    # ---------------------------------------------------------
    if _female_rate is None and _male_rate is None:

        _gender_finding = (
            "Insufficient gender data is available to support "
            "a churn comparison under the current filter."
        )

    elif _female_rate is None:

        _gender_finding = (
            f"Only male customers are represented in the current "
            f"filtered population, with an observed churn rate of "
            f"<b>{_male_rate:.1f}%</b>."
        )

    elif _male_rate is None:

        _gender_finding = (
            f"Only female customers are represented in the current "
            f"filtered population, with an observed churn rate of "
            f"<b>{_female_rate:.1f}%</b>."
        )

    else:

        _gender_gap = _female_rate - _male_rate

        if abs(_gender_gap) < 0.1:

            _gender_finding = (
                f"Female and male customers show approximately the "
                f"same observed churn rate: <b>{_female_rate:.1f}%</b> "
                f"for female customers and <b>{_male_rate:.1f}%</b> "
                f"for male customers."
            )

        elif _gender_gap > 0:

            _gender_finding = (
                f"Female customers record the higher observed churn "
                f"rate at <b>{_female_rate:.1f}%</b>, compared with "
                f"<b>{_male_rate:.1f}%</b> for male customers — a "
                f"difference of <b>{abs(_gender_gap):.1f} percentage "
                f"points</b>."
            )

        else:

            _gender_finding = (
                f"Male customers record the higher observed churn "
                f"rate at <b>{_male_rate:.1f}%</b>, compared with "
                f"<b>{_female_rate:.1f}%</b> for female customers — a "
                f"difference of <b>{abs(_gender_gap):.1f} percentage "
                f"points</b>."
            )

    # ---------------------------------------------------------
    # Dynamic gender statistics
    # ---------------------------------------------------------
    _gender_stats = []

    if _female_rate is not None:
        _gender_stats.append(("Female churn", f"{_female_rate:.1f}%"))

    if _male_rate is not None:
        _gender_stats.append(("Male churn", f"{_male_rate:.1f}%"))

    _gender_stats.append(("Portfolio average", f"{overall_rate:.1f}%"))

    chart_row(
        fig8,
        "Gender Breakdown",
        _gender_stats,
        _gender_finding,
    )

    st.markdown(
        "<h3 class='section-header'>Financial Stability vs Churn Comparison</h3>",
        unsafe_allow_html=True,
    )
    st.caption(
        "Credit score, product holding, and card ownership are treated here as financial stability indicators and compared against churn outcome."
    )

    credit_churn = (
        df.groupby("CreditBand", observed=True)["Exited"]
        .mean()
        .mul(100)
        .round(1)
        .reset_index()
    )
    credit_churn.columns = ["Credit Score Band", "Churn Rate (%)"]
    fig13 = px.bar(
        credit_churn,
        x="Credit Score Band",
        y="Churn Rate (%)",
        text="Churn Rate (%)",
        color_discrete_sequence=["#22303C"],
        title="Churn Rate by Credit Score Band",
    )
    fig13.update_traces(texttemplate="%{text}%", textposition="outside")
    chart_row(
        fig13,
        "Credit Score Breakdown",
        rate_breakdown_stats(
            credit_churn, "Credit Score Band", "Churn Rate (%)", overall=overall_rate
        ),
        rate_insight_text(
            credit_churn, "Credit Score Band", "Churn Rate (%)", overall=overall_rate
        ),
    )

    prod_churn = (
        df.groupby("NumOfProducts", observed=True)["Exited"]
        .mean()
        .mul(100)
        .round(1)
        .reset_index()
    )
    prod_churn.columns = ["Number of Products", "Churn Rate (%)"]
    fig14 = px.bar(
        prod_churn,
        x="Number of Products",
        y="Churn Rate (%)",
        text="Churn Rate (%)",
        color_discrete_sequence=["#B08D57"],
        title="Churn Rate by Product Holding",
    )
    fig14.update_traces(texttemplate="%{text}%", textposition="outside")
    chart_row(
        fig14,
        "Product Holding Breakdown",
        rate_breakdown_stats(
            prod_churn, "Number of Products", "Churn Rate (%)", overall=overall_rate
        ),
        rate_insight_text(
            prod_churn, "Number of Products", "Churn Rate (%)", overall=overall_rate
        ),
    )

    card_churn = (
        df.groupby("HasCrCard", observed=True)["Exited"]
        .mean()
        .mul(100)
        .round(1)
        .reset_index()
    )
    card_churn["HasCrCard"] = card_churn["HasCrCard"].map(
        {1: "Has Credit Card", 0: "No Credit Card"}
    )
    card_churn.columns = ["Credit Card Ownership", "Churn Rate (%)"]
    fig15 = px.bar(
        card_churn,
        x="Credit Card Ownership",
        y="Churn Rate (%)",
        text="Churn Rate (%)",
        color_discrete_sequence=["#7A1F2B"],
        title="Churn Rate by Credit Card Ownership",
    )
    fig15.update_traces(texttemplate="%{text}%", textposition="outside")
    chart_row(
        fig15,
        "Card Ownership Breakdown",
        rate_breakdown_stats(
            card_churn, "Credit Card Ownership", "Churn Rate (%)", overall=overall_rate
        ),
        rate_insight_text(
            card_churn, "Credit Card Ownership", "Churn Rate (%)", overall=overall_rate
        ),
    )

    
    
    

    fig16 = px.box(
        df,
        x="ChurnStatus",
        y="Balance",
        color="ChurnStatus",
        color_discrete_map={"Retained": "#22303C", "Churned": "#7A1F2B"},
        title="Balance Distribution: Churned vs Retained",
    )

    fig16.update_layout(showlegend=False)

    # ---------------------------------------------------------
    # Dynamic Balance Distribution Finding
    # ---------------------------------------------------------
    _bal_churned_med = (
        df.loc[df["Exited"] == 1, "Balance"].median()
        if (df["Exited"] == 1).any()
        else 0
    )

    _bal_retained_med = (
        df.loc[df["Exited"] == 0, "Balance"].median()
        if (df["Exited"] == 0).any()
        else 0
    )

    _bal_difference = _bal_churned_med - _bal_retained_med

    # Dynamic direction and interpretation
    if abs(_bal_difference) < 1000:
        _balance_finding = (
            f"Churned and retained customers show broadly similar median account "
            f"balances, at <b>€{_bal_churned_med:,.0f}</b> and "
            f"<b>€{_bal_retained_med:,.0f}</b>, respectively. "
            f"The median difference is only <b>€{abs(_bal_difference):,.0f}</b>, "
            f"suggesting limited separation in typical account balances between "
            f"the two groups within the current selection."
        )

    elif _bal_difference > 0:
        _balance_finding = (
            f"Churned customers show a higher median account balance of "
            f"<b>€{_bal_churned_med:,.0f}</b> compared with "
            f"<b>€{_bal_retained_med:,.0f}</b> among retained customers — "
            f"a difference of <b>€{abs(_bal_difference):,.0f}</b>. "
            f"This indicates that higher typical account balances are not "
            f"associated with lower observed churn within the current selection."
        )

    else:
        _balance_finding = (
            f"Churned customers show a lower median account balance of "
            f"<b>€{_bal_churned_med:,.0f}</b> compared with "
            f"<b>€{_bal_retained_med:,.0f}</b> among retained customers — "
            f"a difference of <b>€{abs(_bal_difference):,.0f}</b>. "
            f"This indicates that lower typical account balances are associated "
            f"with higher observed churn within the current selection."
        )

    chart_row(
        fig16,
        "Balance Distribution Breakdown",
        [
            ("Median balance — Churned", f"€{_bal_churned_med:,.0f}"),
            ("Median balance — Retained", f"€{_bal_retained_med:,.0f}"),
            ("Median difference", f"€{_bal_difference:,.0f}"),
        ],
        _balance_finding,
    )

    
    _prod_sorted = prod_churn.sort_values("Churn Rate (%)", ascending=False)
    _worst_prod = _prod_sorted.iloc[0] if len(_prod_sorted) > 0 else None
    if _worst_prod is not None:
        _prod_median = prod_churn["Number of Products"].median()
        _is_low_end = _worst_prod["Number of Products"] <= _prod_median
        _best_prod_rate = prod_churn.sort_values("Churn Rate (%)").iloc[0]
        _prod_gap = _worst_prod["Churn Rate (%)"] - _best_prod_rate["Churn Rate (%)"]
        if _is_low_end:
            _prod_interpretation = (
                f"This pattern reflects a low-embeddedness dynamic: customers holding only "
                f"{_worst_prod['Number of Products']} product(s) carry {magnitude_word(_prod_gap)} switching costs "
                f"relative to the {_best_prod_rate['Number of Products']}-product segment, which churns at just "
                f"{_best_prod_rate['Churn Rate (%)']:.1f}%. Fewer products means less friction to leave when "
                f"dissatisfied, and less of the relationship depth that typically anchors retention."
            )
            _prod_recommendation = (
                f"Cross-check the {_worst_prod['Number of Products']}-product segment against onboarding date and "
                f"channel to identify under-served new customers, and prioritise a structured cross-sell programme "
                f"to deepen product relationships before disengagement sets in. This {urgency_clause(_prod_gap)}."
            )
        else:
            _prod_interpretation = (
                f"This pattern is consistent with a potential over-selling dynamic: an unusually high product count "
                f"of {_worst_prod['Number of Products']} correlates with a {magnitude_word(_prod_gap)} elevation in "
                f"churn ({_prod_gap:.1f} points above the best-performing {_best_prod_rate['Number of Products']}-product "
                f"segment at {_best_prod_rate['Churn Rate (%)']:.1f}%), suggesting these customers may feel mis-sold "
                f"rather than genuinely loyal."
            )
            _prod_recommendation = (
                f"Cross-check the {_worst_prod['Number of Products']}-product segment against sales channel and "
                f"bundling practices to determine whether products were sold to match genuine need, and consider "
                f"a service-quality review for this cohort. This {urgency_clause(_prod_gap)}."
            )
        # finding_box(
        #     "05",
        #     "Product Holding Is the Strongest Financial Stability Signal",
        #     f"Customers holding {_worst_prod['Number of Products']} product(s) churn at {_worst_prod['Churn Rate (%)']:.1f}%, "
        #     f"the highest churn rate across all product-count bands in the current selection.",
        #     _prod_interpretation,
        #     _prod_recommendation,
        # )


        # =========================================================
    # CONSOLIDATED FINDING — ENTIRE GEOGRAPHY-WISE CHURN TAB
    # =========================================================

    # ---------------------------------------------------------
    # 1. Geography-level statistics
    # ---------------------------------------------------------
    _geo_summary = (
        df.groupby("Geography", observed=True)
        .agg(
            Customers=("Exited", "count"),
            Churned=("Exited", "sum"),
            Churn_Rate=("Exited", "mean"),
        )
        .reset_index()
    )

    _geo_summary["Churn_Rate"] = (
        _geo_summary["Churn_Rate"] * 100
    )

    # Sort from highest to lowest churn rate
    _geo_ranked = _geo_summary.sort_values(
        "Churn_Rate",
        ascending=False
    ).reset_index(drop=True)


    # ---------------------------------------------------------
    # 2. Handle insufficient geography data safely
    # ---------------------------------------------------------
    if len(_geo_ranked) == 0:

        finding_box(
            "02",
            "Geography-Wise Churn Comparison — Overall Finding",
            (
                "Insufficient geographic data is available to produce "
                "a reliable geography-wise churn comparison under the "
                "current filter selection."
            ),
            (
                "The dashboard requires at least one valid geography "
                "with customer-level churn information before regional "
                "differences can be interpreted."
            ),
            (
                "Review the active filters and ensure that customer records "
                "are available across the geography dimension."
            ),
        )

    else:

        # ---------------------------------------------------------
        # 3. Highest and lowest geography
        # ---------------------------------------------------------
        _geo_high = _geo_ranked.iloc[0]
        _geo_low = _geo_ranked.iloc[-1]

        _geo_high_rate = _geo_high["Churn_Rate"]
        _geo_low_rate = _geo_low["Churn_Rate"]

        _geo_gap = _geo_high_rate - _geo_low_rate

        _geo_high_customers = int(_geo_high["Customers"])
        _geo_high_churned = int(_geo_high["Churned"])


        # ---------------------------------------------------------
        # 4. Geographic Risk Index
        # ---------------------------------------------------------
        _geo_high_risk_index = (
            _geo_high_rate / overall_rate
            if overall_rate > 0
            else 0
        )

        _geo_low_risk_index = (
            _geo_low_rate / overall_rate
            if overall_rate > 0
            else 0
        )

        _geo_high_vs_overall = (
            _geo_high_rate - overall_rate
        )


        # ---------------------------------------------------------
        # 5. Geographic concentration of churn
        # ---------------------------------------------------------
        _total_churned = int(df["Exited"].sum())

        _geo_high_churn_share = (
            (_geo_high_churned / _total_churned) * 100
            if _total_churned > 0
            else 0
        )


        # ---------------------------------------------------------
        # 6. Dynamic risk interpretation
        # ---------------------------------------------------------
        if _geo_high_vs_overall > 5:

            _geo_risk_text = (
                f"{_geo_high['Geography']} is materially above the "
                f"portfolio-wide churn baseline by "
                f"<b>{_geo_high_vs_overall:.1f} percentage points</b>."
            )

        elif _geo_high_vs_overall > 0:

            _geo_risk_text = (
                f"{_geo_high['Geography']} is above the portfolio-wide "
                f"churn baseline by "
                f"<b>{_geo_high_vs_overall:.1f} percentage points</b>, "
                f"indicating higher observed regional churn exposure."
            )

        elif _geo_high_vs_overall < 0:

            _geo_risk_text = (
                f"Even the highest-churn geography remains "
                f"<b>{abs(_geo_high_vs_overall):.1f} percentage points</b> "
                f"below the portfolio-wide churn rate."
            )

        else:

            _geo_risk_text = (
                f"The highest-churn geography is aligned with the "
                f"portfolio-wide churn baseline."
            )


        # ---------------------------------------------------------
        # 7. Geographic spread interpretation
        # ---------------------------------------------------------
        if _geo_gap >= 10:

            _geo_spread_text = (
                f"The <b>{_geo_gap:.1f}-percentage-point</b> spread between "
                f"the highest- and lowest-churn geographies represents a "
                f"substantial geographic difference in observed churn."
            )

        elif _geo_gap >= 5:

            _geo_spread_text = (
                f"The <b>{_geo_gap:.1f}-percentage-point</b> spread indicates "
                f"a meaningful difference in observed churn across "
                f"geographies."
            )

        else:

            _geo_spread_text = (
                f"The geographic spread is relatively limited at "
                f"<b>{_geo_gap:.1f} percentage points</b>, indicating that "
                f"overall churn performance is comparatively closer across "
                f"the reviewed geographies."
            )


        # ---------------------------------------------------------
        # 8. Geography × Age interaction
        # ---------------------------------------------------------
        _heat_data_summary = (
            df.groupby(
                ["Geography", "AgeGroup"],
                observed=True
            )["Exited"]
            .mean()
            .mul(100)
            .reset_index(name="Churn_Rate")
        )

        if len(_heat_data_summary) > 0:

            _hot_cell = _heat_data_summary.loc[
                _heat_data_summary["Churn_Rate"].idxmax()
            ]

            _cold_cell = _heat_data_summary.loc[
                _heat_data_summary["Churn_Rate"].idxmin()
            ]

            _interaction_text = (
                f"The Geography × Age analysis further identifies "
                f"<b>{_hot_cell['Geography']} / "
                f"{_hot_cell['AgeGroup']}</b> as the highest observed "
                f"combined-risk cell, with a churn rate of "
                f"<b>{_hot_cell['Churn_Rate']:.1f}%</b>. "
                f"The lowest observed Geography × Age cell is "
                f"<b>{_cold_cell['Geography']} / "
                f"{_cold_cell['AgeGroup']}</b> at "
                f"<b>{_cold_cell['Churn_Rate']:.1f}%</b>."
            )

        else:

            _interaction_text = (
                "The Geography × Age interaction cannot be reliably "
                "evaluated because sufficient combined geography and "
                "age-group data is not available under the current filter."
            )


    # ---------------------------------------------------------
    # 9. Build the single consolidated finding
        # ---------------------------------------------------------
    finding_box(
        "02",
        "Geography-Wise Churn Comparison — Overall Finding",

        (
            f"<ol>"
            
            f"<li><b>Regional churn position:</b> "
            f"<b>{_geo_high['Geography']}</b> records the highest "
            f"observed churn rate at <b>{_geo_high_rate:.1f}%</b>, "
            f"while <b>{_geo_low['Geography']}</b> records the lowest "
            f"at <b>{_geo_low_rate:.1f}%</b>. "
            f"The difference between these two geographies is "
            f"<b>{_geo_gap:.1f} percentage points</b>.</li>"

            f"<li><b>Highest-risk geography:</b> "
            f"{_geo_high['Geography']} has "
            f"<b>{_geo_high_customers:,}</b> customers in the current "
            f"selection, of whom <b>{_geo_high_churned:,}</b> have "
            f"churned. Its Geographic Risk Index is "
            f"<b>{_geo_high_risk_index:.2f}×</b> the portfolio average "
            f"of <b>{overall_rate:.1f}%</b>. "
            f"{_geo_risk_text}</li>"

            f"<li><b>Geographic spread:</b> "
            f"{_geo_spread_text} "
            f"This means geography should be considered an important "
            f"descriptive dimension when prioritising retention analysis, "
            f"although the result does not by itself establish that "
            f"geography causes customer churn.</li>"

            f"<li><b>Churn concentration:</b> "
            f"The highest-churn geography contributes "
            f"<b>{_geo_high_churn_share:.1f}%</b> of all observed "
            f"churned customers in the current filtered population. "
            f"This distinguishes the <b>churn rate</b> within a geography "
            f"from the <b>volume of churn</b> coming from that geography.</li>"

            f"<li><b>Geography × Age perspective:</b> "
            f"{_interaction_text} "
            f"This interaction view is useful because a geography with "
            f"high overall churn may contain specific age groups where "
            f"the observed churn concentration is even stronger.</li>"

            f"</ol>"
        ),

        # (
        #     f"<b>Overall interpretation:</b> "
        #     f"The geographic analysis shows that customer churn is not "
        #     f"uniform across the reviewed regions. "
        #     f"<b>{_geo_high['Geography']}</b> has the highest observed "
        #     f"regional churn rate at <b>{_geo_high_rate:.1f}%</b>, "
        #     f"compared with <b>{_geo_low_rate:.1f}%</b> in "
        #     f"<b>{_geo_low['Geography']}</b>. "
        #     f"The resulting <b>{_geo_gap:.1f}-percentage-point</b> "
        #     f"spread provides a clear descriptive measure of regional "
        #     f"difference. "
        #     f"{_geo_risk_text} "
        #     f"The Geography × Age analysis adds another layer of detail "
        #     f"by showing whether the highest observed churn is concentrated "
        #     f"within a particular age group rather than being evenly "
        #     f"distributed across the geography."
        # ),

        # (
        #     f"<b>Recommended action:</b> "
        #     f"Use <b>{_geo_high['Geography']}</b> as the first geography "
        #     f"for deeper retention investigation, while also monitoring "
        #     f"the lowest-churn geography as a useful internal comparison "
        #     f"point. "
        #     f"Within the highest-risk region, review the "
        #     f"Geography × Age combination identified above and then "
        #     f"cross-check customer engagement, product holding, balance, "
        #     f"tenure and other churn indicators before deciding on "
        #     f"retention actions. "
        #     f"Regional churn should therefore be treated as a "
        #     f"<b>prioritisation signal</b>, not as evidence that geography "
        #     f"itself is the cause of churn."
        # ),
    )









# ---------------------------------------------------------
# TAB 3 — SEGMENTATION EXPLORER
# ---------------------------------------------------------
with tab3:
    st.markdown(
        "<h3 class='section-header'>Segment Churn Contribution</h3>",
        unsafe_allow_html=True,
    )
    st.caption(
        "Churn Rate = risk WITHIN a segment. Churn Contribution = share of TOTAL churned customers coming FROM that segment."
    )

    dim_choice = st.selectbox(
        "Choose a segmentation dimension to drill into:", options=SEGMENT_DIMS
    )
    total_churned_all = safe_get_value(df, "Exited", "sum")

    seg_table = (
        df.groupby(dim_choice, observed=True)
        .agg(Customers=("Exited", "count"), Churned=("Exited", "sum"))
        .reset_index()
    )
    seg_table["Churn_Rate_%"] = (
        seg_table["Churned"] / seg_table["Customers"] * 100
    ).round(1)
    seg_table["Churn_Contribution_%"] = (
        (seg_table["Churned"] / total_churned_all * 100).round(1)
        if total_churned_all > 0
        else 0
    )
    seg_table = seg_table.sort_values("Churn_Rate_%", ascending=False)

    fig9 = px.bar(
        seg_table,
        x=dim_choice,
        y="Churn_Rate_%",
        text="Churn_Rate_%",
        color_discrete_sequence=["#22303C"],
        title="Churn Rate by Segment",
    )
    fig9.update_traces(texttemplate="%{text}%", textposition="outside")
    _rate_sorted = seg_table.sort_values("Churn_Rate_%", ascending=False)
    chart_row(
        fig9,
        f"{dim_choice} \u2014 Churn Rate",
        [
            (
                "Highest rate",
                f"{_rate_sorted.iloc[0][dim_choice]} \u2014 {_rate_sorted.iloc[0]['Churn_Rate_%']:.1f}%",
            ),
            (
                "Lowest rate",
                f"{_rate_sorted.iloc[-1][dim_choice]} \u2014 {_rate_sorted.iloc[-1]['Churn_Rate_%']:.1f}%",
            ),
            ("Segments shown", f"{len(seg_table)}"),
            ("Portfolio average", f"{overall_rate:.1f}%"),
        ],
        f"Within the {dim_choice} segmentation, <b>{_rate_sorted.iloc[0][dim_choice]}</b> reflects the highest observed attrition rate at {_rate_sorted.iloc[0]['Churn_Rate_%']:.1f}%, "
        f"relative to the {overall_rate:.1f}% portfolio-wide baseline.",
    )

    fig10 = px.bar(
        seg_table,
        x=dim_choice,
        y="Churn_Contribution_%",
        text="Churn_Contribution_%",
        color_discrete_sequence=["#B08D57"],
        title="Share of Total Churn Volume",
    )
    fig10.update_traces(texttemplate="%{text}%", textposition="outside")
    _contrib_sorted = seg_table.sort_values("Churn_Contribution_%", ascending=False)
    chart_row(
        fig10,
        f"{dim_choice} \u2014 Churn Contribution",
        [
            (
                "Largest contributor",
                f"{_contrib_sorted.iloc[0][dim_choice]} \u2014 {_contrib_sorted.iloc[0]['Churn_Contribution_%']:.1f}%",
            ),
            (
                "Smallest contributor",
                f"{_contrib_sorted.iloc[-1][dim_choice]} \u2014 {_contrib_sorted.iloc[-1]['Churn_Contribution_%']:.1f}%",
            ),
            ("Segments shown", f"{len(seg_table)}"),
            ("Total churned (all segments)", f"{int(total_churned_all):,}"),
        ],
        f"<b>{_contrib_sorted.iloc[0][dim_choice]}</b> accounts for the largest proportion of total churn volume, representing {_contrib_sorted.iloc[0]['Churn_Contribution_%']:.1f}% of all customers who exited within the current selection.",
    )

    st.markdown("**Segment Detail Table**")
    st.dataframe(
        seg_table.rename(
            columns={
                "Customers": "Total Customers",
                "Churned": "Churned Customers",
                "Churn_Rate_%": "Churn Rate (%)",
                "Churn_Contribution_%": "Contribution to Total Churn (%)",
            }
        ),
        hide_index=True,
        use_container_width=True,
    )

    _top_rate_row = (
        seg_table.sort_values("Churn_Rate_%", ascending=False).iloc[0]
        if len(seg_table) > 0
        else None
    )
    _top_contrib_row = (
        seg_table.sort_values("Churn_Contribution_%", ascending=False).iloc[0]
        if len(seg_table) > 0
        else None
    )
    if _top_rate_row is not None and _top_contrib_row is not None:
        _same_segment = _top_rate_row[dim_choice] == _top_contrib_row[dim_choice]
        _rate_vs_overall = (
            (_top_rate_row["Churn_Rate_%"] / overall_rate) if overall_rate > 0 else 0
        )
        if _same_segment:
            print("gg")
            # finding_box(
            #     "06",
            #     f"Risk and Volume Align for {dim_choice}",
            #     f"Within {dim_choice}, the segment '{_top_rate_row[dim_choice]}' has both the highest churn rate "
            #     f"({_top_rate_row['Churn_Rate_%']:.1f}%) and the highest share of total churn volume "
            #     f"({_top_contrib_row['Churn_Contribution_%']:.1f}%).",
            #     (
            #         f"When the highest-risk segment and the highest-volume segment are the same group, the priority "
            #         f"call is unambiguous. This segment's {int(_top_rate_row['Customers']):,} customers churn at "
            #         f"{_rate_vs_overall:.1f}\u00d7 the {overall_rate:.1f}% portfolio average, making it simultaneously "
            #         f"the most dangerous segment per customer and the most consequential in aggregate — a "
            #         f"{magnitude_word((_rate_vs_overall - 1) * 100, thresholds=(20, 50, 100))} concentration of risk "
            #         f"in a single, clearly identifiable group."
            #     ),
            #     (
            #         f"Make '{_top_rate_row[dim_choice]}' the first target for a {dim_choice.lower()}-specific retention "
            #         f"initiative covering its {int(_top_rate_row['Customers']):,} customers — the business case is "
            #         f"strong on both risk and volume grounds. This {urgency_clause((_rate_vs_overall - 1) * 100, thresholds=(20, 50, 100))}."
            #     ),
            # )
        else:
            _size_ratio = (
                (_top_contrib_row["Customers"] / _top_rate_row["Customers"])
                if _top_rate_row["Customers"] > 0
                else 0
            )
            # finding_box(
            #     "06",
            #     f"Risk and Volume Diverge for {dim_choice}",
            #     f"Within {dim_choice}, '{_top_rate_row[dim_choice]}' has the highest churn RATE ({_top_rate_row['Churn_Rate_%']:.1f}%), "
            #     f"but '{_top_contrib_row[dim_choice]}' contributes the largest SHARE of total churned customers "
            #     f"({_top_contrib_row['Churn_Contribution_%']:.1f}%).",
            #     (
            #         f"This is the classic risk-versus-volume trade-off. '{_top_rate_row[dim_choice]}' carries "
            #         f"{int(_top_rate_row['Customers']):,} customers at {_rate_vs_overall:.1f}\u00d7 the portfolio "
            #         f"average rate, while '{_top_contrib_row[dim_choice]}' is roughly {_size_ratio:.1f}\u00d7 larger "
            #         f"at {int(_top_contrib_row['Customers']):,} customers — its more moderate rate still produces the "
            #         f"larger absolute count of lost customers simply because of its size."
            #     ),
            #     (
            #         f"Address '{_top_contrib_row[dim_choice]}' first if the goal is to reduce total churned customers "
            #         f"across the portfolio, or '{_top_rate_row[dim_choice]}' first if the goal is to fix the segment "
            #         f"with the worst per-customer retention outcome. Given the {_size_ratio:.1f}\u00d7 size difference "
            #         f"between the two, a volume-first approach {'is likely to move the headline churn count faster' if _size_ratio >= 1.5 else 'offers only a modest speed advantage over a risk-first approach'}."
            #     ),
            # )

    st.markdown(
        "<h3 class='section-header'>Filtered Customer Records</h3>",
        unsafe_allow_html=True,
    )
    page_size = 25
    total_rows = len(df)
    total_pages = max(1, (total_rows - 1) // page_size + 1)
    page_num = st.number_input(
        "Page", min_value=1, max_value=total_pages, value=1, step=1
    )
    start_idx = (page_num - 1) * page_size
    end_idx = start_idx + page_size

    display_cols = [
        "CustomerId",
        "Geography",
        "Gender",
        "Age",
        "CreditBand",
        "TenureGroup",
        "BalanceSegment",
        "ActivityStatus",
        "ChurnStatus",
    ]
    st.dataframe(
        df[display_cols].iloc[start_idx:end_idx],
        hide_index=True,
        use_container_width=True,
    )
    st.caption(
        f"Showing rows {start_idx + 1}-{min(end_idx, total_rows)} of {total_rows:,}"
    )

    csv_data = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download Filtered Data (CSV)",
        data=csv_data,
        file_name="filtered_customer_segments.csv",
        mime="text/csv",
    )








        # =========================================================
    # SEGMENT EXPLORER — SINGLE FULLY DYNAMIC EXECUTIVE FINDING
    # =========================================================

    if len(seg_table) > 0 and total_churned_all > 0:

        # -----------------------------------------------------
        # 1. Highest churn-rate segment
        # -----------------------------------------------------

        _rate_sorted = seg_table.sort_values(
            "Churn_Rate_%",
            ascending=False
        )

        _top_rate_row = _rate_sorted.iloc[0]

        _risk_segment = _top_rate_row[dim_choice]
        _risk_rate = float(_top_rate_row["Churn_Rate_%"])


        # -----------------------------------------------------
        # 2. Lowest churn-rate segment
        # -----------------------------------------------------

        _bottom_rate_row = _rate_sorted.iloc[-1]

        _lowest_segment = _bottom_rate_row[dim_choice]
        _lowest_rate = float(_bottom_rate_row["Churn_Rate_%"])


        # -----------------------------------------------------
        # 3. Churn-rate spread
        # -----------------------------------------------------

        _risk_spread = _risk_rate - _lowest_rate


        # -----------------------------------------------------
        # 4. Largest churn-volume contributor
        # -----------------------------------------------------

        _contrib_sorted = seg_table.sort_values(
            "Churn_Contribution_%",
            ascending=False
        )

        _top_contrib_row = _contrib_sorted.iloc[0]

        _volume_segment = _top_contrib_row[dim_choice]

        _volume_share = float(
            _top_contrib_row["Churn_Contribution_%"]
        )


        # -----------------------------------------------------
        # 5. Risk and volume alignment
        # -----------------------------------------------------

        _same_segment = (
            _risk_segment == _volume_segment
        )


        # -----------------------------------------------------
        # 6. Dynamic interpretation of risk spread
        # -----------------------------------------------------

        if _risk_spread >= 15:

            _spread_interpretation = (
                "a substantial separation in observed churn risk "
                "across the segments"
            )

        elif _risk_spread >= 10:

            _spread_interpretation = (
                "a pronounced separation in observed churn risk "
                "across the segments"
            )

        elif _risk_spread >= 5:

            _spread_interpretation = (
                "a meaningful separation in observed churn risk "
                "across the segments"
            )

        elif _risk_spread > 0:

            _spread_interpretation = (
                "a relatively narrow separation in observed churn "
                "risk across the segments"
            )

        else:

            _spread_interpretation = (
                "no observed separation in churn risk across "
                "the segments"
            )


        # -----------------------------------------------------
        # 7. Dynamic risk-volume interpretation
        # -----------------------------------------------------

        if _same_segment:

            _alignment_finding = (
                f"Risk and churn volume are concentrated in the same "
                f"<b>{dim_choice}</b> segment, <b>{_risk_segment}</b>. "
                f"This segment has the highest observed churn rate and "
                f"also contributes the largest share of churned customers "
                f"at <b>{_volume_share:.1f}%</b>."
            )

        else:

            _alignment_finding = (
                f"The highest observed churn risk and the largest churn "
                f"volume occur in different <b>{dim_choice}</b> segments. "
                f"<b>{_risk_segment}</b> has the highest churn rate at "
                f"<b>{_risk_rate:.1f}%</b>, whereas <b>{_volume_segment}</b> "
                f"contributes the largest share of churned customers at "
                f"<b>{_volume_share:.1f}%</b>."
            )


        # -----------------------------------------------------
        # 8. Final single executive finding
        # -----------------------------------------------------

        segment_explorer_finding = (
            f"Across the current <b>{dim_choice}</b> segmentation, "
            f"<b>{_risk_segment}</b> records the highest observed churn "
            f"rate at <b>{_risk_rate:.1f}%</b>, while "
            f"<b>{_lowest_segment}</b> records the lowest at "
            f"<b>{_lowest_rate:.1f}%</b> — a difference of "
            f"<b>{_risk_spread:.1f} percentage points</b>, indicating "
            f"{_spread_interpretation}. "
            f"{_alignment_finding} "
            f"These results provide a descriptive view of where churn "
            f"risk and churn volume are concentrated within the current "
            f"customer selection and can be used to prioritise further "
            f"segment-level retention analysis."
        )

    else:

        segment_explorer_finding = (
            "Insufficient data is available to produce a segment-level "
            "churn finding under the current filter selection."
        )



    finding_box1(1,"Findings",segment_explorer_finding)
 









# ---------------------------------------------------------
# TAB 4 — HIGH-VALUE CUSTOMER REVIEW
# ---------------------------------------------------------
with tab4:
    st.markdown(
        "<h3 class='section-header'>High-Value Customer Churn Explorer</h3>",
        unsafe_allow_html=True,
    )
    st.caption(
        "High-Value = top 25% by Balance OR top 25% by Estimated Salary (computed on the full portfolio, before filtering)."
    )

    hv_churn_rate = hv_rate
    std_churn_rate = (
        safe_get_value(df[df["HighValueCustomer"] == "Standard"], "Exited", "mean")
        * 100
        if "Standard" in df["HighValueCustomer"].values
        else 0
    )
    revenue_at_risk = safe_get_value(
        df[(df["HighValueCustomer"] == "High-Value") & (df["Exited"] == 1)],
        "Balance",
        "sum",
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        kpi_card("High-Value Churn Ratio", f"{hv_churn_rate:.1f}", "%")
    with c2:
        kpi_card("Standard Customer Churn Rate", f"{std_churn_rate:.1f}", "%")
    with c3:
        kpi_card("Revenue at Risk (Balance)", f"\u20ac{revenue_at_risk:,.0f}")

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("**Salary vs Balance — Churn Quadrant**")
    sample_df = df.sample(min(2000, len(df)), random_state=42)
    fig11 = px.scatter(
        sample_df,
        x="EstimatedSalary",
        y="Balance",
        color="ChurnStatus",
        color_discrete_map={"Retained": "#22303C", "Churned": "#7A1F2B"},
        opacity=0.6,
    )
    _sal_churned_avg = (
        sample_df[sample_df["ChurnStatus"] == "Churned"]["EstimatedSalary"].mean()
        if (sample_df["ChurnStatus"] == "Churned").any()
        else 0
    )
    _bal_churned_avg2 = (
        sample_df[sample_df["ChurnStatus"] == "Churned"]["Balance"].mean()
        if (sample_df["ChurnStatus"] == "Churned").any()
        else 0
    )
    _sal_retained_avg = (
        sample_df[sample_df["ChurnStatus"] == "Retained"]["EstimatedSalary"].mean()
        if (sample_df["ChurnStatus"] == "Retained").any()
        else 0
    )
    _bal_retained_avg = (
        sample_df[sample_df["ChurnStatus"] == "Retained"]["Balance"].mean()
        if (sample_df["ChurnStatus"] == "Retained").any()
        else 0
    )
    _sal_sep_pct = (
        (abs(_sal_churned_avg - _sal_retained_avg) / _sal_retained_avg * 100)
        if _sal_retained_avg > 0
        else 0
    )
    _bal_sep_pct = (
        (abs(_bal_churned_avg2 - _bal_retained_avg) / _bal_retained_avg * 100)
        if _bal_retained_avg > 0
        else 0
    )
    _dominant_field = "Estimated Salary" if _sal_sep_pct > _bal_sep_pct else "Balance"
    _dominant_sep = max(_sal_sep_pct, _bal_sep_pct)
    _weaker_field = (
        "Balance" if _dominant_field == "Estimated Salary" else "Estimated Salary"
    )
    _weaker_sep = min(_sal_sep_pct, _bal_sep_pct)
    chart_row(
        fig11,
        "Quadrant Breakdown",
        [
            ("Customers plotted", f"{len(sample_df):,} (sampled)"),
            ("Avg salary \u2014 Churned", f"\u20ac{_sal_churned_avg:,.0f}"),
            ("Avg balance \u2014 Churned", f"\u20ac{_bal_churned_avg2:,.0f}"),
            (f"{_dominant_field} separation", f"{_dominant_sep:.1f}%"),
        ],
        (
            f"<b>{_dominant_field}</b> provides the larger relative separation between churned and retained customers, "
            f"at {_dominant_sep:.1f}% versus {_weaker_sep:.1f}% for {_weaker_field}, "
            f"{'though a material degree of overlap persists across both dimensions' if _dominant_sep < 15 else 'representing a meaningfully clearer split, though overlap remains present'}. "
            f"This underscores the rationale for the multivariate predictive model presented in Tab VIII, which "
            f"considers all ten input variables in combination rather than relying on {_dominant_field.lower()} alone."
        ),
    )

    st.markdown("**High-Value vs Standard Churn Comparison**")
    hv_df = (
        df.groupby("HighValueCustomer", observed=True)["Exited"]
        .mean()
        .mul(100)
        .round(1)
        .reset_index()
    )
    hv_df.columns = ["Customer Tier", "Churn Rate (%)"]
    fig12 = px.bar(
        hv_df,
        x="Customer Tier",
        y="Churn Rate (%)",
        text="Churn Rate (%)",
        color="Customer Tier",
        color_discrete_map={"High-Value": "#7A1F2B", "Standard": "#22303C"},
    )
    fig12.update_traces(texttemplate="%{text}%", textposition="outside")
    chart_row(
        fig12,
        "Tier Breakdown",
        rate_breakdown_stats(
            hv_df, "Customer Tier", "Churn Rate (%)", overall=overall_rate
        ),
        rate_insight_text(
            hv_df, "Customer Tier", "Churn Rate (%)", overall=overall_rate
        ),
    )

    st.markdown("**High-Value Churned Customers**")
    hv_churned = df[
        (df["HighValueCustomer"] == "High-Value") & (df["Exited"] == 1)
    ].sort_values("Balance", ascending=False)
    hv_display_cols = [
        "CustomerId",
        "Geography",
        "Age",
        "Balance",
        "EstimatedSalary",
        "NumOfProducts",
        "TenureGroup",
        "ActivityStatus",
    ]
    st.dataframe(hv_churned[hv_display_cols], hide_index=True, use_container_width=True)

    hv_csv = hv_churned.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download High-Value Churned Customers (CSV)",
        data=hv_csv,
        file_name="high_value_churned_customers.csv",
        mime="text/csv",
    )

    _hv_gap = hv_churn_rate - std_churn_rate
    _hv_direction = "higher" if _hv_gap > 0 else "lower"
    _avg_loss_per_hv_churn = (
        (revenue_at_risk / len(hv_churned)) if len(hv_churned) > 0 else 0
    )
    _portfolio_avg_balance = df["Balance"].mean() if len(df) > 0 else 0
    _balance_multiple = (
        (_avg_loss_per_hv_churn / _portfolio_avg_balance)
        if _portfolio_avg_balance > 0
        else 0
    )
    # finding_box(
    #     "07",
    #     "Revenue Exposure From Premium Customer Churn",
    #     f"High-value customers churn at {hv_churn_rate:.1f}%, {abs(_hv_gap):.1f} percentage points {_hv_direction} than "
    #     f"standard customers ({std_churn_rate:.1f}%), representing \u20ac{revenue_at_risk:,.0f} in account balance "
    #     f"already lost among the {len(hv_churned):,} high-value customers who have churned.",
    #     (
    #         f"Because premium customers are defined by top-quartile balance or salary, each individual departure carries "
    #         f"a {magnitude_word(_hv_gap, thresholds=(2, 5, 10))} incremental revenue impact relative to an average "
    #         f"customer loss — this tier's churn translates into an average of \u20ac{_avg_loss_per_hv_churn:,.0f} lost per "
    #         f"departing customer, {_balance_multiple:.1f}\u00d7 the portfolio's average account balance of "
    #         f"\u20ac{_portfolio_avg_balance:,.0f}."
    #     ),
    #     (
    #         f"{'Prioritise' if abs(_hv_gap) >= 2 else 'Maintain'} a dedicated relationship-manager outreach track for "
    #         f"high-value customers {'rather than folding them into the bank general retention campaign' if _hv_direction == 'higher' else 'to preserve the currently favourable retention performance already observed in this tier'}, "
    #         f"given the \u20ac{revenue_at_risk:,.0f} revenue concentration identified above. This {urgency_clause(_hv_gap, thresholds=(2, 5, 10))}."
    #     ),
    # )

    
    # =========================================================
# CONSOLIDATED FINDING — ENTIRE HIGH-VALUE CUSTOMER TAB
# =========================================================

    # ---------------------------------------------------------
    # 1. High-Value vs Standard customer statistics
    # ---------------------------------------------------------
    _hv_population = df[df["HighValueCustomer"] == "High-Value"]
    _std_population = df[df["HighValueCustomer"] == "Standard"]

    _hv_customers = len(_hv_population)
    _std_customers = len(_std_population)

    _hv_churned_count = int(_hv_population["Exited"].sum())
    _std_churned_count = int(_std_population["Exited"].sum())

    _hv_churn_rate_final = (
        (_hv_churned_count / _hv_customers) * 100
        if _hv_customers > 0
        else 0
    )

    _std_churn_rate_final = (
        (_std_churned_count / _std_customers) * 100
        if _std_customers > 0
        else 0
    )


    # ---------------------------------------------------------
    # 2. High-Value vs Standard churn gap
    # ---------------------------------------------------------
    _hv_gap_final = (
        _hv_churn_rate_final - _std_churn_rate_final
    )

    if _hv_gap_final > 0:
        _hv_direction_final = "higher"
    elif _hv_gap_final < 0:
        _hv_direction_final = "lower"
    else:
        _hv_direction_final = "the same as"


    # ---------------------------------------------------------
    # 3. Revenue / balance exposure
    # ---------------------------------------------------------
    _hv_balance_exposure = (
        _hv_population.loc[
            _hv_population["Exited"] == 1,
            "Balance"
        ].sum()
        if _hv_customers > 0
        else 0
    )

    _avg_balance_per_hv_churn = (
        _hv_balance_exposure / _hv_churned_count
        if _hv_churned_count > 0
        else 0
    )


    # ---------------------------------------------------------
    # 4. Portfolio average balance
    # ---------------------------------------------------------
    _portfolio_avg_balance_final = (
        df["Balance"].mean()
        if len(df) > 0
        else 0
    )

    _balance_multiple_final = (
        _avg_balance_per_hv_churn / _portfolio_avg_balance_final
        if _portfolio_avg_balance_final > 0
        else 0
    )


    # ---------------------------------------------------------
    # 5. Share of total churn represented by High-Value
    # ---------------------------------------------------------
    _total_churned_final = int(df["Exited"].sum())

    _hv_churn_share_final = (
        (_hv_churned_count / _total_churned_final) * 100
        if _total_churned_final > 0
        else 0
    )


    # ---------------------------------------------------------
    # 6. Salary vs Balance separation
    # ---------------------------------------------------------
    _hv_churned_df = _hv_population[
        _hv_population["Exited"] == 1
    ]

    _hv_retained_df = _hv_population[
        _hv_population["Exited"] == 0
    ]

    _hv_churned_salary = (
        _hv_churned_df["EstimatedSalary"].mean()
        if len(_hv_churned_df) > 0
        else 0
    )

    _hv_retained_salary = (
        _hv_retained_df["EstimatedSalary"].mean()
        if len(_hv_retained_df) > 0
        else 0
    )

    _hv_churned_balance = (
        _hv_churned_df["Balance"].mean()
        if len(_hv_churned_df) > 0
        else 0
    )

    _hv_retained_balance = (
        _hv_retained_df["Balance"].mean()
        if len(_hv_retained_df) > 0
        else 0
    )


    # ---------------------------------------------------------
    # 7. Relative separation
    # ---------------------------------------------------------
    _salary_separation_hv = (
        abs(_hv_churned_salary - _hv_retained_salary)
        / _hv_retained_salary * 100
        if _hv_retained_salary > 0
        else 0
    )

    _balance_separation_hv = (
        abs(_hv_churned_balance - _hv_retained_balance)
        / _hv_retained_balance * 100
        if _hv_retained_balance > 0
        else 0
    )

    _dominant_financial_dimension = (
        "Estimated Salary"
        if _salary_separation_hv > _balance_separation_hv
        else "Balance"
    )

    _dominant_financial_separation = max(
        _salary_separation_hv,
        _balance_separation_hv
    )


    # ---------------------------------------------------------
    # 8. Dynamic risk interpretation
    # ---------------------------------------------------------
    if _hv_gap_final >= 10:

        _hv_risk_interpretation = (
            f"High-value customers show a substantially higher observed "
            f"churn rate than standard customers, with a difference of "
            f"<b>{_hv_gap_final:.1f} percentage points</b>. "
            f"This represents a clear retention priority because the "
            f"higher churn rate is occurring within the financially "
            f"important customer tier."
        )

    elif _hv_gap_final >= 5:

        _hv_risk_interpretation = (
            f"High-value customers show a meaningfully higher observed "
            f"churn rate than standard customers, with a difference of "
            f"<b>{_hv_gap_final:.1f} percentage points</b>. "
            f"The pattern warrants focused retention monitoring for this tier."
        )

    elif _hv_gap_final > 0:

        _hv_risk_interpretation = (
            f"High-value customers churn at a higher observed rate than "
            f"standard customers by <b>{_hv_gap_final:.1f} percentage points</b>. "
            f"The difference is positive but should be interpreted together "
            f"with the financial exposure associated with the departing customers."
        )

    elif _hv_gap_final < 0:

        _hv_risk_interpretation = (
            f"High-value customers actually show a lower observed churn rate "
            f"than standard customers by <b>{abs(_hv_gap_final):.1f} percentage points</b>. "
            f"This indicates that the premium tier is currently performing "
            f"better on retention, although its churned customers can still "
            f"represent meaningful balance exposure."
        )

    else:

        _hv_risk_interpretation = (
            "High-value and standard customers show the same observed churn "
            "rate in the current selection. Therefore, the main distinction "
            "between the two groups comes from the financial value attached "
            "to each departing customer rather than from churn-rate differences."
        )


    # ---------------------------------------------------------
    # 9. Financial exposure interpretation
    # ---------------------------------------------------------
    if _hv_balance_exposure > 0:

        _financial_exposure_text = (
            f"The <b>€{_hv_balance_exposure:,.0f}</b> in balance associated "
            f"with churned high-value customers represents the principal "
            f"financial exposure identified by this tab. On average, each "
            f"high-value churned customer represents approximately "
            f"<b>€{_avg_balance_per_hv_churn:,.0f}</b> in account balance, "
            f"equivalent to <b>{_balance_multiple_final:.1f}×</b> the "
            f"portfolio's average customer balance."
        )

    else:

        _financial_exposure_text = (
            "No balance exposure from churned high-value customers is "
            "identified under the current filter selection."
        )


    # ---------------------------------------------------------
    # 10. Salary / Balance interpretation
    # ---------------------------------------------------------
    if _dominant_financial_separation > 0:

        _financial_dimension_text = (
            f"Within the High-Value population, <b>{_dominant_financial_dimension}</b> "
            f"shows the larger relative difference between churned and retained "
            f"customers, with approximately "
            f"<b>{_dominant_financial_separation:.1f}%</b> separation. "
            f"However, this relationship should be treated as a descriptive "
            f"pattern rather than evidence that salary or balance directly "
            f"causes churn."
        )

    else:

        _financial_dimension_text = (
            "Salary and balance do not show meaningful separation between "
            "churned and retained high-value customers in the current selection."
        )


    # ---------------------------------------------------------
    # 11. Final consolidated Finding
    # ---------------------------------------------------------
    finding_box(
        "08",
        "High-Value Customer Churn — Overall Finding",

        (
            f"<ol>"

            f"<li><b>Premium customer retention position:</b> "
            f"High-value customers record an observed churn rate of "
            f"<b>{_hv_churn_rate_final:.1f}%</b>, compared with "
            f"<b>{_std_churn_rate_final:.1f}%</b> among standard customers. "
            f"Therefore, high-value customer churn is "
            f"<b>{abs(_hv_gap_final):.1f} percentage points "
            f"{_hv_direction_final}</b> than the standard-customer rate."
            f"</li>"

            f"<li><b>Financial exposure:</b> "
            f"{_financial_exposure_text}"
            f"</li>"

            f"<li><b>Contribution to overall churn:</b> "
            f"High-value customers account for "
            f"<b>{_hv_churn_share_final:.1f}%</b> of all observed churned "
            f"customers in the current filtered population. "
            f"This shows the importance of evaluating both the number of "
            f"high-value customers leaving and the financial value associated "
            f"with those departures."
            f"</li>"

            f"<li><b>Salary and balance profile:</b> "
            f"{_financial_dimension_text}"
            f"</li>"

            f"<li><b>Overall business implication:</b> "
            f"{_hv_risk_interpretation} "
            f"Because high-value customers are specifically defined using "
            f"balance or salary thresholds, churn within this group deserves "
            f"greater financial attention than a simple customer-count "
            f"analysis would provide."
            f"</li>"

            f"</ol>"
        ),

        # (
        #     f"<b>Overall interpretation:</b> "
        #     f"The High-Value Customer Churn Explorer shows that the "
        #     f"retention problem should be evaluated not only by "
        #     f"<b>how many customers are leaving</b>, but also by "
        #     f"<b>which customers are leaving and what financial value "
        #     f"is associated with them</b>. "
        #     f"High-value customers churn at <b>{_hv_churn_rate_final:.1f}%</b>, "
        #     f"while standard customers churn at "
        #     f"<b>{_std_churn_rate_final:.1f}%</b>. "
        #     f"{_hv_risk_interpretation} "
        #     f"{_financial_exposure_text} "
        #     f"Together, these results make high-value customer churn an "
        #     f"important relationship-management and retention consideration."
        # ),

        # (
        #     f"<b>Recommended action:</b> "
        #     f"Use the high-value churn rate and the "
        #     f"<b>€{_hv_balance_exposure:,.0f}</b> balance exposure as the "
        #     f"primary signals for prioritising premium-customer retention. "
        #     f"Customers identified as high-value and showing additional "
        #     f"risk indicators should receive more focused retention review "
        #     f"than the general customer population. "
        #     f"At the same time, use the Salary × Balance analysis and the "
        #     f"customer-level churned list to identify common characteristics "
        #     f"within departing high-value customers. "
        #     f"The observed patterns should be validated against engagement, "
        #     f"product usage, geography, age and tenure before attributing "
        #     f"churn to any single financial characteristic."
        # ),
    )






# ---------------------------------------------------------
# TAB 5 — AGE & TENURE CHURN COMPARISON
# ---------------------------------------------------------
with tab5:

    st.markdown(
        "<h3 class='section-header'>Age & Tenure Churn Comparison</h3>",
        unsafe_allow_html=True,
    )

    age_churn = (
        df.groupby("AgeGroup", observed=True)["Exited"]
        .mean()
        .mul(100)
        .round(1)
        .reset_index()
    )
    age_churn.columns = ["Age Group", "Churn Rate (%)"]
    fig5 = px.bar(
        age_churn,
        x="Age Group",
        y="Churn Rate (%)",
        text="Churn Rate (%)",
        color_discrete_sequence=["#22303C"],
    )
    fig5.update_traces(texttemplate="%{text}%", textposition="outside")
    chart_row(
        fig5,
        "Age Group Breakdown",
        rate_breakdown_stats(
            age_churn, "Age Group", "Churn Rate (%)", overall=overall_rate
        ),
        rate_insight_text(
            age_churn, "Age Group", "Churn Rate (%)", overall=overall_rate
        ),
    )

    tenure_churn = (
        df.groupby("TenureGroup", observed=True)["Exited"]
        .mean()
        .mul(100)
        .round(1)
        .reset_index()
    )
    tenure_churn.columns = ["Tenure Group", "Churn Rate (%)"]
    fig6 = px.bar(
        tenure_churn,
        x="Tenure Group",
        y="Churn Rate (%)",
        text="Churn Rate (%)",
        color_discrete_sequence=["#B08D57"],
    )
    fig6.update_traces(texttemplate="%{text}%", textposition="outside")
    fig6.update_layout(yaxis_title="Churn Rate (%)")
    chart_row(
        fig6,
        "Tenure Group Breakdown",
        rate_breakdown_stats(
            tenure_churn, "Tenure Group", "Churn Rate (%)", overall=overall_rate
        ),
        rate_insight_text(
            tenure_churn, "Tenure Group", "Churn Rate (%)", overall=overall_rate
        ),
    )

    st.markdown(
        "<h3 class='section-header'>Engagement Rate Across the Customer Lifecycle</h3>",
        unsafe_allow_html=True,
    )
    st.caption(
        "How account activity and churn shift as customers move from New to Long-term relationships with the bank."
    )

    tenure_life = (
        df.groupby("TenureGroup", observed=True)
        .agg(
            Customers=("Exited", "count"),
            Churned=("Exited", "sum"),
            ActiveCount=("ActivityStatus", lambda s: (s == "Active").sum()),
        )
        .reset_index()
    )
    tenure_life["Churn_Rate"] = (
        tenure_life["Churned"] / tenure_life["Customers"] * 100
    ).round(1)
    tenure_life["Active_Rate"] = (
        tenure_life["ActiveCount"] / tenure_life["Customers"] * 100
    ).round(1)

    fig17 = px.bar(
        tenure_life,
        x="TenureGroup",
        y="Active_Rate",
        text="Active_Rate",
        color_discrete_sequence=["#22303C"],
        title="Active Customer Rate by Tenure Stage",
    )
    fig17.update_traces(texttemplate="%{text}%", textposition="outside")
    fig17.update_layout(yaxis_title="Active Rate (%)")

    fig18 = px.bar(
        tenure_life,
        x="TenureGroup",
        y="Churn_Rate",
        text="Churn_Rate",
        color_discrete_sequence=["#7A1F2B"],
        title="Churn Rate by Tenure Stage",
    )
    fig18.update_traces(texttemplate="%{text}%", textposition="outside")
    fig18.update_layout(yaxis_title="Churn Rate (%)")

    _tl_sorted_active = tenure_life.sort_values("Active_Rate", ascending=False)
    
    
    
        # ============================================================
    # ACTIVE CUSTOMER RATE BY TENURE STAGE — FULLY DYNAMIC
    # ============================================================

    _tl_sorted_active = tenure_life.sort_values(
        "Active_Rate",
        ascending=False
    ).reset_index(drop=True)

    if len(_tl_sorted_active) > 0:

        _active_high = _tl_sorted_active.iloc[0]
        _active_low = _tl_sorted_active.iloc[-1]

        _active_spread = (
            _active_high["Active_Rate"]
            - _active_low["Active_Rate"]
        )

        if _active_spread >= 15:
            _active_interpretation = (
                f"This represents a substantial engagement gap of "
                f"<b>{_active_spread:.1f} percentage points</b> between "
                f"the highest- and lowest-active tenure stages."
            )

        elif _active_spread >= 5:
            _active_interpretation = (
                f"This represents a noticeable engagement gap of "
                f"<b>{_active_spread:.1f} percentage points</b> across "
                f"the observed tenure stages."
            )

        else:
            _active_interpretation = (
                f"Active-customer representation is relatively consistent "
                f"across tenure stages, with only a "
                f"<b>{_active_spread:.1f} percentage-point</b> spread."
            )

    _active_finding = (
        f"The <b>{_active_high['TenureGroup']}</b> tenure stage has the "
        f"highest active-customer rate at "
        f"<b>{_active_high['Active_Rate']:.1f}%</b>, while "
        f"<b>{_active_low['TenureGroup']}</b> has the lowest at "
        f"<b>{_active_low['Active_Rate']:.1f}%</b>. "
        f"{_active_interpretation}"
    )

    chart_row(
        fig17,
        "Active Rate Breakdown",
        [
            (
                "Highest active rate",
                f"{_active_high['TenureGroup']} — "
                f"{_active_high['Active_Rate']:.1f}%"
            ),
            (
                "Lowest active rate",
                f"{_active_low['TenureGroup']} — "
                f"{_active_low['Active_Rate']:.1f}%"
            ),
            (
                "Active-rate spread",
                f"{_active_spread:.1f} pts"
            ),
            (
                "Tenure stages shown",
                f"{len(_tl_sorted_active)}"
            ),
        ],
        _active_finding,
    )
    
    







    
    # chart_row(
    #     fig17,
    #     "Active Rate Breakdown",
    #     [
    #         (
    #             "Highest active rate",
    #             f"{_tl_sorted_active.iloc[0]['TenureGroup']} \u2014 {_tl_sorted_active.iloc[0]['Active_Rate']:.1f}%",
    #         ),
    #         (
    #             "Lowest active rate",
    #             f"{_tl_sorted_active.iloc[-1]['TenureGroup']} \u2014 {_tl_sorted_active.iloc[-1]['Active_Rate']:.1f}%",
    #         ),
    #         ("Tenure stages shown", f"{len(tenure_life)}"),
    #     ],
    #     f"Active-customer representation is highest among <b>{_tl_sorted_active.iloc[0]['TenureGroup']}</b> customers, at {_tl_sorted_active.iloc[0]['Active_Rate']:.1f}%, indicating this cohort maintains the strongest degree of ongoing engagement with the institution.",
    # )



    

    # ============================================================
# CHURN RATE BY TENURE STAGE — FULLY DYNAMIC
# ============================================================

    _tenure_churn = tenure_life[
        ["TenureGroup", "Churn_Rate", "Customers"]
    ].copy()

    _tenure_churn = _tenure_churn.rename(
        columns={"Churn_Rate": "Churn Rate (%)"}
    )

    _tenure_churn = _tenure_churn.sort_values(
        "Churn Rate (%)",
        ascending=False
    ).reset_index(drop=True)

    if len(_tenure_churn) > 0:

        _tc_high = _tenure_churn.iloc[0]
        _tc_low = _tenure_churn.iloc[-1]

        _tc_spread = (
            _tc_high["Churn Rate (%)"]
            - _tc_low["Churn Rate (%)"]
        )

        _tc_difference_from_overall = (
            _tc_high["Churn Rate (%)"]
            - overall_rate
        )

        if _tc_spread >= 10:
            _tc_interpretation = (
                f"The difference between the highest- and lowest-risk "
                f"tenure stages is large at "
                f"<b>{_tc_spread:.1f} percentage points</b>, indicating "
                f"substantial variation in observed churn across tenure stages."
            )

        elif _tc_spread >= 5:
            _tc_interpretation = (
                f"The highest- and lowest-risk tenure stages differ by "
                f"<b>{_tc_spread:.1f} percentage points</b>, indicating "
                f"a noticeable variation in observed churn across tenure stages."
            )

        else:
            _tc_interpretation = (
                f"Churn rates are relatively close across tenure stages, "
                f"with a spread of only "
                f"<b>{_tc_spread:.1f} percentage points</b>."
            )

        if _tc_difference_from_overall > 0:
            _tc_comparison = (
                f"The highest-risk stage is "
                f"<b>{_tc_difference_from_overall:.1f} points above</b> "
                f"the current portfolio churn rate of "
                f"<b>{overall_rate:.1f}%</b>."
            )

        elif _tc_difference_from_overall < 0:
            _tc_comparison = (
                f"The highest-risk stage remains "
                f"<b>{abs(_tc_difference_from_overall):.1f} points below</b> "
                f"the current portfolio churn rate of "
                f"<b>{overall_rate:.1f}%</b>."
            )

        else:
            _tc_comparison = (
                f"The highest-risk stage is exactly aligned with the "
                f"current portfolio churn rate of "
                f"<b>{overall_rate:.1f}%</b>."
            )

    _tc_finding = (
        f"<b>{_tc_high['TenureGroup']}</b> has the highest observed "
        f"churn rate at <b>{_tc_high['Churn Rate (%)']:.1f}%</b>, "
        f"while <b>{_tc_low['TenureGroup']}</b> has the lowest at "
        f"<b>{_tc_low['Churn Rate (%)']:.1f}%</b>. "
        f"{_tc_interpretation} {_tc_comparison}"
    )

    chart_row(
        fig18,
        "Tenure Stage Breakdown",
        [
            (
                "Highest churn",
                f"{_tc_high['TenureGroup']} — "
                f"{_tc_high['Churn Rate (%)']:.1f}%"
            ),
            (
                "Lowest churn",
                f"{_tc_low['TenureGroup']} — "
                f"{_tc_low['Churn Rate (%)']:.1f}%"
            ),
            (
                "Churn-rate spread",
                f"{_tc_spread:.1f} pts"
            ),
            (
                "Portfolio average",
                f"{overall_rate:.1f}%"
            ),
        ],
        _tc_finding,
    )




    # chart_row(
    #     fig18,
    #     "Tenure Stage Breakdown",
    #     rate_breakdown_stats(
    #         tenure_life.rename(columns={"Churn_Rate": "Churn Rate (%)"}),
    #         "TenureGroup",
    #         "Churn Rate (%)",
    #         overall=overall_rate,
    #     ),
    #     rate_insight_text(
    #         tenure_life.rename(columns={"Churn_Rate": "Churn Rate (%)"}),
    #         "TenureGroup",
    #         "Churn Rate (%)",
    #         overall=overall_rate,
    #     ),
    # )

    st.markdown(
        "<h3 class='section-header'>Tenure x Activity Status Interaction</h3>",
        unsafe_allow_html=True,
    )
    tenure_act_heat = (
        df.groupby(["TenureGroup", "ActivityStatus"], observed=True)["Exited"]
        .mean()
        .mul(100)
        .round(1)
        .reset_index()
    )
    heat_pivot2 = tenure_act_heat.pivot(
        index="TenureGroup", columns="ActivityStatus", values="Exited"
    )
    fig19 = px.imshow(
        heat_pivot2,
        text_auto=True,
        color_continuous_scale=["#F7F5F0", "#7A1F2B"],
        labels=dict(color="Churn Rate (%)"),
    )


   


    # ============================================================
    # TENURE × ACTIVITY STATUS — FULLY DYNAMIC
    # ============================================================

    _heat2_sorted = tenure_act_heat.sort_values(
        "Exited",
        ascending=False
    ).reset_index(drop=True)

    if len(_heat2_sorted) > 0:

        _hot2 = _heat2_sorted.iloc[0]
        _cold2 = _heat2_sorted.iloc[-1]

        _interaction_spread = (
            _hot2["Exited"] - _cold2["Exited"]
        )

        _hot2_rate = _hot2["Exited"]
        _cold2_rate = _cold2["Exited"]

        # --------------------------------------------------------
        # Dynamic spread interpretation
        # --------------------------------------------------------

        if _interaction_spread >= 20:

            _interaction_pattern = (
                f"The highest- and lowest-churn combinations differ by "
                f"<b>{_interaction_spread:.1f} percentage points</b>, "
                f"showing a substantial variation in observed churn "
                f"across tenure and activity combinations."
            )

        elif _interaction_spread >= 10:

            _interaction_pattern = (
                f"The highest- and lowest-churn combinations differ by "
                f"<b>{_interaction_spread:.1f} percentage points</b>, "
                f"showing a noticeable variation in observed churn "
                f"across the interaction cells."
            )

        elif _interaction_spread >= 5:

            _interaction_pattern = (
                f"The interaction cells show a moderate "
                f"<b>{_interaction_spread:.1f} percentage-point</b> "
                f"difference between the highest and lowest observed "
                f"churn rates."
            )

        else:

            _interaction_pattern = (
                f"Observed churn rates are relatively similar across "
                f"the interaction cells, with a spread of only "
                f"<b>{_interaction_spread:.1f} percentage points</b>."
            )

    # --------------------------------------------------------
    # Dynamic risk comparison
    # --------------------------------------------------------

    if _hot2_rate > overall_rate:

        _interaction_comparison = (
            f"The highest-risk combination is "
            f"<b>{_hot2_rate - overall_rate:.1f} points above</b> "
            f"the current portfolio churn rate of "
            f"<b>{overall_rate:.1f}%</b>."
        )

    elif _hot2_rate < overall_rate:

        _interaction_comparison = (
            f"Even the highest-risk combination is "
            f"<b>{overall_rate - _hot2_rate:.1f} points below</b> "
            f"the current portfolio churn rate of "
            f"<b>{overall_rate:.1f}%</b>."
        )

    else:

        _interaction_comparison = (
            f"The highest-risk combination is aligned with the "
            f"current portfolio churn rate of "
            f"<b>{overall_rate:.1f}%</b>."
        )

    _interaction_finding = (
        f"The <b>{_hot2['TenureGroup']} / "
        f"{_hot2['ActivityStatus']}</b> combination has the highest "
        f"observed churn rate at <b>{_hot2_rate:.1f}%</b>, while "
        f"<b>{_cold2['TenureGroup']} / "
        f"{_cold2['ActivityStatus']}</b> has the lowest at "
        f"<b>{_cold2_rate:.1f}%</b>. "
        f"{_interaction_pattern} "
        f"{_interaction_comparison}"
    )

    chart_row(
        fig19,
        "Interaction Breakdown",
        [
            (
                "Highest-risk cell",
                f"{_hot2['TenureGroup']} / "
                f"{_hot2['ActivityStatus']} — "
                f"{_hot2_rate:.1f}%"
            ),
            (
                "Lowest-risk cell",
                f"{_cold2['TenureGroup']} / "
                f"{_cold2['ActivityStatus']} — "
                f"{_cold2_rate:.1f}%"
            ),
            (
                "Cell spread",
                f"{_interaction_spread:.1f} pts"
            ),
            (
                "Cells shown",
                f"{len(_heat2_sorted)}"
            ),
        ],
        _interaction_finding,
    )




    _heat2_sorted = tenure_act_heat.sort_values("Exited", ascending=False)
    # if len(_heat2_sorted) > 0:
    #     _hot2, _cold2 = _heat2_sorted.iloc[0], _heat2_sorted.iloc[-1]
    #     chart_row(
    #         fig19,
    #         "Interaction Breakdown",
    #         [
    #             (
    #                 "Highest-risk cell",
    #                 f"{_hot2['TenureGroup']} / {_hot2['ActivityStatus']} \u2014 {_hot2['Exited']:.1f}%",
    #             ),
    #             (
    #                 "Lowest-risk cell",
    #                 f"{_cold2['TenureGroup']} / {_cold2['ActivityStatus']} \u2014 {_cold2['Exited']:.1f}%",
    #             ),
    #             ("Cells shown", f"{len(_heat2_sorted)}"),
    #         ],
    #         f"The <b>{_hot2['TenureGroup']} / {_hot2['ActivityStatus']}</b> combination reflects the highest attrition observed within this interaction, at {_hot2['Exited']:.1f}%, warranting closer examination as a compounded risk factor.",
    #     )

    st.markdown(
        "<h3 class='section-header'>Churn Rate Across Exact Tenure Year-by-Year</h3>",
        unsafe_allow_html=True,
    )
    st.caption(
        "A finer-grained view than the three tenure bands above — reveals lifecycle moments the grouped bands can hide."
    )
    yearly_tenure = (
        df.groupby("Tenure", observed=True)["Exited"]
        .mean()
        .mul(100)
        .round(1)
        .reset_index()
    )
    yearly_tenure.columns = ["Tenure (Years)", "Churn Rate (%)"]
    fig20 = px.line(
        yearly_tenure,
        x="Tenure (Years)",
        y="Churn Rate (%)",
        markers=True,
        color_discrete_sequence=["#22303C"],
    )
    fig20.add_hline(
        y=overall_rate,
        line_dash="dash",
        line_color="#B08D57",
        annotation_text="Portfolio Average",
        annotation_position="top left",
    )
    
    
    
# ============================================================
# EXACT TENURE YEAR-BY-YEAR — FULLY DYNAMIC
# ============================================================

    if len(yearly_tenure) > 0:

        _yt_sorted = yearly_tenure.sort_values(
            "Churn Rate (%)",
            ascending=False
        ).reset_index(drop=True)

        _yt_peak = _yt_sorted.iloc[0]
        _yt_low = _yt_sorted.iloc[-1]

        _yt_peak_year = int(_yt_peak["Tenure (Years)"])
        _yt_peak_rate = _yt_peak["Churn Rate (%)"]

        _yt_low_year = int(_yt_low["Tenure (Years)"])
        _yt_low_rate = _yt_low["Churn Rate (%)"]

        _yt_spread = _yt_peak_rate - _yt_low_rate
        _yt_peak_vs_overall = _yt_peak_rate - overall_rate

        # --------------------------------------------------------
        # Dynamic spread interpretation
        # --------------------------------------------------------

        if _yt_spread >= 15:
            _yt_pattern = (
                f"The year-by-year churn profile shows a substantial "
                f"<b>{_yt_spread:.1f} percentage-point</b> difference "
                f"between the highest- and lowest-churn tenure years."
            )

        elif _yt_spread >= 5:
            _yt_pattern = (
                f"The year-by-year churn profile shows a noticeable "
                f"<b>{_yt_spread:.1f} percentage-point</b> difference "
                f"between the highest- and lowest-churn tenure years."
            )

        else:
            _yt_pattern = (
                f"Churn rates remain relatively close across the observed "
                f"tenure years, with a spread of only "
                f"<b>{_yt_spread:.1f} percentage points</b>."
            )

    # --------------------------------------------------------
    # Dynamic comparison with portfolio average
    # --------------------------------------------------------

    if _yt_peak_vs_overall > 0:

        _yt_peak_comparison = (
            f"The peak year is <b>{_yt_peak_vs_overall:.1f} points above</b> "
            f"the current portfolio average of "
            f"<b>{overall_rate:.1f}%</b>."
        )

    elif _yt_peak_vs_overall < 0:

        _yt_peak_comparison = (
            f"The peak year is <b>{abs(_yt_peak_vs_overall):.1f} points below</b> "
            f"the current portfolio average of "
            f"<b>{overall_rate:.1f}%</b>."
        )

    else:

        _yt_peak_comparison = (
            f"The peak year is aligned with the current portfolio "
            f"average of <b>{overall_rate:.1f}%</b>."
        )

    # --------------------------------------------------------
    # Dynamic location of peak
    # --------------------------------------------------------

    if _yt_peak_year <= 1:

        _yt_location = (
            "The highest observed churn occurs during the earliest "
            "tenure year."
        )

    elif _yt_peak_year <= 2:

        _yt_location = (
            "The highest observed churn occurs during the early "
            "relationship period."
        )

    elif _yt_peak_year <= 4:

        _yt_location = (
            "The highest observed churn occurs during the early-to-mid "
            "relationship period."
        )

    else:

        _yt_location = (
            "The highest observed churn occurs during a later stage "
            "of the customer relationship."
        )

    _yt_finding = (
        f"Tenure year <b>{_yt_peak_year}</b> records the highest "
        f"observed churn rate at <b>{_yt_peak_rate:.1f}%</b>, while "
        f"tenure year <b>{_yt_low_year}</b> records the lowest at "
        f"<b>{_yt_low_rate:.1f}%</b>. "
        f"{_yt_pattern} {_yt_peak_comparison} "
        f"{_yt_location}"
    )

    chart_row(
        fig20,
        "Year-by-Year Breakdown",
        [
            (
                "Peak churn year",
                f"Year {_yt_peak_year} — "
                f"{_yt_peak_rate:.1f}%"
            ),
            (
                "Lowest churn year",
                f"Year {_yt_low_year} — "
                f"{_yt_low_rate:.1f}%"
            ),
            (
                "Churn-rate spread",
                f"{_yt_spread:.1f} pts"
            ),
            (
                "Years tracked",
                f"{len(yearly_tenure)}"
            ),
            (
                "Portfolio average",
                f"{overall_rate:.1f}%"
            ),
        ],
        _yt_finding,
    )
    
    
    

    # if len(yearly_tenure) > 0:
    #     _yt_sorted = yearly_tenure.sort_values("Churn Rate (%)", ascending=False)
    #     chart_row(
    #         fig20,
    #         "Year-by-Year Breakdown",
    #         [
    #             (
    #                 "Peak churn year",
    #                 f"Year {int(_yt_sorted.iloc[0]['Tenure (Years)'])} \u2014 {_yt_sorted.iloc[0]['Churn Rate (%)']:.1f}%",
    #             ),
    #             (
    #                 "Lowest churn year",
    #                 f"Year {int(_yt_sorted.iloc[-1]['Tenure (Years)'])} \u2014 {_yt_sorted.iloc[-1]['Churn Rate (%)']:.1f}%",
    #             ),
    #             ("Years tracked", f"{len(yearly_tenure)}"),
    #             ("Portfolio average", f"{overall_rate:.1f}%"),
    #         ],
    #         f"Tenure year <b>{int(_yt_sorted.iloc[0]['Tenure (Years)'])}</b> registers the single highest attrition rate at {_yt_sorted.iloc[0]['Churn Rate (%)']:.1f}%, exceeding the {overall_rate:.1f}% portfolio-wide average denoted by the reference line.",
    #     )

    if len(tenure_life) >= 2 and len(yearly_tenure) > 0:
        _new_row = tenure_life[tenure_life["TenureGroup"] == "New (0-2yr)"]
        _long_row = tenure_life[tenure_life["TenureGroup"] == "Long-term (7-10yr)"]
        _peak_year = yearly_tenure.sort_values("Churn Rate (%)", ascending=False).iloc[
            0
        ]

        if len(_new_row) > 0 and len(_long_row) > 0:
            _new_active = _new_row["Active_Rate"].values[0]
            _long_active = _long_row["Active_Rate"].values[0]
            _eng_direction = "declines" if _long_active < _new_active else "improves"
            _eng_gap_lifecycle = abs(_new_active - _long_active)
        else:
            _eng_direction, _eng_gap_lifecycle, _new_active, _long_active = (
                "is not determinable",
                0,
                0,
                0,
            )


        _is_early_spike = int(_peak_year["Tenure (Years)"]) <= 2
        _peak_excess = _peak_year["Churn Rate (%)"] - overall_rate

        if _is_early_spike:
            _tenure_interpretation = (
                f"Grouped tenure bands can mask sharp single-year effects. The year-{int(_peak_year['Tenure (Years)'])} "
                f"spike identified here falls within the earliest phase of the relationship — commonly termed "
                f"'honeymoon churn' — and sits {_peak_excess:.1f} points above the {overall_rate:.1f}% portfolio "
                f"average, a {magnitude_word(_peak_excess)} deviation that points to onboarding or early-experience "
                f"friction rather than a slow erosion of engagement over time."
            )
            _tenure_recommendation = (
                f"Focus the onboarding and first-year engagement programme on tenure year "
                f"{int(_peak_year['Tenure (Years)'])} specifically, since the risk is concentrated early rather than "
                f"building gradually. This {urgency_clause(_peak_excess)}."
            )
        else:
            _tenure_interpretation = (
                f"Grouped tenure bands can mask sharp single-year effects. The year-{int(_peak_year['Tenure (Years)'])} "
                f"spike identified here occurs well into the relationship rather than at onboarding, sitting "
                f"{_peak_excess:.1f} points above the {overall_rate:.1f}% portfolio average — a "
                f"{magnitude_word(_peak_excess)} deviation more consistent with a mid-life loyalty or fatigue effect "
                f"than an early-experience problem."
            )
            _tenure_recommendation = (
                f"Prioritise a mid-life loyalty or re-engagement touchpoint timed around tenure year "
                f"{int(_peak_year['Tenure (Years)'])} for long-standing customers, rather than concentrating spend on "
                f"the onboarding programme alone. This {urgency_clause(_peak_excess)}."
            )

        # finding_box(
        #     "08",
        #     "How Engagement and Churn Shift Across the Customer Lifecycle",
        #     f"Active-customer rate {_eng_direction} from {_new_active:.1f}% among New (0-2yr) customers to "
        #     f"{_long_active:.1f}% among Long-term (7-10yr) customers, a shift of {_eng_gap_lifecycle:.1f} points. "
        #     f"Separately, Tenure year {int(_peak_year['Tenure (Years)'])} shows the single highest churn rate at "
        #     f"{_peak_year['Churn Rate (%)']:.1f}%, against a {overall_rate:.1f}% portfolio average.",
        #     _tenure_interpretation,
        #     _tenure_recommendation,
        # )

        # key_finding_banner(
        #    "Engagement & Tenure Patterns",
        #    f"Engagement <b>{_eng_direction}</b> across the customer lifecycle, moving from {_new_active:.1f}% active "
        #   f"among new customers to {_long_active:.1f}% among long-term customers, while tenure year "
        #   f"<b>{int(_peak_year['Tenure (Years)'])}</b> stands out as the single highest-risk point in the relationship "
        #   f"at {_peak_year['Churn Rate (%)']:.1f}% churn.",
        # )
# else:
#    key_finding_banner(
#       "Engagement & Tenure Patterns",
#      "Insufficient data is available under the current filter selection to establish a reliable lifecycle pattern.",
# )
   




   



    # ============================================================
    # TAB V — CONSOLIDATED AGE & TENURE FINDING
    # ============================================================

    # -----------------------------
    # 1. AGE-BASED CHURN PROFILE
    # -----------------------------
    _age_summary = (
        df.groupby("AgeGroup", observed=True)
        .agg(
            Customers=("Exited", "count"),
            Churned=("Exited", "sum"),
        )
        .reset_index()
    )

    _age_summary["Churn Rate"] = (
        _age_summary["Churned"] / _age_summary["Customers"] * 100
    ).round(1)

    _age_summary = _age_summary.sort_values(
        "Churn Rate", ascending=False
    ).reset_index(drop=True)

    _age_high = _age_summary.iloc[0]
    _age_low = _age_summary.iloc[-1]

    _age_spread = _age_high["Churn Rate"] - _age_low["Churn Rate"]
    _age_high_vs_overall = _age_high["Churn Rate"] - overall_rate

    if _age_spread >= 10:
        _age_pattern = (
            f"Age groups show a substantial churn difference of "
            f"<b>{_age_spread:.1f} percentage points</b> between the "
            f"highest- and lowest-churn groups."
        )
    elif _age_spread >= 5:
        _age_pattern = (
            f"Age groups show a noticeable churn difference of "
            f"<b>{_age_spread:.1f} percentage points</b> between the "
            f"highest- and lowest-churn groups."
        )
    else:
        _age_pattern = (
            f"Churn rates are relatively close across age groups, "
            f"with a spread of only <b>{_age_spread:.1f} percentage points</b>."
        )


    # ------------------------------------
    # 2. TENURE-BASED CHURN PROFILE
    # ------------------------------------
    _tenure_summary = tenure_life[
        ["TenureGroup", "Customers", "Churned", "Churn_Rate", "Active_Rate"]
    ].copy()

    _tenure_summary = _tenure_summary.sort_values(
        "Churn_Rate", ascending=False
    ).reset_index(drop=True)

    _tenure_high = _tenure_summary.iloc[0]
    _tenure_low = _tenure_summary.iloc[-1]

    _tenure_spread = (
        _tenure_high["Churn_Rate"] - _tenure_low["Churn_Rate"]
    )

    _tenure_high_vs_overall = (
        _tenure_high["Churn_Rate"] - overall_rate
    )


    # ------------------------------------
    # 3. TENURE × ACTIVITY INTERACTION
    # ------------------------------------
    _interaction_summary = tenure_act_heat.copy()

    _interaction_summary = _interaction_summary.sort_values(
        "Exited", ascending=False
    ).reset_index(drop=True)

    _interaction_high = _interaction_summary.iloc[0]
    _interaction_low = _interaction_summary.iloc[-1]

    _interaction_spread = (
        _interaction_high["Exited"] -
        _interaction_low["Exited"]
    )


    # ------------------------------------
    # 4. EXACT TENURE YEAR ANALYSIS
    # ------------------------------------
    _year_summary = yearly_tenure.copy()

    _year_summary = _year_summary.sort_values(
        "Churn Rate (%)",
        ascending=False
    ).reset_index(drop=True)

    _peak_year = _year_summary.iloc[0]
    _low_year = _year_summary.iloc[-1]

    _peak_year_number = int(_peak_year["Tenure (Years)"])
    _peak_year_rate = _peak_year["Churn Rate (%)"]

    _low_year_number = int(_low_year["Tenure (Years)"])
    _low_year_rate = _low_year["Churn Rate (%)"]

    _year_spread = _peak_year_rate - _low_year_rate
    _peak_year_vs_overall = _peak_year_rate - overall_rate


    # ------------------------------------
    # 5. LIFECYCLE ENGAGEMENT
    # ------------------------------------
    _new_row = tenure_life[
        tenure_life["TenureGroup"] == "New (0-2yr)"
    ]

    _long_row = tenure_life[
        tenure_life["TenureGroup"] == "Long-term (7-10yr)"
    ]

    if len(_new_row) > 0 and len(_long_row) > 0:

        _new_active_rate = _new_row["Active_Rate"].iloc[0]
        _long_active_rate = _long_row["Active_Rate"].iloc[0]

        _engagement_change = (
            _long_active_rate - _new_active_rate
        )

        if _engagement_change < 0:
            _engagement_direction = "declines"
        elif _engagement_change > 0:
            _engagement_direction = "increases"
        else:
            _engagement_direction = "remains broadly stable"

    else:

        _new_active_rate = 0
        _long_active_rate = 0
        _engagement_change = 0
        _engagement_direction = "is not determinable"


    # ------------------------------------
    # 6. DYNAMIC MANAGEMENT INTERPRETATION
    # ------------------------------------
    if _peak_year_number <= 2:

        _lifecycle_interpretation = (
            f"The most important tenure signal occurs early in the relationship. "
            f"Tenure year <b>{_peak_year_number}</b> records the highest observed "
            f"churn rate, which is <b>{abs(_peak_year_vs_overall):.1f} points "
            f"{'above' if _peak_year_vs_overall >= 0 else 'below'}</b> the "
            f"portfolio average. This makes early-stage customer experience and "
            f"onboarding an important area for retention monitoring."
        )

    elif _peak_year_number <= 4:

        _lifecycle_interpretation = (
            f"The strongest tenure signal occurs during the early-to-mid "
            f"relationship period. Tenure year <b>{_peak_year_number}</b> records "
            f"the highest observed churn rate at <b>{_peak_year_rate:.1f}%</b>, "
            f"indicating that retention attention should not be limited to the "
            f"initial onboarding period."
        )

    else:

        _lifecycle_interpretation = (
            f"The strongest tenure signal occurs later in the customer lifecycle. "
            f"Tenure year <b>{_peak_year_number}</b> records the highest observed "
            f"churn rate at <b>{_peak_year_rate:.1f}%</b>, suggesting that "
            f"longer-standing customers should also receive targeted "
            f"re-engagement and retention monitoring."
        )


    # ------------------------------------
    # 7. CONSOLIDATED FINDING
    # ------------------------------------
    finding_box(
        "08",
        "Age, Tenure and Customer Lifecycle Churn — Overall Finding",

        (
            "<ol>"
            
            f"<li><b>Age-based risk:</b> "
            f"<b>{_age_high['AgeGroup']}</b> has the highest observed age-group "
            f"churn rate at <b>{_age_high['Churn Rate']:.1f}%</b>, while "
            f"<b>{_age_low['AgeGroup']}</b> has the lowest at "
            f"<b>{_age_low['Churn Rate']:.1f}%</b>. "
            f"{_age_pattern}</li>"

            f"<li><b>Tenure-based risk:</b> "
            f"<b>{_tenure_high['TenureGroup']}</b> records the highest grouped "
            f"tenure churn rate at <b>{_tenure_high['Churn_Rate']:.1f}%</b>, "
            f"compared with <b>{_tenure_low['TenureGroup']}</b> at "
            f"<b>{_tenure_low['Churn_Rate']:.1f}%</b>. "
            f"The overall tenure spread is "
            f"<b>{_tenure_spread:.1f} percentage points</b>. "
            f"The highest-risk tenure group is "
            f"<b>{abs(_tenure_high_vs_overall):.1f} points "
            f"{'above' if _tenure_high_vs_overall >= 0 else 'below'}</b> "
            f"the portfolio churn rate of <b>{overall_rate:.1f}%</b>.</li>"

            f"<li><b>Engagement across the lifecycle:</b> "
            f"The active-customer rate {_engagement_direction} from "
            f"<b>{_new_active_rate:.1f}%</b> among New (0-2yr) customers to "
            f"<b>{_long_active_rate:.1f}%</b> among Long-term (7-10yr) customers, "
            f"a difference of <b>{abs(_engagement_change):.1f} percentage points</b>. "
            f"This shows whether customer engagement is being maintained as the "
            f"relationship matures.</li>"

            f"<li><b>Tenure × activity risk:</b> "
            f"The <b>{_interaction_high['TenureGroup']} / "
            f"{_interaction_high['ActivityStatus']}</b> combination has the "
            f"highest observed churn rate at <b>{_interaction_high['Exited']:.1f}%</b>, "
            f"while <b>{_interaction_low['TenureGroup']} / "
            f"{_interaction_low['ActivityStatus']}</b> has the lowest at "
            f"<b>{_interaction_low['Exited']:.1f}%</b>. "
            f"The interaction spread is <b>{_interaction_spread:.1f} percentage "
            f"points</b>, showing that tenure and activity should be reviewed "
            f"together rather than in isolation.</li>"

            f"<li><b>Exact tenure-year signal:</b> "
            f"Tenure year <b>{_peak_year_number}</b> has the highest individual "
            f"year-level churn rate at <b>{_peak_year_rate:.1f}%</b>, while "
            f"year <b>{_low_year_number}</b> has the lowest at "
            f"<b>{_low_year_rate:.1f}%</b>. "
            f"The year-level spread is <b>{_year_spread:.1f} percentage points</b>. "
            f"{_lifecycle_interpretation}</li>"

            f"<li><b>Management implication:</b> "
            f"Age and tenure analysis identifies where churn is concentrated "
            f"within the customer lifecycle. The bank should use these patterns "
            f"to prioritise retention monitoring for the highest-risk age and "
            f"tenure cohorts, while using activity status as an additional "
            f"segmentation layer. These results describe observed churn "
            f"associations and should be combined with financial, geographic, "
            f"product and behavioural variables before making individual-level "
            f"retention decisions.</li>"

            "</ol>"
        ),

        # (
        #     f"<b>Overall interpretation:</b> "
        #     f"The Age & Tenure analysis shows that customer churn is not evenly "
        #     f"distributed across the customer lifecycle. Differences are visible "
        #     f"across age groups, tenure stages, activity levels and individual "
        #     f"tenure years. The most useful retention strategy is therefore "
        #     f"segment-specific rather than applying the same intervention to "
        #     f"every customer. {_lifecycle_interpretation}"
        # ),

        # (
        #     f"<b>Recommended action:</b> "
        #     f"Prioritise the highest-risk age group "
        #     f"(<b>{_age_high['AgeGroup']}</b>), the highest-risk tenure stage "
        #     f"(<b>{_tenure_high['TenureGroup']}</b>) and the highest-risk "
        #     f"tenure/activity combination "
        #     f"(<b>{_interaction_high['TenureGroup']} / "
        #     f"{_interaction_high['ActivityStatus']}</b>) for targeted retention "
        #     f"monitoring. Use the exact tenure-year pattern to determine when "
        #     f"engagement interventions should be introduced, rather than relying "
        #     f"only on broad New/Mid-term/Long-term bands."
        # ),
    )


# ---------------------------------------------------------
# TAB 7 — PREDICTIVE CHURN RISK MODEL
# ---------------------------------------------------------
with tab7:
    st.markdown(
        "<h3 class='section-header'>Model Overview</h3>", unsafe_allow_html=True
    )
    st.caption(
        "A Gradient Boosting classifier trained on the full portfolio (n="
        + f"{len(df_raw):,}"
        + "), evaluated on a held-out 20% test set. Scores below reflect true out-of-sample performance, "
        "not accuracy on data the model has already seen."
    )

    m = _churn_model_bundle["metrics"]
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        kpi_report_card(
            "Accuracy",
            "Overall correct predictions",
            f"{m['Accuracy']*100:.1f}%",
            f"Test set: {_churn_model_bundle['test_size']:,} customers",
        )
    with c2:
        kpi_report_card(
            "Precision",
            "Of predicted churners, % correct",
            f"{m['Precision']*100:.1f}%",
            "",
        )
    with c3:
        kpi_report_card(
            "Recall", "Of actual churners, % caught", f"{m['Recall']*100:.1f}%", ""
        )
    with c4:
        kpi_report_card("F1 Score", "Precision/Recall balance", f"{m['F1']:.2f}", "")
    with c5:
        kpi_report_card(
            "ROC-AUC",
            "Overall discrimination ability",
            f"{m['ROC-AUC']:.3f}",
            "1.0 = perfect, 0.5 = random",
        )

    st.markdown(
        "<h3 class='section-header'>Feature Importance — Random Forest Interpretation</h3>",
        unsafe_allow_html=True,
    )
    imp_df = _churn_model_bundle["importances"].reset_index()
    imp_df.columns = ["Feature", "Importance"]
    imp_df["Importance"] = (imp_df["Importance"] * 100).round(1)
    fig21 = px.bar(
        imp_df.sort_values("Importance"),
        x="Importance",
        y="Feature",
        orientation="h",
        color_discrete_sequence=["#22303C"],
        title="Feature Importance (Random Forest)",
    )
    fig21.update_layout(xaxis_title="Relative Importance (%)", yaxis_title="")
    _imp_sorted = imp_df.sort_values("Importance", ascending=False)
    _top3_sum = _imp_sorted.head(3)["Importance"].sum()
    chart_row(
        fig21,
        "Importance Breakdown",
        [
            (
                "Top feature",
                f"{_imp_sorted.iloc[0]['Feature']} \u2014 {_imp_sorted.iloc[0]['Importance']:.1f}%",
            ),
            (
                "2nd feature",
                f"{_imp_sorted.iloc[1]['Feature']} \u2014 {_imp_sorted.iloc[1]['Importance']:.1f}%",
            ),
            ("Top-3 combined", f"{_top3_sum:.1f}%"),
            ("Total features used", f"{len(imp_df)}"),
        ],
        f"<b>{_imp_sorted.iloc[0]['Feature']}</b> constitutes the single largest determinant of the model's classifications, accounting for {_imp_sorted.iloc[0]['Importance']:.1f}% of  total feature importance; "
        f"the top 3 features together explain {_top3_sum:.1f}% of all decisions.",
    )

    st.markdown(
        "<h3 class='section-header'>Customer-Level Churn Risk Scores</h3>",
        unsafe_allow_html=True,
    )
    st.caption(
        "Every customer in the current filter selection, scored 0-100% by predicted churn probability and bucketed into Low / Medium / High risk."
    )

    scored_df = score_churn_risk(df, _churn_model_bundle)

    risk_summary = (
        scored_df.groupby("RiskBucket", observed=True)
        .agg(Customers=("ChurnRiskScore", "count"), ActualChurnRate=("Exited", "mean"))
        .reset_index()
    )
    risk_summary["ActualChurnRate"] = (risk_summary["ActualChurnRate"] * 100).round(1)
    risk_order = {"Low": 0, "Medium": 1, "High": 2}
    risk_summary["_order"] = risk_summary["RiskBucket"].map(risk_order)
    risk_summary = risk_summary.sort_values("_order").drop(columns="_order")
    st.markdown("**Customers by Predicted Risk Bucket**")
    fig22 = px.bar(
        risk_summary,
        x="RiskBucket",
        y="Customers",
        text="Customers",
        color="RiskBucket",
        color_discrete_map={"Low": "#22303C", "Medium": "#B08D57", "High": "#7A1F2B"},
        title="Customers by Risk Bucket",
    )
    fig22.update_layout(showlegend=False)
    _rb_high = risk_summary[risk_summary["RiskBucket"] == "High"]
    _rb_low = risk_summary[risk_summary["RiskBucket"] == "Low"]
    chart_row(
        fig22,
        "Risk Bucket Breakdown",
        [
            (
                "High risk",
                (
                    f"{int(_rb_high['Customers'].values[0]):,}"
                    if len(_rb_high) > 0
                    else "0"
                ),
            ),
            (
                "Medium risk",
                (
                    f"{int(risk_summary[risk_summary['RiskBucket']=='Medium']['Customers'].values[0]):,}"
                    if (risk_summary["RiskBucket"] == "Medium").any()
                    else "0"
                ),
            ),
            (
                "Low risk",
                f"{int(_rb_low['Customers'].values[0]):,}" if len(_rb_low) > 0 else "0",
            ),
            ("Total scored", f"{len(scored_df):,}"),
        ],
        (
            f"{int(_rb_high['Customers'].values[0]):,} customers fall into the High Risk bucket under the current filter selection."
            if len(_rb_high) > 0
            else "No customers currently fall into the High Risk bucket under this filter selection."
        ),
    )

    fig23 = px.bar(
        risk_summary,
        x="RiskBucket",
        y="ActualChurnRate",
        text="ActualChurnRate",
        color="RiskBucket",
        color_discrete_map={"Low": "#22303C", "Medium": "#B08D57", "High": "#7A1F2B"},
        title="Actual Churn Rate Within Each Predicted Bucket",
    )
    fig23.update_traces(texttemplate="%{text}%", textposition="outside")
    fig23.update_layout(showlegend=False, yaxis_title="Actual Churn Rate (%)")
    chart_row(
        fig23,
        "Bucket Accuracy Breakdown",
        [
            (
                "High bucket actual churn",
                (
                    f"{_rb_high['ActualChurnRate'].values[0]:.1f}%"
                    if len(_rb_high) > 0
                    else "N/A"
                ),
            ),
            (
                "Low bucket actual churn",
                (
                    f"{_rb_low['ActualChurnRate'].values[0]:.1f}%"
                    if len(_rb_low) > 0
                    else "N/A"
                ),
            ),
            ("Portfolio average", f"{overall_rate:.1f}%"),
        ],
        (
            f"Customers the model flags High Risk actually churn at {_rb_high['ActualChurnRate'].values[0]:.1f}%, "
            f"versus {_rb_low['ActualChurnRate'].values[0]:.1f}% for those flagged Low Risk — indicating that the model meaningfully separates customers by observed churn risk."
            if len(_rb_high) > 0 and len(_rb_low) > 0
            else "Insufficient data across risk buckets to compare under the current filter."
        ),
    )

    st.markdown("**Highest-Risk Customers (Top 25 by Predicted Probability)**")
    top_risk = scored_df.sort_values("ChurnRiskScore", ascending=False).head(25)
    risk_display_cols = [
        "CustomerId",
        "Geography",
        "Age",
        "Balance",
        "NumOfProducts",
        "IsActiveMember",
        "ChurnRiskScore",
        "RiskBucket",
    ]
    top_risk_display = top_risk[risk_display_cols].copy()
    top_risk_display["ChurnRiskScore"] = top_risk_display["ChurnRiskScore"].round(1)
    st.dataframe(top_risk_display, hide_index=True, use_container_width=True)

    risk_csv = (
        scored_df[
            [
                "CustomerId",
                "Geography",
                "Age",
                "Balance",
                "ChurnRiskScore",
                "RiskBucket",
                "Exited",
            ]
        ]
        .to_csv(index=False)
        .encode("utf-8")
    )
    st.download_button(
        "Download All Customer Risk Scores (CSV)",
        data=risk_csv,
        file_name="customer_churn_risk_scores.csv",
        mime="text/csv",
    )

    _high_bucket = risk_summary[risk_summary["RiskBucket"] == "High"]
    if (
        len(_high_bucket) > 0
        and _high_bucket["ActualChurnRate"].values[0] > overall_rate
    ):
        _high_actual = _high_bucket["ActualChurnRate"].values[0]
        _high_count = int(_high_bucket["Customers"].values[0])
        _model_multiple = (_high_actual / overall_rate) if overall_rate > 0 else 0
        _manual_multiple = (
            (top_seg["ChurnRate"] / overall_rate)
            if (top_seg is not None and overall_rate > 0)
            else 0
        )
        _model_advantage_pts = (
            (_model_multiple - _manual_multiple) * overall_rate
            if overall_rate > 0
            else 0
        )
        # finding_box(
        #     "10",
        #     "The Model Materially Outperforms Manual Segmentation",
        #     f"The {_high_count:,} customers the model flags as High Risk actually churn at {_high_actual:.1f}%, "
        #     f"versus the {overall_rate:.1f}% portfolio average — a concentration manual segmentation alone "
        #     f"cannot achieve, since it is limited to one or two dimensions at a time.",
        #     (
        #         f"The model combines all ten input variables simultaneously and learns their interactions "
        #         f"automatically. This High Risk bucket concentrates risk at {_model_multiple:.1f}\u00d7 the portfolio "
        #         f"average"
        #         + (
        #             f", exceeding the {_manual_multiple:.1f}\u00d7 concentration achieved by the single best manual "
        #             f"segment identified in Tab IV ({top_seg['Dimension']} = {top_seg['Value']}) by "
        #             f"{_model_advantage_pts:+.1f} points"
        #             if top_seg is not None
        #             else ""
        #         )
        #         + f". Cross-validated ROC-AUC of {m['ROC-AUC']:.3f} confirms this holds beyond a single lucky test split."
        #     ),
        #     (
        #         f"Route the High Risk list directly to the retention team as a prioritised call list, ranked by "
        #         f"ChurnRiskScore, rather than relying solely on the segment-level targeting used in Tabs III-VII. "
        #         f"This {urgency_clause((_model_multiple - 1) * 100, thresholds=(20, 50, 100))}."
        #     ),
        # )
    #     key_finding_banner(
    #         "Predictive Churn Risk Model",
    #         f"The Gradient Boosting model (ROC-AUC {m['ROC-AUC']:.3f}) identifies <b>{_high_count:,} customers</b> as "
    #         f"High Risk, who churn at an actual rate of <b>{_high_actual:.1f}%</b> against a {overall_rate:.1f}% "
    #         f"portfolio average — a level of risk concentration that manual, one- or two-dimension segmentation cannot "
    #         f"replicate, and which should now serve as the primary prioritisation tool for retention outreach.",
    #     )
    # else:
    #     key_finding_banner(
    #         "Predictive Churn Risk Model",
    #         f"The Gradient Boosting model achieves a cross-validated ROC-AUC of <b>{m['ROC-AUC']:.3f}</b> on the full "
    #         f"portfolio; under the current filter selection, insufficient high-risk customers remain to demonstrate "
    #         f"risk concentration beyond the {overall_rate:.1f}% portfolio average.",
    #     )
    



        # ============================================================
    # TAB VI — CONSOLIDATED PREDICTIVE MODEL FINDING
    # ============================================================

    # ------------------------------------------------------------
    # MODEL PERFORMANCE
    # ------------------------------------------------------------
    _model_accuracy = m["Accuracy"] * 100
    _model_precision = m["Precision"] * 100
    _model_recall = m["Recall"] * 100
    _model_f1 = m["F1"]
    _model_auc = m["ROC-AUC"]


    # ------------------------------------------------------------
    # FEATURE IMPORTANCE
    # ------------------------------------------------------------
    _imp_sorted = imp_df.sort_values(
        "Importance",
        ascending=False
    ).reset_index(drop=True)

    _top_feature = _imp_sorted.iloc[0]["Feature"]
    _top_feature_importance = _imp_sorted.iloc[0]["Importance"]

    _second_feature = (
        _imp_sorted.iloc[1]["Feature"]
        if len(_imp_sorted) > 1
        else "N/A"
    )

    _second_feature_importance = (
        _imp_sorted.iloc[1]["Importance"]
        if len(_imp_sorted) > 1
        else 0
    )

    _top3_importance = _imp_sorted.head(3)["Importance"].sum()


    # ------------------------------------------------------------
    # RISK BUCKETS
    # ------------------------------------------------------------
    _high_bucket = risk_summary[
        risk_summary["RiskBucket"] == "High"
    ]

    _medium_bucket = risk_summary[
        risk_summary["RiskBucket"] == "Medium"
    ]

    _low_bucket = risk_summary[
        risk_summary["RiskBucket"] == "Low"
    ]


    _high_count = (
        int(_high_bucket["Customers"].values[0])
        if len(_high_bucket) > 0
        else 0
    )

    _medium_count = (
        int(_medium_bucket["Customers"].values[0])
        if len(_medium_bucket) > 0
        else 0
    )

    _low_count = (
        int(_low_bucket["Customers"].values[0])
        if len(_low_bucket) > 0
        else 0
    )


    _high_actual = (
        _high_bucket["ActualChurnRate"].values[0]
        if len(_high_bucket) > 0
        else 0
    )

    _medium_actual = (
        _medium_bucket["ActualChurnRate"].values[0]
        if len(_medium_bucket) > 0
        else 0
    )

    _low_actual = (
        _low_bucket["ActualChurnRate"].values[0]
        if len(_low_bucket) > 0
        else 0
    )


    # ------------------------------------------------------------
    # HIGH-RISK CONCENTRATION
    # ------------------------------------------------------------
    if overall_rate > 0 and _high_count > 0:

        _high_risk_multiple = (
            _high_actual / overall_rate
        )

        _high_risk_gap = (
            _high_actual - overall_rate
        )

    else:

        _high_risk_multiple = 0
        _high_risk_gap = 0


    # ------------------------------------------------------------
    # MODEL VS MANUAL SEGMENTATION
    # ------------------------------------------------------------
    if (
        top_seg is not None
        and overall_rate > 0
    ):

        _manual_segment_rate = top_seg["ChurnRate"]

        _manual_risk_multiple = (
            _manual_segment_rate / overall_rate
        )

    else:

        _manual_segment_rate = 0
        _manual_risk_multiple = 0


    if overall_rate > 0:

        _model_vs_manual_gap = (
            _high_risk_multiple -
            _manual_risk_multiple
        ) * overall_rate

    else:

        _model_vs_manual_gap = 0


    # ------------------------------------------------------------
    # MODEL INTERPRETATION
    # ------------------------------------------------------------
    if _model_auc >= 0.80:

        _auc_interpretation = (
            "The ROC-AUC indicates strong overall ability to distinguish "
            "customers with higher observed churn risk from those with lower risk."
        )

    elif _model_auc >= 0.70:

        _auc_interpretation = (
            "The ROC-AUC indicates acceptable discrimination between "
            "customers with higher and lower observed churn risk."
        )

    else:

        _auc_interpretation = (
            "The ROC-AUC indicates that the model's ability to distinguish "
            "between higher- and lower-risk customers should be monitored "
            "before using it as the sole retention prioritisation tool."
        )


    # ------------------------------------------------------------
    # RECALL INTERPRETATION
    # ------------------------------------------------------------
    if _model_recall >= 80:

        _recall_interpretation = (
            f"The model captures {_model_recall:.1f}% of actual churners in "
            "the held-out test set, which is particularly important when "
            "the business objective is to identify as many potential "
            "churners as possible."
        )

    elif _model_recall >= 60:

        _recall_interpretation = (
            f"The model captures {_model_recall:.1f}% of actual churners "
            "in the held-out test set, providing useful but not complete "
            "coverage of customers who eventually churn."
        )

    else:

        _recall_interpretation = (
            f"The model captures {_model_recall:.1f}% of actual churners "
            "in the held-out test set, so additional model tuning may be "
            "required before using it for broad retention targeting."
        )


    # ------------------------------------------------------------
    # RISK SEPARATION
    # ------------------------------------------------------------
    if (
        _high_count > 0
        and _low_count > 0
    ):

        _risk_separation = (
            f"The High Risk group records an actual churn rate of "
            f"<b>{_high_actual:.1f}%</b>, compared with "
            f"<b>{_low_actual:.1f}%</b> for the Low Risk group, "
            f"creating a separation of "
            f"<b>{abs(_high_actual - _low_actual):.1f} percentage points</b>."
        )

    else:

        _risk_separation = (
            "There is insufficient representation across the risk buckets "
            "to establish a reliable High-versus-Low risk separation under "
            "the current filter selection."
        )


    # ------------------------------------------------------------
    # # FINAL CONSOLIDATED FINDING
    # ------------------------------------------------------------
    finding_box(
        "09",
        "Predictive Churn Risk Model — Overall Finding",

        (
            "<ol>"

            f"<li><b>Model performance:</b> "
            f"The Gradient Boosting model was evaluated on a held-out "
            f"20% test set, meaning the reported performance reflects "
            f"customers that were not used for model fitting. The model "
            f"achieves <b>{_model_accuracy:.1f}% accuracy</b>, "
            f"<b>{_model_precision:.1f}% precision</b>, "
            f"<b>{_model_recall:.1f}% recall</b>, "
            f"an <b>F1 score of {_model_f1:.2f}</b>, and a "
            f"<b>ROC-AUC of {_model_auc:.3f}</b>. "
            f"{_auc_interpretation}</li>"

            f"<li><b>Ability to identify actual churners:</b> "
            f"{_recall_interpretation} "
            f"Precision of <b>{_model_precision:.1f}%</b> indicates that "
            f"approximately this proportion of customers classified as "
            f"churners were actual churners in the test evaluation. "
            f"Together, Precision and Recall provide a more meaningful "
            f"view of churn-detection performance than Accuracy alone.</li>"

            f"<li><b>What drives the model:</b> "
            f"<b>{_top_feature}</b> is the most important feature in the "
            f"Random Forest interpretation, contributing "
            f"<b>{_top_feature_importance:.1f}%</b> of total feature importance. "
            f"The second-ranked feature is <b>{_second_feature}</b> at "
            f"<b>{_second_feature_importance:.1f}%</b>, while the top three "
            f"features together account for <b>{_top3_importance:.1f}%</b> "
            f"of total feature importance. This shows which customer "
            f"characteristics the model relies on most heavily when "
            f"separating higher- and lower-risk customers. Feature importance "
            f"should be interpreted as model contribution, not as proof that "
            f"a feature independently causes churn.</li>"

            f"<li><b>Customer-level risk identification:</b> "
            f"The model converts each customer's predicted churn probability "
            f"into a <b>ChurnRiskScore from 0% to 100%</b> and assigns the "
            f"customer to Low, Medium or High Risk. Under the current "
            f"selection, <b>{_high_count:,}</b> customers are classified as "
            f"High Risk, <b>{_medium_count:,}</b> as Medium Risk and "
            f"<b>{_low_count:,}</b> as Low Risk. This changes the dashboard "
            f"from a descriptive churn analysis into a customer-level "
            f"prioritisation system.</li>"

            f"<li><b>Risk concentration:</b> "
            f"{_risk_separation} "
            f"{'The High Risk group is also ' + f'<b>{_high_risk_gap:.1f} points above</b> ' + 'the portfolio average churn rate of ' + f'<b>{overall_rate:.1f}%</b>, ' + f'corresponding to approximately <b>{_high_risk_multiple:.1f}×</b> the portfolio average risk.' if _high_count > 0 and _high_actual > overall_rate else 'The current High Risk group does not exceed the portfolio-wide churn rate, so the risk concentration should be interpreted cautiously under the current filter selection.'}</li>"

            f"<li><b>Why the predictive model adds value:</b> "
            f"The earlier dashboard tabs identify risk using individual "
            f"dimensions such as geography, age, tenure, activity and "
            f"customer value. The predictive model evaluates multiple "
            f"customer characteristics together and therefore provides a "
            f"more integrated customer-level risk assessment. "
            f"{'The model therefore provides stronger prioritisation than relying only on a single manual segment, with the current High Risk group showing approximately ' + f'<b>{_high_risk_multiple:.1f}×</b> ' + 'the portfolio average churn rate.' if _high_count > 0 and overall_rate > 0 else 'Its value should be assessed through continued validation and monitoring as the portfolio changes.'}</li>"

            f"<li><b>Operational use:</b> "
            f"The <b>Top 25 highest-risk customers</b> should be treated as "
            f"a prioritised review list rather than as an automatic churn "
            f"decision. Relationship managers can use the ChurnRiskScore "
            f"together with customer profile information to determine which "
            f"customers should receive proactive retention outreach. "
            f"The downloadable risk-score file can also support downstream "
            f"retention workflows and monitoring.</li>"

            f"<li><b>Management conclusion:</b> "
            f"The predictive model provides the bank with a practical way "
            f"to move from broad churn reporting to <b>risk-based customer "
            f"prioritisation</b>. Instead of treating all customers as "
            f"equally likely to leave, the bank can focus limited retention "
            f"resources on customers with the strongest model-indicated risk, "
            f"while continuously monitoring model performance and actual "
            f"outcomes over time.</li>"

            "</ol>"
        ),

    #     (
    #         f"<b>Overall interpretation:</b> "
    #         f"The predictive model is the natural culmination of the "
    #         f"dashboard's earlier descriptive analysis. Tabs I–V explain "
    #         f"where churn is concentrated, while this tab estimates "
    #         f"customer-level churn risk using multiple variables together. "
    #         f"{_auc_interpretation} "
    #         f"The most important business value is therefore not simply "
    #         f"the model's accuracy, but its ability to create a ranked and "
    #         f"actionable list of customers for proactive retention."
    #     ),

    #     (
    #         f"<b>Recommended action:</b> "
    #         f"Use <b>ChurnRiskScore</b> to prioritise retention outreach, "
    #         f"starting with the highest-risk customers and then expanding "
    #         f"to the Medium Risk group according to available retention "
    #         f"capacity. Track predicted risk against actual churn outcomes "
    #         f"regularly, and re-evaluate model performance as new customer "
    #         f"data becomes available. The model should support human "
    #         f"decision-making rather than automatically determining whether "
    #         f"a customer should receive or be denied a service."
    #     ),
     )



# =========================================================
# FINDING 01 — OVERALL CHURN SUMMARY
# FULL DATASET / UNFILTERED PORTFOLIO VIEW
# =========================================================

# IMPORTANT:
# Use df_raw here, NOT df.
# df = filtered customer population.
# df_raw = complete portfolio.

_full_df = df_raw.copy()

# ---------------------------------------------------------
# 1. Overall portfolio metrics
# ---------------------------------------------------------
_full_total_customers = len(_full_df)

_full_churned = int(
    _full_df["Exited"].sum()
) if _full_total_customers > 0 else 0

_full_retained = (
    _full_total_customers - _full_churned
)

_full_churn_rate = (
    (_full_churned / _full_total_customers) * 100
    if _full_total_customers > 0
    else 0
)

_full_retention_rate = (
    (_full_retained / _full_total_customers) * 100
    if _full_total_customers > 0
    else 0
)

# Retention-to-churn relationship
_full_retained_to_churned = (
    _full_retained / _full_churned
    if _full_churned > 0
    else 0
)


# ---------------------------------------------------------
# 2. Full-dataset engagement analysis
# ---------------------------------------------------------
_full_active = _full_df[
    _full_df["ActivityStatus"] == "Active"
]

_full_inactive = _full_df[
    _full_df["ActivityStatus"] == "Inactive"
]

_full_active_rate = (
    _full_active["Exited"].mean() * 100
    if len(_full_active) > 0
    else 0
)

_full_inactive_rate = (
    _full_inactive["Exited"].mean() * 100
    if len(_full_inactive) > 0
    else 0
)

_full_engagement_gap = (
    _full_inactive_rate - _full_active_rate
)


# ---------------------------------------------------------
# 3. Full-dataset geographic risk
# ---------------------------------------------------------
_full_geo = (
    _full_df.groupby("Geography", observed=True)
    .agg(
        Customers=("Exited", "count"),
        Churned=("Exited", "sum"),
        ChurnRate=("Exited", "mean"),
    )
    .reset_index()
)

if len(_full_geo) > 0:

    _full_geo["ChurnRate"] = (
        _full_geo["ChurnRate"] * 100
    )

    _full_geo["RiskIndex"] = (
        _full_geo["ChurnRate"] / _full_churn_rate
        if _full_churn_rate > 0
        else 0
    )

    _full_top_geo = (
        _full_geo
        .sort_values("ChurnRate", ascending=False)
        .iloc[0]
    )

else:
    _full_top_geo = None


# ---------------------------------------------------------
# 4. Full-dataset highest-risk segment
# ---------------------------------------------------------
_segment_candidates = []

for _dimension in SEGMENT_DIMS:

    _tmp_seg = (
        _full_df.groupby(_dimension, observed=True)
        .agg(
            Customers=("Exited", "count"),
            Churned=("Exited", "sum"),
        )
        .reset_index()
    )

    if len(_tmp_seg) == 0:
        continue

    _tmp_seg["ChurnRate"] = (
        _tmp_seg["Churned"]
        / _tmp_seg["Customers"]
        * 100
    )

    for _, _row in _tmp_seg.iterrows():

        # Avoid treating extremely small groups as automatically
        # important because of unstable percentages.
        if _row["Customers"] >= 50:

            _segment_candidates.append(
                {
                    "Dimension": _dimension,
                    "Value": _row[_dimension],
                    "Customers": int(_row["Customers"]),
                    "Churned": int(_row["Churned"]),
                    "ChurnRate": float(_row["ChurnRate"]),
                }
            )

if _segment_candidates:

    _full_segment_df = pd.DataFrame(
        _segment_candidates
    )

    _full_top_segment = (
        _full_segment_df
        .sort_values("ChurnRate", ascending=False)
        .iloc[0]
    )

else:
    _full_top_segment = None


# ---------------------------------------------------------
# 5. Full-dataset high-value customer analysis
# ---------------------------------------------------------
_full_hv = _full_df[
    _full_df["HighValueCustomer"] == "High-Value"
]

_full_hv_churn_rate = (
    _full_hv["Exited"].mean() * 100
    if len(_full_hv) > 0
    else 0
)

_full_standard = _full_df[
    _full_df["HighValueCustomer"] == "Standard"
]

_full_standard_churn_rate = (
    _full_standard["Exited"].mean() * 100
    if len(_full_standard) > 0
    else 0
)

_full_hv_gap = (
    _full_hv_churn_rate
    - _full_standard_churn_rate
)


# ---------------------------------------------------------
# 6. Dynamic interpretation of overall churn severity
# ---------------------------------------------------------
if _full_churn_rate >= 30:

    _full_severity = (
        "a high level of customer attrition requiring "
        "<b>strong retention attention</b>"
    )

elif _full_churn_rate >= 20:

    _full_severity = (
        "a substantial level of customer attrition warranting "
        "<b>focused retention attention</b>"
    )

elif _full_churn_rate >= 15:

    _full_severity = (
        "a meaningful level of customer attrition that should be "
        "<b>actively monitored and addressed</b>"
    )

elif _full_churn_rate >= 10:

    _full_severity = (
        "a moderate level of customer attrition requiring "
        "<b>continued monitoring</b>"
    )

else:

    _full_severity = (
        "a comparatively lower level of customer attrition, "
        "although retention monitoring remains important"
    )


# ---------------------------------------------------------
# 7. Dynamic engagement interpretation
# ---------------------------------------------------------
if _full_engagement_gap > 0:

    _engagement_direction = (
        f"Inactive customers show a higher churn rate than Active "
        f"customers by <b>{_full_engagement_gap:.1f} percentage points</b>."
    )

elif _full_engagement_gap < 0:

    _engagement_direction = (
        f"Active customers show a higher observed churn rate than "
        f"Inactive customers by <b>{abs(_full_engagement_gap):.1f} "
        f"percentage points</b>."
    )

else:

    _engagement_direction = (
        "Active and Inactive customers show the same observed churn rate."
    )


# ---------------------------------------------------------
# 8. Dynamic geographic interpretation
# ---------------------------------------------------------
if _full_top_geo is not None:

    _geo_risk_multiple = (
        _full_top_geo["ChurnRate"] / _full_churn_rate
        if _full_churn_rate > 0
        else 0
    )

    _geo_clause = (
        f"<b>{_full_top_geo['Geography']}</b> records the highest "
        f"observed geographic churn rate at "
        f"<b>{_full_top_geo['ChurnRate']:.1f}%</b>, approximately "
        f"<b>{_geo_risk_multiple:.1f}×</b> the portfolio-wide churn rate."
    )

else:

    _geo_clause = (
        "A highest-risk geography cannot be determined from the "
        "available portfolio data."
    )


# ---------------------------------------------------------
# 9. Dynamic segment interpretation
# ---------------------------------------------------------
if _full_top_segment is not None:

    _segment_multiple = (
        _full_top_segment["ChurnRate"] / _full_churn_rate
        if _full_churn_rate > 0
        else 0
    )

    _segment_clause = (
        f"The highest-risk identifiable segment is "
        f"<b>{_full_top_segment['Dimension']} = "
        f"{_full_top_segment['Value']}</b>, with a churn rate of "
        f"<b>{_full_top_segment['ChurnRate']:.1f}%</b> across "
        f"<b>{_full_top_segment['Customers']:,}</b> customers. "
        f"This is approximately <b>{_segment_multiple:.1f}×</b> "
        f"the portfolio-wide churn rate."
    )

else:

    _segment_clause = (
        "No sufficiently sized customer segment was available "
        "to establish a stable highest-risk segment."
    )


# ---------------------------------------------------------
# 10. Dynamic high-value interpretation
# ---------------------------------------------------------
if len(_full_hv) > 0 and len(_full_standard) > 0:

    if _full_hv_gap > 0:

        _hv_clause = (
            f"High-value customers show a churn rate of "
            f"<b>{_full_hv_churn_rate:.1f}%</b>, which is "
            f"<b>{_full_hv_gap:.1f} percentage points higher</b> "
            f"than the { _full_standard_churn_rate:.1f}% rate among "
            f"standard customers."
        )

    elif _full_hv_gap < 0:

        _hv_clause = (
            f"High-value customers show a churn rate of "
            f"<b>{_full_hv_churn_rate:.1f}%</b>, which is "
            f"<b>{abs(_full_hv_gap):.1f} percentage points lower</b> "
            f"than the { _full_standard_churn_rate:.1f}% rate among "
            f"standard customers."
        )

    else:

        _hv_clause = (
            f"High-value and standard customers show approximately "
            f"the same churn rate at <b>{_full_hv_churn_rate:.1f}%</b>."
        )

else:

    _hv_clause = (
        "There is insufficient high-value versus standard customer "
        "coverage to establish a reliable comparison."
    )


# # =========================================================
# # SINGLE KEY FINDING FOR THE ENTIRE DATASET
# # =========================================================

# finding_box(
#     "01",
#     "Overall Customer Churn — Full Portfolio Key Finding",

#     (
#         f"<ol>"

#         f"<li>"
#         f"<b>1. Overall portfolio churn position:</b> "
#         f"The full portfolio contains <b>{_full_total_customers:,}</b> "
#         f"customers, of whom <b>{_full_churned:,}</b> have churned. "
#         f"This produces an overall churn rate of "
#         f"<b>{_full_churn_rate:.1f}%</b>, representing "
#         f"{_full_severity}."
#         f"</li>"

#         f"<li>"
#         f"<b>2. Retention remains the larger part of the portfolio:</b> "
#         f"<b>{_full_retained:,}</b> customers remain retained, equivalent "
#         f"to <b>{_full_retention_rate:.1f}%</b> of the full customer base. "
#         f"For every 1 customer who has churned, approximately "
#         f"<b>{_full_retained_to_churned:.1f}</b> customers remain retained. "
#         f"This means the bank still has a substantial retained customer "
#         f"base, but the existing churn level represents a meaningful "
#         f"retention opportunity."
#         f"</li>"

#         f"<li>"
#         f"<b>3. Customer engagement is an important portfolio-level "
#         f"signal:</b> "
#         f"Active customers have an observed churn rate of "
#         f"<b>{_full_active_rate:.1f}%</b>, compared with "
#         f"<b>{_full_inactive_rate:.1f}%</b> among Inactive customers. "
#         f"{_engagement_direction} "
#         f"This indicates that customer engagement status should be "
#         f"considered when identifying customers requiring retention "
#         f"attention, while recognising that this analysis shows an "
#         f"association rather than proving causation."
#         f"</li>"

#         f"<li>"
#         f"<b>4. Churn is not necessarily distributed evenly across "
#         f"geographies:</b> "
#         f"{_geo_clause} "
#         f"The geographic result establishes where the bank has the "
#         f"strongest observed regional retention exposure and provides "
#         f"a useful baseline for deeper geographic analysis."
#         f"</li>"

#         f"<li>"
#         f"<b>5. Customer risk is concentrated in identifiable "
#         f"segments:</b> "
#         f"{_segment_clause} "
#         f"This demonstrates why the overall churn percentage alone is "
#         f"not sufficient for management action; the portfolio-wide "
#         f"average should be used as a benchmark for identifying "
#         f"higher-risk customer groups."
#         f"</li>"

#         f"<li>"
#         f"<b>6. High-value customer retention requires separate "
#         f"attention:</b> "
#         f"{_hv_clause} "
#         f"Because the high-value definition is based on the full "
#         f"portfolio's top 25% by Balance or Estimated Salary, this "
#         f"comparison should be treated as a portfolio-level indicator "
#         f"of premium-customer retention exposure."
#         f"</li>"

#         f"</ol>"
#     ),

#     (
#         f"<b>Overall interpretation:</b> "
#         f"The full European banking portfolio shows an observed churn "
#         f"rate of <b>{_full_churn_rate:.1f}%</b> across "
#         f"<b>{_full_total_customers:,}</b> customers. "
#         f"The most important management insight is that churn should "
#         f"not be viewed as a single portfolio-wide percentage. "
#         f"The overall rate provides the baseline, while engagement, "
#         f"geography, customer segmentation and high-value status reveal "
#         f"where retention exposure is more concentrated. "
#         f"The analysis therefore supports a <b>targeted retention "
#         f"strategy</b> rather than applying the same intervention to "
#         f"every customer."
#     ),

#     (
#         f"<b>Recommended management action:</b> "
#         f"Use the full-portfolio churn rate as the benchmark for all "
#         f"subsequent dashboard analysis. Prioritize retention analysis "
#         f"where observed churn is materially above this baseline, "
#         f"particularly across high-risk segments, higher-risk "
#         f"geographies, less-engaged customers and financially important "
#         f"high-value customers. "
#         f"Use the later segmentation and predictive-risk tabs to move "
#         f"from portfolio-level diagnosis to customer-level prioritisation."
#     ),
# )


# key_finding_banner(
#     "Executive KPI Summary",
#     f"The portfolio's five headline indicators, taken together, point to a churn profile that is "
#     f"concentrated rather than uniform: <b>{top_geo['Geography'] if top_geo is not None else 'the highest-risk region'}</b> "
#     f"and the engagement gap of <b>{engagement_gap:.1f} points</b> are the two indicators most deserving of "
#     f"immediate management attention, against an overall churn rate of <b>{overall_rate:.1f}%</b>.",
# )


# key_finding_banner(
#     "Portfolio Overview",
#     f"The portfolio retains <b>{100 - overall_rate:.1f}%</b> of customers overall, with attrition concentrated "
#     f"among inactive customers and most pronounced in <b>{_widest['Metric']}</b>, the profile field showing the "
#     f"widest divergence between churned and retained customers at <b>{abs(_widest['Difference (%)']):.1f}%</b>.",
# )


# key_finding_banner(
#     "Segmentation Explorer",
#     f"Within the <b>{dim_choice}</b> segmentation, risk and volume "
#     f"{'converge on a single segment, sharpening the case for immediate targeted action' if _rate_sorted.iloc[0][dim_choice] == _contrib_sorted.iloc[0][dim_choice] else 'diverge across two distinct segments, requiring separate risk-based and volume-based prioritisation'}. "
#     f"The highest-risk segment, <b>{_rate_sorted.iloc[0][dim_choice]}</b>, churns at <b>{_rate_sorted.iloc[0]['Churn_Rate_%']:.1f}%</b>, "
#     f"against a portfolio-wide baseline of {overall_rate:.1f}%.",
# )


# key_finding_banner(
#     "High-Value Customer Review",
#     f"Premium customers churn at <b>{hv_churn_rate:.1f}%</b>, {abs(_hv_gap):.1f} points {_hv_direction} than "
#     f"standard customers, exposing the portfolio to <b>\u20ac{revenue_at_risk:,.0f}</b> in already-realised "
#     f"balance loss. This concentration justifies dedicated retention coverage distinct from the general programme.",
# )




# =========================================================
# AGE & TENURE — FULL PORTFOLIO EXECUTIVE FINDING
# Uses df_raw so the finding is NOT affected by filters
# =========================================================

_age_tenure_full = df_raw.copy()

if len(_age_tenure_full) > 0:

    # -----------------------------------------------------
    # 1. Full portfolio churn rate
    # -----------------------------------------------------

    _at_total_customers = len(_age_tenure_full)

    _at_total_churned = int(
        _age_tenure_full["Exited"].sum()
    )

    _at_overall_rate = (
        _at_total_churned
        / _at_total_customers
        * 100
        if _at_total_customers > 0
        else 0
    )


    # =====================================================
    # 2. AGE-BASED CHURN PROFILE
    # =====================================================

    _age_full = (
        _age_tenure_full
        .groupby("AgeGroup", observed=True)
        .agg(
            Customers=("Exited", "count"),
            Churned=("Exited", "sum")
        )
        .reset_index()
    )

    _age_full["Churn_Rate"] = (
        _age_full["Churned"]
        / _age_full["Customers"]
        * 100
    )

    _age_full = _age_full.sort_values(
        "Churn_Rate",
        ascending=False
    ).reset_index(drop=True)

    _age_high_full = _age_full.iloc[0]
    _age_low_full = _age_full.iloc[-1]

    _age_spread_full = (
        _age_high_full["Churn_Rate"]
        - _age_low_full["Churn_Rate"]
    )

    _age_high_vs_overall_full = (
        _age_high_full["Churn_Rate"]
        - _at_overall_rate
    )


    if _age_spread_full >= 10:

        _age_pattern_full = (
            f"Age groups show a substantial observed churn "
            f"difference of <b>{_age_spread_full:.1f} "
            f"percentage points</b>."
        )

    elif _age_spread_full >= 5:

        _age_pattern_full = (
            f"Age groups show a noticeable observed churn "
            f"difference of <b>{_age_spread_full:.1f} "
            f"percentage points</b>."
        )

    else:

        _age_pattern_full = (
            f"Churn rates remain relatively close across age "
            f"groups, with a spread of only "
            f"<b>{_age_spread_full:.1f} percentage points</b>."
        )


    # =====================================================
    # 3. TENURE-STAGE CHURN PROFILE
    # =====================================================

    _tenure_full = (
        _age_tenure_full
        .groupby("TenureGroup", observed=True)
        .agg(
            Customers=("Exited", "count"),
            Churned=("Exited", "sum"),
            ActiveCount=(
                "ActivityStatus",
                lambda s: (s == "Active").sum()
            )
        )
        .reset_index()
    )

    _tenure_full["Churn_Rate"] = (
        _tenure_full["Churned"]
        / _tenure_full["Customers"]
        * 100
    )

    _tenure_full["Active_Rate"] = (
        _tenure_full["ActiveCount"]
        / _tenure_full["Customers"]
        * 100
    )

    _tenure_full = _tenure_full.sort_values(
        "Churn_Rate",
        ascending=False
    ).reset_index(drop=True)

    _tenure_high_full = _tenure_full.iloc[0]
    _tenure_low_full = _tenure_full.iloc[-1]

    _tenure_spread_full = (
        _tenure_high_full["Churn_Rate"]
        - _tenure_low_full["Churn_Rate"]
    )

    _tenure_high_vs_overall_full = (
        _tenure_high_full["Churn_Rate"]
        - _at_overall_rate
    )


    # =====================================================
    # 4. LIFECYCLE ENGAGEMENT
    # =====================================================

    _new_full = _tenure_full[
        _tenure_full["TenureGroup"]
        == "New (0-2yr)"
    ]

    _long_full = _tenure_full[
        _tenure_full["TenureGroup"]
        == "Long-term (7-10yr)"
    ]


    if len(_new_full) > 0 and len(_long_full) > 0:

        _new_active_full = float(
            _new_full["Active_Rate"].iloc[0]
        )

        _long_active_full = float(
            _long_full["Active_Rate"].iloc[0]
        )

        _engagement_change_full = (
            _long_active_full
            - _new_active_full
        )

        if _engagement_change_full < 0:

            _engagement_direction_full = "declines"

        elif _engagement_change_full > 0:

            _engagement_direction_full = "increases"

        else:

            _engagement_direction_full = (
                "remains broadly stable"
            )

    else:

        _new_active_full = 0
        _long_active_full = 0
        _engagement_change_full = 0

        _engagement_direction_full = (
            "is not determinable"
        )


    # =====================================================
    # 5. TENURE × ACTIVITY INTERACTION
    # =====================================================

    _interaction_full = (
        _age_tenure_full
        .groupby(
            ["TenureGroup", "ActivityStatus"],
            observed=True
        )["Exited"]
        .mean()
        .mul(100)
        .reset_index()
    )

    _interaction_full.columns = [
        "TenureGroup",
        "ActivityStatus",
        "Churn_Rate"
    ]

    _interaction_full = _interaction_full.sort_values(
        "Churn_Rate",
        ascending=False
    ).reset_index(drop=True)

    _interaction_high_full = (
        _interaction_full.iloc[0]
    )

    _interaction_low_full = (
        _interaction_full.iloc[-1]
    )

    _interaction_spread_full = (
        _interaction_high_full["Churn_Rate"]
        - _interaction_low_full["Churn_Rate"]
    )


    # =====================================================
    # 6. EXACT TENURE YEAR ANALYSIS
    # =====================================================

    _year_full = (
        _age_tenure_full
        .groupby("Tenure", observed=True)["Exited"]
        .mean()
        .mul(100)
        .reset_index()
    )

    _year_full.columns = [
        "Tenure_Years",
        "Churn_Rate"
    ]

    _year_full = _year_full.sort_values(
        "Churn_Rate",
        ascending=False
    ).reset_index(drop=True)

    _peak_year_full = _year_full.iloc[0]
    _low_year_full = _year_full.iloc[-1]

    _peak_year_number_full = int(
        _peak_year_full["Tenure_Years"]
    )

    _peak_year_rate_full = float(
        _peak_year_full["Churn_Rate"]
    )

    _low_year_number_full = int(
        _low_year_full["Tenure_Years"]
    )

    _low_year_rate_full = float(
        _low_year_full["Churn_Rate"]
    )

    _year_spread_full = (
        _peak_year_rate_full
        - _low_year_rate_full
    )

    _peak_year_vs_overall_full = (
        _peak_year_rate_full
        - _at_overall_rate
    )


    # =====================================================
    # 7. LIFECYCLE INTERPRETATION
    # =====================================================

    if _peak_year_number_full <= 2:

        _lifecycle_full = (
            f"The strongest observed tenure signal occurs "
            f"early in the relationship. Tenure year "
            f"<b>{_peak_year_number_full}</b> records the "
            f"highest churn rate at "
            f"<b>{_peak_year_rate_full:.1f}%</b>. "
            f"Early relationship experience and onboarding "
            f"therefore deserve closer retention monitoring."
        )

    elif _peak_year_number_full <= 4:

        _lifecycle_full = (
            f"The strongest observed tenure signal occurs "
            f"during the early-to-mid relationship period. "
            f"Tenure year <b>{_peak_year_number_full}</b> "
            f"records the highest churn rate at "
            f"<b>{_peak_year_rate_full:.1f}%</b>. "
            f"Retention monitoring should therefore continue "
            f"beyond the initial onboarding period."
        )

    else:

        _lifecycle_full = (
            f"The strongest observed tenure signal occurs "
            f"later in the customer lifecycle. Tenure year "
            f"<b>{_peak_year_number_full}</b> records the "
            f"highest churn rate at "
            f"<b>{_peak_year_rate_full:.1f}%</b>. "
            f"Longer-standing customers therefore also "
            f"deserve targeted re-engagement monitoring."
        )


    # =====================================================
    # 8. FINAL FULL-PORTFOLIO FINDING
    # =====================================================




    



    
#     finding_box(
#         "08",
#         "Age, Tenure and Customer Lifecycle — Full Portfolio Key Findings",

#         f"""
#         <ol>

#         <li>
#         <b>Age-based churn risk:</b>
#         Across the complete portfolio of
#         <b>{_at_total_customers:,}</b> customers,
#         <b>{_age_high_full["AgeGroup"]}</b> has the highest
#         observed age-group churn rate at
#         <b>{_age_high_full["Churn_Rate"]:.1f}%</b>, while
#         <b>{_age_low_full["AgeGroup"]}</b> has the lowest at
#         <b>{_age_low_full["Churn_Rate"]:.1f}%</b>.
#         The difference is
#         <b>{_age_spread_full:.1f} percentage points</b>.
#         {_age_pattern_full}
#         </li>


#         <li>
#         <b>Tenure-stage risk:</b>
#         <b>{_tenure_high_full["TenureGroup"]}</b> records the
#         highest grouped tenure churn rate at
#         <b>{_tenure_high_full["Churn_Rate"]:.1f}%</b>,
#         compared with
#         <b>{_tenure_low_full["TenureGroup"]}</b> at
#         <b>{_tenure_low_full["Churn_Rate"]:.1f}%</b>.
#         The tenure-stage spread is
#         <b>{_tenure_spread_full:.1f} percentage points</b>.
#         The highest-risk tenure group is
#         <b>{abs(_tenure_high_vs_overall_full):.1f} points
#         {'above' if _tenure_high_vs_overall_full >= 0 else 'below'}</b>
#         the full-portfolio churn rate of
#         <b>{_at_overall_rate:.1f}%</b>.
#         </li>


#         <li>
#         <b>Lifecycle engagement:</b>
#         The active-customer rate
#         <b>{_engagement_direction_full}</b> from
#         <b>{_new_active_full:.1f}%</b> among New
#         (0–2 year) customers to
#         <b>{_long_active_full:.1f}%</b> among Long-term
#         (7–10 year) customers, a difference of
#         <b>{abs(_engagement_change_full):.1f} percentage points</b>.
#         This shows whether customer engagement is being maintained
#         as relationships mature.
#         </li>


#         <li>
#         <b>Tenure × activity risk:</b>
#         The
#         <b>{_interaction_high_full["TenureGroup"]} /
#         {_interaction_high_full["ActivityStatus"]}</b>
#         combination has the highest observed churn rate at
#         <b>{_interaction_high_full["Churn_Rate"]:.1f}%</b>,
#         while
#         <b>{_interaction_low_full["TenureGroup"]} /
#         {_interaction_low_full["ActivityStatus"]}</b>
#         has the lowest at
#         <b>{_interaction_low_full["Churn_Rate"]:.1f}%</b>.
#         The interaction spread is
#         <b>{_interaction_spread_full:.1f} percentage points</b>,
#         showing that tenure and customer activity are more
#         informative when reviewed together rather than separately.
#         </li>


#         <li>
#         <b>Exact tenure-year signal:</b>
#         Tenure year <b>{_peak_year_number_full}</b> records the
#         highest individual year-level churn rate at
#         <b>{_peak_year_rate_full:.1f}%</b>, while year
#         <b>{_low_year_number_full}</b> records the lowest at
#         <b>{_low_year_rate_full:.1f}%</b>.
#         The year-level spread is
#         <b>{_year_spread_full:.1f} percentage points</b>.
#         {_lifecycle_full}
#         </li>


#         <li>
#         <b>Management implication:</b>
#         The full-portfolio analysis shows that churn is not evenly
#         distributed across the customer lifecycle. Retention
#         monitoring should therefore be prioritised around the
#         highest-risk age group, tenure stage and tenure/activity
#         combination. The exact tenure-year pattern can help identify
#         when additional engagement should be introduced.
#         These findings represent observed associations and should be
#         combined with product usage, geography, financial and
#         behavioural indicators before making individual customer
#         retention decisions.
#         </li>

#         </ol>
#         """,

#         f"""
#         <b>Overall interpretation:</b>
#         The full-portfolio Age & Tenure analysis shows that
#         customer churn varies across <b>age, relationship tenure,
#         customer activity and specific tenure years</b>.
#         The results therefore support a lifecycle-based retention
#         strategy rather than applying the same intervention to every
#         customer. The strongest observed patterns should be treated
#         as indicators for further investigation, not as evidence that
#         age or tenure directly causes churn.
#         """
#     )

# else:

#     finding_box(
#         "08",
#         "Age, Tenure and Customer Lifecycle — Full Portfolio Key Findings",
#         "Insufficient full-portfolio data is available to produce the age and tenure finding."
#     )

# ============================================================
# TAB VI — FULL PORTFOLIO PREDICTIVE MODEL KEY FINDING
# Uses df_raw so this executive finding is NOT affected
# by sidebar filters.
# ============================================================

# ------------------------------------------------------------
# 1. MODEL PERFORMANCE — HELD-OUT TEST SET
# ------------------------------------------------------------
_pf_accuracy = m["Accuracy"] * 100
_pf_precision = m["Precision"] * 100
_pf_recall = m["Recall"] * 100
_pf_f1 = m["F1"]
_pf_auc = m["ROC-AUC"]


# ------------------------------------------------------------
# 2. FEATURE IMPORTANCE — FULL MODEL INTERPRETATION
# ------------------------------------------------------------
_pf_imp = _churn_model_bundle["importances"].reset_index()
_pf_imp.columns = ["Feature", "Importance"]
_pf_imp["Importance"] = _pf_imp["Importance"] * 100

_pf_imp_sorted = (
    _pf_imp
    .sort_values("Importance", ascending=False)
    .reset_index(drop=True)
)

_pf_top_feature = _pf_imp_sorted.iloc[0]["Feature"]
_pf_top_feature_importance = _pf_imp_sorted.iloc[0]["Importance"]

_pf_second_feature = _pf_imp_sorted.iloc[1]["Feature"]
_pf_second_feature_importance = _pf_imp_sorted.iloc[1]["Importance"]

_pf_top3_importance = _pf_imp_sorted.head(3)["Importance"].sum()


# ------------------------------------------------------------
# 3. SCORE THE ENTIRE UNFILTERED PORTFOLIO
# ------------------------------------------------------------
_pf_scored = score_churn_risk(
    df_raw.copy(),
    _churn_model_bundle
)


# ------------------------------------------------------------
# 4. PORTFOLIO-WIDE RISK BUCKET SUMMARY
# ------------------------------------------------------------
_pf_risk_summary = (
    _pf_scored
    .groupby("RiskBucket", observed=True)
    .agg(
        Customers=("ChurnRiskScore", "count"),
        ActualChurnRate=("Exited", "mean"),
    )
    .reset_index()
)

_pf_risk_summary["ActualChurnRate"] = (
    _pf_risk_summary["ActualChurnRate"] * 100
)

_pf_high = _pf_risk_summary[
    _pf_risk_summary["RiskBucket"] == "High"
]

_pf_medium = _pf_risk_summary[
    _pf_risk_summary["RiskBucket"] == "Medium"
]

_pf_low = _pf_risk_summary[
    _pf_risk_summary["RiskBucket"] == "Low"
]


_pf_high_count = (
    int(_pf_high["Customers"].iloc[0])
    if len(_pf_high) > 0 else 0
)

_pf_medium_count = (
    int(_pf_medium["Customers"].iloc[0])
    if len(_pf_medium) > 0 else 0
)

_pf_low_count = (
    int(_pf_low["Customers"].iloc[0])
    if len(_pf_low) > 0 else 0
)


_pf_high_actual = (
    float(_pf_high["ActualChurnRate"].iloc[0])
    if len(_pf_high) > 0 else 0
)

_pf_low_actual = (
    float(_pf_low["ActualChurnRate"].iloc[0])
    if len(_pf_low) > 0 else 0
)

_pf_medium_actual = (
    float(_pf_medium["ActualChurnRate"].iloc[0])
    if len(_pf_medium) > 0 else 0
)


# ------------------------------------------------------------
# 5. FULL PORTFOLIO OVERALL CHURN RATE
# ------------------------------------------------------------
_pf_overall_churn = (
    df_raw["Exited"].mean() * 100
)


# ------------------------------------------------------------
# 6. RISK SEPARATION
# ------------------------------------------------------------
_pf_risk_gap = _pf_high_actual - _pf_low_actual


if _pf_overall_churn > 0 and _pf_high_count > 0:

    _pf_high_multiple = (
        _pf_high_actual / _pf_overall_churn
    )

    _pf_high_vs_overall_gap = (
        _pf_high_actual - _pf_overall_churn
    )

else:

    _pf_high_multiple = 0
    _pf_high_vs_overall_gap = 0


# ------------------------------------------------------------
# 7. MODEL INTERPRETATION
# ------------------------------------------------------------
if _pf_auc >= 0.80:

    _pf_auc_text = (
        f"The model shows strong discrimination, with a "
        f"ROC-AUC of <b>{_pf_auc:.3f}</b>, indicating good ability "
        f"to distinguish customers with higher observed churn risk "
        f"from those with lower observed risk."
    )

elif _pf_auc >= 0.70:

    _pf_auc_text = (
        f"The model shows acceptable discrimination, with a "
        f"ROC-AUC of <b>{_pf_auc:.3f}</b>, indicating useful ability "
        f"to distinguish customers with higher and lower observed "
        f"churn risk."
    )

else:

    _pf_auc_text = (
        f"The model records a ROC-AUC of <b>{_pf_auc:.3f}</b>; "
        f"its ability to distinguish higher- and lower-risk "
        f"customers should therefore be monitored carefully."
    )


# ------------------------------------------------------------
# 8. RISK-SEPARATION INTERPRETATION
# ------------------------------------------------------------
if (
    _pf_high_count > 0
    and _pf_low_count > 0
):

    if _pf_high_actual > _pf_low_actual:

        _pf_separation_text = (
            f"The High Risk group shows an observed churn rate of "
            f"<b>{_pf_high_actual:.1f}%</b>, compared with "
            f"<b>{_pf_low_actual:.1f}%</b> for the Low Risk group — "
            f"a separation of <b>{_pf_risk_gap:.1f} percentage points</b>."
        )

    else:

        _pf_separation_text = (
            f"The High Risk group does not show a higher observed "
            f"churn rate than the Low Risk group in the full portfolio, "
            f"so risk-bucket separation should be interpreted cautiously."
        )

else:

    _pf_separation_text = (
        "There is insufficient representation across the High and "
        "Low Risk buckets to assess portfolio-wide risk separation."
    )


# ------------------------------------------------------------
# 9. TOP 25 RISK PRIORITY
# ------------------------------------------------------------
_pf_top25 = (
    _pf_scored
    .sort_values("ChurnRiskScore", ascending=False)
    .head(25)
)

_pf_top25_avg_score = (
    _pf_top25["ChurnRiskScore"].mean()
    if len(_pf_top25) > 0 else 0
)


# # ------------------------------------------------------------
# # 10. EXECUTIVE KEY FINDING
# # ------------------------------------------------------------
# finding_box(
#     "06",
#     "Predictive Churn Risk Model — Full Portfolio Key Findings",

#     (
#         f"<ol>"

#         f"<li><b>Model performance:</b> "
#         f"{_pf_auc_text} "
#         f"The model records <b>{_pf_precision:.1f}% precision</b> "
#         f"and <b>{_pf_recall:.1f}% recall</b> on the held-out test set. "
#         f"These measures provide a more useful view of churn-detection "
#         f"performance than accuracy alone.</li>"

#         f"<li><b>What drives predicted risk:</b> "
#         f"The Random Forest interpretation identifies "
#         f"<b>{_pf_top_feature}</b> as the most important feature at "
#         f"<b>{_pf_top_feature_importance:.1f}%</b> of total feature "
#         f"importance, followed by <b>{_pf_second_feature}</b> at "
#         f"<b>{_pf_second_feature_importance:.1f}%</b>. "
#         f"The top three features together account for "
#         f"<b>{_pf_top3_importance:.1f}%</b> of total importance. "
#         f"These values indicate model reliance, not that the features "
#         f"independently cause churn.</li>"

#         f"<li><b>Portfolio-wide risk distribution:</b> "
#         f"Across the full unfiltered portfolio, the model classifies "
#         f"<b>{_pf_high_count:,}</b> customers as High Risk, "
#         f"<b>{_pf_medium_count:,}</b> as Medium Risk and "
#         f"<b>{_pf_low_count:,}</b> as Low Risk. "
#         f"This converts the broad churn analysis into a customer-level "
#         f"risk-prioritisation framework.</li>"

#         f"<li><b>Risk concentration:</b> "
#         f"{_pf_separation_text} "
#         f"The full portfolio churn rate is "
#         f"<b>{_pf_overall_churn:.1f}%</b>. "
#         f"{(
#             f"The High Risk group is "
#             f"<b>{_pf_high_vs_overall_gap:.1f} percentage points</b> "
#             f"above the portfolio average, representing approximately "
#             f"<b>{_pf_high_multiple:.1f}×</b> the portfolio-wide churn rate."
#             if _pf_high_count > 0
#             and _pf_high_actual > _pf_overall_churn
#             else
#             "The High Risk group does not currently exceed the "
#             "portfolio-wide churn rate, so additional validation "
#             "should be used before treating it as a strong risk concentration."
#         )}</li>"

#         f"<li><b>Highest-priority customers:</b> "
#         f"The model ranks customers using a <b>ChurnRiskScore from "
#         f"0% to 100%</b>. The <b>Top 25 highest-risk customers</b> "
#         f"provide a focused review list for relationship managers. "
#         f"These customers should be treated as priority cases for "
#         f"retention review, not as automatic churn decisions.</li>"

#         f"<li><b>Management implication:</b> "
#         f"The predictive model moves the bank from "
#         f"<b>descriptive churn reporting to risk-based customer "
#         f"prioritisation</b>. Retention resources can be directed first "
#         f"towards customers with the strongest model-indicated risk, "
#         f"while Medium Risk customers can be addressed according to "
#         f"available retention capacity. Model predictions should be "
#         f"monitored against future actual churn outcomes and reviewed "
#         f"regularly as customer behaviour changes.</li>"

#         f"</ol>"
#     ),

#     (
#         f"<b>Overall interpretation:</b> "
#         f"The Predictive Churn Risk Model provides the final "
#         f"customer-level layer of the dashboard. Earlier tabs identify "
#         f"where churn is concentrated across geography, segments, "
#         f"customer value, age and tenure; this model combines multiple "
#         f"customer characteristics to estimate individual churn risk. "
#         f"{_pf_auc_text} "
#         f"The primary business value is therefore the ability to "
#         f"prioritise retention activity using a consistent, ranked "
#         f"risk score rather than treating every customer equally."
#     ),

#     (
#         f"<b>Recommended action:</b> "
#         f"Use <b>ChurnRiskScore</b> as a prioritisation tool for "
#         f"retention outreach. Start with the highest-risk customers, "
#         f"then expand to the Medium Risk group according to available "
#         f"resources. Track predicted risk against actual churn and "
#         f"periodically re-evaluate model performance. The model should "
#         f"support relationship-manager judgement rather than replace "
#         f"human decision-making."
#     ),
# )


# =========================================================
# KEY FINDINGS — FULL PORTFOLIO
# IMPORTANT:
# Every calculation in this section uses df_raw (the complete,
# unfiltered dataset) rather than the sidebar-filtered df used
# throughout the rest of the app. These four cards are meant to
# read as fixed, portfolio-wide facts about the whole bank's
# customer base — they must NOT change when the user adjusts the
# Filter Panel, unlike the per-tab Key Finding banners above,
# which deliberately do respond to filters.
# =========================================================

st.markdown('<div class="styled-divider"></div>', unsafe_allow_html=True)
st.markdown("### \U0001F4CB Key Findings \u2014 Full Portfolio")
st.caption(
    "The four findings below are calculated on the entire customer base "
    "(n={:,}) regardless of any filters currently applied above.".format(len(df_raw))
)

# ---------------------------------------------------------
# Key Findings — Engagement Analysis (whole portfolio)
# ---------------------------------------------------------
_kf_active_all = df_raw[df_raw["ActivityStatus"] == "Active"]
_kf_inactive_all = df_raw[df_raw["ActivityStatus"] == "Inactive"]
_kf_active_churn_all = (
    _kf_active_all["Exited"].mean() * 100 if len(_kf_active_all) > 0 else 0
)
_kf_inactive_churn_all = (
    _kf_inactive_all["Exited"].mean() * 100 if len(_kf_inactive_all) > 0 else 0
)
_kf_engagement_gap_all = _kf_inactive_churn_all - _kf_active_churn_all

st.markdown(
    f"""
<div class="finding-box">
<div class="finding-title">\U0001F4CB Key Findings \u2014 Engagement Analysis (Full Portfolio)</div>
\u26A0 Inactive customers churn at <strong>{_kf_inactive_churn_all:.1f}%</strong>
versus <strong>{_kf_active_churn_all:.1f}%</strong> for active customers, across the entire customer base.<br>
\U0001F4CA This represents a <strong>{magnitude_word(_kf_engagement_gap_all)}</strong>
gap of <strong>{_kf_engagement_gap_all:.1f} percentage points</strong>.<br>
\U0001F4A1 Engagement status {urgency_clause(_kf_engagement_gap_all)} at the full-portfolio level.
</div>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# Key Findings — Product Holding (whole portfolio)
# ---------------------------------------------------------
_kf_prod_all = (
    df_raw.groupby("NumOfProducts", observed=True)
    .agg(Customers=("Exited", "count"), ChurnRate=("Exited", lambda x: x.mean() * 100))
    .reset_index()
)

if len(_kf_prod_all) > 0:
    _kf_best_prod = _kf_prod_all.loc[_kf_prod_all["ChurnRate"].idxmin()]
    _kf_worst_prod = _kf_prod_all.loc[_kf_prod_all["ChurnRate"].idxmax()]
    _kf_prod_gap = _kf_worst_prod["ChurnRate"] - _kf_best_prod["ChurnRate"]

    st.markdown(
        f"""
<div class="finding-box">
<div class="finding-title">\U0001F4CB Key Findings \u2014 Product Holding (Full Portfolio)</div>
\u2705 Customers holding <strong>{int(_kf_best_prod['NumOfProducts'])}</strong> product(s)
churn least, at <strong>{_kf_best_prod['ChurnRate']:.1f}%</strong>, across the full customer base.<br>
\u26A0 Customers holding <strong>{int(_kf_worst_prod['NumOfProducts'])}</strong> product(s)
churn most, at <strong>{_kf_worst_prod['ChurnRate']:.1f}%</strong>.<br>
\U0001F4C8 The {_kf_prod_gap:.1f}-point spread is {magnitude_word(_kf_prod_gap)},
and {urgency_clause(_kf_prod_gap)} at the portfolio level.
</div>
""",
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------
# Key Findings — High-Value Customer Risk (whole portfolio)
# ---------------------------------------------------------
_kf_hv_all = df_raw[df_raw["HighValueCustomer"] == "High-Value"]
_kf_std_all = df_raw[df_raw["HighValueCustomer"] == "Standard"]
_kf_hv_count_all = len(_kf_hv_all)
_kf_hv_pct_all = (
    (_kf_hv_count_all / len(df_raw) * 100) if len(df_raw) > 0 else 0
)
_kf_hv_churn_all = (
    _kf_hv_all["Exited"].mean() * 100 if len(_kf_hv_all) > 0 else 0
)
_kf_std_churn_all = (
    _kf_std_all["Exited"].mean() * 100 if len(_kf_std_all) > 0 else 0
)
_kf_revenue_at_risk_all = (
    _kf_hv_all[_kf_hv_all["Exited"] == 1]["Balance"].sum()
    if len(_kf_hv_all) > 0 else 0
)
_kf_hv_gap_all = _kf_hv_churn_all - _kf_std_churn_all

st.markdown(
    f"""
<div class="finding-box">
<div class="finding-title">\U0001F4CB Key Findings \u2014 High-Value Customer Risk (Full Portfolio)</div>
\u26A0 <strong>{_kf_hv_count_all:,}</strong> high-value customers identified
(<strong>{_kf_hv_pct_all:.1f}%</strong> of the entire customer base).<br>
\U0001F4CA They churn at <strong>{_kf_hv_churn_all:.1f}%</strong> versus
<strong>{_kf_std_churn_all:.1f}%</strong> for standard customers
\u2014 a {magnitude_word(_kf_hv_gap_all)} differential.<br>
\U0001F4B0 Revenue already at risk from churned high-value customers: <strong>\u20ac{_kf_revenue_at_risk_all:,.0f}</strong>.
Dedicated retention coverage for this tier {urgency_clause(_kf_hv_gap_all)}.
</div>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# Key Findings — Geographic Risk (whole portfolio)
# ---------------------------------------------------------
_kf_geo_all = (
    df_raw.groupby("Geography", observed=True)
    .agg(Customers=("Exited", "count"), ChurnRate=("Exited", lambda x: x.mean() * 100))
    .reset_index()
)

if len(_kf_geo_all) > 0:
    _kf_best_geo = _kf_geo_all.loc[_kf_geo_all["ChurnRate"].idxmin()]
    _kf_worst_geo = _kf_geo_all.loc[_kf_geo_all["ChurnRate"].idxmax()]
    _kf_overall_rate_all = df_raw["Exited"].mean() * 100 if len(df_raw) > 0 else 0
    _kf_geo_index_all = (
        (_kf_worst_geo["ChurnRate"] / _kf_overall_rate_all)
        if _kf_overall_rate_all > 0 else 0
    )

    st.markdown(
        f"""
<div class="finding-box">
<div class="finding-title">\U0001F4CB Key Findings \u2014 Geographic Risk (Full Portfolio)</div>
\U0001F534 <strong>{_kf_worst_geo['Geography']}</strong> has the highest churn across the
entire customer base at <strong>{_kf_worst_geo['ChurnRate']:.1f}%</strong>.<br>
\U0001F7E2 <strong>{_kf_best_geo['Geography']}</strong> has the lowest, at
<strong>{_kf_best_geo['ChurnRate']:.1f}%</strong>.<br>
\U0001F4C8 {_kf_worst_geo['Geography']}'s Geographic Risk Index of
<strong>{_kf_geo_index_all:.2f}\u00d7</strong> the portfolio average
{urgency_clause((_kf_geo_index_all - 1) * 100, thresholds=(20, 50, 100))}.
</div>
""",
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------
# Key Findings — Predictive Churn Risk Model (whole portfolio)
# IMPORTANT: the model itself is always trained once on the full
# df_raw (see train_churn_model above) regardless of this section —
# what changes here is that we SCORE it against df_raw as well,
# rather than the sidebar-filtered df used in Tab VIII, so this
# card reports the model's full-portfolio risk concentration.
# ---------------------------------------------------------
_kf_scored_all = score_churn_risk(df_raw, _churn_model_bundle)
_kf_high_risk_all = _kf_scored_all[_kf_scored_all["RiskBucket"] == "High"]
_kf_low_risk_all = _kf_scored_all[_kf_scored_all["RiskBucket"] == "Low"]
_kf_high_risk_count_all = len(_kf_high_risk_all)
_kf_high_risk_pct_all = (
    (_kf_high_risk_count_all / len(df_raw) * 100) if len(df_raw) > 0 else 0
)
_kf_high_risk_actual_all = (
    _kf_high_risk_all["Exited"].mean() * 100 if len(_kf_high_risk_all) > 0 else 0
)
_kf_low_risk_actual_all = (
    _kf_low_risk_all["Exited"].mean() * 100 if len(_kf_low_risk_all) > 0 else 0
)
_kf_model_overall_rate = df_raw["Exited"].mean() * 100 if len(df_raw) > 0 else 0
_kf_model_multiple_all = (
    (_kf_high_risk_actual_all / _kf_model_overall_rate)
    if _kf_model_overall_rate > 0 else 0
)
_kf_model_auc = _churn_model_bundle["metrics"]["ROC-AUC"]

st.markdown(
    f"""
<div class="finding-box">
<div class="finding-title">\U0001F4CB Key Findings \u2014 Predictive Churn Risk Model (Full Portfolio)</div>
\U0001F916 The Gradient Boosting model (ROC-AUC <strong>{_kf_model_auc:.3f}</strong>) flags
<strong>{_kf_high_risk_count_all:,}</strong> customers as High Risk
(<strong>{_kf_high_risk_pct_all:.1f}%</strong> of the entire customer base).<br>
\U0001F4CA This group actually churns at <strong>{_kf_high_risk_actual_all:.1f}%</strong>,
versus <strong>{_kf_low_risk_actual_all:.1f}%</strong> for customers flagged Low Risk
\u2014 a <strong>{_kf_model_multiple_all:.1f}\u00d7</strong> concentration of risk relative to the
{_kf_model_overall_rate:.1f}% portfolio average.<br>
\U0001F4A1 This concentration {urgency_clause((_kf_model_multiple_all - 1) * 100, thresholds=(20, 50, 100))}
and should anchor the retention team's prioritised outreach list, ranked by ChurnRiskScore.
</div>
""",
    unsafe_allow_html=True,
)












# =========================================================
# EXECUTIVE SUMMARY — FULL PORTFOLIO
# IMPORTANT:
# Uses df_raw so this summary is NOT affected by sidebar filters
# =========================================================

_exec_df = df_raw.copy()

# ---------------------------------------------------------
# 1. Overall portfolio metrics
# ---------------------------------------------------------
_exec_total_customers = len(_exec_df)

_exec_churned = (
    int(_exec_df["Exited"].sum())
    if _exec_total_customers > 0
    else 0
)

_exec_retained = _exec_total_customers - _exec_churned

_exec_churn_rate = (
    (_exec_churned / _exec_total_customers) * 100
    if _exec_total_customers > 0
    else 0
)

_exec_retention_rate = (
    (_exec_retained / _exec_total_customers) * 100
    if _exec_total_customers > 0
    else 0
)


# ---------------------------------------------------------
# 2. Engagement risk gap
# ---------------------------------------------------------
_exec_active = _exec_df[
    _exec_df["ActivityStatus"] == "Active"
]

_exec_inactive = _exec_df[
    _exec_df["ActivityStatus"] == "Inactive"
]

_exec_active_churn = (
    _exec_active["Exited"].mean() * 100
    if len(_exec_active) > 0
    else 0
)

_exec_inactive_churn = (
    _exec_inactive["Exited"].mean() * 100
    if len(_exec_inactive) > 0
    else 0
)

_exec_engagement_gap = (
    _exec_inactive_churn - _exec_active_churn
)


# ---------------------------------------------------------
# 3. Highest-churn geography
# ---------------------------------------------------------
_exec_geo = (
    _exec_df
    .groupby("Geography", observed=True)
    .agg(
        Customers=("Exited", "count"),
        Churned=("Exited", "sum"),
        ChurnRate=("Exited", "mean")
    )
    .reset_index()
)

_exec_geo["ChurnRate"] = (
    _exec_geo["ChurnRate"] * 100
)

_exec_top_geo = (
    _exec_geo
    .sort_values("ChurnRate", ascending=False)
    .iloc[0]
    if len(_exec_geo) > 0
    else None
)


# ---------------------------------------------------------
# 4. Highest-risk customer segment
# ---------------------------------------------------------
_exec_segment_candidates = []

for _dim in SEGMENT_DIMS:

    _seg = (
        _exec_df
        .groupby(_dim, observed=True)
        .agg(
            Customers=("Exited", "count"),
            Churned=("Exited", "sum"),
            ChurnRate=("Exited", "mean")
        )
        .reset_index()
    )

    _seg["ChurnRate"] = _seg["ChurnRate"] * 100

    for _, _row in _seg.iterrows():

        # Minimum sample size prevents a tiny group
        # from dominating the executive summary.
        if _row["Customers"] >= 50:

            _exec_segment_candidates.append({
                "Dimension": _dim,
                "Value": _row[_dim],
                "Customers": int(_row["Customers"]),
                "Churned": int(_row["Churned"]),
                "ChurnRate": float(_row["ChurnRate"])
            })


if _exec_segment_candidates:

    _exec_top_segment = max(
        _exec_segment_candidates,
        key=lambda x: x["ChurnRate"]
    )

else:

    _exec_top_segment = {
        "Dimension": "N/A",
        "Value": "N/A",
        "Customers": 0,
        "Churned": 0,
        "ChurnRate": 0
    }


# ---------------------------------------------------------
# 5. Strategy message
# ---------------------------------------------------------
if _exec_engagement_gap >= 10:

    _exec_strategy = (
        "Prioritise re-engagement of inactive customers, "
        "with targeted retention actions for higher-risk "
        "customer segments."
    )

elif _exec_engagement_gap >= 5:

    _exec_strategy = (
        "Strengthen customer engagement and focus retention "
        "efforts on inactive and higher-risk customer groups."
    )

else:

    _exec_strategy = (
        "Maintain broad retention activity while using "
        "segment-level analysis to target customers with "
        "higher observed churn risk."
    )



# st.markdown(
#     f"""
#     <h3>📊 Executive Summary</h3>

#     <div class="exec-row">
#         📋 Total Customers Analyzed:
#         <strong>{_exec_total_customers:,}</strong>
#     </div>

#     <div class="exec-row">
#         📉 Overall Churn Rate:
#         <strong>{_exec_churn_rate:.1f}%</strong>
#     </div>

#     <div class="exec-row">
#         📈 Overall Retention Rate:
#         <strong>{_exec_retention_rate:.1f}%</strong>
#     </div>

#     <div class="exec-row">
#         ⚠ Engagement Risk Gap:
#         <strong>{_exec_engagement_gap:.1f} percentage points</strong>
#     </div>

#     <div class="exec-row">
#         🔴 Highest Churn Geography:
#         <strong>
#             {_exec_top_geo['Geography']}
#             ({_exec_top_geo['ChurnRate']:.1f}% churn)
#         </strong>
#     </div>

#     <div class="exec-row">
#         🎯 Highest-Risk Customer Segment:
#         <strong>
#             {_exec_top_segment['Dimension']} =
#             {_exec_top_segment['Value']}
#             ({_exec_top_segment['ChurnRate']:.1f}% churn)
#         </strong>
#     </div>

#     <div class="exec-strategy">
#         💡 <strong style="color:#FFFFFF;">Recommended Strategy:</strong>
#         {_exec_strategy}
#     </div>
#     """,
#     unsafe_allow_html=True,
# )



# st.markdown(
#     f"""
#     <div class="exec-summary-card">
#         <h3>📊 Executive Summary</h3>

#         <div class="exec-row">
#             📋 Total Customers Analyzed:
#             <strong>{_exec_total_customers:,}</strong>
#         </div>

#         <div class="exec-row">
#             📉 Overall Churn Rate:
#             <strong>{_exec_churn_rate:.1f}%</strong>
#         </div>

#         <div class="exec-row">
#             📈 Overall Retention Rate:
#             <strong>{_exec_retention_rate:.1f}%</strong>
#         </div>

#         <div class="exec-row">
#             ⚠ Engagement Risk Gap:
#             <strong>{_exec_engagement_gap:.1f} percentage points</strong>
#         </div>

#         <div class="exec-row alert">
#             🔴 Highest Churn Geography:
#             <strong>{_exec_top_geo['Geography']} ({_exec_top_geo['ChurnRate']:.1f}% churn)</strong>
#         </div>

#         <div class="exec-row alert">
#             🎯 Highest-Risk Customer Segment:
#             <strong>{_exec_top_segment['Dimension']} = {_exec_top_segment['Value']} ({_exec_top_segment['ChurnRate']:.1f}% churn)</strong>
#         </div>

#         <div class="exec-strategy">
#             💡 <strong style="color:#FFFFFF;">Recommended Strategy:</strong>
#             {_exec_strategy}
#         </div>
#     </div>
#     """,
#     unsafe_allow_html=True,
# )

# ====================================

def render_executive_summary(
    total_customers,
    churn_rate,
    retention_rate,
    engagement_gap,
    top_geo,
    top_segment,
    strategy_text,
    high_risk_count=None,
    high_risk_rate=None,
):
    """Renders the Executive Summary card, line by line -- one stat per row,
    matching the structural pattern used in sdashboard30.py (icon + label +
    bold value on a single line, stacked vertically), rendered in this
    project's own charcoal/burgundy/bronze palette rather than sdashboard30's
    navy/gold, consistent with the earlier decision to keep this project
    visually distinct from that ECB dashboard.
    Every line of the HTML string below starts at column 0 deliberately --
    do not re-indent this block to match the surrounding Python code, or
    Markdown will print it as literal text instead of rendering it."""
    _risk_line = ""
    if high_risk_count is not None and high_risk_rate is not None:
        _risk_line = (
            '<div class="exec-row alert">\U0001F916 Model-Flagged High Risk: '
            '<strong>{high_risk_count:,} customers ({high_risk_rate:.1f}% actual churn)</strong></div>'
        ).format(high_risk_count=high_risk_count, high_risk_rate=high_risk_rate)

    html = """
<div class="exec-summary-card">
<h3>\U0001F4CA Executive Summary</h3>
<div class="exec-row">\U0001F4CB Total Customers Analyzed: <strong>{total_customers:,}</strong></div>
<div class="exec-row">\U0001F4C9 Overall Churn Rate: <strong>{churn_rate:.1f}%</strong></div>
<div class="exec-row">\U0001F4C8 Overall Retention Rate: <strong>{retention_rate:.1f}%</strong></div>
<div class="exec-row">\u26A0\uFE0F Engagement Risk Gap: <strong>{engagement_gap:.1f} percentage points</strong></div>
<div class="exec-row alert">\U0001F534 Highest Churn Geography: <strong>{top_geo_name} ({top_geo_rate:.1f}% churn)</strong></div>
<div class="exec-row alert">\U0001F3AF Highest-Risk Customer Segment: <strong>{seg_dim} = {seg_val} ({seg_rate:.1f}% churn)</strong></div>
{risk_line}
<div class="exec-strategy">\U0001F4A1 <strong style="color:#FFFFFF;">Recommended Strategy:</strong> {strategy_text}</div>
</div>
""".format(
        total_customers=total_customers,
        churn_rate=churn_rate,
        retention_rate=retention_rate,
        engagement_gap=engagement_gap,
        top_geo_name=top_geo["Geography"],
        top_geo_rate=top_geo["ChurnRate"],
        seg_dim=top_segment["Dimension"],
        seg_val=top_segment["Value"],
        seg_rate=top_segment["ChurnRate"],
        strategy_text=strategy_text,
        risk_line=_risk_line,
    )
    st.markdown(html, unsafe_allow_html=True)

_exec_scored_df = score_churn_risk(df_raw, _churn_model_bundle)
_exec_high_risk = _exec_scored_df[_exec_scored_df["RiskBucket"] == "High"]
_exec_high_risk_count = len(_exec_high_risk)
_exec_high_risk_rate = (
    _exec_high_risk["Exited"].mean() * 100 if len(_exec_high_risk) > 0 else 0
)

render_executive_summary(
    _exec_total_customers,
    _exec_churn_rate,
    _exec_retention_rate,
    _exec_engagement_gap,
    _exec_top_geo,
    _exec_top_segment,
    _exec_strategy,
    high_risk_count=_exec_high_risk_count,
    high_risk_rate=_exec_high_risk_rate,
)



# =========================================================
# FOOTER
# =========================================================
st.markdown(
    '<div class="report-footer">This report is generated for internal analytical purposes only. '
    "All figures are derived from the European Bank customer dataset (n="
    + f"{len(df_raw):,}"
    + ") and reflect the "
    "population currently selected in the Filter Panel. Prepared by the Data Analytics Division &mdash; "
    "Unified Mentor Data Analyst Internship Programme.</div>",
    unsafe_allow_html=True,
)
