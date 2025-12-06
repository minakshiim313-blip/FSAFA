# app.py
"""
Benford's Law Analyzer - Streamlit app
Single-file prototype for upload -> automatic Benford analysis -> results + exports

Usage:
    pip install -r requirements.txt
    streamlit run app.py

Requirements (suggested):
    streamlit
    pandas
    numpy
    scipy
    matplotlib
    fpdf  # optional, for PDF export

This file implements:
- File upload (CSV/XLSX)
- Auto-detect numeric columns (and try to coerce)
- Column selection + options
- Benford first-digit analysis (core), optional first-two-digit & last-digit
- MAD, Chi-square, simple "AI" interpretation (heuristic)
- CSV export and PDF report export (simple)
"""

import io
import math
from collections import Counter, defaultdict
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import chisquare

# Optional import for PDF export
try:
    from fpdf import FPDF
    FPDF_AVAILABLE = True
except Exception:
    FPDF_AVAILABLE = False

st.set_page_config(page_title="Benford's Law Analyzer", layout="wide")

# -------------------------
# Utility functions
# -------------------------
def benford_expected_first_digit():
    """Return expected proportions for digits 1..9"""
    return np.array([math.log10(1 + 1 / d) for d in range(1, 10)])


def first_significant_digit(val):
    """Extract first significant digit of a numeric value.
    Returns int 1..9 or None if not findable.
    """
    try:
        if pd.isna(val):
            return None
        # Convert to float safely
        f = float(val)
        if f == 0:
            return None
        s = '{:.15e}'.format(abs(f))  # scientific notation
        # find first non-zero digit in mantissa
        mantissa = s.split('e')[0].replace('.', '').lstrip('0')
        if mantissa:
            d = int(mantissa[0])
            if 1 <= d <= 9:
                return d
    except Exception:
        pass
    return None


def first_two_digits(val):
    """Return integer 10..99 for first two significant digits or None"""
    try:
        if pd.isna(val):
            return None
        f = float(val)
        if f == 0:
            return None
        s = '{:.15e}'.format(abs(f))
        mantissa = s.split('e')[0].replace('.', '').lstrip('0')
        if len(mantissa) >= 2:
            return int(mantissa[:2])
        elif len(mantissa) == 1:
            return int(mantissa[0])  # fallback single digit
    except Exception:
        pass
    return None


def last_digit(val):
    """Return last digit of integer representation or None"""
    try:
        if pd.isna(val):
            return None
        # treat numbers like 123.45 -> consider integer part? We'll consider integer part
        f = float(val)
        n = int(abs(f))
        d = n % 10
        return int(d)
    except Exception:
        pass
    return None


def compute_first_digit_stats(series):
    """Given a pandas Series, compute observed counts, freq, expected, MAD, chi2 & p"""
    # Extract first digits
    digits = [first_significant_digit(x) for x in series.dropna()]
    digits = [d for d in digits if d is not None]
    n = len(digits)
    counts = np.array([Counter(digits).get(d, 0) for d in range(1, 10)])
    obs_freq = counts / counts.sum() if counts.sum() > 0 else np.zeros(9)
    expected_prop = benford_expected_first_digit()
    expected_counts = expected_prop * counts.sum()
    # MAD
    mad = np.mean(np.abs(obs_freq - expected_prop)) if n > 0 else np.nan
    # chi-square (handle degenerate)
    try:
        chi2_stat, p_value = chisquare(f_obs=counts, f_exp=expected_counts)
    except Exception:
        chi2_stat, p_value = (np.nan, np.nan)
    return {
        "n": n,
        "counts": counts,
        "obs_freq": obs_freq,
        "expected_prop": expected_prop,
        "expected_counts": expected_counts,
        "mad": mad,
        "chi2": chi2_stat,
        "p_value": p_value,
        "raw_digits": digits,
    }


def compute_two_digit_stats(series):
    """First-two-digit stats for 10..99"""
    digits = [first_two_digits(x) for x in series.dropna()]
    digits = [d for d in digits if d is not None and d >= 10]
    n = len(digits)
    counts = Counter(digits)
    # expected proportion for 10..99: P(d) = log10(1 + 1/d)
    expected = {d: math.log10(1 + 1 / d) for d in range(10, 100)}
    obs_counts = np.array([counts.get(d, 0) for d in range(10, 100)])
    expected_props = np.array([expected[d] for d in range(10, 100)])
    expected_counts = expected_props * obs_counts.sum()
    obs_freq = obs_counts / obs_counts.sum() if obs_counts.sum() > 0 else np.zeros_like(obs_counts)
    mad = np.mean(np.abs(obs_freq - expected_props)) if n > 0 else np.nan
    try:
        chi2_stat, p_value = chisquare(f_obs=obs_counts, f_exp=expected_counts)
    except Exception:
        chi2_stat, p_value = (np.nan, np.nan)
    return {
        "n": n,
        "obs_counts": obs_counts,
        "obs_freq": obs_freq,
        "expected_props": expected_props,
        "expected_counts": expected_counts,
        "mad": mad,
        "chi2": chi2_stat,
        "p_value": p_value,
    }


def compute_last_digit_stats(series):
    """Last digit (0-9) uniformity test"""
    digits = [last_digit(x) for x in series.dropna()]
    digits = [d for d in digits if d is not None]
    n = len(digits)
    counts = np.array([Counter(digits).get(d, 0) for d in range(0, 10)])
    expected_counts = np.array([n / 10.0] * 10)
    obs_freq = counts / counts.sum() if counts.sum() > 0 else np.zeros_like(counts)
    mad = np.mean(np.abs(obs_freq - (1/10))) if n > 0 else np.nan
    try:
        chi2_stat, p_value = chisquare(f_obs=counts, f_exp=expected_counts)
    except Exception:
        chi2_stat, p_value = (np.nan, np.nan)
    return {
        "n": n,
        "counts": counts,
        "obs_freq": obs_freq,
        "expected_counts": expected_counts,
        "mad": mad,
        "chi2": chi2_stat,
        "p_value": p_value,
    }


def nigrini_mad_verdict(mad):
    """Nigrini thresholds for first-digit MAD (heuristic)"""
    if np.isnan(mad):
        return ("Insufficient data", "gray")
    if mad <= 0.006:
        return ("Close conformity", "green")
    if mad <= 0.012:
        return ("Acceptable conformity", "green")
    if mad <= 0.015:
        return ("Marginal conformity", "orange")
    return ("Non-conforming (possible red flag)", "red")


def generate_ai_explanation(col_name, stats_first, stats_two=None, stats_last=None):
    """Generate a short heuristic explanation; not an LLM."""
    lines = []
    n = stats_first.get("n", 0)
    mad = stats_first.get("mad", np.nan)
    verdict, _ = nigrini_mad_verdict(mad)
    lines.append(f"Column '{col_name}' (n={n}) -> {verdict}. MAD={mad:.4f}.")
    # find top overrepresented digits
    if n and isinstance(stats_first.get("obs_freq", None), np.ndarray):
        obs = stats_first["obs_freq"]
        exp = stats_first["expected_prop"]
        diffs = obs - exp
        top_idx = np.argsort(-diffs)[:3]
        for i in top_idx:
            if diffs[i] > 0.02:  # threshold for mention
                lines.append(f"Digit {i+1} overrepresented by {(diffs[i]*100):.1f}% (obs {(obs[i]*100):.1f}% vs exp {(exp[i]*100):.1f}%).")
    # simple suggestions
    if verdict.startswith("Non-conforming"):
        lines.append("Recommendation: prioritize sampling records where the first digit is overrepresented; check month-end adjustments, rounding, or manual entries.")
    else:
        lines.append("No immediate red flag from Benford's first-digit distribution; complement with other tests (last-digit, time-series checks).")
    return " ".join(lines)


def prepare_pdf_report(title, analyses, generated_by="Benford Analyzer"):
    """Create a simple PDF report using fpdf (if available).
    analyses: list of dicts with keys: col_name, stats_first, explanation
    Returns bytes (PDF) or None if FPDF not installed.
    """
    if not FPDF_AVAILABLE:
        return None
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", size=16)
    pdf.cell(0, 10, title, ln=True, align="L")
    pdf.set_font("Helvetica", size=10)
    pdf.cell(0, 8, f"Generated: {datetime.utcnow().isoformat()} UTC", ln=True)
    pdf.cell(0, 6, f"By: {generated_by}", ln=True)
    pdf.ln(4)
    for a in analyses:
        pdf.set_font("Helvetica", size=12, style="B")
        pdf.cell(0, 8, f"Column: {a['col_name']}", ln=True)
        pdf.set_font("Helvetica", size=10)
        s = a.get("explanation", "")
        # wrap text
        for chunk in list(split_text(s, 90)):
            pdf.multi_cell(0, 6, chunk)
        pdf.ln(2)
        # add small table: MAD, n, chi2 p
        stats = a["stats_first"]
        pdf.cell(0, 6, f"Observations: {stats.get('n', 'NA')}   MAD: {stats.get('mad', np.nan):.4f}   Chi2 p-value: {stats.get('p_value', 'NA')}", ln=True)
        pdf.ln(4)
    return pdf.output(dest='S').encode('latin-1')


def split_text(text, width):
    """Yield chunks of text no longer than width characters (simple wrapper)."""
    import textwrap
    for paragraph in text.split("\n"):
        for chunk in textwrap.wrap(paragraph, width=width):
            yield chunk


# -------------------------
# Streamlit UI
# -------------------------
st.title("Benford's Law Analyzer — Prototype")
st.markdown(
    """
Upload a company's financial data (CSV or Excel). This tool automatically detects numeric columns and runs Benford analyses.
**Important:** Benford deviations are *red flags*, not proof of fraud. Use together with domain knowledge.
"""
)

# Layout columns: left 30% (upload & instructions), middle 30% controls, right 40% results
col_left, col_mid, col_right = st.columns([1, 1, 1.6])

# --- Left column: upload and instructions
with col_left:
    st.header("Upload & Instructions")
    uploaded = st.file_uploader("Upload CSV or Excel (XLSX)", type=["csv", "xlsx"], help="Drag and drop or click to upload.")
    st.caption("Accepted: transaction lists, ledger extracts, invoices, account balances.")
    st.markdown("**Quick instructions**")
    st.markdown(
        """
- Prefer transaction-level data (invoices, receipts) rather than pre-aggregated monthly totals.
- Benford works best with >100 observations and when numbers span multiple orders of magnitude.
- Exclude IDs, account numbers, and purely assigned numbers.
- Use the options in the center column to exclude zeros/negatives or group by segment (e.g., vendor, period).
"""
    )
    with st.expander("Detailed docs & best practices"):
        st.markdown(
            """
**When Benford applies**
- Natural financial amounts (sales, invoice amounts, balances) across wide ranges.
- Not suitable for assigned numbers, prices with fixed format, or small samples.

**Preprocessing tips**
- Remove zeros if they represent missing data.
- Consider analyzing by vendor / period / branch to surface segment-level anomalies.
- Combine Benford with last-digit tests, time-series checks, and manual sampling.
"""
        )
    st.markdown("---")
    st.markdown("**Detected / Suggested numeric columns**")
    # we will populate suggested_cols below after reading file

# --- Middle column: analysis controls
with col_mid:
    st.header("Analysis Controls")
    # Defaults
    analyze_first = st.checkbox("First-digit analysis (default)", value=True)
    analyze_two = st.checkbox("First-two-digit analysis", value=False)
    analyze_last = st.checkbox("Last-digit (uniformity) analysis", value=False)

    st.markdown("**Preprocessing options**")
    exclude_zeros = st.checkbox("Exclude zeros", True)
    exclude_negatives = st.checkbox("Exclude negative values", False)
    log_transform = st.checkbox("Log-transform values before analysis (not usually recommended)", False)

    st.markdown("**Grouping**")
    group_by = st.selectbox("Group by (optional):", options=["(none)"], index=0)
    st.markdown("---")
    st.markdown("**MAD thresholds (Nigrini, editable)**")
    mad_close = st.number_input("Close conformity <= ", value=0.006, format="%.6f")
    mad_accept = st.number_input("Acceptable conformity <= ", value=0.012, format="%.6f")
    mad_marginal = st.number_input("Marginal conformity <= ", value=0.015, format="%.6f")

    run_button = st.button("Run analysis", type="primary")

# --- Right column: results area
with col_right:
    st.header("Results")
    results_placeholder = st.empty()  # will fill after analysis

# -------------------------
# Read uploaded file, detect numeric columns
# -------------------------
df = None
if uploaded:
    try:
        if uploaded.name.lower().endswith(".csv"):
            df = pd.read_csv(uploaded)
        else:
            df = pd.read_excel(uploaded)
    except Exception as e:
        st.error(f"Could not read uploaded file: {e}")

if df is not None:
    # preview
    with col_left:
        st.write("Preview (first 5 rows):")
        st.dataframe(df.head(5))

    # detect numeric columns (dtype numeric or coercible)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    coerced = []
    if not numeric_cols:
        # try coercion
        for c in df.columns:
            coerced_col = pd.to_numeric(df[c], errors="coerce")
            if coerced_col.notna().sum() > 0:
                coerced.append(c)
                df[c] = coerced_col
        numeric_cols = coerced

    # suggest monetary-like columns by name matching
    monetary_suggestions = []
    lower_names = {c: c.lower() for c in df.columns}
    patterns = ["amount", "amt", "invoice", "sale", "revenue", "total", "balance", "price"]
    for c, lname in lower_names.items():
        if any(p in lname for p in patterns):
            if c in numeric_cols:
                monetary_suggestions.append(c)

    # populate group_by options (allow any non-numeric columns for grouping)
    group_options = ["(none)"] + df.select_dtypes(exclude=[np.number]).columns.tolist()
    # update group_by widget (we can't directly update selectbox from here, but we can show suggestions)
    # Inform user if group_by currently "(none)"
    with col_mid:
        if len(group_options) > 1 and group_by == "(none) and False":
            pass  # dummy to satisfy linter

    # show detected columns in left column
    with col_left:
        if numeric_cols:
            # simple info table
            info_rows = []
            for c in numeric_cols:
                col_series = pd.to_numeric(df[c], errors="coerce")
                info_rows.append({
                    "column": c,
                    "count": int(col_series.count()),
                    "% zeros": f"{(col_series == 0).sum() / max(1, col_series.count()) * 100:.1f}%",
                    "suggested": "✔" if c in monetary_suggestions else ""
                })
            info_df = pd.DataFrame(info_rows)
            st.dataframe(info_df, height=200)
        else:
            st.warning("No numeric columns detected. Try cleaning/coercing your file or ensure amounts are numeric.")

    # allow column selection: show multiselect in middle column
    with col_mid:
        chosen_columns = st.multiselect("Select numeric columns to analyze", options=numeric_cols, default=monetary_suggestions[:1] if monetary_suggestions else (numeric_cols[:1] if numeric_cols else []))
        # update group_by selectbox with available grouping columns
        group_by = st.selectbox("Group by (optional):", options=group_options, index=0)

    # Run analysis when button pressed
    if run_button:
        if not chosen_columns:
            st.warning("Please select at least one numeric column to analyze.")
        else:
            analyses = []  # collect for export
            # If grouping requested, create group keys
            perform_grouping = (group_by != "(none)")

            # Create a results display in right column
            with results_placeholder.container():
                st.success(f"Running analysis on {len(chosen_columns)} column(s)...")
                for col in chosen_columns:
                    st.subheader(f"Column: {col}")
                    # prepare series
                    s = pd.to_numeric(df[col], errors="coerce")
                    if exclude_zeros:
                        s = s[s != 0]
                    if exclude_negatives:
                        s = s[s >= 0]
                    # apply log transform if requested (works only for positive numbers)
                    if log_transform:
                        s = s[s > 0]
                        s = np.log10(s)

                    # If grouping, iterate groups
                    if perform_grouping:
                        grp = df[group_by].fillna("(missing)")
                        groups = grp.unique()
                        # show a table of MAD per group
                        mad_rows = []
                        for g in sorted(groups, key=lambda x: str(x))[:50]:  # limit to first 50 groups for UI sanity
                            mask = (grp == g)
                            ss = pd.to_numeric(df.loc[mask, col], errors="coerce")
                            if exclude_zeros:
                                ss = ss[ss != 0]
                            if exclude_negatives:
                                ss = ss[ss >= 0]
                            if log_transform:
                                ss = ss[ss > 0]
                                ss = np.log10(ss)
                            stats = compute_first_digit_stats(ss)
                            mad_rows.append({"group": g, "n": stats["n"], "mad": stats["mad"]})
                        mad_df = pd.DataFrame(mad_rows).sort_values(by="mad", ascending=False).reset_index(drop=True)
                        st.markdown(f"**Group-level MAD (top 20 by MAD) — grouping by {group_by}**")
                        st.dataframe(mad_df.head(20))
                        # allow user to pick a group to drill down
                        chosen_group = st.selectbox(f"Choose a group to drill down for column '{col}'", options=["(none)"] + list(sorted(mad_df["group"].astype(str).unique())), index=0)
                        if chosen_group != "(none)":
                            # compute stats for chosen group
                            mask = (df[group_by].astype(str) == str(chosen_group))
                            ss = pd.to_numeric(df.loc[mask, col], errors="coerce")
                            if exclude_zeros:
                                ss = ss[ss != 0]
                            if exclude_negatives:
                                ss = ss[ss >= 0]
                            if log_transform:
                                ss = ss[ss > 0]
                                ss = np.log10(ss)
                            stats_first = compute_first_digit_stats(ss)
                            # display card for the group
                            v, color = nigrini_mad_verdict(stats_first.get("mad", np.nan))
                            st.markdown(f"**Group: {chosen_group} — {v}**")
                            # plot observed vs expected
                            plot_first_digit(stats_first)
                            explanation = generate_ai_explanation(f"{col} — {chosen_group}", stats_first)
                            st.write(explanation)
                            # add to analyses collection
                            analyses.append({"col_name": f"{col} — {chosen_group}", "stats_first": stats_first, "explanation": explanation})
                    else:
                        # single overall
                        stats_first = compute_first_digit_stats(s)
                        # optionally do two-digit and last-digit analyses
                        stats_two = compute_two_digit_stats(s) if analyze_two else None
                        stats_last = compute_last_digit_stats(s) if analyze_last else None

                        # display metrics
                        n = stats_first.get("n", 0)
                        mad = stats_first.get("mad", np.nan)
                        pval = stats_first.get("p_value", np.nan)
                        verdict, color = nigrini_mad_verdict(mad)
                        col1, col2 = st.columns([1, 2])
                        with col1:
                            st.metric("Observations used", n)
                            st.metric("MAD", f"{mad:.5f}" if not np.isnan(mad) else "NA")
                            st.metric("Chi-square p-value", f"{pval:.4f}" if not np.isnan(pval) else "NA")
                            st.markdown(f"**Verdict:** <span style='color:{color};font-weight:600'>{verdict}</span>", unsafe_allow_html=True)
                        with col2:
                            plot_first_digit(stats_first)
                        # show small table of counts
                        table = pd.DataFrame({
                            "digit": list(range(1, 10)),
                            "observed_count": stats_first["counts"],
                            "observed_freq": stats_first["obs_freq"],
                            "expected_freq": stats_first["expected_prop"],
                            "expected_count": stats_first["expected_counts"]
                        })
                        st.dataframe(table.style.format({"observed_freq": "{:.4f}", "expected_freq": "{:.4f}", "expected_count": "{:.1f}"}))

                        explanation = generate_ai_explanation(col, stats_first, stats_two, stats_last)
                        st.write(explanation)
                        analyses.append({"col_name": col, "stats_first": stats_first, "explanation": explanation})

                # Export buttons
                st.markdown("---")
                st.write("Exports")
                # CSV export: compile analyses into CSV
                export_rows = []
                for a in analyses:
                    s = a["stats_first"]
                    for i in range(9):
                        export_rows.append({
                            "column": a["col_name"],
                            "digit": i+1,
                            "observed_count": int(s["counts"][i]) if hasattr(s["counts"], "__len__") else None,
                            "observed_freq": float(s["obs_freq"][i]) if hasattr(s["obs_freq"], "__len__") else None,
                            "expected_freq": float(s["expected_prop"][i]) if hasattr(s["expected_prop"], "__len__") else None,
                            "mad": float(s["mad"]) if s.get("mad", None) is not None else None,
                            "chi2": float(s["chi2"]) if s.get("chi2", None) is not None else None,
                            "p_value": float(s["p_value"]) if s.get("p_value", None) is not None else None,
                        })
                export_df = pd.DataFrame(export_rows)
                csv_bytes = export_df.to_csv(index=False).encode("utf-8")
                st.download_button("Download results CSV", data=csv_bytes, file_name="benford_results.csv", mime="text/csv")

                if FPDF_AVAILABLE:
                    pdf_bytes = prepare_pdf_report("Benford Analysis Report", analyses, generated_by="Benford Analyzer")
                    if pdf_bytes:
                        st.download_button("Download PDF report", data=pdf_bytes, file_name="benford_report.pdf", mime="application/pdf")
                else:
                    st.info("PDF export not available (install 'fpdf' to enable PDF reports).")

# -------------------------
# Helper plotting (placed at bottom to avoid duplication above)
# -------------------------
def plot_first_digit(stats):
    """Plot observed vs expected first-digit distribution inline."""
    obs = stats.get("obs_freq", np.zeros(9))
    expected = stats.get("expected_prop", benford_expected_first_digit())
    digits = np.arange(1, 10)
    fig, ax = plt.subplots(figsize=(6, 2.8))
    ax.bar(digits - 0.15, obs, width=0.3, label="Observed")
    ax.bar(digits + 0.15, expected, width=0.3, label="Benford expected")
    ax.set_xticks(digits)
    ax.set_xlabel("First digit")
    ax.set_ylabel("Proportion")
    ax.set_ylim(0, max(max(obs) * 1.2 if obs.sum() else 0.1, max(expected) * 1.2))
    ax.legend()
    st.pyplot(fig)
