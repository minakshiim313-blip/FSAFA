# app.py
"""
Benford's Law Analyzer — Focused & Polished Streamlit App
- First-digit Benford only (no extra unused options)
- Minimal, clear UI: upload -> select columns -> analyze
- Includes MAD + chi-square, chart, csv/pdf export, and a short AI-style explanation
Run:
    pip install -r requirements.txt
    streamlit run app.py
"""

import math
from collections import Counter
from datetime import datetime
import io

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import chisquare

# Optional: PDF export
try:
    from fpdf import FPDF
    FPDF_AVAILABLE = True
except Exception:
    FPDF_AVAILABLE = False

# Page config & basic styling
st.set_page_config(page_title="Benford's Law Analyzer", layout="wide")
st.markdown(
    """
<style>
h1 {font-size:42px; margin-bottom: 0.1rem;}
.header-sub {color: #555; margin-top: 0;}
.card {background: #fff; border-radius: 10px; padding: 16px; box-shadow: 0 6px 18px rgba(20,20,30,0.05);}
.small-muted {color:#666; font-size:0.9rem;}
.badge-green {color:#fff; background:#16a34a; padding:6px 10px; border-radius:6px;}
.badge-orange {color:#fff; background:#f97316; padding:6px 10px; border-radius:6px;}
.badge-red {color:#fff; background:#dc2626; padding:6px 10px; border-radius:6px;}
.badge-gray {color:#fff; background:#6b7280; padding:6px 10px; border-radius:6px;}
</style>
""",
    unsafe_allow_html=True,
)

# ---------- Utilities ----------
def benford_expected_first_digit():
    return np.array([math.log10(1 + 1 / d) for d in range(1, 10)])


def first_significant_digit(val):
    try:
        if pd.isna(val):
            return None
        f = float(val)
        if f == 0:
            return None
        s = "{:.15e}".format(abs(f))
        mantissa = s.split("e")[0].replace(".", "").lstrip("0")
        if mantissa:
            d = int(mantissa[0])
            if 1 <= d <= 9:
                return d
    except Exception:
        pass
    return None


def compute_first_digit_stats(series):
    digits = [first_significant_digit(x) for x in series.dropna()]
    digits = [d for d in digits if d is not None]
    n = len(digits)
    counts = np.array([Counter(digits).get(d, 0) for d in range(1, 10)])
    obs_freq = counts / counts.sum() if counts.sum() > 0 else np.zeros(9)
    expected_prop = benford_expected_first_digit()
    expected_counts = expected_prop * counts.sum()
    mad = np.mean(np.abs(obs_freq - expected_prop)) if n > 0 else np.nan
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


def nigrini_mad_verdict(mad):
    if np.isnan(mad):
        return ("Insufficient data", "gray")
    if mad <= 0.006:
        return ("Close conformity", "green")
    if mad <= 0.012:
        return ("Acceptable conformity", "green")
    if mad <= 0.015:
        return ("Marginal conformity", "orange")
    return ("Non-conforming (possible red flag)", "red")


def generate_ai_explanation(col_name, stats_first):
    n = stats_first.get("n", 0)
    mad = stats_first.get("mad", np.nan)
    verdict, _ = nigrini_mad_verdict(mad)
    parts = [f"Column '{col_name}' — {verdict} (n={n}, MAD={mad:.4f})."]
    # highlight top deviations
    if n and isinstance(stats_first.get("obs_freq", None), np.ndarray):
        obs = stats_first["obs_freq"]
        exp = stats_first["expected_prop"]
        diffs = obs - exp
        over = np.where(diffs > 0.02)[0]  # >2% over expected
        if len(over):
            bullets = []
            for i in over:
                bullets.append(f"digit {i+1} overrepresented by {(diffs[i]*100):.1f}%")
            parts.append("Notable deviations: " + "; ".join(bullets) + ".")
    if verdict.startswith("Non-conforming"):
        parts.append("Recommendation: sample records with overrepresented leading digits and inspect for manual adjustments, rounding, or batch edits.")
    else:
        parts.append("No immediate Benford red-flag. Combine with sampling and last-digit checks if concerned.")
    return " ".join(parts)


def make_pdf_report(title, analyses):
    if not FPDF_AVAILABLE:
        return None
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=12)
    pdf.add_page()
    pdf.set_font("Helvetica", size=16)
    pdf.cell(0, 10, title, ln=True)
    pdf.set_font("Helvetica", size=9)
    pdf.cell(0, 6, f"Generated: {datetime.utcnow().isoformat()} UTC", ln=True)
    pdf.ln(4)
    for a in analyses:
        pdf.set_font("Helvetica", size=12, style="B")
        pdf.cell(0, 8, f"Column: {a['col']}", ln=True)
        pdf.set_font("Helvetica", size=10)
        pdf.multi_cell(0, 6, a["explanation"])
        stats = a["stats"]
        pdf.cell(0, 6, f"Observations: {stats['n']}   MAD: {stats['mad']:.4f}   Chi2 p: {stats['p_value']:.4f}", ln=True)
        pdf.ln(4)
    return pdf.output(dest="S").encode("latin-1")


def plot_first_digit(stats, ax=None):
    obs = stats.get("obs_freq", np.zeros(9))
    expected = stats.get("expected_prop", benford_expected_first_digit())
    digits = np.arange(1, 10)
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 2.6))
    bars1 = ax.bar(digits - 0.15, obs, width=0.3, label="Observed")
    bars2 = ax.bar(digits + 0.15, expected, width=0.3, label="Benford expected")
    ax.set_xticks(digits)
    ax.set_xlabel("First digit")
    ax.set_ylabel("Proportion")
    ax.set_ylim(0, max(0.12, max(obs.max(), expected.max()) * 1.2))
    ax.legend()
    for bar in bars1:
        height = bar.get_height()
        if height > 0:
            ax.annotate(f"{height:.2f}", xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=8)
    return ax

# ---------- UI ----------
st.markdown("<h1>Benford's Law Analyzer — Prototype</h1>", unsafe_allow_html=True)
st.markdown('<div class="header-sub small-muted">Upload financial transaction data (CSV or Excel). This tool performs a focused first-digit Benford analysis — a red-flag/triage test, not proof of fraud.</div>', unsafe_allow_html=True)
st.write("")

left_col, mid_col, right_col = st.columns([1, 1, 1.2])

# Left: Upload + instructions
with left_col:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload CSV or Excel (XLSX)", type=["csv", "xlsx"], help="Drag & drop or click to upload.")
    st.write("")
    st.markdown("**Quick instructions**")
    st.markdown(
        """
- Use transaction-level amounts (invoices, receipts, payments).  
- Benford works best with **≥100 observations** and numbers spanning several orders of magnitude.  
- Exclude IDs or assigned numbers; upload CSV if Excel reading fails.
"""
    )
    with st.expander("Detailed notes"):
        st.markdown(
            """
Benford checks the distribution of the *first significant digit* (1 appears ≈30% of the time, etc.).
**Important:** Treat deviations as *signals* to investigate, not proof of fraud. Combine with sampling and context knowledge.
"""
        )
    st.markdown("</div>", unsafe_allow_html=True)

# Mid: options (minimal)
with mid_col:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Analysis options")
    exclude_zeros = st.checkbox("Exclude zeros (recommended)", value=True, key="exclude_zeros")
    exclude_negatives = st.checkbox("Exclude negative values", value=False, key="exclude_negatives")
    st.write("")
    st.markdown("**Sample guidance**")
    st.markdown("<div class='small-muted'>Results become meaningful with sample size ≥100. Small samples may be unreliable.</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# Right: results area (placeholder)
with right_col:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Results")
    result_area = st.empty()
    st.markdown("</div>", unsafe_allow_html=True)

# ---------- Read file and detect numeric columns ----------
df = None
if uploaded:
    try:
        if uploaded.name.lower().endswith(".csv"):
            df = pd.read_csv(uploaded)
        else:
            try:
                df = pd.read_excel(uploaded, engine="openpyxl")
            except ImportError:
                st.error("Reading .xlsx requires the 'openpyxl' package. Add it to requirements or upload CSV.")
            except Exception:
                try:
                    df = pd.read_excel(uploaded)
                except Exception as e:
                    st.error(f"Could not read Excel file: {e}")
    except Exception as e:
        st.error(f"Could not read file: {e}")

if df is not None:
    # detect numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        # try coercion once
        coerced = []
        for c in df.columns:
            coerced_col = pd.to_numeric(df[c], errors="coerce")
            if coerced_col.notna().sum() > 0:
                coerced.append(c)
                df[c] = coerced_col
        numeric_cols = coerced

    # monetary suggestions (simple heuristic)
    patterns = ["amount", "amt", "invoice", "sale", "revenue", "total", "balance", "price"]
    monetary_suggestions = [c for c in numeric_cols if any(p in c.lower() for p in patterns)]

    # show detected columns on left
    with left_col:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("**Detected numeric columns**")
        if numeric_cols:
            for c in numeric_cols:
                col_ser = pd.to_numeric(df[c], errors="coerce")
                pct_zero = (col_ser == 0).sum() / max(1, col_ser.count()) * 100
                st.write(f"- **{c}** — {col_ser.count()} values, {pct_zero:.1f}% zeros {'(suggested)' if c in monetary_suggestions else ''}")
        else:
            st.warning("No numeric columns detected. Ensure amounts are numeric, or try uploading a CSV.")
        st.markdown("</div>", unsafe_allow_html=True)

    # selection & run button in middle
    with mid_col:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        chosen = st.multiselect("Select column(s) to analyze", options=numeric_cols, default=monetary_suggestions[:1] if monetary_suggestions else (numeric_cols[:1] if numeric_cols else []), key="chosen_cols")
        run = st.button("Run Benford analysis", type="primary", key="run_benford")
        st.markdown("</div>", unsafe_allow_html=True)

    # Run analysis when clicked
    if run:
        if not chosen:
            st.warning("Choose at least one numeric column to analyze.")
        else:
            analyses = []
            # build results display
            with result_area.container():
                for col in chosen:
                    st.markdown(f"#### {col}")
                    series = pd.to_numeric(df[col], errors="coerce").dropna()
                    if exclude_zeros:
                        series = series[series != 0]
                    if exclude_negatives:
                        series = series[series >= 0]
                    stats = compute_first_digit_stats(series)

                    # show sample size warning
                    if stats["n"] < 50:
                        st.warning(f"Only {stats['n']} usable observations for '{col}'. Benford less reliable for small samples.")
                    # verdict badge
                    verdict, color = nigrini_mad_verdict(stats.get("mad", np.nan))
                    badge_class = {"green": "badge-green", "orange": "badge-orange", "red": "badge-red", "gray": "badge-gray"}[color]
                    st.markdown(f"<div style='display:flex; gap:12px; align-items:center'><div class='{badge_class}'>{verdict}</div><div class='small-muted'>n = {stats['n']} &nbsp;&nbsp; MAD = {stats['mad']:.5f} &nbsp;&nbsp; Chi2 p = {stats['p_value']:.4f}</div></div>", unsafe_allow_html=True)

                    # plot
                    fig, ax = plt.subplots(figsize=(7, 2.6))
                    plot_first_digit(stats, ax=ax)
                    st.pyplot(fig)

                    # table
                    table = pd.DataFrame({
                        "digit": list(range(1, 10)),
                        "observed_count": stats["counts"],
                        "observed_freq": stats["obs_freq"],
                        "expected_freq": stats["expected_prop"],
                        "expected_count": stats["expected_counts"]
                    })
                    st.dataframe(table.style.format({"observed_freq": "{:.4f}", "expected_freq": "{:.4f}", "expected_count": "{:.1f}"}), height=220)

                    # AI-style explanation
                    explanation = generate_ai_explanation(col, stats)
                    st.markdown(f"**Interpretation**: {explanation}")

                    analyses.append({"col": col, "stats": stats, "explanation": explanation})
                    st.markdown("---")

                # Exports
                st.markdown("### Exports")
                # build CSV rows
                rows = []
                for a in analyses:
                    s = a["stats"]
                    for i in range(9):
                        rows.append({
                            "column": a["col"],
                            "digit": i + 1,
                            "observed_count": int(s["counts"][i]),
                            "observed_freq": float(s["obs_freq"][i]),
                            "expected_freq": float(s["expected_prop"][i]),
                            "mad": float(s["mad"]),
                            "chi2": float(s["chi2"]),
                            "p_value": float(s["p_value"]),
                        })
                out_df = pd.DataFrame(rows)
                csv_bytes = out_df.to_csv(index=False).encode("utf-8")
                st.download_button("Download results CSV", data=csv_bytes, file_name="benford_results.csv", mime="text/csv")
                if FPDF_AVAILABLE:
                    pdf_bytes = make_pdf_report("Benford Analysis Report", analyses)
                    if pdf_bytes:
                        st.download_button("Download PDF report", data=pdf_bytes, file_name="benford_report.pdf", mime="application/pdf")
                else:
                    st.info("PDF export not installed. Add 'fpdf' to requirements to enable PDF report.")

# End of file
