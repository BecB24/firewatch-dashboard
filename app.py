# app.py — FireWatch (public-friendly dashboard)

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from datetime import date
import re

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay



# Page setup

st.set_page_config("🎆 FireWatch: Bonfire Night Dashboard", layout="wide")



# Styling (CSS)

st.markdown(
    """
<style>
.big-title { font-size: 2.3rem; font-weight: 800; margin-bottom: .2rem; color: #F97316; }
.subtitle  { font-size: 1.05rem; opacity: .85; margin-bottom: 1.1rem; color: #E5E7EB; }

.kpi-card {
  border-radius:16px;
  padding:16px 18px;
  background:rgba(255,255,255,.04);
  border:1px solid rgba(255,255,255,.08);
}
.kpi-card h4{ margin:0 0 6px 0; font-weight:600; font-size:0.95rem; opacity:.85; }
.kpi-card h2{ margin:0; font-size:1.8rem; font-weight:800; }
.kpi-good{ color:#22c55e; }
.kpi-warn{ color:#f59e0b; }
.small-note { opacity:.75; font-size:.9rem; }

/* Prediction card */
.pred-card{
  border-radius:16px;
  padding:20px 20px;
  border:1px solid rgba(255,255,255,.14);
  box-shadow: 0 10px 24px rgba(0,0,0,.25);
}
.pred-float{
  animation: floaty 3.6s ease-in-out infinite;
}
@keyframes floaty{
  0%   { transform: translateY(0px); }
  50%  { transform: translateY(-6px); }
  100% { transform: translateY(0px); }
}
.pred-title{ margin:0 0 8px 0; font-weight:700; opacity:.9; }
.pred-big{ margin:0; font-size:2.0rem; font-weight:900; letter-spacing:.2px; }
.pred-why{ margin-top:12px; opacity:.88; line-height:1.4; }
.pred-meta{ margin-top:10px; opacity:.75; font-size:.9rem; }
</style>

<div class="big-title">🎆 FireWatch</div>
<div class="subtitle">Scottish fire incidents around Guy Fawkes Night — made simple and clear for everyone.</div>
""",
    unsafe_allow_html=True,
)

with st.expander("ℹ️ About this dashboard"):
    st.write(
        "FireWatch visualises Scottish fire incidents with a focus on **Bonfire Week (Nov 3–7)** "
        "and **Guy Fawkes Night (Nov 5)**. It includes a simple example machine-learning model and "
        "a public-friendly summary of the firefighters’ questionnaire."
    )



# Helper functions

def norm_text(x: str) -> str:
    """Normalise text so it is easier to match columns by name."""
    return re.sub(r"\s+", " ", str(x)).strip().lower()


def find_col(df: pd.DataFrame, keywords: list[str]) -> str | None:
    """
    Try to find a column by checking if any keyword appears in the column name.
    Returns the column name or None if not found.
    """
    for c in df.columns:
        c_norm = norm_text(c)
        if any(k in c_norm for k in keywords):
            return c
    return None


def is_good_pie_col(series: pd.Series) -> bool:
    """
    Decide if a column is suitable for pie/bar chart.
    We avoid columns with very long answers (usually free text).
    """
    s = series.dropna().astype(str)
    if len(s) < 3:
        return False
    avg_len = s.str.len().mean()
    return avg_len < 120


def is_free_text_question(series: pd.Series) -> bool:
    """
    Simple check for free-text columns:
    - long average answer length OR
    - lots of unique answers
    """
    s = series.dropna().astype(str)
    if len(s) < 3:
        return True

    avg_len = s.str.len().mean()
    unique_ratio = s.nunique() / len(s)

    return avg_len > 60 or unique_ratio > 0.6



# Data loaders

@st.cache_data
def load_fire_data(path: str = "data/fire_scotland_daily.csv") -> pd.DataFrame:
    """Load the fire incidents CSV and tidy up the key columns."""
    df = pd.read_csv(path)

    # Convert date column into real datetime values
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=False)

    # Convert incidents into numbers (removing commas)
    if "incidents" in df.columns:
        df["incidents"] = (
            df["incidents"].astype(str).str.replace(",", "", regex=False).str.strip()
        )
        df["incidents"] = pd.to_numeric(df["incidents"], errors="coerce")

    # Convert year to integer type
    if "year" in df.columns:
        df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")

    # Ensure day_of_year exists
    if "day_of_year" in df.columns:
        df["day_of_year"] = pd.to_numeric(df["day_of_year"], errors="coerce")

    # Sort by date 
    if "date" in df.columns:
        df = df.sort_values("date")

    # Recompute flags from date 
    if "date" in df.columns:
        bonfire_bool = (df["date"].dt.month.eq(11)) & (df["date"].dt.day.between(3, 7))
        df["is_bonfire_window"] = bonfire_bool.map({True: "Yes", False: "No"})

        weekend_bool = df["date"].dt.dayofweek.isin([5, 6])  # Sat=5 Sun=6
        df["is_weekend"] = weekend_bool.map({True: "Yes", False: "No"})

        # If day_of_year is missing/empty, compute from date
        if "day_of_year" not in df.columns or df["day_of_year"].isna().all():
            df["day_of_year"] = df["date"].dt.dayofyear

    return df


@st.cache_data
def load_survey_json(path: str = "data/data.cleaned.json") -> pd.DataFrame:
    """Load the cleaned questionnaire JSON."""
    return pd.read_json(path)



# Load and validate fire data

df = load_fire_data()

required = {"date", "year", "incidents", "day_of_year"}
missing = required - set(df.columns)
if missing:
    st.error(f"Your fire dataset is missing required columns: {missing}")
    st.stop()

if df["date"].isna().all():
    st.error("All dates failed to parse. Check the 'date' column format in your CSV.")
    st.stop()

if df["incidents"].notna().sum() == 0:
    st.error("No numeric incident values found. Check your 'incidents' column in the CSV.")
    st.stop()



# Year selector

years = sorted(df["year"].dropna().unique().tolist())
if not years:
    st.error("No valid years found in the dataset.")
    st.stop()

sel_year = st.selectbox("Select Year", years, index=len(years) - 1)

sub = df[df["year"] == sel_year].copy().sort_values("date")
sub_plot = sub.dropna(subset=["incidents", "date"]).copy()
if sub_plot.empty:
    st.warning("No usable rows to display for this year.")
    st.stop()



# KPI calculations

is_event = (sub_plot["date"].dt.month == 11) & (sub_plot["date"].dt.day.isin([3, 4, 5, 6, 7]))
baseline = sub_plot.loc[~is_event, "incidents"].mean()

nov5_mask = (sub_plot["date"].dt.month == 11) & (sub_plot["date"].dt.day == 5)
nov5_avg = sub_plot.loc[nov5_mask, "incidents"].mean()

pct_increase = None
if pd.notna(baseline) and baseline > 0 and pd.notna(nov5_avg):
    pct_increase = 100 * (nov5_avg - baseline) / baseline

total_incidents = int(sub_plot["incidents"].sum())
peak_row = sub_plot.loc[sub_plot["incidents"].idxmax()]
peak_day = peak_row["date"].date()
peak_val = int(peak_row["incidents"])



# KPI cards

c1, c2, c3 = st.columns(3, gap="medium")

with c1:
    st.markdown(
        f"<div class='kpi-card'><h4>Total incidents in {sel_year}</h4><h2>{total_incidents:,}</h2></div>",
        unsafe_allow_html=True,
    )

with c2:
    pct_txt = "—" if pct_increase is None else f"{pct_increase:.1f}%"
    cls = "kpi-good" if (pct_increase is not None and pct_increase > 0) else "kpi-warn"
    st.markdown(
        f"<div class='kpi-card'><h4>% ↑ on Nov 5 vs baseline</h4><h2 class='{cls}'>{pct_txt}</h2></div>",
        unsafe_allow_html=True,
    )

with c3:
    st.markdown(
        f"<div class='kpi-card'><h4>Peak day</h4><h2>{peak_day} ({peak_val:,})</h2></div>",
        unsafe_allow_html=True,
    )

st.caption("Baseline = average daily incidents excluding Nov 3–7 (Bonfire Week).")



# Tabs

tab1, tab2, tab3 = st.tabs(["📈 Incidents", "🧠 Prediction Model", "📋 Survey Results"])



# TAB 1: Incidents

with tab1:
    st.subheader("Daily fire incidents (with Bonfire Week highlighted)")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(sub_plot["date"], sub_plot["incidents"], "-", linewidth=2)

    nov3 = pd.Timestamp(int(sel_year), 11, 3)
    nov7 = pd.Timestamp(int(sel_year), 11, 7)
    nov5 = pd.Timestamp(int(sel_year), 11, 5)

    ax.axvspan(nov3, nov7, alpha=0.12)
    ax.axvline(nov5, linestyle="--", linewidth=1)
    ax.annotate(
        "Nov 5",
        xy=(nov5, ax.get_ylim()[1]),
        xytext=(0, 6),
        textcoords="offset points",
        ha="center",
        fontsize=9,
    )

    ax.set_title(f"Daily Fire Incidents in {sel_year}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Incidents")
    fig.autofmt_xdate()
    st.pyplot(fig, clear_figure=True)

    st.markdown(
        "<div class='small-note'>Tip: Scroll on the chart area to zoom (trackpad/mousewheel).</div>",
        unsafe_allow_html=True,
    )

    # Snapshot (Nov 1–10)
    st.subheader("Data snapshot (easy-to-read) — Nov 1–10")

    show_cols = [c for c in ["date", "incidents", "is_weekend", "is_bonfire_window"] if c in sub_plot.columns]
    nov_window = sub_plot[
        (sub_plot["date"].dt.month == 11) & (sub_plot["date"].dt.day.between(1, 10))
    ][show_cols].copy()

    nov_window["date"] = nov_window["date"].dt.date.astype(str)

    if nov_window.empty:
        st.warning("No rows found for Nov 1–10 in this year (check your dataset dates).")
    else:
        st.dataframe(nov_window, use_container_width=True, hide_index=True)

    st.download_button(
        "⬇️ Download this year as CSV",
        data=sub_plot.to_csv(index=False),
        file_name=f"firewatch_{sel_year}.csv",
        mime="text/csv",
    )



# TAB 2: Model + Confusion Matrix + Prediction card

with tab2:
    st.subheader("A simple model that flags “high-incident days”")
    st.write(
        "This example model uses **day of year** to predict whether a day is likely to be **higher than usual**. "
        "It’s intentionally simple, so the public can understand it."
    )

    # Build a dataset for the model
    model_df = df.dropna(subset=["day_of_year", "incidents"]).copy()

    # Create the label: above median = 1 (high), else 0 (low)
    threshold = model_df["incidents"].median()
    model_df["high_risk"] = (model_df["incidents"] > threshold).astype(int)

    X = model_df[["day_of_year"]]
    y = model_df["high_risk"]

    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )

    # Train a simple model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Predict on test set
    y_pred = model.predict(X_test)

    # Layout: matrix left, prediction right
    left, right = st.columns([1.2, 1.0], gap="large")

    with left:
        st.markdown("**Model performance (confusion matrix)**")

        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(cm, display_labels=["Low", "High"])

        fig2, ax2 = plt.subplots(figsize=(2.6, 2.6))
        disp.plot(ax=ax2, colorbar=False, values_format="d")
        ax2.set_title("Confusion Matrix", fontsize=9)
        ax2.tick_params(labelsize=8)
        plt.tight_layout(pad=0.4)
        st.pyplot(fig2, clear_figure=True)

        st.caption("Shows how often the model correctly (and incorrectly) labels days as Low/High.")

    with right:
        st.markdown("**Try a prediction**")

        # Date picker (default to Nov 5 of selected year)
        default_pick = date(int(sel_year), 11, 5)
        picked = st.date_input("Pick a date", value=default_pick)

        # Convert picked date into day-of-year number
        day_of_year = int(picked.timetuple().tm_yday)

        # Make prediction
        pred = model.predict(pd.DataFrame({"day_of_year": [day_of_year]}))[0]
        label = "High-risk" if pred == 1 else "Low-risk"

        # Style the prediction card
        if pred == 1:
            bg = "rgba(239, 68, 68, 0.18)"
            border = "rgba(239, 68, 68, 0.45)"
            icon = "🔴"
            why = (
                "This date falls in a part of the year where incident levels have often been higher "
                "in the historical data. The model has learned that this time period is more likely "
                "to be a high-incident day."
            )
            extra_class = "pred-float"  # animate only for high risk
        else:
            bg = "rgba(34, 197, 94, 0.16)"
            border = "rgba(34, 197, 94, 0.40)"
            icon = "🟢"
            why = (
                "This date falls in a part of the year where incident levels have usually been lower "
                "in the historical data, based on the model’s learned pattern."
            )
            extra_class = ""

        st.markdown(
            f"""
            <div class="pred-card {extra_class}" style="background:{bg}; border:1px solid {border};">
              <div class="pred-title">{icon} Predicted risk level</div>
              <div class="pred-big">{label}</div>

              <div class="pred-why">
                <b>Why?</b> {why}
              </div>

              <div class="pred-meta">
                Model input: <b>day {day_of_year}</b> of the year (based on your selected date).
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )



# TAB 3: Survey summary + dropdown chart + collapsible table

with tab3:
    st.subheader("Firefighters’ questionnaire results (summary)")
    st.write(
        "This section summarises responses from the firefighters’ questionnaire. "
        "Charts highlight key patterns, with full responses available below if you want to explore further."
    )

    try:
        survey = load_survey_json("data/data.cleaned.json").copy()

        # Identify key columns
        col_role = find_col(survey, ["role", "position", "rank"])
        col_years = find_col(survey, ["years", "service"])

        # Two charts side-by-side
        a, b = st.columns(2, gap="large")

        with a:
            if col_role:
                st.markdown("### 🧑‍🚒 Roles represented")
                role_counts = survey[col_role].value_counts()

                fig1, ax1 = plt.subplots(figsize=(5, 3))
                role_counts.plot(kind="barh", ax=ax1, color="#fb7185")
                ax1.set_xlabel("Number of responses")
                ax1.set_ylabel("")
                ax1.invert_yaxis()
                plt.tight_layout()
                st.pyplot(fig1, clear_figure=True)
            else:
                st.info("Role/Position column not detected in the survey file.")

        with b:
            if col_years:
                st.markdown("### ⏱ Years of service")
                years_counts = survey[col_years].value_counts()

                figy, axy = plt.subplots(figsize=(5, 3))
                years_counts.plot(kind="bar", ax=axy, color="#60a5fa")
                axy.set_ylabel("Responses")
                axy.set_xlabel("")
                axy.tick_params(axis="x", labelrotation=25)
                plt.tight_layout()
                st.pyplot(figy, clear_figure=True)
            else:
                st.info("Years of service column not detected in the survey file.")

        st.markdown("---")
        st.markdown("## 🥧 Pick a question to visualise")

        # Find suitable columns for pie/bar chart
        pie_cols = [c for c in survey.columns if is_good_pie_col(survey[c])]

        if not pie_cols:
            st.info("No suitable survey columns found for a chart.")
        else:
            pie_col = st.selectbox("Choose a survey question", pie_cols)

            # Check if the question is mostly free-text
            free_text = is_free_text_question(survey[pie_col])

            # Only show split option if it is not free text
            split_multi = False
            if not free_text:
                split_multi = st.checkbox("Split multi-select answers (e.g., 'A, B, C')", value=True)
            else:
                st.info(
                    "ℹ️ This question has mostly free-text answers, so it is shown as example responses "
                    "instead of a category chart."
                )

            # Display either sample responses or a chart
            if free_text:
                st.subheader("📝 Sample responses")
                st.caption("These are raw answers (not turned into a chart).")
                st.dataframe(
                    survey[[pie_col]].dropna().head(15),
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                # Turn the selected column into a clean series
                series = survey[pie_col].dropna().astype(str).str.strip()

                # Split answers if they are multi-select in one cell
                if split_multi:
                    series = series.apply(lambda x: re.split(r"\s*[,;]\s*", x))
                    series = series.explode().dropna().astype(str).str.strip()
                    series = series[series != ""]

                # Count the answers
                counts = series.value_counts()

                # Let user choose how many categories to display
                TOP_N = st.slider("How many categories to show", 3, 12, 6)

                # Group the rest into "Other"
                if len(counts) > TOP_N:
                    top = counts.head(TOP_N)
                    other = counts.iloc[TOP_N:].sum()
                    counts_plot = pd.concat([top, pd.Series({"Other": other})])
                else:
                    counts_plot = counts

                # If too many categories, use a bar chart (pie becomes unreadable)
                if len(counts_plot) > 9:
                    st.caption("Too many categories for a clear pie chart — showing a bar chart instead.")
                    figp, axp = plt.subplots(figsize=(6, 3.2))
                    counts_plot.sort_values().plot(kind="barh", ax=axp, color="#a78bfa")
                    axp.set_xlabel("Responses")
                    axp.set_ylabel("")
                    plt.tight_layout()
                    st.pyplot(figp, clear_figure=True)
                else:
                    # Fixed-size pie chart with legend (prevents giant charts)
                    figp, axp = plt.subplots(figsize=(3.8, 3.8))

                    colors = plt.cm.Set3.colors
                    wedges, _, _ = axp.pie(
                        counts_plot.values,
                        autopct="%1.0f%%",
                        startangle=90,
                        colors=colors,
                        textprops={"fontsize": 9},
                        pctdistance=0.75,
                    )
                    axp.axis("equal")

                    axp.legend(
                        wedges,
                        counts_plot.index,
                        title="Responses",
                        loc="center left",
                        bbox_to_anchor=(1.02, 0.5),
                        fontsize=8,
                        title_fontsize=9,
                        frameon=False,
                    )

                    plt.tight_layout()
                    st.pyplot(figp, clear_figure=True)

                st.caption(f"Showing distribution for: {pie_col}")

        st.markdown("---")

        # Full survey data (collapsible)
        with st.expander("📋 View full questionnaire responses"):
            st.dataframe(survey, use_container_width=True, hide_index=True)

            st.download_button(
                "⬇️ Download survey as CSV",
                data=survey.to_csv(index=False),
                file_name="firewatch_survey.csv",
                mime="text/csv",
            )

    except Exception as e:
        st.info("Survey data not found yet. Add `data/data.cleaned.json` to display results.")
        st.caption(f"Details (for you): {e}")



# Footer

st.markdown("---")
st.markdown("© FireWatch | Built with Python, Streamlit, Matplotlib, and scikit-learn")
