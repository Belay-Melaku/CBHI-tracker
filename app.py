import streamlit as st
import pandas as pd
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime
import plotly.express as px
import json
import traceback
from io import StringIO

# --- 1. CONFIGURATION & PLAN DATA ---
st.set_page_config(page_title="CBHI Performance Tracker", layout="wide", page_icon="📈")

PLANS = {
    "01 Merged Health Post": {"high": 453, "medium": 551, "free": 474, "new": 251, "total": 1729},
    "02 Densa Zuriya Health Post": {"high": 147, "medium": 316, "free": 155, "new": 0, "total": 618},
    "03 Derew Health Post": {"high": 456, "medium": 557, "free": 478, "new": 429, "total": 1920},
    "04 Wejed Health Post": {"high": 246, "medium": 346, "free": 249, "new": 0, "total": 841},
    "06 Gert Health Post": {"high": 237, "medium": 298, "free": 255, "new": 22, "total": 812},
    "07 Lenguat Health Post": {"high": 240, "medium": 328, "free": 244, "new": 0, "total": 812},
    "08 Alegeta Health Post": {"high": 217, "medium": 252, "free": 248, "new": 22, "total": 739},
    "09 Sensa Health Post": {"high": 173, "medium": 272, "free": 179, "new": 0, "total": 624}
}
INSTITUTIONS = list(PLANS.keys())
METRICS = ["High", "Medium", "Free", "New"]

# --- 2. DATABASE CONNECTION HELPERS ---
def get_gs_client():
    """
    Returns an authorized gspread client (or None on failure).
    Uses st.secrets['gcp_service_account'] which can be either a dict or JSON string.
    """
    try:
        creds_raw = st.secrets.get("gcp_service_account")
        if not creds_raw:
            st.warning("Google Sheets credentials not found in st.secrets['gcp_service_account']. Running in offline mode.")
            return None

        if isinstance(creds_raw, str):
            creds_dict = json.loads(creds_raw)
        elif isinstance(creds_raw, dict):
            creds_dict = creds_raw
        else:
            creds_dict = dict(creds_raw)

        scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
        creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
        client = gspread.authorize(creds)
        return client
    except Exception:
        st.error("Could not create Google Sheets client. Running in offline mode.")
        st.exception(traceback.format_exc())
        return None

def get_records_worksheet(client):
    """
    Convenience for opening the Records worksheet (the app uses this for raw entries).
    Returns worksheet or None.
    """
    if not client:
        return None
    try:
        sh = client.open("CBHI_Data_Database")
        return sh.worksheet("Records")
    except Exception:
        st.error("Could not open 'CBHI_Data_Database' or 'Records' worksheet.")
        st.exception(traceback.format_exc())
        return None

def write_performance_to_sheet(client, perf_df, sheet_name="Performance"):
    """
    Writes (or creates) a worksheet called sheet_name with the perf_df contents.
    perf_df is a pandas DataFrame. Overwrites existing worksheet contents.
    """
    if client is None:
        st.warning("No Google Sheets client available — cannot sync performance.")
        return False

    try:
        sh = client.open("CBHI_Data_Database")
    except Exception:
        st.error("Could not open spreadsheet 'CBHI_Data_Database'.")
        st.exception(traceback.format_exc())
        return False

    try:
        try:
            ws = sh.worksheet(sheet_name)
            ws.clear()
        except gspread.exceptions.WorksheetNotFound:
            # create with at least len rows and columns
            rows = max(100, len(perf_df) + 5)
            cols = max(10, len(perf_df.columns) + 2)
            ws = sh.add_worksheet(title=sheet_name, rows=str(rows), cols=str(cols))

        # Prepare data (list of lists)
        headers = list(perf_df.columns)
        values = perf_df.fillna("").astype(str).values.tolist()
        ws.update([headers] + values)
        return True
    except Exception:
        st.error("Failed to write performance worksheet.")
        st.exception(traceback.format_exc())
        return False

# --- 3. PERFORMANCE & TREND COMPUTATION ---
def compute_performance_from_records(df_records):
    """
    Input: raw records DataFrame expected to contain columns 'Health Institution',
           'High','Medium','Free','New' (case-insensitive tolerated).
    Output:
      - per_metric_df: long table with Institution, Metric, Plan, Achieved, Achieved %
      - total_summary_df: one-row-per-institution summary with Plan_total, Achieved_total, Achieved %
      - global_summary: dict with global plans, achieved and % by metric and totals
    """
    # Normalize column names
    if df_records is None or df_records.empty:
        per_metric_df = pd.DataFrame(columns=["Institution","Metric","Plan","Achieved","Achieved %"])
        total_summary_df = pd.DataFrame(columns=["Institution","Plan Total","Achieved Total","Achieved %"])
        global_summary = {}
        return per_metric_df, total_summary_df, global_summary

    df = df_records.copy()
    # detect institution column
    possible_inst_cols = [c for c in df.columns if c.strip().lower() in ("health institution", "institution", "health_institution", "health institution ", "health institution")]
    inst_col = possible_inst_cols[0] if possible_inst_cols else next((c for c in df.columns if "inst" in c.lower()), df.columns[0])

    # Ensure metric columns exist and numeric
    for m in METRICS:
        if m not in df.columns:
            if m.lower() in df.columns:
                df[m] = pd.to_numeric(df[m.lower()], errors='coerce').fillna(0)
            else:
                df[m] = 0
        else:
            df[m] = pd.to_numeric(df[m], errors='coerce').fillna(0)

    rows = []
    totals = []
    for inst_name, plan in PLANS.items():
        inst_df = df[df[inst_col] == inst_name] if inst_col in df.columns else pd.DataFrame()
        achieved_by_metric = inst_df[METRICS].sum() if not inst_df.empty else pd.Series({m: 0 for m in METRICS})
        plan_total = plan.get("total", sum(plan.get(k.lower(), 0) for k in METRICS))
        achieved_total = int(achieved_by_metric.sum())
        total_pct = (achieved_total / plan_total * 100) if plan_total > 0 else 0

        for m in METRICS:
            plan_value = int(plan.get(m.lower(), 0))
            achieved_value = int(achieved_by_metric.get(m, 0))
            pct = (achieved_value / plan_value * 100) if plan_value > 0 else 0
            rows.append({
                "Institution": inst_name,
                "Metric": m,
                "Plan": plan_value,
                "Achieved": achieved_value,
                "Achieved %": round(pct, 1)
            })

        totals.append({
            "Institution": inst_name,
            "Plan Total": plan_total,
            "Achieved Total": achieved_total,
            "Achieved %": round(total_pct, 1),
            "Status": "✅ Good" if total_pct > 70 else ("⚠️ Low" if total_pct > 0 else "❌ None")
        })

    per_metric_df = pd.DataFrame(rows)
    total_summary_df = pd.DataFrame(totals)

    # Global summary
    global_plan = {m: sum(p.get(m.lower(), 0) for p in PLANS.values()) for m in METRICS}
    global_achieved = {m: int(per_metric_df[per_metric_df["Metric"] == m]["Achieved"].sum()) for m in METRICS}
    global_total_plan = sum(global_plan.values())
    global_total_achieved = sum(global_achieved.values())
    global_summary = {
        "by_metric": {
            m: {
                "Plan": global_plan[m],
                "Achieved": global_achieved[m],
                "Achieved %": round((global_achieved[m] / global_plan[m] * 100) if global_plan[m] > 0 else 0, 1)
            } for m in METRICS
        },
        "totals": {
            "Plan Total": global_total_plan,
            "Achieved Total": global_total_achieved,
            "Achieved %": round((global_total_achieved / global_total_plan * 100) if global_total_plan > 0 else 0, 1)
        }
    }

    return per_metric_df, total_summary_df, global_summary

def compute_trend_delta(df, value_col_hint=None, date_col_hint=None):
    """
    Robust computation of previous, delta and delta% for a timeseries dataframe.
    - Will detect the numeric column to use (prefers 'Actual', 'Achieved', 'Value', or first numeric column).
    - Will detect a date column if present (columns containing 'date').
    - Returns a dataframe with columns: Date (if available), Actual, Prev, Delta, Delta %
    Raises KeyError if no numeric column can be found.
    """
    if df is None or df.empty:
        raise KeyError("Input dataframe for trend is empty.")

    df_work = df.copy()

    # Detect date column (if any)
    date_col = None
    if date_col_hint and date_col_hint in df_work.columns:
        date_col = date_col_hint
    else:
        for c in df_work.columns:
            if "date" in c.lower() or "day" in c.lower():
                date_col = c
                break

    if date_col:
        df_work[date_col] = pd.to_datetime(df_work[date_col], errors="coerce")
        df_work = df_work.sort_values(by=date_col).reset_index(drop=True)
    else:
        df_work = df_work.reset_index(drop=True)

    # Detect value column
    candidates = []
    if value_col_hint:
        candidates.append(value_col_hint)
    candidates += ["Actual", "Actuals", "Achieved", "Value", "Count", "Total", "Amount"]
    found_value_col = None
    for cand in candidates:
        if cand in df_work.columns:
            found_value_col = cand
            break

    if not found_value_col:
        # fallback to first numeric column
        numeric_cols = df_work.select_dtypes(include="number").columns.tolist()
        if numeric_cols:
            found_value_col = numeric_cols[0]
        else:
            # try numeric-like columns (strings that parse to numbers)
            for c in df_work.columns:
                try:
                    pd.to_numeric(df_work[c], errors="raise")
                    found_value_col = c
                    break
                except Exception:
                    continue

    if not found_value_col:
        raise KeyError("No numeric column found for trend computation. Expected columns like 'Actual' or 'Achieved' or any numeric column.")

    # Prepare Actual column
    df_work["Actual"] = pd.to_numeric(df_work[found_value_col], errors="coerce").fillna(0).astype(int)

    # Compute Prev, Delta, Delta %
    df_work["Prev"] = df_work["Actual"].shift(1).fillna(0).astype(int)
    df_work["Delta"] = df_work["Actual"] - df_work["Prev"]

    def safe_pct(row):
        prev = row["Prev"]
        if prev == 0:
            # if both prev and actual are zero -> 0%, if prev zero but actual >0 -> show 100% (or Infinity)
            return 100.0 if row["Actual"] > 0 else 0.0
        return round((row["Delta"] / prev) * 100, 1)

    df_work["Delta %"] = df_work.apply(safe_pct, axis=1)

    # Return a clean subset
    cols = []
    if date_col:
        cols.append(date_col)
    cols += ["Actual", "Prev", "Delta", "Delta %"]
    return df_work[cols]

# --- 4. ADMIN SECURITY & UI ---
st.sidebar.title("🔐 Admin Access")
user = st.sidebar.text_input("Username")
pw = st.sidebar.text_input("Password", type="password")

# Login Check
if user == "Belay Melaku" and pw == "@densa1972":
    st.sidebar.success("Welcome, Belay")
    menu = st.sidebar.selectbox("Main Navigation", ["📝 Data Entry", "📊 Performance Dashboard", "⚙️ Export Data"])

    client = get_gs_client()
    sheet = get_records_worksheet(client)

    if menu == "📝 Data Entry":
        st.header("Daily Achievement Entry")
        with st.container():
            col1, col2 = st.columns(2)
            reporter = col1.text_input("Reporter Name *")
            phone = col2.text_input("Phone Number *")
            inst = st.selectbox("Health Institution", INSTITUTIONS)
            date_rep = st.date_input("Report Date", datetime.now())

            st.markdown("---")
            st.subheader("1. Membership Achievement Counts")
            r1, r2, r3, r4 = st.columns(4)
            h_val = r1.number_input("High", min_value=0, step=1)
            m_val = r2.number_input("Medium", min_value=0, step=1)
            f_val = r3.number_input("Free", min_value=0, step=1)
            n_val = r4.number_input("New", min_value=0, step=1)

            calc_money = (h_val * 1710) + (m_val * 1260) + (n_val * 1260)

            st.markdown("---")
            st.subheader("2. Financial Status")
            f1, f2 = st.columns(2)
            f1.metric("Calculated Collected (ETB)", f"{calc_money:,.2f}")
            saved = f2.number_input("Actual Amount Saved to Bank (ETB) *", min_value=0.0)

            if st.button("🚀 Submit Final Report"):
                if not reporter or not phone:
                    st.error("Please fill in Name and Phone.")
                else:
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    new_row = [str(date_rep), reporter, phone, inst, int(h_val), int(m_val), int(f_val), int(n_val), float(calc_money), float(saved), timestamp]

                    if sheet:
                        try:
                            sheet.append_row(new_row)
                            st.success(f"✅ Success! Report for {inst} has been synced.")
                            st.balloons()
                        except Exception:
                            st.error("Failed to append to Google Sheets. See details below.")
                            st.exception(traceback.format_exc())
                    else:
                        st.warning("No Google Sheets connection detected. Preparing downloadable CSV with the report row.")
                        df_local = pd.DataFrame([{
                            "Report Date": str(date_rep),
                            "Reporter": reporter,
                            "Phone": phone,
                            "Health Institution": inst,
                            "High": int(h_val),
                            "Medium": int(m_val),
                            "Free": int(f_val),
                            "New": int(n_val),
                            "Calculated Collected (ETB)": float(calc_money),
                            "Saved to Bank (ETB)": float(saved),
                            "Submitted At": timestamp
                        }])
                        csv_buffer = df_local.to_csv(index=False)
                        st.download_button("📥 Download report as CSV", csv_buffer, file_name="cbhi_report_offline.csv", mime="text/csv")
                        st.success("Local report prepared for download.")

    elif menu == "📊 Performance Dashboard":
        st.header("Real-Time KPI Achievements — Plan vs Achievement")
        # Load raw records
        records_df = pd.DataFrame()
        if sheet:
            try:
                records = sheet.get_all_records()
                records_df = pd.DataFrame(records)
            except Exception:
                st.error("Failed to read data from Google Sheets.")
                st.exception(traceback.format_exc())
                records_df = pd.DataFrame()
        else:
            st.info("No Google Sheets connection. Dashboard will work with local data only (if uploaded).")

        st.markdown("## Upload local CSV (optional)")
        uploaded = st.file_uploader("Upload a CSV of records (will be used instead of sheet data for analysis)", type=["csv"])
        if uploaded:
            try:
                records_df = pd.read_csv(uploaded)
                st.success("Local CSV loaded and will be used for performance calculations.")
            except Exception:
                st.error("Failed to read uploaded CSV.")
                st.exception(traceback.format_exc())

        per_metric_df, total_summary_df, global_summary = compute_performance_from_records(records_df)

        # Display global KPIs
        st.subheader("Global Achievements (All Institutions)")
        gs1, gs2, gs3, gs4 = st.columns(4)
        for i, m in enumerate(METRICS):
            g_plan = global_summary.get("by_metric", {}).get(m, {}).get("Plan", 0)
            g_ach = global_summary.get("by_metric", {}).get(m, {}).get("Achieved", 0)
            g_pct = global_summary.get("by_metric", {}).get(m, {}).get("Achieved %", 0)
            [gs1, gs2, gs3, gs4][i].metric(f"{m}", f"{g_ach:,}/{g_plan:,}", f"{g_pct}%")

        st.markdown("---")
        # Institution-level summary table (totals)
        st.subheader("Institutional Performance Summary")
        st.table(total_summary_df.style.format({"Plan Total": "{:,.0f}", "Achieved Total": "{:,.0f}", "Achieved %": "{:.1f}%"}))

        st.markdown("---")
        # Per-metric long table
        st.subheader("Plan vs Achievement — by Institution & Metric")
        st.dataframe(per_metric_df.sort_values(["Institution","Metric"]).reset_index(drop=True).style.format({"Plan": "{:,.0f}", "Achieved": "{:,.0f}", "Achieved %": "{:.1f}%"}), height=360)

        # Attractive visuals
        st.markdown("---")
        st.subheader("Visual Comparison — Plans vs Achievements")

        # Bar chart: Achieved vs Plan per institution (totals)
        if not total_summary_df.empty:
            fig_totals = px.bar(
                total_summary_df,
                x="Institution",
                y=["Plan Total", "Achieved Total"],
                title="Plan Total vs Achieved Total per Institution",
                barmode="group",
                labels={"value": "Count", "variable": "Series"},
                height=450
            )
            st.plotly_chart(fig_totals, use_container_width=True)

        # Metric breakdown per institution: interactive filter
        st.markdown("### Metric Breakdown per Institution")
        col_left, col_right = st.columns([3,1])
        institution_filter = col_right.selectbox("Select Institution", options=["All Institutions"] + INSTITUTIONS)
        if institution_filter == "All Institutions":
            df_plot = per_metric_df.groupby("Metric").sum().reset_index()
            fig_metric = px.bar(df_plot, x="Metric", y=["Plan","Achieved"], barmode="group", title="Global Plan vs Achieved by Metric", height=420)
        else:
            df_plot = per_metric_df[per_metric_df["Institution"] == institution_filter]
            fig_metric = px.bar(df_plot, x="Metric", y=["Plan","Achieved"], barmode="group", title=f"Plan vs Achieved — {institution_filter}", height=420)
        st.plotly_chart(fig_metric, use_container_width=True)

        st.markdown("---")
        # Timeseries trend (safe)
        st.subheader("Timeseries Trend — total Achievements over time")
        show_trend = st.checkbox("Show trend (requires a date column in records)", value=True)
        ts_df = pd.DataFrame()
        # Build time series by summing metrics per date (best-effort)
        if not records_df.empty:
            # find a date-like column
            date_col_candidates = [c for c in records_df.columns if "date" in c.lower() or "day" in c.lower()]
            date_col = date_col_candidates[0] if date_col_candidates else None

            if date_col:
                tmp = records_df.copy()
                # normalize metric columns to numeric
                for m in METRICS:
                    if m not in tmp.columns and m.lower() in tmp.columns:
                        tmp[m] = pd.to_numeric(tmp[m.lower()], errors="coerce").fillna(0)
                    elif m not in tmp.columns:
                        tmp[m] = 0
                    else:
                        tmp[m] = pd.to_numeric(tmp[m], errors="coerce").fillna(0)
                tmp[date_col] = pd.to_datetime(tmp[date_col], errors="coerce")
                ts_df = tmp.groupby(tmp[date_col].dt.date)[METRICS].sum().reset_index().rename(columns={date_col: "Date"})
                # create an 'Actual' column as total achieved across metrics for that date
                ts_df["Actual"] = ts_df[METRICS].sum(axis=1).astype(int)
            else:
                st.info("No date column detected in records. Upload a CSV with a date column (e.g., 'Report Date' or 'Submitted At') to enable trend view.")

        if ts_df.empty:
            st.info("No timeseries data available.")
        else:
            try:
                ts_trend = compute_trend_delta(ts_df, value_col_hint="Actual", date_col_hint="Date") if show_trend else ts_df
                # Display small table and chart
                st.dataframe(ts_trend.head(20))
                fig_ts = px.line(ts_trend, x=ts_trend.columns[0] if "Date" in ts_trend.columns else ts_trend.index, y="Actual", title="Actuals over time", markers=True)
                st.plotly_chart(fig_ts, use_container_width=True)
            except Exception as e:
                st.error("Failed to compute trend. See details below.")
                st.exception(traceback.format_exc())

        st.markdown("---")
        # Allow syncing computed performance to Google Sheets or download CSV
        st.subheader("Export / Sync Performance Measures")
        perf_to_export = per_metric_df.copy()
        # include totals per institution as extra rows
        totals_for_export = total_summary_df.copy()
        totals_for_export = totals_for_export.rename(columns={"Plan Total":"Plan", "Achieved Total":"Achieved", "Achieved %":"Achieved %"})
        totals_for_export["Metric"] = "TOTAL"
        totals_for_export = totals_for_export[["Institution","Metric","Plan","Achieved","Achieved %","Status"]] if "Status" in totals_for_export.columns else totals_for_export[["Institution","Metric","Plan","Achieved","Achieved %"]]
        perf_export_df = pd.concat([perf_to_export, totals_for_export], ignore_index=True, sort=False).fillna("")

        csv_perf = perf_export_df.to_csv(index=False)
        st.download_button("📥 Download Performance CSV", csv_perf, file_name="cbhi_performance.csv", mime="text/csv")

        if client:
            if st.button("🔁 Sync Performance to Google Sheets (creates/updates 'Performance' worksheet)"):
                success = write_performance_to_sheet(client, perf_export_df, sheet_name="Performance")
                if success:
                    st.success("Performance worksheet updated in Google Sheets ✅")
                else:
                    st.error("Failed to update performance worksheet.")
        else:
            st.info("No Google Sheets client — sign in or provide credentials to enable sync.")

        # Show global summary numbers in a compact card
        st.markdown("---")
        st.subheader("Overall Progress")
        cols = st.columns(4)
        totals = global_summary.get("totals", {})
        for i, label in enumerate(["Plan Total", "Achieved Total", "Achieved %", "Status"]):
            if label == "Status":
                status = "✅ On Track" if totals.get("Achieved %", 0) > 70 else ("⚠️ Needs Attention" if totals.get("Achieved %", 0) > 0 else "❌ No Progress")
                cols[i].metric(label, status)
            else:
                val = totals.get(label, "")
                display = f"{val:,}" if isinstance(val, int) else f"{val}"
                cols[i].metric(label, display)

    elif menu == "⚙️ Export Data":
        st.header("Data Management")
        if sheet:
            try:
                df = pd.DataFrame(sheet.get_all_records())
                csv_data = df.to_csv(index=False)
                st.download_button("📥 Download Records as CSV", csv_data, "cbhi_records.csv", mime="text/csv")
            except Exception:
                st.error("Failed to export data from Google Sheets.")
                st.exception(traceback.format_exc())
        else:
            st.info("No Google Sheets connection. Nothing to export.")

else:
    st.info("🏥 Welcome to the CBHI System. Please log in to continue.")