import streamlit as st
import pandas as pd
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime, timedelta
import plotly.express as px
import json
import traceback
from io import StringIO, BytesIO

# --- UI / Page config ---
st.set_page_config(page_title="CBHI Performance Tracker", layout="wide", page_icon="📈")

# --- 1. CONFIGURATION & PLAN DATA ---
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

# --- 2. DATABASE CONNECTION ---
def connect_to_gsheets():
    """
    Connect to Google Sheets using service account credentials stored in Streamlit secrets
    under the key 'gcp_service_account'. The secret may be either a dict or JSON string.
    Returns a worksheet object or None on failure.
    """
    try:
        creds_raw = st.secrets.get("gcp_service_account")
        if not creds_raw:
            st.warning("Google Sheets credentials not found in st.secrets['gcp_service_account']. Running in offline/upload mode.")
            return None

        # Accept dict or JSON string
        if isinstance(creds_raw, str):
            creds_dict = json.loads(creds_raw)
        elif isinstance(creds_raw, dict):
            creds_dict = creds_raw
        else:
            creds_dict = dict(creds_raw)

        scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
        creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
        client = gspread.authorize(creds)
        return client.open("CBHI_Data_Database").worksheet("Records")
    except Exception:
        st.error("Could not connect to Google Sheets. Running in offline/upload mode.")
        st.exception(traceback.format_exc())
        return None

# --- 3. UTILITIES ---
def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize column names and types:
    - Recognize 'Report Date' or similar and parse as datetime
    - Ensure metric columns exist and are numeric
    - Ensure 'Health Institution' column exists
    """
    df = df.copy()
    # Strip column whitespace
    df.columns = [c.strip() if isinstance(c, str) else c for c in df.columns]

    # Find date column
    date_cols = [c for c in df.columns if "date" in c.lower() or "report" in c.lower()]
    if date_cols:
        date_col = date_cols[0]
        try:
            df["Report Date"] = pd.to_datetime(df[date_col], errors="coerce")
        except Exception:
            df["Report Date"] = pd.to_datetime(df[date_col].astype(str), errors="coerce")
    else:
        # If no date column, create one from current timestamp to avoid errors
        df["Report Date"] = pd.NaT

    # Health Institution column detection
    inst_cols = [c for c in df.columns if "institution" in c.lower() or "health" in c.lower() or "facility" in c.lower()]
    if inst_cols:
        df["Health Institution"] = df[inst_cols[0]].astype(str)
    else:
        # Try positional default if available
        if df.shape[1] >= 4:
            df["Health Institution"] = df.iloc[:, 3].astype(str)
        else:
            df["Health Institution"] = ""

    # Metric columns: create/normalize
    for metric in METRICS:
        possible = [c for c in df.columns if c.lower().strip() == metric.lower()]
        if possible:
            df[metric] = pd.to_numeric(df[possible[0]], errors="coerce").fillna(0).astype(int)
        else:
            # try lowercase names
            possible_lower = [c for c in df.columns if c.lower() == metric.lower()]
            if possible_lower:
                df[metric] = pd.to_numeric(df[possible_lower[0]], errors="coerce").fillna(0).astype(int)
            else:
                # If not present, set 0
                df[metric] = 0

    return df

def build_plan_df(plans: dict) -> pd.DataFrame:
    rows = []
    for inst, p in plans.items():
        row = {"Health Institution": inst,
               "Plan_High": p.get("high", 0),
               "Plan_Medium": p.get("medium", 0),
               "Plan_Free": p.get("free", 0),
               "Plan_New": p.get("new", 0),
               "Plan_Total": p.get("total", 0)}
        rows.append(row)
    return pd.DataFrame(rows)

def compute_report(plan_df: pd.DataFrame, actual_df: pd.DataFrame, selected_metrics=None):
    """
    Merge plan and actual and compute percentages.
    Returns a wide table with for each institution: Plan_xxx, Actual_xxx, Perc_xxx.
    """
    if selected_metrics is None:
        selected_metrics = METRICS

    # Aggregate actuals by institution
    agg = actual_df.groupby("Health Institution")[METRICS].sum().reset_index()
    # Ensure all institutions appear
    plan_copy = plan_df.copy()
    merged = plan_copy.merge(agg, left_on="Health Institution", right_on="Health Institution", how="left")
    merged.fillna(0, inplace=True)

    # Create columns
    report_rows = []
    for _, r in merged.iterrows():
        inst = r["Health Institution"]
        row = {"Health Institution": inst}
        total_plan = r.get("Plan_Total", 0)
        total_actual = sum(int(r.get(m, 0)) for m in METRICS)
        row["Plan_Total"] = total_plan
        row["Actual_Total"] = int(total_actual)
        row["Perc_Total"] = (total_actual / total_plan * 100) if total_plan else 0
        for metric in selected_metrics:
            p = int(r.get(f"Plan_{metric}", r.get(metric.lower(), 0)))
            a = int(r.get(metric, 0))
            perc = (a / p * 100) if p > 0 else (100.0 if a == 0 else 0.0)  # if plan 0 and actual 0 treat as 100%, else 0%
            row[f"Plan_{metric}"] = p
            row[f"Actual_{metric}"] = a
            row[f"Perc_{metric}"] = perc
        report_rows.append(row)

    report_df = pd.DataFrame(report_rows)
    # Add an "Overall" totals row
    totals = {"Health Institution": "ALL INSTITUTIONS",
              "Plan_Total": report_df["Plan_Total"].sum(),
              "Actual_Total": report_df["Actual_Total"].sum(),
              "Perc_Total": (report_df["Actual_Total"].sum() / report_df["Plan_Total"].sum() * 100) if report_df["Plan_Total"].sum() else 0}
    for metric in selected_metrics:
        totals[f"Plan_{metric}"] = report_df[f"Plan_{metric}"].sum()
        totals[f"Actual_{metric}"] = report_df[f"Actual_{metric}"].sum()
        # percentage aggregated from sums
        plan_sum = totals[f"Plan_{metric}"]
        actual_sum = totals[f"Actual_{metric}"]
        totals[f"Perc_{metric}"] = (actual_sum / plan_sum * 100) if plan_sum else 0
    report_df = pd.concat([report_df, pd.DataFrame([totals])], ignore_index=True)
    return report_df

def to_excel_bytes(df: pd.DataFrame):
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Report")
    buffer.seek(0)
    return buffer

# --- 4. ADMIN SECURITY ---
st.sidebar.title("🔐 Admin Access")
user = st.sidebar.text_input("Username")
pw = st.sidebar.text_input("Password", type="password")

# Public header (for all users)
st.markdown("<h1 style='text-align:center;'>📈 CBHI Performance Tracker</h1>", unsafe_allow_html=True)
st.markdown("Use the sidebar to log in (admin) or to upload data when offline. The report generator is customizable and provides plan vs achievement across High / Medium / Free / New.")

# Connect to sheet (admin only needed for sync; we'll still attempt)
sheet = connect_to_gsheets()

# If no sheet, allow CSV upload
st.sidebar.markdown("## Data Source")
data_source = st.sidebar.radio("Choose data source", ("Google Sheets (if connected)", "Upload CSV (offline)"))
uploaded_file = None
if data_source.startswith("Upload"):
    uploaded_file = st.sidebar.file_uploader("Upload CSV of records", type=["csv"], help="CSV should contain columns for Reporter, Phone, Health Institution, High, Medium, Free, New, Report Date")

# Login Check
if user == "Belay Melaku" and pw == "@densa1972":
    st.sidebar.success("Welcome, Belay")
    menu = st.sidebar.selectbox("Main Navigation", ["📝 Data Entry", "📊 Performance Dashboard", "⚙️ Export & Custom Reports"])
else:
    menu = st.sidebar.selectbox("Main Navigation", ["📊 Performance Dashboard", "⚙️ Export & Custom Reports"])

# --- Load data: from sheet or upload or none ---
def load_data():
    if data_source == "Google Sheets (if connected)":
        if sheet:
            try:
                records = sheet.get_all_records()
                df = pd.DataFrame(records)
                df = normalize_df(df)
                return df
            except Exception:
                st.error("Failed to read from Google Sheets.")
                st.exception(traceback.format_exc())
                return pd.DataFrame()
        else:
            st.info("No Google Sheets connection. Please upload CSV in the sidebar.")
            return pd.DataFrame()
    else:
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                df = normalize_df(df)
                return df
            except Exception:
                st.error("Failed to parse uploaded CSV.")
                st.exception(traceback.format_exc())
                return pd.DataFrame()
        else:
            return pd.DataFrame()

# Page: Data Entry (admin)
if menu == "📝 Data Entry" and user == "Belay Melaku" and pw == "@densa1972":
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

        # Calculate money (assuming Free category does not generate collection)
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
                        # Try to append row with header mapping considered
                        sheet.append_row(new_row)
                        st.success(f"✅ Success! Report for {inst} has been synced.")
                        st.balloons()
                    except Exception:
                        st.error("Failed to append to Google Sheets. See details below.")
                        st.exception(traceback.format_exc())
                else:
                    # Fallback: provide the row as a downloadable CSV if there's no Google Sheets connection
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

# --- Performance Dashboard ---
if menu == "📊 Performance Dashboard":
    st.header("Real-Time KPI Achievements")
    df = load_data()

    if df.empty:
        st.info("No data available. Use the sidebar to choose Google Sheets or upload a CSV file.")
    else:
        # Filters
        with st.sidebar.expander("Report Filters & Customization", expanded=True):
            insts = st.multiselect("Select Institutions (empty = all)", INSTITUTIONS, default=INSTITUTIONS)
            today = datetime.now().date()
            min_date = df["Report Date"].min().date() if not df["Report Date"].isna().all() else today - timedelta(days=365)
            max_date = df["Report Date"].max().date() if not df["Report Date"].isna().all() else today
            date_range = st.date_input("Date range", value=(min_date, max_date))
            metrics_sel = st.multiselect("Metrics to include", METRICS, default=METRICS)
            aggregation = st.selectbox("Aggregation level", ["By Institution (default)", "By Date", "By Institution & Date"])
            show_charts = st.checkbox("Show charts", value=True)
            highlight_low = st.slider("Highlight institutions below % of plan (Total)", 0, 100, 70)

        # Apply filters
        df_filtered = df.copy()
        if insts:
            df_filtered = df_filtered[df_filtered["Health Institution"].isin(insts)]
        # Date filtering if the Report Date has values
        if not df_filtered["Report Date"].isna().all():
            start_date, end_date = date_range
            mask = (df_filtered["Report Date"].dt.date >= start_date) & (df_filtered["Report Date"].dt.date <= end_date)
            df_filtered = df_filtered[mask]

        plan_df = build_plan_df(PLANS)
        report_df = compute_report(plan_df[plan_df["Health Institution"].isin(insts)] if insts else plan_df, df_filtered, selected_metrics=metrics_sel)

        # Top KPI summary (overall)
        st.subheader("Overall KPI Summary")
        k1, k2, k3, k4 = st.columns(4)
        overall = report_df[report_df["Health Institution"] == "ALL INSTITUTIONS"].iloc[0]
        def kpi_widget(col, metric_name):
            plan = overall.get(f"Plan_{metric_name}", 0) if metric_name!="Total" else overall.get("Plan_Total",0)
            actual = overall.get(f"Actual_{metric_name}", 0) if metric_name!="Total" else overall.get("Actual_Total",0)
            perc = overall.get(f"Perc_{metric_name}", 0) if metric_name!="Total" else overall.get("Perc_Total",0)
            col.metric(f"{metric_name} (Act / Plan)", f"{int(actual):,} / {int(plan):,}", f"{perc:.1f}%")
        kpi_widget(k1, "High")
        kpi_widget(k2, "Medium")
        kpi_widget(k3, "Free")
        kpi_widget(k4, "New")

        st.markdown("---")
        st.subheader("Institutional Performance")
        # Add status column (Good/Low) based on total percentage
        display_df = report_df.copy()
        display_df["Status"] = display_df["Perc_Total"].apply(lambda v: "✅ Good" if v >= highlight_low else "⚠️ Low")
        # Reorder columns for readability
        cols_order = ["Health Institution"]
        for metric in metrics_sel:
            cols_order += [f"Plan_{metric}", f"Actual_{metric}", f"Perc_{metric}"]
        cols_order += ["Plan_Total", "Actual_Total", "Perc_Total", "Status"]
        # Ensure columns exist
        cols_order = [c for c in cols_order if c in display_df.columns]
        st.dataframe(display_df[cols_order].sort_values("Health Institution").reset_index(drop=True), use_container_width=True)

        # Charts
        if show_charts:
            st.markdown("---")
            st.subheader("Visual Comparison - Plan vs Achievement")
            # For each metric, show a grouped bar chart (Plan vs Actual) by institution
            for metric in metrics_sel:
                fig = px.bar(
                    report_df[report_df["Health Institution"] != "ALL INSTITUTIONS"],
                    x="Health Institution",
                    y=[f"Plan_{metric}", f"Actual_{metric}"],
                    barmode="group",
                    title=f"{metric} — Plan vs Actual by Institution",
                    labels={"value": metric, "variable": "Series"}
                )
                fig.update_layout(height=400, xaxis_tickangle=-45, margin=dict(t=50, b=120))
                st.plotly_chart(fig, use_container_width=True)

        # Download buttons
        st.markdown("---")
        st.subheader("Export Report")
        csv_bytes = report_df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download report as CSV", csv_bytes, file_name="cbhi_report_summary.csv", mime="text/csv")
        try:
            excel_buffer = to_excel_bytes(report_df)
            st.download_button("📥 Download report as Excel", excel_buffer, file_name="cbhi_report_summary.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        except Exception:
            st.info("Excel export requires openpyxl. Install it if you want Excel export.")

# --- Export & Custom Reports (more fine-grained) ---
if menu == "⚙️ Export & Custom Reports":
    st.header("Custom Report Generator")
    st.markdown("Create a customized report (plan vs achievement) and export it. Select metrics, institutions, date range, and output format.")

    df = load_data()
    if df.empty:
        st.info("No data available. Choose Google Sheets or upload a CSV in the sidebar.")
    else:
        with st.form("report_form"):
            colA, colB = st.columns(2)
            sel_insts = colA.multiselect("Institutions (leave empty = all)", INSTITUTIONS, default=INSTITUTIONS)
            sel_metrics = colB.multiselect("Metrics to include", METRICS, default=METRICS)
            # Date range
            min_date = df["Report Date"].min().date() if not df["Report Date"].isna().all() else datetime.now().date() - timedelta(days=365)
            max_date = df["Report Date"].max().date() if not df["Report Date"].isna().all() else datetime.now().date()
            drange = st.date_input("Report Date Range", value=(min_date, max_date))
            output_format = st.selectbox("Export format", ["CSV", "Excel"])
            include_charts = st.checkbox("Include charts in the app view", value=True)
            submitted = st.form_submit_button("Generate Report")

        if submitted:
            # Apply filters
            df_filtered = df.copy()
            if sel_insts:
                df_filtered = df_filtered[df_filtered["Health Institution"].isin(sel_insts)]
            if not df_filtered["Report Date"].isna().all():
                start_date, end_date = drange
                mask = (df_filtered["Report Date"].dt.date >= start_date) & (df_filtered["Report Date"].dt.date <= end_date)
                df_filtered = df_filtered[mask]

            plan_df = build_plan_df(PLANS)
            plan_df = plan_df[plan_df["Health Institution"].isin(sel_insts)] if sel_insts else plan_df
            report_df = compute_report(plan_df, df_filtered, selected_metrics=sel_metrics)

            st.success("Report generated")
            st.dataframe(report_df, use_container_width=True)

            if include_charts:
                for metric in sel_metrics:
                    fig = px.bar(
                        report_df[report_df["Health Institution"] != "ALL INSTITUTIONS"],
                        x="Health Institution",
                        y=[f"Plan_{metric}", f"Actual_{metric}"],
                        barmode="group",
                        title=f"{metric} — Plan vs Actual by Institution",
                        labels={"value": metric, "variable": "Series"}
                    )
                    fig.update_layout(height=400, xaxis_tickangle=-45, margin=dict(t=50, b=120))
                    st.plotly_chart(fig, use_container_width=True)

            # Export
            if output_format == "CSV":
                csv_bytes = report_df.to_csv(index=False).encode("utf-8")
                st.download_button("📥 Download CSV", csv_bytes, file_name="cbhi_custom_report.csv", mime="text/csv")
            else:
                try:
                    excel_buffer = to_excel_bytes(report_df)
                    st.download_button("📥 Download Excel", excel_buffer, file_name="cbhi_custom_report.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
                except Exception:
                    st.error("Excel export failed. Ensure openpyxl is installed.")

# Non-admin info message
if not (user == "Belay Melaku" and pw == "@densa1972"):
    st.sidebar.info("Login as admin to add data. Anyone can view and generate reports from uploaded CSV or Google Sheets (if connected).")