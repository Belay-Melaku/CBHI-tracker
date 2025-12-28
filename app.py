import streamlit as st
import pandas as pd
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime
import plotly.express as px  # Added for better visuals
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

# --- 2. DATABASE CONNECTION ---
def connect_to_gsheets():
    """
    Connect to Google Sheets using service account credentials stored in Streamlit secrets
    under the key 'gcp_service_account'. The secret may be either a dict or a JSON string.
    Returns the worksheet object or None on failure.
    """
    try:
        creds_raw = st.secrets.get("gcp_service_account")
        if not creds_raw:
            st.warning("Google Sheets credentials not found in st.secrets['gcp_service_account']. Running in offline mode.")
            return None

        # Accept both a dict (recommended) or a JSON string
        if isinstance(creds_raw, str):
            try:
                creds_dict = json.loads(creds_raw)
            except Exception:
                # If it's a multiline secret stored as Streamlit's secret TOML style, try converting
                creds_dict = dict(json.loads(creds_raw))
        elif isinstance(creds_raw, dict):
            creds_dict = creds_raw
        else:
            creds_dict = dict(creds_raw)

        scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
        creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
        client = gspread.authorize(creds)
        return client.open("CBHI_Data_Database").worksheet("Records")
    except Exception as e:
        # Provide the traceback in the app for easier debugging (only for the admin user)
        st.error("Could not connect to Google Sheets. Running in offline mode.")
        st.exception(traceback.format_exc())
        return None

# --- 3. ADMIN SECURITY ---
st.sidebar.title("🔐 Admin Access")
user = st.sidebar.text_input("Username")
pw = st.sidebar.text_input("Password", type="password")

# Login Check
if user == "Belay Melaku" and pw == "@densa1972":
    st.sidebar.success("Welcome, Belay")
    menu = st.sidebar.selectbox("Main Navigation", ["📝 Data Entry", "📊 Performance Dashboard", "⚙️ Export Data"])

    sheet = connect_to_gsheets()

    if menu == "📝 Data Entry":
        st.header("Daily Achievement Entry")
        # st.container does not accept a 'border' kwarg in some Streamlit versions -> use plain container
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

    elif menu == "📊 Performance Dashboard":
        st.header("Real-Time KPI Achievements")
        if sheet:
            try:
                records = sheet.get_all_records()
                df = pd.DataFrame(records)
            except Exception:
                st.error("Failed to read data from Google Sheets.")
                st.exception(traceback.format_exc())
                df = pd.DataFrame()

            if not df.empty:
                # Normalize expected column names (best effort)
                # Ensure numeric columns exist
                for col in ["High", "Medium", "Free", "New"]:
                    if col not in df.columns:
                        # try lowercase alternative
                        if col.lower() in df.columns:
                            df[col] = pd.to_numeric(df[col.lower()], errors='coerce').fillna(0)
                        else:
                            df[col] = 0
                    else:
                        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

                # Global Metrics
                st.subheader("Global Achievement (All Institutions)")
                m1, m2, m3, m4 = st.columns(4)
                for i, label in enumerate(["High", "Medium", "Free", "New"]):
                    act = int(df[label].sum())
                    plan = sum(p.get(label.lower(), 0) for p in PLANS.values())
                    perc = (act / plan * 100) if plan > 0 else 0
                    [m1, m2, m3, m4][i].metric(label, f"{act}/{plan}", f"{perc:.1f}%")

                st.markdown("---")
                st.subheader("Institutional Performance Table")
                matrix = []
                for name, p in PLANS.items():
                    inst_df = df[df.get("Health Institution", "") == name] if "Health Institution" in df.columns else df[df.iloc[:, 3] == name] if df.shape[1] >= 4 else pd.DataFrame()
                    i_act = inst_df[["High", "Medium", "Free", "New"]].sum() if not inst_df.empty else pd.Series({"High": 0, "Medium": 0, "Free": 0, "New": 0})
                    t_perc = (i_act.sum() / p["total"] * 100) if p["total"] > 0 else 0
                    matrix.append({"Institution": name, "Perf %": f"{t_perc:.1f}%", "Status": "✅ Good" if t_perc > 70 else "⚠️ Low"})
                st.table(pd.DataFrame(matrix))
            else:
                st.info("No data found in the worksheet.")
        else:
            st.info("No Google Sheets connection. Dashboard cannot show live data.")

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