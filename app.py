import streamlit as st
import pandas as pd
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime
import plotly.express as px # Added for better visuals

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
    try:
        scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
        creds_dict = dict(st.secrets["gcp_service_account"])
        creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
        client = gspread.authorize(creds)
        return client.open("CBHI_Data_Database").worksheet("Records")
    except Exception as e:
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
        with st.container(border=True):
            col1, col2 = st.columns(2)
            reporter = col1.text_input("Reporter Name *")
            phone = col2.text_input("Phone Number *")
            inst = st.selectbox("Health Institution", INSTITUTIONS)
            date_rep = st.date_input("Report Date", datetime.now())
            
            st.divider()
            st.subheader("1. Membership Achievement Counts")
            r1, r2, r3, r4 = st.columns(4)
            h_val = r1.number_input("High", min_value=0, step=1)
            m_val = r2.number_input("Medium", min_value=0, step=1)
            f_val = r3.number_input("Free", min_value=0, step=1)
            n_val = r4.number_input("New", min_value=0, step=1)
            
            calc_money = (h_val * 1710) + (m_val * 1260) + (n_val * 1260)
            
            st.divider()
            st.subheader("2. Financial Status")
            f1, f2 = st.columns(2)
            f1.metric("Calculated Collected (ETB)", f"{calc_money:,.2f}")
            saved = f2.number_input("Actual Amount Saved to Bank (ETB) *", min_value=0.0)

            if st.button("🚀 Submit Final Report"):
                if not reporter or not phone:
                    st.error("Please fill in Name and Phone.")
                elif sheet:
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    new_row = [str(date_rep), reporter, phone, inst, h_val, m_val, f_val, n_val, calc_money, saved, timestamp]
                    sheet.append_row(new_row)
                    st.success(f"✅ Success! Report for {inst} has been synced.")
                    st.balloons()

    elif menu == "📊 Performance Dashboard":
        st.header("Real-Time KPI Achievements")
        if sheet:
            df = pd.DataFrame(sheet.get_all_records())
            if not df.empty:
                # [Data Processing and Logic exactly as you had it]
                for col in ["High", "Medium", "Free", "New"]:
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
                
                # Global Metrics
                st.subheader("Global Achievement (All Institutions)")
                m1, m2, m3, m4 = st.columns(4)
                for i, label in enumerate(["High", "Medium", "Free", "New"]):
                    act = df[label].sum()
                    plan = sum(p[label.lower()] for p in PLANS.values())
                    perc = (act/plan*100) if plan > 0 else 0
                    [m1,m2,m3,m4][i].metric(label, f"{int(act)}/{plan}", f"{perc:.1f}%")

                st.divider()
                st.subheader("Institutional Performance Table")
                matrix = []
                for name, p in PLANS.items():
                    i_act = df[df["Health Institution"] == name][["High", "Medium", "Free", "New"]].sum()
                    t_perc = (i_act.sum() / p["total"] * 100) if p["total"] > 0 else 0
                    matrix.append({"Institution": name, "Perf %": f"{t_perc:.1f}%", "Status": "✅ Good" if t_perc > 70 else "⚠️ Low"})
                st.table(pd.DataFrame(matrix))
            else:
                st.info("No data found.")

    elif menu == "⚙️ Export Data":
        st.header("Data Management")
        if sheet:
            df = pd.DataFrame(sheet.get_all_records())
            st.download_button("📥 Download Records as CSV", df.to_csv(index=False), "cbhi_records.csv")

else:
    st.info("🏥 Welcome to the CBHI System. Please log in to continue.")
