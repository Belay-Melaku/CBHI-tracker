import streamlit as st
import pandas as pd
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime
import yagmail
import io

# --- 1. CONFIGURATION & PLAN DATA ---
st.set_page_config(page_title="CBHI Performance Tracker", layout="wide", page_icon="📈")

# Finalized Plan Data (High, Medium, Free, New)
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
        st.error(f"⚠️ Connection Error: {e}")
        return None

# --- 3. UI NAVIGATION ---
st.title("🏥 CBHI Achievement Tracking System")
menu = st.sidebar.selectbox("Main Navigation", ["📝 Data Entry", "📊 Performance Dashboard", "⚙️ Admin & Export"])

# --- DATA ENTRY PAGE ---
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
        
        # Automated Calculation: High*1710 + Medium*1260 + New*1260
        calc_money = (h_val * 1710) + (m_val * 1260) + (n_val * 1260)
        
        st.divider()
        st.subheader("2. Financial Status")
        f1, f2 = st.columns(2)
        f1.metric("Calculated Collected (ETB)", f"{calc_money:,.2f}")
        saved = f2.number_input("Actual Amount Saved to Bank (ETB) *", min_value=0.0)

        if st.button("🚀 Submit Final Report"):
            if not reporter or not phone:
                st.error("Please fill in Name and Phone.")
            else:
                sheet = connect_to_gsheets()
                if sheet:
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    new_row = [str(date_rep), reporter, phone, inst, h_val, m_val, f_val, n_val, calc_money, saved, timestamp]
                    sheet.append_row(new_row)
                    st.success(f"✅ Success! Report for {inst} has been synced to Google Sheets.")
                    st.balloons()

# --- PERFORMANCE DASHBOARD ---
elif menu == "📊 Performance Dashboard":
    st.header("Real-Time KPI Achievements")
    sheet = connect_to_gsheets()
    if sheet:
        data = sheet.get_all_records()
        df = pd.DataFrame(data)
        
        if not df.empty:
            for col in ["High", "Medium", "Free", "New", "Collected (ETB)", "Saved to Bank (ETB)"]:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

            selected = st.selectbox("Filter by Health Post", ["Summary (All)"] + INSTITUTIONS)
            
            if selected == "Summary (All)":
                actuals = df[["High", "Medium", "Free", "New"]].sum()
                plans = {k: sum(p[k] for p in PLANS.values()) for k in ["high", "medium", "free", "new"]}
                st.subheader("Global Achievement (All Institutions)")
            else:
                actuals = df[df["Health Institution"] == selected][["High", "Medium", "Free", "New"]].sum()
                plans = PLANS[selected]
                st.subheader(f"Performance Details: {selected}")

            # Achievement Metrics
            m1, m2, m3, m4 = st.columns(4)
            metrics = [("High", "high"), ("Medium", "medium"), ("Free", "free"), ("New", "new")]
            cols = [m1, m2, m3, m4]
            
            for i, (label, key) in enumerate(metrics):
                act, plan = int(actuals[label]), plans[key]
                perc = (act / plan * 100) if plan > 0 else 0
                cols[i].metric(label, f"{act} / {plan}", f"{perc:.1f}% achievement")
            
            st.divider()
            
            # Master Performance Matrix
            st.subheader("Institutional Performance Table")
            matrix = []
            for name, p in PLANS.items():
                inst_df = df[df["Health Institution"] == name]
                i_act = inst_df[["High", "Medium", "Free", "New"]].sum()
                total_act = i_act.sum()
                total_perc = (total_act / p["total"] * 100) if p["total"] > 0 else 0
                matrix.append({
                    "Institution": name,
                    "High (Act/Plan)": f"{int(i_act['High'])} / {p['high']}",
                    "Medium (Act/Plan)": f"{int(i_act['Medium'])} / {p['medium']}",
                    "New (Act/Plan)": f"{int(i_act['New'])} / {p['new']}",
                    "Status": "✅ Excellent" if total_perc >= 90 else "⚠️ Warning" if total_perc < 50 else "📊 Progressing",
                    "Total Perf %": f"{total_perc:.1f}%"
                })
            st.table(pd.DataFrame(matrix))
        else:
            st.info("The database is currently empty. Please submit a report first.")
