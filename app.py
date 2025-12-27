import streamlit as st
import pandas as pd
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime
import io

# --- 1. CONFIGURATION & TARGET DATA ---
st.set_page_config(page_title="CBHI Tracker Pro", layout="wide", page_icon="📊")

# Official Plan Data (Targets)
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

# --- 2. USER INTERFACE ---
st.title("🏥 CBHI Achievement Tracking System")
menu = st.sidebar.selectbox("Main Menu", ["📝 Daily Entry", "📊 Performance Dashboard", "⚙️ Admin & Export"])

# --- PAGE: DATA ENTRY ---
if menu == "📝 Daily Entry":
    st.header("Enter Daily Achievement")
    with st.container(border=True):
        c1, c2 = st.columns(2)
        reporter = c1.text_input("Staff Name *")
        phone = c2.text_input("Contact Phone *")
        inst = st.selectbox("Health Institution", INSTITUTIONS)
        date_rep = st.date_input("Reporting Date", datetime.now())
        
        st.subheader("Membership Counts")
        r1, r2, r3, r4 = st.columns(4)
        h = r1.number_input("High", min_value=0)
        m = r2.number_input("Medium", min_value=0)
        f = r3.number_input("Free", min_value=0)
        n = r4.number_input("New", min_value=0)
        
        # Calculation: High*1710 + Medium*1260 + New*1260 (Free is 0 ETB)
        calc_coll = (h * 1710) + (m * 1260) + (n * 1260)
        
        st.divider()
        st.subheader("Financial Verification")
        f1, f2 = st.columns(2)
        f1.metric("Calculated Collection (ETB)", f"{calc_coll:,.2f}")
        saved = f2.number_input("Amount Saved to Bank (ETB) *", min_value=0.0)

        if st.button("🚀 Sync Report to Database"):
            if reporter and phone:
                sheet = connect_to_gsheets()
                if sheet:
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
                    sheet.append_row([str(date_rep), reporter, phone, inst, h, m, f, n, calc_coll, saved, timestamp])
                    st.success(f"✅ Data for {inst} has been recorded!")
                    st.balloons()

# --- PAGE: PERFORMANCE DASHBOARD ---
elif menu == "📊 Performance Dashboard":
    st.header("KPI Achievement Analysis")
    sheet = connect_to_gsheets()
    if sheet:
        data = sheet.get_all_records()
        df = pd.DataFrame(data)
        
        if not df.empty:
            for col in ["High", "Medium", "Free", "New"]:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

            view = st.selectbox("Select View", ["Total Summary"] + INSTITUTIONS)
            
            if view == "Total Summary":
                act = df[["High", "Medium", "Free", "New"]].sum()
                plan = {k: sum(p[k] for p in PLANS.values()) for k in ["high", "medium", "free", "new"]}
            else:
                act = df[df["Health Institution"] == view][["High", "Medium", "Free", "New"]].sum()
                plan = PLANS[view]

            st.write(f"### Achievement Summary: {view}")
            cols = st.columns(4)
            metrics = [("High", "high"), ("Medium", "medium"), ("Free", "free"), ("New", "new")]
            for i, (label, key) in enumerate(metrics):
                a_val, p_val = int(act[label]), plan[key]
                perc = (a_val / p_val * 100) if p_val > 0 else 0
                cols[i].metric(label, f"{a_val} / {p_val}", f"{perc:.1f}%")
            
            st.divider()
            
            # REVISED MATRIX: Includes Free and Total Calculation (H+M+F+N)
            st.subheader("Institutional Performance Matrix")
            matrix = []
            for name, p in PLANS.items():
                i_df = df[df["Health Institution"] == name]
                i_act = i_df[["High", "Medium", "Free", "New"]].sum()
                
                # Formula: (High + Medium + Free + New) / Total Plan
                total_achieved = i_act["High"] + i_act["Medium"] + i_act["Free"] + i_act["New"]
                total_perc = (total_achieved / p["total"] * 100) if p["total"] > 0 else 0
                
                matrix.append({
                    "Health Post": name,
                    "High (A/P)": f"{int(i_act['High'])}/{p['high']}",
                    "Medium (A/P)": f"{int(i_act['Medium'])}/{p['medium']}",
                    "Free (A/P)": f"{int(i_act['Free'])}/{p['free']}",
                    "New (A/P)": f"{int(i_act['New'])}/{p['new']}",
                    "Total Perf %": f"{total_perc:.1f}%",
                    "Status": "🟢 Excellent" if total_perc >= 90 else "🟡 Good" if total_perc >= 50 else "🛑 Low"
                })
            st.table(pd.DataFrame(matrix))
