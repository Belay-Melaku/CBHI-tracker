import streamlit as st
import pandas as pd
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime
import yagmail
import io

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(page_title="CBHI Performance Tracker", layout="wide", page_icon="📊")

# --- 2. PLAN DATA (Based on user provided metrics) ---
PLAN_DATA = {
    "01 Merged Health Post": {"high": 453, "med": 551, "free": 474, "total": 1478},
    "02 Densa Zuriya Health Post": {"high": 147, "med": 316, "free": 155, "total": 618},
    "03 Derew Health Post": {"high": 456, "med": 557, "free": 478, "total": 1491},
    "04 Wejed Health Post": {"high": 246, "med": 346, "free": 249, "total": 841},
    "06 Gert Health Post": {"high": 237, "med": 298, "free": 255, "total": 790},
    "07 Lenguat Health Post": {"high": 240, "med": 328, "free": 244, "total": 812},
    "08 Alegeta Health Post": {"high": 217, "med": 252, "free": 248, "total": 717},
    "09 Sensa Health Post": {"high": 173, "med": 272, "free": 179, "total": 624}
}

INSTITUTIONS = list(PLAN_DATA.keys())

# --- 3. DATABASE CONNECTION ---
def connect_to_gsheets():
    try:
        scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
        creds_dict = dict(st.secrets["gcp_service_account"])
        creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
        client = gspread.authorize(creds)
        sheet = client.open("CBHI_Data_Database").worksheet("Records")
        return sheet
    except Exception as e:
        st.error(f"❌ Connection Error: {e}")
        return None

# --- 4. MAIN APP LOGIC ---
st.title("🏥 CBHI Daily Report & Performance Tracker")

menu = st.sidebar.selectbox("Navigation Menu", ["📝 Data Entry", "📊 Admin Dashboard"])

# --- DATA ENTRY PAGE ---
if menu == "📝 Data Entry":
    st.header("Daily Reporting Form")
    with st.form("reporting_form", clear_on_submit=True):
        col1, col2 = st.columns(2)
        with col1:
            reporter = st.text_input("Reporter Full Name *")
            phone = st.text_input("Phone Number *")
        with col2:
            inst = st.selectbox("Health Institution", INSTITUTIONS)
            report_date = st.date_input("Report Date", datetime.now())

        st.divider()
        st.subheader("Section 1: Membership Renewals")
        r1, r2, r3 = st.columns(3)
        ren_high = r1.number_input("Higher Paid (Renewal)", min_value=0)
        ren_med = r2.number_input("Medium Paid (Renewal)", min_value=0)
        ren_free = r3.number_input("Free (Renewal)", min_value=0)

        st.subheader("Section 2: New Memberships")
        n1, n2 = st.columns(2)
        new_high = n1.number_input("Higher Paid (New)", min_value=0)
        new_med = n2.number_input("Medium Paid (New)", min_value=0)

        st.subheader("Section 3: Financial Totals (ETB)")
        f1, f2 = st.columns(2)
        collected = f1.number_input("Total Money Collected", min_value=0.0)
        saved = f2.number_input("Total Saved to Bank", min_value=0.0)

        if st.form_submit_button("🚀 Submit Daily Report"):
            sheet = connect_to_gsheets()
            if sheet and reporter and phone:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                new_row = [str(report_date), reporter, phone, inst, ren_high, ren_med, ren_free, new_high, new_med, collected, saved, timestamp]
                sheet.append_row(new_row)
                st.success(f"✅ Data saved for {inst}!")

# --- ADMIN DASHBOARD ---
elif menu == "📊 Admin Dashboard":
    admin_user = st.sidebar.text_input("Admin Username")
    admin_pass = st.sidebar.text_input("Admin Password", type="password")

    if admin_user == st.secrets["admin"]["user"] and admin_pass == st.secrets["admin"]["password"]:
        st.header("Performance Indicators vs Plan")
        sheet = connect_to_gsheets()
        if sheet:
            df = pd.DataFrame(sheet.get_all_records())
            if not df.empty:
                # Convert columns to numeric
                num_cols = ["Renew (High)", "Renew (Med)", "Renew (Free)", "New (High)", "New (Med)"]
                for col in num_cols:
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
                
                # Calculate aggregated performance per Institution
                performance_list = []
                for inst, plans in PLAN_DATA.items():
                    inst_df = df[df["Health Institution"] == inst]
                    # Total achievement is sum of Renewals + New Members
                    ach_high = inst_df["Renew (High)"].sum() + inst_df["New (High)"].sum()
                    ach_med = inst_df["Renew (Med)"].sum() + inst_df["New (Med)"].sum()
                    ach_free = inst_df["Renew (Free)"].sum() # Usually new members aren't 'free' unless renewals
                    total_achieved = ach_high + ach_med + ach_free
                    
                    performance_list.append({
                        "Institution": inst,
                        "High Plan": plans["high"],
                        "High Achieved": ach_high,
                        "High %": round((ach_high/plans["high"])*100, 1) if plans["high"] > 0 else 0,
                        "Med Plan": plans["med"],
                        "Med Achieved": ach_med,
                        "Med %": round((ach_med/plans["med"])*100, 1) if plans["med"] > 0 else 0,
                        "Total Plan": plans["total"],
                        "Total Achieved": total_achieved,
                        "Overall %": round((total_achieved/plans["total"])*100, 1)
                    })
                
                perf_df = pd.DataFrame(performance_list)
                st.subheader("Institution Performance Summary")
                st.dataframe(perf_df, use_container_width=True)

                # Visual Achievement Gauges
                st.divider()
                st.subheader("Overall Achievement Highlights")
                selected_inst = st.selectbox("View Details for Institution", INSTITUTIONS)
                row = perf_df[perf_df["Institution"] == selected_inst].iloc[0]
                
                m1, m2, m3 = st.columns(3)
                m1.metric("Total Progress", f"{row['Total Achieved']} / {row['Total Plan']}", f"{row['Overall %']}%")
                m2.metric("High Paid Achievement", f"{row['High Achieved']}", f"{row['High %']}%")
                m3.metric("Medium Paid Achievement", f"{row['Med Achieved']}", f"{row['Med %']}%")
            else:
                st.info("No data available yet.")
    else:
        st.warning("🔒 Please enter Admin credentials.")
