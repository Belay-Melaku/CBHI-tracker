import streamlit as st
import pandas as pd
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime
import plotly.express as px

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="CBHI Tracker", layout="wide")

# This matches the names in your uploaded Summary sheet
PLANS = {
    "01 Merged HP": {"high": 453, "medium": 551, "free": 474, "new": 251, "total": 1729},
    "02 Densa Zuriya HP": {"high": 147, "medium": 316, "free": 155, "new": 0, "total": 618},
    "03 Derew HP": {"high": 456, "medium": 557, "free": 478, "new": 429, "total": 1920},
    "04 Wejed HP": {"high": 246, "medium": 346, "free": 249, "new": 0, "total": 841},
    "06 Gert HP": {"high": 237, "medium": 298, "free": 255, "new": 22, "total": 812},
    "07 Lenguat HP": {"high": 240, "medium": 328, "free": 244, "new": 0, "total": 812},
    "08 Alegeta HP": {"high": 217, "medium": 252, "free": 248, "new": 22, "total": 739},
    "09 Sensa HP": {"high": 173, "medium": 272, "free": 179, "new": 0, "total": 624}
}

def connect_to_gsheets():
    try:
        scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
        # Use secrets for GCP credentials
        creds_dict = dict(st.secrets["gcp_service_account"])
        creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
        client = gspread.authorize(creds)
        return client.open("CBHI_Data_Database").worksheet("Records")
    except Exception as e:
        st.error("Google Sheets Connection Failed. Check your secrets.")
        return None

# --- 2. SECURE ADMIN LOGIN ---
st.sidebar.title("🔐 Admin Control")
user_input = st.sidebar.text_input("User Name")
pass_input = st.sidebar.text_input("Password", type="password")

# Verify against secrets
if user_input == st.secrets["admin"]["user"] and pass_input == st.secrets["admin"]["password"]:
    st.sidebar.success(f"Welcome, {user_input}")
    menu = st.sidebar.selectbox("Navigation", ["📊 Performance Graphics", "📝 Data Entry"])
    
    sheet = connect_to_gsheets()
    if sheet:
        df = pd.DataFrame(sheet.get_all_records())
        
        if menu == "📊 Performance Graphics":
            st.title("📈 Performance Visualization")
            if not df.empty:
                # Prepare graph data
                graph_data = []
                for name, p in PLANS.items():
                    # Matching logic using start of string (e.g., '01')
                    act_rows = df[df["Health Institution"].str.contains(name[:2], na=False)]
                    total_act = pd.to_numeric(act_rows[["High", "Medium", "Free", "New"]].sum().sum())
                    graph_data.append({"Institution": name, "Actual": total_act, "Plan": p["total"]})
                
                v_df = pd.DataFrame(graph_data)

                # BAR CHART: Comparison
                fig = px.bar(v_df, x="Institution", y=["Plan", "Actual"], barmode="group",
                             title="Achievement vs Plan by Health Post",
                             color_discrete_sequence=["#D3D3D3", "#1f77b4"])
                st.plotly_chart(fig, use_container_width=True)

                # RUN CHART: Daily Progress
                df['Date'] = pd.to_datetime(df['Date'])
                daily = df.groupby('Date')[["High", "Medium", "Free", "New"]].sum().sum(axis=1).reset_index()
                fig_line = px.line(daily, x="Date", y=0, title="Daily Performance Trend", markers=True)
                fig_line.update_layout(yaxis_title="Total Members Enrolled")
                st.plotly_chart(fig_line, use_container_width=True)
            else:
                st.info("No data available yet.")

        elif menu == "📝 Data Entry":
            # (Standard Data Entry Form logic goes here)
            st.title("📝 Data Entry Form")
            st.write("Submit daily health post achievements below.")
            # ... [Previous form code] ...

else:
    st.title("🏥 CBHI Achievement Tracking System")
    st.warning("Please enter your admin credentials in the sidebar to access the system.")
