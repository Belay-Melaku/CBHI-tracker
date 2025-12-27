import streamlit as st
import pandas as pd
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime
import plotly.express as px
import io

# --- 1. CONFIGURATION (Admin Control) ---
st.set_page_config(page_title="CBHI Performance Pro", layout="wide", page_icon="📈")

# The "Keys" for calculation - Admin can see these in the Settings tab
COLLECTION_KEYS = {"high": 1710, "medium": 1260, "new": 1260}

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

def connect_to_gsheets():
    try:
        scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
        creds_dict = dict(st.secrets["gcp_service_account"])
        creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
        return gspread.authorize(creds).open("CBHI_Data_Database").worksheet("Records")
    except Exception as e:
        st.error(f"Authentication Error: Check your Secrets configuration.")
        return None

# --- 2. NAVIGATION ---
st.title("🏥 CBHI Smart Performance System")
menu = st.sidebar.selectbox("Navigate", ["📝 Daily Entry", "📊 Performance Dashboard", "🔒 Admin Control"])

# --- PAGE: DAILY ENTRY ---
if menu == "📝 Daily Entry":
    st.header("Daily Achievement Entry")
    with st.form("entry_form"):
        c1, c2 = st.columns(2)
        rep = c1.text_input("Reporter Name")
        inst = c2.selectbox("Institution", list(PLANS.keys()))
        
        r1, r2, r3, r4 = st.columns(4)
        h = r1.number_input("High", 0)
        m = r2.number_input("Medium", 0)
        f = r3.number_input("Free", 0)
        n = r4.number_input("New", 0)
        
        calc = (h * COLLECTION_KEYS["high"]) + (m * COLLECTION_KEYS["medium"]) + (n * COLLECTION_KEYS["new"])
        st.info(f"**Calculated Collection:** {calc:,.2f} ETB")
        bank = st.number_input("Amount Saved to Bank", 0.0)
        
        if st.form_submit_button("🚀 Submit to Database"):
            sheet = connect_to_gsheets()
            if sheet:
                sheet.append_row([str(datetime.now().date()), rep, "N/A", inst, h, m, f, n, calc, bank, str(datetime.now())])
                st.success("Data Synced Successfully!")

# --- PAGE: DASHBOARD (Visuals) ---
elif menu == "📊 Performance Dashboard":
    sheet = connect_to_gsheets()
    if sheet:
        df = pd.DataFrame(sheet.get_all_records())
        if not df.empty:
            for col in ["High", "Medium", "Free", "New"]: df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            
            # --- TOP LEVEL KPI ---
            st.subheader("Global Achievement Metrics")
            m1, m2, m3, m4 = st.columns(4)
            for i, (lbl, key) in enumerate([("High", "high"), ("Medium", "medium"), ("Free", "free"), ("New", "new")]):
                act, plan = df[lbl].sum(), sum(p[key] for p in PLANS.values())
                [m1, m2, m3, m4][i].metric(lbl, f"{int(act)}", f"{(act/plan*100):.1f}% of Plan")

            st.divider()

            # --- VISUAL CHARTS ---
            col_chart1, col_chart2 = st.columns(2)
            
            # 1. Bar Chart: Performance by Institution
            matrix = []
            for name, p in PLANS.items():
                act = df[df["Health Institution"] == name][["High", "Medium", "Free", "New"]].sum().sum()
                matrix.append({"Institution": name, "Achieved %": round((act/p['total']*100), 2)})
            
            fig_bar = px.bar(pd.DataFrame(matrix), x='Institution', y='Achieved %', color='Achieved %',
                             title="Total Performance by Institution (%)", color_continuous_scale='RdYlGn')
            col_chart1.plotly_chart(fig_bar, use_container_width=True)

            # 2. Pie Chart: Category Contribution
            totals = df[["High", "Medium", "Free", "New"]].sum().reset_index()
            totals.columns = ['Category', 'Total']
            fig_pie = px.pie(totals, values='Total', names='Category', title="Overall Achievement Distribution", hole=0.4)
            col_chart2.plotly_chart(fig_pie, use_container_width=True)

# --- PAGE: ADMIN CONTROL (Secure) ---
elif menu == "🔒 Admin Control":
    st.header("Administrative Management")
    pw = st.sidebar.text_input("Enter Admin Password", type="password")
    
    if pw == st.secrets["admin"]["password"]:
        st.success("Admin Access Granted")
        
        tab1, tab2 = st.tabs(["⚙️ Performance Keys & Plans", "📥 Data Extraction"])
        
        with tab1:
            st.subheader("Current Performance Indicators (Targets)")
            st.write("These indicators govern the % calculations in the dashboard.")
            st.dataframe(pd.DataFrame(PLANS).T)
            
            st.subheader("Financial Extraction Keys")
            st.json(COLLECTION_KEYS)
            
        with tab2:
            st.subheader("Master Data Export")
            sheet = connect_to_gsheets()
            if sheet:
                df_raw = pd.DataFrame(sheet.get_all_records())
                st.dataframe(df_raw)
                
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                    df_raw.to_excel(writer, index=False, sheet_name='RawData')
                st.download_button("📥 Download Database (Excel)", data=output.getvalue(), file_name="CBHI_Master_Data.xlsx")
    else:
        st.warning("Please enter the correct password in the sidebar to access Admin tools.")
