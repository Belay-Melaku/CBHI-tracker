import streamlit as st
import pandas as pd
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime
import yagmail
import io

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(page_title="CBHI Daily Report", layout="wide", page_icon="🏥")

# --- 2. CONSTANTS ---
INSTITUTIONS = [
    "01 Merged Health Post", "02 Densa Zuriya Health Post", 
    "03 Derew Health Post", "04 Wejed Health Post", 
    "06 Gert Health Post", "07 Lenguat Health Post", 
    "08 Alegeta Health Post", "09 Sensa Health Post"
]

# --- 3. DATABASE CONNECTION ---
def connect_to_gsheets():
    try:
        scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
        # Convert secrets to a dictionary
        creds_dict = dict(st.secrets["gcp_service_account"])
        creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
        client = gspread.authorize(creds)
        
        # Connect to the Sheet
        # IMPORTANT: Make sure your Sheet Name is EXACTLY this:
        sheet = client.open("CBHI_Data_Database").worksheet("Records")
        return sheet
    except gspread.exceptions.SpreadsheetNotFound:
        st.error("❌ Error: The Google Sheet named 'CBHI_Data_Database' was not found.")
        return None
    except Exception as e:
        st.error(f"❌ Connection Error: {e}")
        return None

# --- 4. MAIN APP LOGIC ---
st.title("🏥 CBHI Daily Report Tracking Website")

menu = st.sidebar.selectbox("Navigation Menu", ["📝 Data Entry", "📊 Admin Dashboard", "📤 Bulk Upload"])

# --- DATA ENTRY PAGE ---
if menu == "📝 Data Entry":
    st.header("Daily Reporting Form")
    st.info("Please fill in all mandatory fields marked with (*)")

    with st.form("reporting_form", clear_on_submit=True):
        col1, col2 = st.columns(2)
        with col1:
            reporter = st.text_input("Reporter Full Name *")
            phone = st.text_input("Phone Number *")
        with col2:
            inst = st.selectbox("Health Institution", INSTITUTIONS)
            report_date = st.date_input("Report Date", datetime.now())

        st.divider()
        
        # Thematic Categories
        st.subheader("Section 1: Membership Renewals")
        r1, r2, r3 = st.columns(3)
        ren_high = r1.number_input("Higher Paid (Renewal)", min_value=0, step=1)
        ren_med = r2.number_input("Medium Paid (Renewal)", min_value=0, step=1)
        ren_free = r3.number_input("Free (Renewal)", min_value=0, step=1)

        st.subheader("Section 2: New Memberships")
        n1, n2 = st.columns(2)
        new_high = n1.number_input("Higher Paid (New)", min_value=0, step=1)
        new_med = n2.number_input("Medium Paid (New)", min_value=0, step=1)

        st.subheader("Section 3: Financial Totals (ETB)")
        f1, f2 = st.columns(2)
        collected = f1.number_input("Total Money Collected", min_value=0.0, format="%.2f")
        saved = f2.number_input("Total Saved to Bank", min_value=0.0, format="%.2f")

        submitted = st.form_submit_button("🚀 Submit Daily Report")

        if submitted:
            if not reporter or not phone:
                st.warning("⚠️ Please enter both your Name and Phone Number to proceed.")
            else:
                sheet = connect_to_gsheets()
                if sheet:
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    new_row = [
                        str(report_date), reporter, phone, inst, 
                        ren_high, ren_med, ren_free, 
                        new_high, new_med, collected, saved, timestamp
                    ]
                    sheet.append_row(new_row)
                    st.success(f"✅ Success! Report for {inst} has been saved.")
                    st.balloons()

# --- ADMIN DASHBOARD ---
elif menu == "📊 Admin Dashboard":
    st.sidebar.markdown("---")
    admin_user = st.sidebar.text_input("Admin Username")
    admin_pass = st.sidebar.text_input("Admin Password", type="password")

    if admin_user == st.secrets["admin"]["user"] and admin_pass == st.secrets["admin"]["password"]:
        st.header("Admin Management Panel")
        
        sheet = connect_to_gsheets()
        if sheet:
            data = sheet.get_all_records()
            df = pd.DataFrame(data)

            if not df.empty:
                # Filtering System
                st.subheader("Filter & Search Reports")
                c1, c2 = st.columns(2)
                selected_inst = c1.multiselect("Filter by Institution", INSTITUTIONS)
                
                # Apply filters
                filtered_df = df.copy()
                if selected_inst:
                    filtered_df = filtered_df[filtered_df["Health Institution"].isin(selected_inst)]
                
                st.dataframe(filtered_df, use_container_width=True)

                # Summary Section
                st.subheader("📈 Aggregated Totals")
                num_cols = ["Renew (High)", "Renew (Med)", "Renew (Free)", "New (High)", "New (Med)", "Collected (ETB)", "Saved to Bank (ETB)"]
                # Convert columns to numeric for calculation
                for col in num_cols:
                    if col in filtered_df.columns:
                        filtered_df[col] = pd.to_numeric(filtered_df[col], errors='coerce').fillna(0)
                
                totals = filtered_df[num_cols].sum().to_frame().T
                st.table(totals)

                # Export and Email
                st.divider()
                e1, e2 = st.columns(2)
                
                # Download
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    filtered_df.to_excel(writer, index=False, sheet_name='CBHI_Report')
                
                e1.download_button(
                    label="📥 Download as Excel",
                    data=output.getvalue(),
                    file_name=f"CBHI_Report_{datetime.now().date()}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

                if e2.button("📧 Email Report to Melaku"):
                    try:
                        yag = yagmail.SMTP(st.secrets["email"]["user"], st.secrets["email"]["password"])
                        yag.send(
                            to="melakubelay47@gmail.com",
                            subject=f"CBHI Daily Report - {datetime.now().date()}",
                            contents="Please find the attached CBHI Daily Report Excel file.",
                            attachments=io.BytesIO(output.getvalue())
                        )
                        st.success("📩 Email sent successfully to melakubelay47@gmail.com")
                    except Exception as e:
                        st.error(f"Failed to send email: {e}")
            else:
                st.info("The database is currently empty.")
    else:
        st.warning("🔒 Please enter Admin credentials in the sidebar to view data.")

# --- BULK UPLOAD ---
elif menu == "📤 Bulk Upload":
    st.header("Excel Data Import")
    st.write("Upload an Excel file to add multiple rows at once.")
    
    file = st.file_uploader("Choose Excel File", type=["xlsx"])
    if file:
        df_upload = pd.read_excel(file)
        st.write("Preview of data to be uploaded:")
        st.dataframe(df_upload.head())
        
        if st.button("Confirm & Upload to Database"):
            sheet = connect_to_gsheets()
            if sheet:
                # Add a timestamp to the uploaded data
                df_upload['Timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                sheet.append_rows(df_upload.values.tolist())
                st.success("✅ Bulk data uploaded successfully!")
