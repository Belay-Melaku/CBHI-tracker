import streamlit as st
import pandas as pd
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime, timedelta
import io
import yagmail

# --- PAGE SETTINGS ---
st.set_page_config(page_title="CBHI Daily Report", layout="wide")

# --- LIST OF HEALTH POSTS ---
INSTITUTIONS = [
    "01 Merged Health Post", "02 Densa Zuriya Health Post", 
    "03 Derew Health Post", "04 Wejed Health Post", 
    "06 Gert Health Post", "07 Lenguat Health Post", 
    "08 Alegeta Health Post", "09 Sensa Health Post"
]

# --- CONNECT TO GOOGLE SHEETS ---
@st.cache_resource
def connect_to_gsheets():
    # This connects using the secrets we will set up on Streamlit.com
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    creds = ServiceAccountCredentials.from_json_keyfile_dict(dict(st.secrets["gcp_service_account"]), scope)
    client = gspread.authorize(creds)
    sheet = client.open("CBHI_Data_Database").worksheet("Records")
    return sheet

# --- MAIN APP LAYOUT ---
def main():
    st.title("🏥 CBHI Daily Report Tracking Website")

    # Sidebar Menu
    menu = st.sidebar.radio("Navigate", ["📝 Data Entry", "📊 Dashboard & Admin", "📤 Upload Excel"])

    # 1. DATA ENTRY PAGE
    if menu == "📝 Data Entry":
        st.subheader("Enter Daily Data")
        
        with st.form("entry_form"):
            col1, col2 = st.columns(2)
            reporter_name = col1.text_input("Reporter Full Name (Mandatory)")
            phone_number = col2.text_input("Phone Number (Mandatory)")
            
            report_date = st.date_input("Date of Report", datetime.now())
            institution = st.selectbox("Health Institution", INSTITUTIONS)
            
            st.markdown("---")
            st.write("**1. Membership Renewal**")
            c1, c2, c3 = st.columns(3)
            renew_high = c1.number_input("Higher Paid (Renewal)", min_value=0)
            renew_med = c2.number_input("Medium Paid (Renewal)", min_value=0)
            renew_free = c3.number_input("Free (Renewal)", min_value=0)
            
            st.write("**2. New Membership**")
            c4, c5 = st.columns(2)
            new_high = c4.number_input("Higher Paid (New)", min_value=0)
            new_med = c5.number_input("Medium Paid (New)", min_value=0)
            
            st.write("**3. Financials**")
            c6, c7 = st.columns(2)
            collected = c6.number_input("Money Collected (ETB)", min_value=0.0)
            saved = c7.number_input("Money Saved to Bank (ETB)", min_value=0.0)
            
            submitted = st.form_submit_button("Submit Data")
            
            if submitted:
                if not reporter_name or not phone_number:
                    st.error("Error: You must enter Name and Phone Number.")
                else:
                    try:
                        sheet = connect_to_gsheets()
                        # Add timestamp
                        timestamp = str(datetime.now())
                        row_data = [
                            str(report_date), reporter_name, phone_number, institution,
                            renew_high, renew_med, renew_free, new_high, new_med,
                            collected, saved, timestamp
                        ]
                        sheet.append_row(row_data)
                        st.success("✅ Data successfully sent to database!")
                    except Exception as e:
                        st.error(f"Connection Error: {e}")

    # 2. ADMIN & DASHBOARD PAGE
    elif menu == "📊 Dashboard & Admin":
        st.subheader("Admin Login")
        user_input = st.sidebar.text_input("Admin Name")
        pass_input = st.sidebar.text_input("Password", type="password")
        
        # Check Login (Using Secrets)
        if user_input == st.secrets["admin"]["user"] and pass_input == st.secrets["admin"]["password"]:
            st.success(f"Welcome, {user_input}")
            
            # Load Data
            try:
                sheet = connect_to_gsheets()
                data = sheet.get_all_records()
                df = pd.DataFrame(data)
                
                if not df.empty:
                    # Filters
                    st.write("### Filter Reports")
                    f_inst = st.multiselect("Filter by Institution", INSTITUTIONS)
                    if f_inst:
                        df = df[df["Health Institution"].isin(f_inst)]
                    
                    st.dataframe(df)
                    
                    # Totals
                    st.write("### Aggregated Totals (Sum)")
                    numeric_cols = ["Renew (High)", "Renew (Med)", "Renew (Free)", "New (High)", "New (Med)", "Collected (ETB)", "Saved to Bank (ETB)"]
                    # Ensure columns are numbers
                    for col in numeric_cols:
                        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
                        
                    totals = df[numeric_cols].sum()
                    st.dataframe(totals.to_frame().T)

                    # Download and Email
                    col_d1, col_d2 = st.columns(2)
                    
                    # Excel Download
                    buffer = io.BytesIO()
                    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                        df.to_excel(writer, index=False)
                    
                    col_d1.download_button(
                        label="📥 Download Excel",
                        data=buffer.getvalue(),
                        file_name="CBHI_Report.xlsx",
                        mime="application/vnd.ms-excel"
                    )
                    
                    # Email Feature
                    if col_d2.button("📧 Send to Melaku"):
                        try:
                            yag = yagmail.SMTP(st.secrets["email"]["user"], st.secrets["email"]["password"])
                            yag.send(
                                to="melakubelay47@gmail.com",
                                subject="CBHI Daily Report",
                                contents="Here is the latest report.",
                                attachments=io.BytesIO(buffer.getvalue()) # Send the buffer we just created
                            )
                            st.success("Email sent!")
                        except Exception as e:
                            st.error(f"Email failed. Check App Password settings. Error: {e}")
                            
                else:
                    st.info("No data in the database yet.")
            except Exception as e:
                st.error(f"Database Error: {e}")
        else:
            st.info("Please login to view reports.")

    # 3. UPLOAD PAGE
    elif menu == "📤 Upload Excel":
        st.subheader("Bulk Upload via Excel")
        uploaded_file = st.file_uploader("Choose Excel File", type=['xlsx'])
        if uploaded_file:
            df_up = pd.read_excel(uploaded_file)
            st.write(df_up.head())
            if st.button("Append to Database"):
                sheet = connect_to_gsheets()
                sheet.append_rows(df_up.values.tolist())
                st.success("Uploaded!")

if __name__ == "__main__":
    main()

