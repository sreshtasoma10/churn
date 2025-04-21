import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from io import BytesIO
import matplotlib.pyplot as plt
import seaborn as sns
import openpyxl
import base64

# Configure page
st.set_page_config(page_title="Telecom Churn Analysis", layout="wide")

# Custom CSS for dark theme
st.markdown("""
<style>
    /* Main app styling */
    .main {
        background-color: #000000;
        color: #ffffff;
    }
    
    /* Set all text to white */
    body, .css-1v0mbdj, .tab, .stRadio > div, 
    .stFileUploader > div, .stAlert, .stMarkdown, 
    .stDataFrame, .st-expander, .stContainer,
    .stTextInput>div>div>input, .stTextArea>div>div>textarea,
    .stSelectbox>div>div>select, .stNumberInput>div>div>input {
        color: #ffffff !important;
    }
    
    /* Backgrounds for components */
    .css-1v0mbdj, .tab, .stRadio > div, 
    .stFileUploader > div, .stAlert, .st-expander,
    .stDataFrame, .stContainer {
        background-color: #121212 !important;
    }
    
    /* Specific component styling */
    .stButton>button {
        background-color: #4e73df; 
        color: white !important; 
        border-radius: 5px;
    }
    
    .stDownloadButton>button {
        background-color: #1cc88a; 
        color: white !important; 
        border-radius: 5px;
    }
    
    .css-1aumxhk {
        background-color: #121212 !important;
        border-radius: 8px;
        color: #ffffff !important;
    }
    
    /* File uploader styling */
    .stFileUploader > div {
        background-color: #1e1e1e !important;
        border-radius: 8px;
        padding: 20px;
        border: 2px dashed #4e73df;
        transition: all 0.3s ease;
        color: #ffffff !important;
    }
    
    .stFileUploader > div:hover {
        border-color: #2e59d9;
        background-color: #2a2a2a !important;
    }
    
    .stFileUploader > div > div > small {
        color: #aaaaaa !important;
        font-size: 0.9em;
    }
    
    /* Radio button styling */
    .stRadio > div {
        background-color: #1e1e1e !important;
        padding: 10px;
        border-radius: 8px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.3);
        color: #ffffff !important;
    }
    
    /* Sidebar styling */
    .css-1v0mbdj {
        background-color: #121212 !important;
        border-radius: 8px;
        padding: 15px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.3);
        color: #ffffff !important;
    }
    
    /* Tab styling */
    .tab {
        padding: 20px; 
        background-color: #121212 !important; 
        border-radius: 8px; 
        margin: 10px 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.3);
        color: #ffffff !important;
    }
    
    /* Dataframe styling */
    .stDataFrame {
        background-color: #121212 !important;
        color: #ffffff !important;
    }
    
    /* Input fields */
    .stTextInput>div>div>input, .stTextArea>div>div>textarea,
    .stSelectbox>div>div>select, .stNumberInput>div>div>input {
        background-color: #1e1e1e !important;
        color: #ffffff !important;
        border: 1px solid #333333 !important;
    }
    
    /* Matplotlib dark theme */
    plt.style.use('dark_background')
    
    /* Seaborn dark theme */
    sns.set_style("darkgrid")
    sns.set_palette("muted")
</style>
""", unsafe_allow_html=True)

# Define the function to clean the data
def clean_data(df):
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].str.replace(u'\xa0', ' ', regex=False)
            df[col] = df[col].str.replace(r'[^\x00-\x7F]+', '', regex=True)
    return df

class RandomDecisionScorer:
    def __init__(self, thresholds):
        self.thresholds = thresholds

    def predict(self, X):
        predictions = []
        for _, row in X.iterrows():
            score = 0
            if row['DaysSinceLastPurchase'] > self.thresholds['DaysSinceLastPurchase']:
                score += 1
            if row['TotalSpent'] < self.thresholds['TotalSpent']:
                score += 1
            if row['TotalPurchases'] < self.thresholds['TotalPurchases']:
                score += 1
            predictions.append(1 if score >= 2 else 0)
        return predictions

def preprocess_telecom_data(df):
    df.columns = df.columns.str.lower().str.replace(' ', '_')
    
    for col in ['totalcharges', 'monthlycharges']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col].fillna(df[col].median(), inplace=True)
    
    conditions = []
    choices = []
    
    tenure_col = next((col for col in df.columns if 'tenure' in col), None)
    if tenure_col:
        conditions.append(df[tenure_col] < 12)
        choices.append(1)
    
    monthly_col = next((col for col in df.columns if 'monthly' in col), None)
    if monthly_col:
        conditions.append(df[monthly_col] > 70)
        choices.append(1)
    
    contract_col = next((col for col in df.columns if 'contract' in col), None)
    if contract_col:
        conditions.append(df[contract_col].str.contains('Month', case=False, na=False))
        choices.append(1)
    
    if conditions:
        df['churn'] = np.select(conditions, choices, default=0)
    else:
        st.error("Could not identify required columns for churn prediction")
        return None
    
    return df

def train_telecom_model(df):
    X = df.drop('churn', axis=1)
    y = df['churn']
    
    categorical_features = X.select_dtypes(include=['object']).columns
    numerical_features = X.select_dtypes(include=['number']).columns
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])
    
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(
            n_estimators=100, 
            random_state=42,
            class_weight='balanced'
        ))
    ])
    
    model.fit(X, y)
    return model, X, y

def generate_download_link(df, format='csv'):
    if format == 'csv':
        data = df.to_csv(index=False).encode('utf-8')
        mime = 'text/csv'
        ext = 'csv'
    else:
        output = BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False, sheet_name='Predictions')
        data = output.getvalue()
        mime = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        ext = 'xlsx'
    
    b64 = base64.b64encode(data).decode()
    href = f'<a href="data:{mime};base64,{b64}" download="churn_predictions.{ext}" style="color: white !important; background-color: #1cc88a; padding: 8px 12px; border-radius: 5px; text-decoration: none;">Download {ext.upper()} File</a>'
    return href

# Main App
st.title("📊 Customer Churn Analysis System")

# Create two main options
option = st.radio("Select Mode:", ("Analysis", "Prediction"), horizontal=True)

if option == "Analysis":
    st.header("🔍 Customer Churn Analysis")
    
    with st.container():
        st.markdown("### Upload your customer data")
        uploaded_file = st.file_uploader(
            "Drag and drop file here\nLimit 200MB per file - CSV, XLSX", 
            type=["csv", "xlsx"],
            key="analysis_uploader",
            help="Upload your customer transaction data"
        )
    
    if uploaded_file:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file, encoding='utf-8')
            else:
                df = pd.read_excel(uploaded_file, engine='openpyxl')
            
            df = clean_data(df)

            st.success("✅ Data loaded successfully!")
            with st.expander("View Data Preview"):
                st.dataframe(df.head())

            # Analysis options
            st.markdown("## Analysis Options")
            
            if st.button("Generate Churn Dataset", key="churn_button"):
                with st.spinner("Processing data..."):
                    try:
                        if 'InvoiceDate' in df.columns:
                            df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], dayfirst=True)
                            df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]
                            df['IsChurned'] = df['DaysSinceLastPurchase'] > 30
                            
                            st.subheader("Churned Customers")
                            st.dataframe(df[df['IsChurned'] == True].head())
                            
                            csv = df[df['IsChurned'] == True].to_csv(index=False).encode('utf-8')
                            st.download_button(
                                "Download Churn Dataset", 
                                data=csv,
                                file_name='churned_customers.csv',
                                mime='text/csv'
                            )

                    except Exception as e:
                        st.error(f"Error processing the dataset: {e}")

            if st.button("Generate Pincode Analysis", key="pincode_button"):
                with st.spinner("Generating analysis..."):
                    try:
                        if 'Pincode' in df.columns:
                            pincode_analysis = df.groupby('Pincode').agg({
                                'CustomerID': 'count',
                                'TotalSpent': 'sum',
                                'TotalPurchases': 'sum',
                                'Description': lambda x: ', '.join(x.unique())
                            }).reset_index()
                            
                            st.subheader("Pincode-wise Analysis")
                            st.dataframe(pincode_analysis)

                            fig, ax = plt.subplots(figsize=(12, 6))
                            sns.barplot(data=pincode_analysis, x='Pincode', y='CustomerID', ax=ax)
                            ax.set_title('Number of Customers by Pincode', fontsize=14)
                            ax.set_xlabel('Pincode', fontsize=12)
                            ax.set_ylabel('Customer Count', fontsize=12)
                            ax.tick_params(axis='x', rotation=45)
                            st.pyplot(fig)

                    except Exception as e:
                        st.error(f"Error generating pincode analysis: {e}")

        except Exception as e:
            st.error(f"Error reading the file: {e}")

elif option == "Prediction":
    st.header("🔮 Telecom Churn Prediction")
    
    with st.container():
        st.markdown("### Upload telecom dataset")
        uploaded_file = st.file_uploader(
            "Drag and drop file here\nLimit 200MB per file - CSV, XLSX, XLS", 
            type=["csv", "xlsx", "xls"],
            key="prediction_uploader",
            help="Upload your telecom customer data"
        )
    
    if uploaded_file is not None:
        with st.spinner('Processing data...'):
            try:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file, engine='openpyxl')
                
                if df is not None:
                    st.success(f"✅ Data loaded successfully! {len(df)} records found.")
                    
                    df = preprocess_telecom_data(df)
                    
                    if df is not None:
                        model, X, y = train_telecom_model(df)
                        
                        df['churn_prediction'] = model.predict(X)
                        df['churn_probability'] = model.predict_proba(X)[:, 1]
                        
                        st.subheader("Prediction Results")
                        st.dataframe(df.head())
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("### Churn Distribution")
                            fig1, ax1 = plt.subplots()
                            df['churn_prediction'].value_counts().plot(kind='bar', color=['#4e73df', '#e74a3b'], ax=ax1)
                            plt.xticks(rotation=0)
                            st.pyplot(fig1)
                        
                        with col2:
                            st.markdown("### Churn Probability")
                            fig2, ax2 = plt.subplots()
                            df['churn_probability'].hist(bins=20, color='#4e73df', ax=ax2)
                            ax2.set_xlabel('Probability')
                            ax2.set_ylabel('Count')
                            st.pyplot(fig2)
                        
                        st.markdown("### Download Options")
                        st.markdown(generate_download_link(df, 'csv'), unsafe_allow_html=True)
                        st.markdown("  ")  # Spacer
                        st.markdown(generate_download_link(df, 'excel'), unsafe_allow_html=True)
            
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
    else:
        st.info("ℹ️ Please upload a telecom dataset to get started")

# Email notification section
st.sidebar.header("✉️ Email Notification")
with st.sidebar.container():
    st.markdown("### Upload churned customers file")
    uploaded_churned_file = st.file_uploader(
        "Drag and drop file here\nLimit 200MB per file - CSV, XLSX", 
        type=["csv", "xlsx"],
        key="email_uploader",
        help="Upload your list of churned customers"
    )

if uploaded_churned_file:
    try:
        if uploaded_churned_file.name.endswith('.csv'):
            df_churned = pd.read_csv(uploaded_churned_file)
        else:
            df_churned = pd.read_excel(uploaded_churned_file, engine='openpyxl')
        
        st.sidebar.success(f"✅ Loaded {len(df_churned)} churned customers")
        
        with st.sidebar.form("email_form"):
            st.markdown("### Email Settings")
            email = st.text_input("Your Email", placeholder="your.email@example.com")
            password = st.text_input("App Password", type="password", placeholder="Enter your app password")
            
            st.markdown("### Email Content")
            subject = st.text_input("Subject", "We Miss You! Special Offer Inside")
            message = st.text_area("Message", """Dear Valued Customer,

We noticed you haven't visited us recently. We'd love to have you back!
Here's a special 20% discount on your next purchase.

Use code: COMEBACK20

The Team""")
            
            if st.form_submit_button("Send Notification Emails"):
                if not email or not password:
                    st.sidebar.error("Please enter both email and password")
                else:
                    try:
                        server = smtplib.SMTP('smtp.gmail.com', 587)
                        server.starttls()
                        server.login(email, password)

                        for _, row in df_churned.head(10).iterrows():  # Limit to first 10 for demo
                            if 'Email' in row:
                                msg = MIMEMultipart()
                                msg['From'] = email
                                msg['To'] = row['Email']
                                msg['Subject'] = subject
                                msg.attach(MIMEText(message, 'plain', 'utf-8'))

                                server.send_message(msg)
                                st.sidebar.write(f"✉️ Sent to {row['Email']}")

                        server.quit()
                        st.sidebar.success("🎉 Emails sent successfully!")
                    except Exception as e:
                        st.sidebar.error(f"Failed to send emails: {str(e)}")
    
    except Exception as e:
        st.sidebar.error(f"Error loading file: {str(e)}")