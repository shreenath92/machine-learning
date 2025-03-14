import streamlit as st
import pandas as pd
import joblib
from io import BytesIO
import time

# --- Fix: set_page_config() must be the first Streamlit command ---
st.set_page_config(
    page_title="🩺 Doctor Survey Targeting AI",
    page_icon="📊",
    layout="wide"
)

# --- Load trained model and encoders ---
@st.cache_resource
def load_model():
    return joblib.load('random_forest_model.pkl')

@st.cache_resource
def load_encoders():
    return joblib.load('label_encoders.pkl')

@st.cache_data
def load_data():
    return pd.read_csv('dummy_npi_data.xlsx - Dataset.csv')

model = load_model()
label_encoders = load_encoders()
df = load_data()

# --- Custom CSS for Styling ---
st.markdown("""
    <style>
        .big-font { font-size:20px !important; font-weight: bold; color: #4CAF50; }
        .small-font { font-size:14px !important; color: #555; }
        .stButton>button { background-color: #4CAF50; color: white; border-radius: 10px; font-size: 16px; }
        .stDownloadButton>button { background-color: #007BFF; color: white; border-radius: 10px; font-size: 16px; }
    </style>
""", unsafe_allow_html=True)

# --- Sidebar ---
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/2/26/Medical_caduceus_symbol.svg/1200px-Medical_caduceus_symbol.svg.png", width=80)
    st.title("🔍 About the App")
    st.markdown("This AI-powered tool predicts the best doctors to target for survey participation based on their login patterns.")
    st.divider()
    st.markdown("📊 **How It Works:**")
    st.markdown("""
    - Input preferred survey time ⏰
    - AI model predicts best doctors ✅
    - Download filtered list 📥
    """)

# --- Main Title ---
st.markdown('<p class="big-font">🩺 AI-Powered Doctor Survey Targeting</p>', unsafe_allow_html=True)

# --- Time Selection ---
selected_hour = st.slider("⏰ Select Preferred Survey Time (Hour)", 0, 23, 12)

# --- Prediction Function ---
def predict_doctors(selected_hour):
    df['Login Time'] = pd.to_datetime(df['Login Time'], errors='coerce')
    df['Logout Time'] = pd.to_datetime(df['Logout Time'], errors='coerce')
    df['Login Hour'] = df['Login Time'].dt.hour

    input_data = df.copy()

    # Apply label encoding
    for col in ['State', 'Region', 'Speciality']:
        if col in label_encoders:
            input_data[col] = label_encoders[col].transform(input_data[col].astype(str))

    # Feature Selection
    feature_cols = ['Login Hour', 'State', 'Region', 'Speciality', 'Usage Time (mins)']
    
    # Show a progress bar while predicting
    with st.spinner("🤖 AI is analyzing data... Please wait."):
        progress_bar = st.progress(0)
        for i in range(100):
            time.sleep(0.01)
            progress_bar.progress(i + 1)
        input_data['Prediction'] = model.predict(input_data[feature_cols])

    # Filter doctors
    filtered_df = input_data[(input_data['Login Hour'] == selected_hour) & (input_data['Prediction'] == 1)]
    
    return filtered_df[['NPI', 'State', 'Region', 'Speciality']]

# --- Convert Data to CSV for Download ---
def convert_df_to_csv(df):
    output = BytesIO()
    df.to_csv(output, index=False, encoding='utf-8')
    output.seek(0)
    return output

# --- Prediction & Display ---
if st.button("🔍 Get List of Doctors"):
    result_df = predict_doctors(selected_hour)
    if not result_df.empty:
        st.success(f"✅ {len(result_df)} doctors found for the selected hour!")
        st.dataframe(result_df, height=400)
        csv_data = convert_df_to_csv(result_df)
        st.download_button("📥 Download CSV", csv_data, "selected_doctors.csv", "text/csv")
        st.balloons()
    else:
        st.warning("⚠️ No doctors found for the selected time.")
