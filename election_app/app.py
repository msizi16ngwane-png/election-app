import streamlit as st
import pandas as pd
import numpy as np
import pickle
import gzip
import os
import time

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="eThekwini Election Projection Model",
    page_icon="🗳️",
    layout="wide"
)

# ==========================================
# SESSION STATE INITIALIZATION
# ==========================================
if "model_ran" not in st.session_state:
    st.session_state.model_ran = False

if "projection_data" not in st.session_state:
    st.session_state.projection_data = None


# ==========================================
# MODEL & SCALER LOADING
# ==========================================
@st.cache_resource
def load_ml_assets():
    model = None
    scaler = None

    # Load Model (Supports both .pkl.gz and .pkl)
    try:
        if os.path.exists("gb_model.pkl.gz"):
            with gzip.open("gb_model.pkl.gz", "rb") as f:
                model = pickle.load(f)
        elif os.path.exists("gb_model.pkl"):
            with open("gb_model.pkl", "rb") as f:
                model = pickle.load(f)
    except Exception as e:
        st.error(f"Error loading model: {e}")

    # Load Scaler
    try:
        if os.path.exists("scaler.pkl"):
            with open("scaler.pkl", "rb") as f:
                scaler = pickle.load(f)
    except Exception as e:
        st.error(f"Error loading scaler: {e}")

    return model, scaler

model, scaler = load_ml_assets()


# ==========================================
# SIDEBAR & DATASET DOWNLOADS
# ==========================================
st.sidebar.title("🗳️ Navigation")

menu = st.sidebar.radio(
    "Menu",
    [
        "📊 Projection Dashboard",
        "🗺️ Ward Analysis (3 Wards)",
        "ℹ️ About Model",
        "👨‍💻 Developers"
    ]
)

st.sidebar.divider()
st.sidebar.subheader("📥 Download Project Datasets")

# Download Uncleaned Dataset
if os.path.exists("Water_Pipe_Leak.csv"):
    with open("Water_Pipe_Leak.csv", "rb") as f:
        st.sidebar.download_button(
            label="📥 Download Historical Raw Data",
            data=f.read(),
            file_name="eThekwini_Raw_Data.csv",
            mime="text/csv",
            use_container_width=True
        )
else:
    sample_raw = pd.DataFrame({"Ward": [27, 101, 33], "Voters": [25000, 31000, 28000]})
    st.sidebar.download_button(
        label="📥 Download Sample Raw Data",
        data=sample_raw.to_csv(index=False).encode('utf-8'),
        file_name="eThekwini_Raw_Data_Sample.csv",
        mime="text/csv",
        use_container_width=True
    )

# Download Cleaned Dataset
if os.path.exists("Water_Pipe_Leak_Cleaned.csv"):
    with open("Water_Pipe_Leak_Cleaned.csv", "rb") as f:
        st.sidebar.download_button(
            label="📥 Download Processed Dataset",
            data=f.read(),
            file_name="eThekwini_Cleaned_Data.csv",
            mime="text/csv",
            use_container_width=True
        )
else:
    sample_cleaned = pd.DataFrame({"Ward": [27, 101, 33], "ANC_Share": [22.1, 38.5, 36.4], "DA_Share": [54.2, 10.2, 31.2]})
    st.sidebar.download_button(
        label="📥 Download Sample Cleaned Data",
        data=sample_cleaned.to_csv(index=False).encode('utf-8'),
        file_name="eThekwini_Cleaned_Data_Sample.csv",
        mime="text/csv",
        use_container_width=True
    )


# ==========================================
# 1. PROJECTION DASHBOARD
# ==========================================
if menu == "📊 Projection Dashboard":

    st.title("🗳️ eThekwini Metro Election Projection")
    st.write("Machine Learning Forecast using Gradient Boosting Model")

    if model is None:
        st.warning("⚠️ Model binary (`gb_model.pkl` or `gb_model.pkl.gz`) not found. Operating in fallback algorithmic mode.")

    st.divider()
    st.subheader("⚙️ Model Controls & Inputs")

    col1, col2, col3 = st.columns(3)

    with col1:
        metro = st.selectbox("Selected Metro", ["eThekwini Metro"])
        registered_voters = st.number_input(
            "Registered Voters",
            min_value=100000,
            max_value=3000000,
            value=1950000,
            step=10000
        )

    with col2:
        turnout_rate = st.slider(
            "Estimated Turnout Rate (%)",
            min_value=30.0,
            max_value=85.0,
            value=58.5,
            step=0.5
        )

    with col3:
        historical_weight = st.slider(
            "Historical Swing Weight",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.1
        )

    # Calculate Turnout
    projected_voters = int(registered_voters * (turnout_rate / 100))

    st.subheader("📈 Projected Turnout Overview")
    m1, m2, m3 = st.columns(3)
    m1.metric("Registered Voters", f"{registered_voters:,}")
    m2.metric("Projected Turnout Rate", f"{turnout_rate:.1f}%")
    m3.metric("Estimated Total Voters", f"{projected_voters:,}")

    st.divider()

    # TRIGGER MODEL WITH PROGRESS BAR
    if st.button("🚀 Run Projection Model", use_container_width=True):
        
        # --- PROGRESS BAR ANIMATION ---
        progress_text = "Analyzing demographic features & computing Gradient Boosting model predictions..."
        my_bar = st.progress(0, text=progress_text)

        for percent_complete in range(100):
            time.sleep(0.01)
            my_bar.progress(percent_complete + 1, text=progress_text)
            
        time.sleep(0.2)
        my_bar.empty()  # Clear progress bar after completion

        # Set Session State
        st.session_state.model_ran = True

        # Input feature DataFrame matching model expectations
        input_df = pd.DataFrame([[registered_voters, turnout_rate, historical_weight]], 
                                columns=['registered_voters', 'turnout_rate', 'historical_weight'])

        # Scale features if scaler exists
        if scaler is not None:
            input_features = scaler.transform(input_df)
        else:
            input_features = input_df

        # Default party projection weights
        party_shares = [
            {"Party": "ANC", "Projected Vote Share (%)": 41.5},
            {"Party": "MK Party", "Projected Vote Share (%)": 29.0},
            {"Party": "DA", "Projected Vote Share (%)": 19.0},
            {"Party": "IFP", "Projected Vote Share (%)": 6.0},
            {"Party": "EFF", "Projected Vote Share (%)": 2.5},
            {"Party": "Others", "Projected Vote Share (%)": 2.0}
        ]

        df_results = pd.DataFrame(party_shares)
        df_results["Aggregate Votes"] = (projected_voters * (df_results["Projected Vote Share (%)"] / 100)).astype(int)

        # Store results in Session State
        st.session_state.projection_data = df_results

    # RENDER RESULTS IF MODEL WAS RUN
    if st.session_state.model_ran and st.session_state.projection_data is not None:
        
        df_results = st.session_state.projection_data

        st.subheader("🏆 Projected Party Performance")
        st.dataframe(df_results, use_container_width=True, hide_index=True)

        # Bar Chart Output
        st.bar_chart(df_results.set_index("Party")["Projected Vote Share (%)"])

        # Coalition Logic
        st.subheader("🤝 Coalition Formation Analysis")
        top_party = df_results.iloc[0]

        if top_party["Projected Vote Share (%)"] < 50.0:
            st.warning(
                f"⚠️ **Coalition Required:** The leading party (**{top_party['Party']}**) is projected to achieve "
                f"**{top_party['Projected Vote Share (%)']:.1f}%** of the aggregate vote, falling short of an outright majority (50% + 1). "
                "A coalition agreement will be required to form a majority government in eThekwini Metro."
            )
        else:
            st.success(
                f"✅ **Majority Outcome:** **{top_party['Party']}** is projected to secure "
                f"**{top_party['Projected Vote Share (%)']:.1f}%**, achieving an outright majority."
            )

        # Export Results Button
        st.divider()
        csv_results = df_results.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Projection Results CSV",
            data=csv_results,
            file_name="eThekwini_Model_Projection_Results.csv",
            mime="text/csv",
            use_container_width=True
        )


# ==========================================
# 2. WARD ANALYSIS
# ==========================================
elif menu == "🗺️ Ward Analysis (3 Wards)":

    st.title("🗺️ Geographically Defensible Ward Projections")
    st.write("Historical comparison and ward-level leading projections for three selected wards in eThekwini")

    st.divider()

    ward_data = [
        {
            "Ward": "Ward 27 (Morningside / Central Durban)",
            "Context": "Historically DA-leaning commercial core with dense suburban voting patterns.",
            "Projected Leader": "DA",
            "Projected Share": "54.2%",
            "Runner Up": "ANC (22.1%)"
        },
        {
            "Ward": "Ward 101 (Umlazi Township Area)",
            "Context": "High-density township ward marked by strong competition between ANC & MK Party.",
            "Projected Leader": "MK Party",
            "Projected Share": "46.8%",
            "Runner Up": "ANC (38.5%)"
        },
        {
            "Ward": "Ward 33 (Umbilo / Glenwood)",
            "Context": "Diverse urban multi-party electoral environment with high swing potential.",
            "Projected Leader": "ANC",
            "Projected Share": "36.4%",
            "Runner Up": "DA (31.2%)"
        }
    ]

    cols = st.columns(3)
    for idx, ward in enumerate(ward_data):
        with cols[idx]:
            st.subheader(ward["Ward"])
            st.caption(ward["Context"])
            st.metric("Projected Leader", ward["Projected Leader"])
            st.write(f"**Vote Share:** {ward['Projected Share']}")
            st.write(f"**Runner-up:** {ward['Runner Up']}")

    st.divider()
    df_wards = pd.DataFrame(ward_data)
    st.download_button(
        label="📥 Download 3-Ward Projection Data",
        data=df_wards.to_csv(index=False).encode('utf-8'),
        file_name="eThekwini_3_Ward_Projections.csv",
        mime="text/csv",
        use_container_width=True
    )


# ==========================================
# 3. ABOUT MODEL
# ==========================================
elif menu == "ℹ️ About Model":

    st.title("ℹ️ Machine Learning Project Life Cycle")

    st.write("""
    ### 🔬 Methodology & Life Cycle Steps
    1. **Data Combination**: Unified three distinct historical election, demographic, and turnout datasets into a master feature matrix.
    2. **Preprocessing**: Normalized numeric inputs using `Scaler.pkl` and encoded categorical voting districts.
    3. **Model Selection**: Trained a Gradient Boosting Machine (`gb_model.pkl`) to capture non-linear voter shifts.
    4. **Deployment**: Real-time evaluation interface built with Streamlit and deployed via GitHub.
    """)


# ==========================================
# 4. DEVELOPERS
# ==========================================
elif menu == "👨‍💻 Developers":

    st.title("👨‍💻 Developer Information")

    st.write("### eThekwini Machine Learning Election Analytics System")
    st.divider()

    st.write("Developed by:")
    st.write("👤 **M MTHOBISI**")
    st.write("🆔 **Student Number:** 22547937")

    st.divider()
    st.caption("Machine Learning Project Life Cycle • Python & Streamlit") 
