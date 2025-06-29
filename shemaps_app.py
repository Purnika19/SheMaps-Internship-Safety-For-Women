import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from math import radians, sin, cos, sqrt, atan2
import joblib

# --- Load Files ---
merged_df = pd.read_csv("merged_dataset.csv")
coords_df = pd.read_csv("All_cities_df.csv")
model = joblib.load("city_recommender_model.pkl")

# Rename 'cities' to 'City' in coords_df
coords_df.rename(columns={'cities': 'City'}, inplace=True)

# --- Helper Functions ---
def dms_to_decimal(dms):
    parts = dms.replace('°', ' ').replace('′', ' ').replace('″', ' ').replace('N','').replace('S','').replace('E','').replace('W','').split()
    degrees = float(parts[0])
    minutes = float(parts[1]) if len(parts) > 1 else 0
    seconds = float(parts[2]) if len(parts) > 2 else 0
    decimal = degrees + minutes / 60 + seconds / 3600
    if 'S' in dms or 'W' in dms:
        decimal *= -1
    return decimal

def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1-a))

# --- Page Setup ---
st.set_page_config(page_title="SheMaps", layout="wide")
st.markdown(
    """
    <style>
    body {
        background-color: #f5f5f5;
        color: #333333;
    }

    h1, h2, h3 {
        color: #7B2CBF;
        font-family: 'Segoe UI', sans-serif;
        transition: color 0.3s ease-in-out;
    }

    h1:hover, h2:hover, h3:hover {
        color: #D63384;
    }

    .stButton>button {
        border-radius: 10px;
        background-color: #6f42c1;
        color: white;
        padding: 0.5em 1em;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #5a32a3;
        color: #fff;
        transition: all 0.3s ease-in-out;
    }

    /* Keep only sliders' label white */
    section[data-testid="stSidebar"] label, .stSlider label {
        color: white !important;
        font-weight: 600;
    }

    /* Dataframe header styling */
    .stDataFrame th {
        background-color: #f3e9ff !important;
        color: #4b0082 !important;
    }

    hr {
        border: 0;
        height: 2px;
        background: linear-gradient(to right, #9d4edd, #7b2cbf, #5a189a);
        margin: 20px 0;
    }

    footer {
        color: #6c757d;
        font-size: 0.85rem;
        text-align: center;
        margin-top: 30px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.image("sheMaps_logo.png", width=100)
st.title("🗺️ SheMaps")
st.markdown("### Where Safety Meets Success ✨")
st.markdown("---")


# --- User Inputs ---
st.markdown("### 📍 Select Your Home City")
user_city = st.selectbox("", sorted(coords_df['City'].unique().tolist()), label_visibility="collapsed")

st.markdown("### 🏙️ Choose 4 Internship Cities")
city_options = sorted(merged_df['City'].unique().tolist())
selected_cities = st.multiselect("", city_options, max_selections=4, label_visibility="collapsed")

st.markdown("### ⚖️ Prioritize What's Important To You")
safety_weight = st.slider("🔒 Safety", 0, 10, 5)
affordability_weight = st.slider("💰 Affordability", 0, 10, 5)
distance_weight = st.slider("📍 Distance from Home", 0, 10, 5)

# --- Processing Logic ---
if st.button("🔍 Find Best City"):
    if len(selected_cities) != 4:
        st.error("❌ Please select exactly 4 cities.")
    else:
        # Get home city coordinates
        home_lat, home_lon = coords_df[coords_df['City'] == user_city][['latitude', 'longitude']].values[0]
        home_lat_dec = dms_to_decimal(home_lat)
        home_lon_dec = dms_to_decimal(home_lon)

        target_df = merged_df[merged_df['City'].isin(selected_cities)].copy()
        target_df['decimal_latitude'] = target_df['latitude'].apply(dms_to_decimal)
        target_df['decimal_longitude'] = target_df['longitude'].apply(dms_to_decimal)

        target_df['Distance_from_Home'] = target_df.apply(
            lambda row: haversine(home_lat_dec, home_lon_dec, row['decimal_latitude'], row['decimal_longitude']), axis=1
        )

        # Normalized Scoring
        normalize = lambda col: (col - col.min()) / (col.max() - col.min())
        target_df['Crime_Score'] = 1 - normalize(target_df['Crime_Count'])
        target_df['Rent_Score'] = 1 - normalize(target_df['Avg_Rent_1BHK_per_Month_INR'])
        target_df['Distance_Score'] = 1 - normalize(target_df['Distance_from_Home'])

        target_df['Final_Score'] = (
            target_df['Crime_Score'] * safety_weight +
            target_df['Rent_Score'] * affordability_weight +
            target_df['Distance_Score'] * distance_weight
        )

        # ML Prediction
        ml_input = target_df[['Crime_Count', 'Avg_Rent_1BHK_per_Month_INR', 'Distance_from_Home']].copy()
        ml_input['Safety_Weight'] = safety_weight
        ml_input['Affordability_Weight'] = affordability_weight
        ml_input['Distance_Weight'] = distance_weight
        target_df['ML_Recommended'] = model.predict(ml_input)

        # Remarks
        top_score = target_df['Final_Score'].max()
        def remark(row):
            if row['Final_Score'] == top_score:
                return "✅ Recommended – Best Match"
            elif row['Final_Score'] >= 0.8 * top_score:
                return "🟢 Good Alternative"
            elif row['Final_Score'] >= 0.5 * top_score:
                return "🟠 Consider with Caution"
            else:
                return "🔴 Least Preferred"
        target_df['Remark'] = target_df.apply(remark, axis=1)

        # --- Display Results ---
        st.markdown("## 📊 City Comparison")
        st.dataframe(
            target_df[['City', 'Crime_Count', 'Avg_Rent_1BHK_per_Month_INR', 'Distance_from_Home', 'Final_Score', 'Remark']]
            .sort_values(by='Final_Score', ascending=False).reset_index(drop=True)
        )

        # --- Recommendation ---
        best = target_df.sort_values(by='Final_Score', ascending=False).iloc[0]
        st.markdown("---")
        st.markdown(f"## 🏆 Recommended City: **{best['City']}**")
        st.markdown(f"""
- 🔒 **Crime Rate**: {best['Crime_Count']}
- 💰 **Rent**: ₹{best['Avg_Rent_1BHK_per_Month_INR']:.0f}
- 📍 **Distance**: {best['Distance_from_Home']:.1f} km
- 📊 **Score**: {best['Final_Score']:.2f}
- 📝 **Remark**: {best['Remark']}
        """)

        # --- Summary ---
        st.markdown("## 📝 Summary")
        st.markdown(f"""
✨ *All the best for your internship journey at* **{best['City']}**! 🚀  
**{best['City']}** offers great potential, diverse culture, and exciting career paths.  
Stay safe, network wisely, and enjoy the journey! 🌆💼
        """)

        # --- Radar Chart ---
        st.markdown("## 📈 Radar Chart: City Metrics")
        radar_df = target_df[['City', 'Crime_Score', 'Rent_Score', 'Distance_Score']].copy()
        radar_df = radar_df.melt(id_vars='City', var_name='Metric', value_name='Value')
        fig = px.line_polar(radar_df, r='Value', theta='Metric', color='City', line_close=True)
        st.plotly_chart(fig)

        # --- Safety Tips ---
        st.markdown("---")
        st.markdown("## 🔐 Safety Tips for Women Interns")
        st.markdown("""
- 📍 Share your live location with trusted contacts  
- 🌙 Avoid late-night travel  
- 🛏 Use verified PG platforms  
- 📞 Keep emergency contacts saved  
        """)

        st.markdown("---")
        st.caption("Made with ❤️ by Purnika & Pallika")

 
 # streamlit run shemaps_app.py
