import streamlit as st
import pandas as pd
import joblib
import os
import numpy as np
from PIL import Image

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------
st.set_page_config(
    page_title="Restaurant Rating Predictor",
    page_icon="🍽️",
    layout="centered",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------
st.markdown("""
<style>
    /* Main background and font styling */
    .reportview-container {
        background-color: #f8f9fa;
    }
    
    /* Rounded cards for sections */
    .css-1r6slb0, .css-18e3th9 {
        padding: 1.5rem;
        background-color: #ffffff;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        margin-bottom: 1.5rem;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #2c3e50;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    /* Result card */
    .result-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 10px 20px rgba(0,0,0,0.1);
        margin: 2rem 0;
    }
    .result-card h2 {
        color: white;
        margin-bottom: 0.5rem;
        font-size: 1.5rem;
    }
    .result-card .rating {
        font-size: 3.5rem;
        font-weight: bold;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    .result-card .interpretation {
        font-size: 1.5rem;
        margin-top: 0.5rem;
        font-weight: 500;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem;
        color: #7f8c8d;
        font-size: 0.9rem;
        margin-top: 3rem;
        border-top: 1px solid #eee;
    }
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# MODEL LOADING
# --------------------------------------------------
@st.cache_resource
def load_model():
    model_path = os.path.join("models", "restaurant_rating_model.pkl")
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

model = load_model()

@st.cache_data
def get_valid_cities(_model):
    if _model is not None:
        try:
            # Extract categories from the OneHotEncoder inside the pipeline
            cities = _model.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot'].categories_[0]
            return list(cities)
        except Exception:
            return ["New Delhi"]
    return ["New Delhi"]

valid_cities = get_valid_cities(model)
default_city_index = valid_cities.index("New Delhi") if "New Delhi" in valid_cities else 0

def get_interpretation(rating):
    """Returns the descriptive category for a given rating."""
    if rating >= 4.5:
        return "⭐ Excellent"
    elif rating >= 4.0:
        return "🌟 Very Good"
    elif rating >= 3.5:
        return "👍 Good"
    elif rating >= 3.0:
        return "🙂 Average"
    else:
        return "⚠️ Needs Improvement"

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
def render_sidebar():
    st.sidebar.title("📊 Project Information")
    
    st.sidebar.markdown("### Model")
    st.sidebar.info("Random Forest Regressor")
    
    st.sidebar.markdown("### Metrics")
    col1, col2 = st.sidebar.columns(2)
    col1.metric("R² Score", "0.6454")
    col2.metric("MAE", "0.2425")
    
    st.sidebar.markdown("### Dataset")
    st.sidebar.success("7,403 rated restaurants")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### About this project")
    st.sidebar.markdown("""
    This application predicts restaurant aggregate ratings using 
    restaurant characteristics such as votes, cuisine, location, 
    price range and available services.
    """)

# --------------------------------------------------
# MAIN APPLICATION
# --------------------------------------------------
def main():
    render_sidebar()
    
    # 1. HEADER
    st.markdown("<h1 style='text-align: center; color: #2c3e50;'>🍽️ Restaurant Rating Predictor</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; color: #7f8c8d; font-weight: normal; margin-bottom: 2rem;'>Predict a restaurant's expected rating using Machine Learning</h3>", unsafe_allow_html=True)
    
    st.info("Enter restaurant details below and our Random Forest model will estimate the restaurant's aggregate rating.")

    if model is None:
        st.error("Model could not be loaded. Please ensure `models/restaurant_rating_model.pkl` exists.")
        return

    # 3. INPUT SECTION
    st.markdown("## 🏪 Restaurant Details")
    
    # Wrap inputs in a subtle container-like logic using standard markdown dividers
    st.markdown("---")
    
    col1, col2 = st.columns(2, gap="large")
    
    with col1:
        st.markdown("#### Location")
        country_code = st.number_input("Country Code", min_value=1, value=1, help="Numerical code representing the country.")
        city = st.selectbox("City", options=valid_cities, index=default_city_index, help="Select the city where the restaurant is located.")
        longitude = st.number_input("Longitude", value=77.2, help="Longitudinal coordinate.")
        latitude = st.number_input("Latitude", value=28.6, help="Latitudinal coordinate.")
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### Services")
        has_table_booking = st.selectbox("Has Table booking?", options=["Yes", "No"])
        has_online_delivery = st.selectbox("Has Online delivery?", options=["Yes", "No"])
        is_delivering_now = st.selectbox("Is delivering now?", options=["Yes", "No"])

    with col2:
        st.markdown("#### Restaurant Info")
        cuisines = st.text_input("Cuisines", value="North Indian, Chinese", help="Comma-separated list of cuisines.")
        avg_cost = st.number_input("Average Cost for two", min_value=0, value=500, help="Approximate cost for two people.")
        price_range = st.selectbox("Price range", options=[1, 2, 3, 4], help="Categorical price bracket.")
        votes = st.number_input("Votes", min_value=0, value=50, help="Number of user ratings/votes received.")

    st.markdown("---")
    
    # 5. PREDICT BUTTON
    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    with col_btn2:
        predict_pressed = st.button("🔮 Predict Restaurant Rating", type="primary", use_container_width=True)

    # 6. PREDICTION RESULT
    if predict_pressed:
        yes_no_map = {"Yes": 1, "No": 0}
        cuisines_clean = cuisines.strip()
        
        input_data = pd.DataFrame({
            'Country Code': [country_code],
            'City': [city],
            'Longitude': [longitude],
            'Latitude': [latitude],
            'Cuisines': [cuisines_clean],
            'Average Cost for two': [avg_cost],
            'Has Table booking': [yes_no_map[has_table_booking]],
            'Has Online delivery': [yes_no_map[has_online_delivery]],
            'Is delivering now': [yes_no_map[is_delivering_now]],
            'Price range': [price_range],
            'Votes': [votes]
        })
        
        try:
            prediction = model.predict(input_data)[0]
            prediction = np.clip(prediction, 1.8, 4.9)
            prediction_rounded = round(prediction, 2)
            interpretation = get_interpretation(prediction_rounded)
            
            # Custom result card
            st.markdown(f"""
            <div class="result-card">
                <h2>⭐ Predicted Restaurant Rating</h2>
                <p class="rating">{prediction_rounded} / 5.0</p>
                <p class="interpretation">{interpretation}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # 7. SUCCESS MESSAGE
            st.success("✅ Prediction generated successfully!")
            
        except Exception as e:
            st.error(f"⚠️ An error occurred during prediction: {e}")

    # 8. MODEL INSIGHTS
    st.markdown("## 📈 Model Insights")
    st.markdown("""
    **Top features influencing predictions:**
    1. Votes — 41.71%
    2. Cuisines — 15.22%
    3. Longitude — 14.30%
    4. Latitude — 9.19%
    5. Country Code — 6.75%
    
    *These percentages represent aggregated feature importance from the Random Forest model. 
    They indicate predictive importance, not direct causation.*
    """)
    
    # 9. FEATURE IMPORTANCE GRAPH
    plot_path = os.path.join("plots", "feature_importance.png")
    if os.path.exists(plot_path):
        image = Image.open(plot_path)
        st.image(image, caption="Aggregated Feature Importance", use_container_width=True)

    # 10. PROJECT WORKFLOW
    st.markdown("## 🔄 How It Works")
    st.markdown("""
    <div style='text-align: center; font-family: monospace; font-size: 1.1rem; color: #34495e; background: #ecf0f1; padding: 1rem; border-radius: 8px;'>
    Restaurant Details<br>
    ↓<br>
    Data Preprocessing<br>
    ↓<br>
    One-Hot Encoding<br>
    ↓<br>
    Random Forest Model<br>
    ↓<br>
    Predicted Rating
    </div>
    """, unsafe_allow_html=True)

    # 11. FOOTER
    st.markdown("""
    <div class="footer">
        Built as part of the Cognifyz Machine Learning Internship
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
