import streamlit as st
import pandas as pd
import joblib
import os
import numpy as np

# ==================================================
# 1. PAGE CONFIGURATION
# ==================================================
st.set_page_config(
    page_title="Restaurant Rating Predictor",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================================================
# 2. CUSTOM CSS
# ==================================================
st.markdown("""
<style>
    /* Global styling */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    h1, h2, h3, h4, h5, h6, p, span {
        font-family: 'Inter', 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    /* Hero section */
    .hero-container {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 3rem 2rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 8px 16px rgba(0,0,0,0.1);
    }
    .hero-title {
        font-size: 2.8rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    .hero-subtitle {
        font-size: 1.3rem;
        font-weight: 500;
        margin-bottom: 1rem;
        color: #e0e0e0;
    }
    .hero-desc {
        font-size: 1rem;
        color: #cfd8dc;
    }
    
    /* Section headers */
    .section-header {
        font-size: 1.8rem;
        font-weight: 600;
        color: #1a237e;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #e8eaf6;
        padding-bottom: 0.5rem;
    }
    
    /* Cards */
    .custom-card {
        background-color: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        border: 1px solid #f0f0f0;
        margin-bottom: 1.5rem;
    }
    .card-title {
        color: #283593;
        font-weight: 600;
        font-size: 1.2rem;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    /* Prediction Result Card */
    .result-card {
        background: linear-gradient(135deg, #4b6cb7 0%, #182848 100%);
        padding: 2.5rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        box-shadow: 0 10px 25px rgba(0,0,0,0.15);
        margin: 2rem 0;
    }
    .result-title {
        font-size: 1.5rem;
        font-weight: 600;
        color: #e8eaf6;
        margin-bottom: 1rem;
    }
    .result-value {
        font-size: 4.5rem;
        font-weight: 800;
        margin: 0;
        text-shadow: 2px 2px 8px rgba(0,0,0,0.2);
    }
    .result-interpretation {
        font-size: 1.8rem;
        font-weight: 600;
        margin-top: 0.5rem;
        color: #ffca28;
    }
    
    /* Progress bar wrapper */
    .progress-wrapper {
        margin-top: 1.5rem;
        text-align: center;
        max-width: 400px;
        margin-left: auto;
        margin-right: auto;
    }
    
    /* Feature importance bars */
    .feat-bar-container {
        display: flex;
        align-items: center;
        margin-bottom: 1rem;
    }
    .feat-name {
        width: 150px;
        font-weight: 500;
        color: #37474f;
    }
    .feat-bar-bg {
        flex-grow: 1;
        background-color: #e0e0e0;
        height: 12px;
        border-radius: 6px;
        margin: 0 1rem;
        overflow: hidden;
    }
    .feat-bar-fill {
        background: linear-gradient(90deg, #3f51b5, #5c6bc0);
        height: 100%;
        border-radius: 6px;
    }
    .feat-val {
        width: 60px;
        text-align: right;
        font-weight: 600;
        color: #1a237e;
    }
    
    /* Metrics section */
    .metric-card {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        border: 1px solid #e0e0e0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.03);
    }
    .metric-title {
        color: #546e7a;
        font-weight: 600;
        font-size: 1.1rem;
    }
    .metric-value {
        color: #1a237e;
        font-weight: 700;
        font-size: 2rem;
        margin-top: 0.5rem;
    }
    
    /* Workflow */
    .workflow-container {
        display: flex;
        justify-content: space-between;
        align-items: center;
        background-color: white;
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #e0e0e0;
        overflow-x: auto;
    }
    .workflow-step {
        text-align: center;
        padding: 1rem;
        background-color: #f5f5f5;
        border-radius: 8px;
        font-weight: 500;
        color: #2c3e50;
        min-width: 120px;
        position: relative;
    }
    .workflow-arrow {
        color: #9e9e9e;
        font-size: 1.5rem;
        font-weight: bold;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem;
        color: #78909c;
        font-size: 0.95rem;
        margin-top: 3rem;
        border-top: 1px solid #eeeeee;
    }
</style>
""", unsafe_allow_html=True)

# ==================================================
# 3. UTILITY FUNCTIONS & MODEL LOADING
# ==================================================
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
    if rating >= 4.5:
        return "⭐ Excellent"
    elif rating >= 4.0:
        return "😊 Very Good"
    elif rating >= 3.5:
        return "🙂 Good"
    elif rating >= 3.0:
        return "😐 Average"
    else:
        return "😕 Needs Improvement"

def get_progress_bar_chars(rating, max_rating=5.0, length=20):
    filled = int((rating / max_rating) * length)
    empty = length - filled
    return "█" * filled + "░" * empty

# ==================================================
# 4. SIDEBAR
# ==================================================
def render_sidebar():
    st.sidebar.markdown("<h2>📊 Project Dashboard</h2>", unsafe_allow_html=True)
    
    st.sidebar.markdown("#### MODEL")
    st.sidebar.info("Random Forest Regressor")
    
    st.sidebar.markdown("#### PERFORMANCE")
    col1, col2 = st.sidebar.columns(2)
    col1.metric("R² Score", "0.6454")
    col2.metric("MAE", "0.2425")
    
    st.sidebar.markdown("#### DATASET")
    st.sidebar.success("7,403 rated restaurants")
    
    st.sidebar.markdown("#### FEATURES")
    st.sidebar.success("11 input features")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("#### About the Project")
    st.sidebar.markdown("""
    <div style='color: #455a64; font-size: 0.95rem; line-height: 1.5;'>
    This application predicts restaurant aggregate ratings using restaurant characteristics such as location, cuisine, price range, votes and available services.
    </div>
    """, unsafe_allow_html=True)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("#### Built with:")
    st.sidebar.markdown("🐍 Python<br>🤖 Scikit-learn<br>📊 Pandas<br>🎨 Streamlit", unsafe_allow_html=True)

# ==================================================
# 5. MAIN APPLICATION
# ==================================================
def main():
    render_sidebar()
    
    # HERO SECTION
    st.markdown("""
    <div class="hero-container">
        <div class="hero-title">🍽️ Restaurant Rating Predictor</div>
        <div class="hero-subtitle">Predict a restaurant's expected customer rating using Machine Learning</div>
        <div class="hero-desc">Enter restaurant characteristics and our Random Forest model will estimate the expected aggregate rating.</div>
    </div>
    """, unsafe_allow_html=True)

    if model is None:
        st.error("Error: Could not find the trained model at `models/restaurant_rating_model.pkl`.")
        return

    # RESTAURANT DETAILS SECTION
    st.markdown("<div class='section-header'>🍽️ Restaurant Details</div>", unsafe_allow_html=True)
    st.markdown("<p style='color: #607d8b; margin-bottom: 1.5rem;'>Provide the restaurant information below</p>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="custom-card">
            <div class="card-title">📍 Location</div>
        </div>
        """, unsafe_allow_html=True)
        country_code = st.number_input("Country Code", min_value=1, value=1, help="Numerical code for the country")
        city = st.selectbox("City", options=valid_cities, index=default_city_index, help="Select the valid training city")
        longitude = st.number_input("Longitude", value=77.2, help="Longitudinal coordinates")
        latitude = st.number_input("Latitude", value=28.6, help="Latitudinal coordinates")

    with col2:
        st.markdown("""
        <div class="custom-card">
            <div class="card-title">🍴 Restaurant Information</div>
        </div>
        """, unsafe_allow_html=True)
        cuisines = st.text_input("Cuisines", value="North Indian, Chinese", help="Comma-separated cuisines")
        avg_cost = st.number_input("Average Cost for two", min_value=0, value=500, help="Expected cost for two people")
        price_range = st.selectbox("Price range", options=[1, 2, 3, 4], help="Categorical price range (1=Cheap, 4=Expensive)")
        votes = st.number_input("Votes", min_value=0, value=50, help="Number of customer votes/reviews")

    with col3:
        st.markdown("""
        <div class="custom-card">
            <div class="card-title">🛎️ Services</div>
        </div>
        """, unsafe_allow_html=True)
        has_table_booking = st.selectbox("Has Table booking?", options=["Yes", "No"], help="Can you book a table?")
        has_online_delivery = st.selectbox("Has Online delivery?", options=["Yes", "No"], help="Do they deliver online?")
        is_delivering_now = st.selectbox("Is delivering now?", options=["Yes", "No"], help="Are they currently delivering?")

    # PREDICTION BUTTON
    st.markdown("<br><br>", unsafe_allow_html=True)
    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    with col_btn2:
        predict_pressed = st.button("⭐ Predict Restaurant Rating", type="primary", use_container_width=True)

    # PREDICTION RESULT
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
            bar_chars = get_progress_bar_chars(prediction_rounded, max_rating=5.0)
            
            st.markdown(f"""
            <div class="result-card">
                <div class="result-title">⭐ Predicted Restaurant Rating</div>
                <div class="result-value">{prediction_rounded} / 5.0</div>
                <div class="result-interpretation">{interpretation}</div>
                <div class="progress-wrapper">
                    <span style="font-family: monospace; font-size: 1.2rem; color: #bbdefb; letter-spacing: 2px;">
                        {bar_chars}
                    </span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.success("✅ Prediction generated successfully!")
            
        except Exception as e:
            st.error(f"⚠️ An error occurred during prediction: {e}")

    # ADDITIONAL INFORMATION
    st.markdown("<br><br>", unsafe_allow_html=True)
    col_info1, col_info2 = st.columns(2, gap="large")
    
    with col1:
        st.markdown("<div class='section-header'>📊 Rating Guide</div>", unsafe_allow_html=True)
        st.markdown("""
        <div style="background: white; padding: 1.5rem; border-radius: 10px; border: 1px solid #eee;">
            <table style="width: 100%; border-collapse: collapse; font-size: 1.05rem;">
                <tr style="border-bottom: 1px solid #f0f0f0;"><td style="padding: 0.8rem 0; color: #455a64;">4.5 – 5.0</td><td style="padding: 0.8rem 0; font-weight: 600;">⭐ Excellent</td></tr>
                <tr style="border-bottom: 1px solid #f0f0f0;"><td style="padding: 0.8rem 0; color: #455a64;">4.0 – 4.49</td><td style="padding: 0.8rem 0; font-weight: 600;">😊 Very Good</td></tr>
                <tr style="border-bottom: 1px solid #f0f0f0;"><td style="padding: 0.8rem 0; color: #455a64;">3.5 – 3.99</td><td style="padding: 0.8rem 0; font-weight: 600;">🙂 Good</td></tr>
                <tr style="border-bottom: 1px solid #f0f0f0;"><td style="padding: 0.8rem 0; color: #455a64;">3.0 – 3.49</td><td style="padding: 0.8rem 0; font-weight: 600;">😐 Average</td></tr>
                <tr><td style="padding: 0.8rem 0; color: #455a64;">Below 3.0</td><td style="padding: 0.8rem 0; font-weight: 600;">😕 Needs Improvement</td></tr>
            </table>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown("<div class='section-header'>📈 Model Insights</div>", unsafe_allow_html=True)
        st.markdown("""
        <div style="background: white; padding: 1.5rem; border-radius: 10px; border: 1px solid #eee;">
            <div class="feat-bar-container">
                <div class="feat-name">Votes</div>
                <div class="feat-bar-bg"><div class="feat-bar-fill" style="width: 41.71%;"></div></div>
                <div class="feat-val">41.71%</div>
            </div>
            <div class="feat-bar-container">
                <div class="feat-name">Cuisines</div>
                <div class="feat-bar-bg"><div class="feat-bar-fill" style="width: 15.22%;"></div></div>
                <div class="feat-val">15.22%</div>
            </div>
            <div class="feat-bar-container">
                <div class="feat-name">Longitude</div>
                <div class="feat-bar-bg"><div class="feat-bar-fill" style="width: 14.30%;"></div></div>
                <div class="feat-val">14.30%</div>
            </div>
            <div class="feat-bar-container">
                <div class="feat-name">Latitude</div>
                <div class="feat-bar-bg"><div class="feat-bar-fill" style="width: 9.19%;"></div></div>
                <div class="feat-val">9.19%</div>
            </div>
            <div class="feat-bar-container" style="margin-bottom: 0;">
                <div class="feat-name">Country Code</div>
                <div class="feat-bar-bg"><div class="feat-bar-fill" style="width: 6.75%;"></div></div>
                <div class="feat-val">6.75%</div>
            </div>
            <p style="margin-top: 1.5rem; font-size: 0.85rem; color: #78909c; font-style: italic;">
                * Feature importance indicates predictive importance, not direct causation.
            </p>
        </div>
        """, unsafe_allow_html=True)

    # MODEL PERFORMANCE
    st.markdown("<div class='section-header'>🤖 Model Performance</div>", unsafe_allow_html=True)
    st.markdown("<p style='color: #546e7a; margin-bottom: 1.5rem;'>The Random Forest model explains approximately 64.5% of the variation in restaurant ratings on the test dataset.</p>", unsafe_allow_html=True)
    
    mp1, mp2, mp3 = st.columns(3)
    with mp1:
        st.markdown("""<div class="metric-card"><div class="metric-title">R² Score</div><div class="metric-value">0.6454</div></div>""", unsafe_allow_html=True)
    with mp2:
        st.markdown("""<div class="metric-card"><div class="metric-title">MAE</div><div class="metric-value">0.2425</div></div>""", unsafe_allow_html=True)
    with mp3:
        st.markdown("""<div class="metric-card"><div class="metric-title">RMSE</div><div class="metric-value">0.3312</div></div>""", unsafe_allow_html=True)
        
    # WORKFLOW
    st.markdown("<div class='section-header'>🔄 Machine Learning Workflow</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class="workflow-container">
        <div class="workflow-step">Dataset</div>
        <div class="workflow-arrow">→</div>
        <div class="workflow-step">Data Cleaning</div>
        <div class="workflow-arrow">→</div>
        <div class="workflow-step">EDA</div>
        <div class="workflow-arrow">→</div>
        <div class="workflow-step">Feature Engineering</div>
        <div class="workflow-arrow">→</div>
        <div class="workflow-step">Preprocessing</div>
        <div class="workflow-arrow">→</div>
        <div class="workflow-step" style="background: #e8eaf6; color: #1a237e; font-weight: 600;">Random Forest</div>
        <div class="workflow-arrow">→</div>
        <div class="workflow-step" style="background: #e8f5e9; color: #1b5e20; font-weight: 600;">Prediction</div>
    </div>
    """, unsafe_allow_html=True)

    # FOOTER
    st.markdown("""
    <div class="footer">
        <strong>Restaurant Rating Prediction</strong><br>
        Built using Python • Scikit-learn • Streamlit<br>
        <span style="color: #90a4ae; font-size: 0.85rem;">Cognifyz Technologies Internship Project</span>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
