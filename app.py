import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="🏠 HDB Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        text-align: center;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        margin: 10px;
        text-align: center;
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def create_dummy_model(model_type):
    """Create a realistic dummy model that has all required methods"""
    class RealisticDummyModel:
        def __init__(self, model_type):
            self.model_type = model_type
            self.n_features_in_ = 9
            self.feature_names_in_ = [
                'floor_area_sqm', 'storey_level', 'flat_age', 'remaining_lease',
                'transaction_year', 'flat_type_encoded', 'town_encoded',
                'flat_model_encoded', 'dummy_feature'
            ]
            self.get_params = lambda deep=True: {}
            self.set_params = lambda **params: self
        
        def predict(self, X):
            if isinstance(X, np.ndarray) and len(X.shape) == 2:
                X = X[0]
            
            floor_area = X[0]
            storey_level = X[1]
            flat_age = X[2]
            town_encoded = X[6]
            
            base_price = floor_area * (4800 + town_encoded * 200)
            storey_bonus = storey_level * 2500
            age_discount = flat_age * 1800
            
            if self.model_type == "xgboost":
                price = base_price + storey_bonus - age_discount + 35000
                if storey_level > 20: price += 15000
                if flat_age < 10: price += 20000
            else:
                price = base_price + storey_bonus - age_discount - 25000
            
            return np.array([max(300000, price)])

    return RealisticDummyModel(model_type)()

@st.cache_resource
def load_model_from_file(filename="best_model_xgboost1.joblib"):
    """Load model from local file with error handling"""
    try:
        if os.path.exists(filename):
            model = joblib.load(filename)
            st.success(f"✅ Successfully loaded model from {filename}")
            
            if not hasattr(model, 'predict'):
                st.error("❌ Loaded object doesn't have predict method")
                return create_dummy_model("xgboost")
                
            # Add missing methods if needed
            if not hasattr(model, 'get_params'):
                model.get_params = lambda deep=True: {}
            if not hasattr(model, 'set_params'):
                model.set_params = lambda **params: model
                
            return model
        else:
            st.warning(f"⚠️  Model file {filename} not found, using dummy model")
            return create_dummy_model("xgboost")
            
    except Exception as e:
        st.error(f"❌ Error loading model from {filename}: {e}")
        return create_dummy_model("xgboost")

@st.cache_data
def create_sample_data():
    """Create sample data for visualization"""
    np.random.seed(42)
    towns = ['ANG MO KIO', 'BEDOK', 'TAMPINES', 'WOODLANDS', 'JURONG WEST', 
             'SENGKANG', 'PUNGGOL', 'YISHUN', 'HOUGANG', 'CHOA CHU KANG']
    flat_types = ['3 ROOM', '4 ROOM', '5 ROOM', 'EXECUTIVE', '2 ROOM']
    flat_models = ['Improved', 'Model A', 'New Generation', 'Standard', 'Premium']
    
    data = []
    for _ in range(200):
        town = np.random.choice(towns)
        flat_type = np.random.choice(flat_types)
        flat_model = np.random.choice(flat_models)
        floor_area = np.random.randint(60, 150)
        storey = np.random.randint(1, 25)
        age = np.random.randint(0, 40)
        
        # Realistic price calculation
        base_price = floor_area * 5000
        town_bonus = towns.index(town) * 15000
        storey_bonus = storey * 2000
        age_discount = age * 1200
        flat_type_bonus = flat_types.index(flat_type) * 25000
        
        resale_price = base_price + town_bonus + storey_bonus - age_discount + flat_type_bonus
        resale_price = max(250000, resale_price + np.random.randint(-15000, 15000))
        
        data.append({
            'town': town, 'flat_type': flat_type, 'flat_model': flat_model,
            'floor_area_sqm': floor_area, 'storey_level': storey,
            'flat_age': age, 'resale_price': resale_price
        })
    
    return pd.DataFrame(data)

def preprocess_input(user_input):
    """Preprocess user input for prediction with correct feature mapping"""
    # Flat type mapping
    flat_type_mapping = {
        '1 ROOM': 1, '2 ROOM': 2, '3 ROOM': 3, '4 ROOM': 4,
        '5 ROOM': 5, 'EXECUTIVE': 6, 'MULTI-GENERATION': 7
    }

    # Town mapping
    town_mapping = {
        'SENGKANG': 0, 'WOODLANDS': 1, 'TAMPINES': 2, 'PUNGGOL': 3,
        'JURONG WEST': 4, 'YISHUN': 5, 'BEDOK': 6, 'HOUGANG': 7,
        'CHOA CHU KANG': 8, 'ANG MO KIO': 9
    }

    # Flat model mapping
    flat_model_mapping = {
        'Model A': 0, 'Improved': 1, 'New Generation': 2,
        'Standard': 3, 'Premium': 4
    }

    # Create input array with features
    input_features = [
        user_input['floor_area_sqm'],           # Feature 1
        user_input['storey_level'],             # Feature 2
        user_input['flat_age'],                 # Feature 3
        99 - user_input['flat_age'],            # Feature 4: remaining_lease
        2024,                                   # Feature 5: transaction_year (current year)
        flat_type_mapping.get(user_input['flat_type'], 4),  # Feature 6
        town_mapping.get(user_input['town'], 0),           # Feature 7
        flat_model_mapping.get(user_input['flat_model'], 0), # Feature 8
        1                                       # Feature 9: (placeholder)
    ]

    return np.array([input_features])

def create_market_insights_chart(data, user_input, predicted_price):
    """Create market insights visualization"""
    if data is None or len(data) == 0:
        return None

    # Filter similar properties
    similar_properties = data[
        (data['flat_type'] == user_input['flat_type']) &
        (data['town'] == user_input['town'])
    ]

    if len(similar_properties) < 5:
        similar_properties = data[data['flat_type'] == user_input['flat_type']]

    if len(similar_properties) > 0:
        fig = px.scatter(
            similar_properties, 
            x='floor_area_sqm', 
            y='resale_price',
            color='flat_model',
            title=f"Market Comparison: {user_input['flat_type']} in {user_input['town']}",
            labels={
                'floor_area_sqm': 'Floor Area (sqm)', 
                'resale_price': 'Resale Price (SGD)',
                'flat_model': 'Flat Model'
            }
        )

        # Add prediction marker
        fig.add_trace(go.Scatter(
            x=[user_input['floor_area_sqm']], 
            y=[predicted_price],
            mode='markers',
            marker=dict(symbol='star', size=20, color='red', line=dict(width=2, color='darkred')),
            name='Your Prediction'
        ))

        fig.update_layout(
            template="plotly_white", 
            height=500,
            showlegend=True,
            font=dict(size=12)
        )
        return fig
    return None

def predict_hdb_price(user_input):
    """Main prediction function"""
    try:
        processed_input = preprocess_input(user_input)
        
        # Get prediction from model
        predicted_price = max(0, float(model.predict(processed_input)[0]))
        
        # Create insights
        remaining_lease = 99 - user_input['flat_age']
        price_per_sqm = predicted_price / user_input['floor_area_sqm']

        insights = f"""
        **Property Summary:**
        - **Location:** {user_input['town']}
        - **Type:** {user_input['flat_type']}
        - **Model:** {user_input['flat_model']}
        - **Area:** {user_input['floor_area_sqm']} sqm
        - **Floor:** Level {user_input['storey_level']}
        - **Age:** {user_input['flat_age']} years
        - **Remaining Lease:** {remaining_lease} years
        - **Price per sqm:** ${price_per_sqm:,.0f}

        **Financing Eligibility:**
        """

        if remaining_lease >= 60:
            insights += "✅ **Bank loan eligible** (≥60 years remaining)"
        elif remaining_lease >= 20:
            insights += "⚠️ **HDB loan eligible only** (20-59 years remaining)"
        else:
            insights += "❌ **Limited financing options** (<20 years remaining)"

        # Create chart
        chart = create_market_insights_chart(data, user_input, predicted_price)

        return predicted_price, insights, chart

    except Exception as e:
        error_msg = f"Prediction failed. Error: {str(e)}"
        st.error(error_msg)
        return None, error_msg, None

# Main app
st.markdown('<h1 class="main-header">🏠 HDB Price Predictor</h1>', unsafe_allow_html=True)
st.markdown("### Estimate HDB resale prices using machine learning")

# Load model and data
with st.spinner("Loading prediction model..."):
    model = load_model_from_file("best_model_xgboost1.joblib")
    data = create_sample_data()

# Define options
towns_list = [
    'ANG MO KIO', 'BEDOK', 'BISHAN', 'BUKIT BATOK', 'BUKIT MERAH',
    'BUKIT PANJANG', 'BUKIT TIMAH', 'CENTRAL AREA', 'CHOA CHU KANG',
    'CLEMENTI', 'GEYLANG', 'HOUGANG', 'JURONG EAST', 'JURONG WEST',
    'KALLANG/WHAMPOA', 'MARINE PARADE', 'PASIR RIS', 'PUNGGOL',
    'QUEENSTOWN', 'SENGKANG', 'SERANGOON', 'TAMPINES', 'TOA PAYOH',
    'WOODLANDS', 'YISHUN'
]

flat_types = ['2 ROOM', '3 ROOM', '4 ROOM', '5 ROOM', 'EXECUTIVE']
flat_models = ['Improved', 'New Generation', 'Model A', 'Standard', 'Premium', 'Simplified', 'Premium Apartment']

# Create input form
with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Property Details")
        town = st.selectbox("Town", sorted(towns_list), index=0)
        flat_type = st.selectbox("Flat Type", flat_types, index=2)
        flat_model = st.selectbox("Flat Model", flat_models, index=0)
    
    with col2:
        st.subheader("Specifications")
        floor_area_sqm = st.slider("Floor Area (sqm)", 30, 200, 100, 5, 
                                 help="Typical HDB sizes: 3-room (60-75 sqm), 4-room (85-105 sqm), 5-room (110-125 sqm)")
        storey_level = st.slider("Storey Level", 1, 50, 8, 1,
                               help="Higher floors typically command premium prices")
        flat_age = st.slider("Flat Age (years)", 0, 50, 10, 1,
                           help="Newer flats generally have higher prices")
    
    # Predict button
    predict_btn = st.form_submit_button("🔮 Predict Resale Price", use_container_width=True)

# Process form submission
if predict_btn:
    user_input = {
        'town': town,
        'flat_type': flat_type,
        'flat_model': flat_model,
        'floor_area_sqm': floor_area_sqm,
        'storey_level': storey_level,
        'flat_age': flat_age
    }
    
    with st.spinner("Analyzing property details..."):
        predicted_price, insights, chart = predict_hdb_price(user_input)
    
    if predicted_price:
        # Display results
        st.markdown("---")
        st.markdown(f'<div class="prediction-box"><h2>💰 Predicted Resale Price: ${predicted_price:,.0f}</h2></div>', unsafe_allow_html=True)
        
        # Create metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            price_per_sqm = predicted_price / floor_area_sqm
            st.metric("Price per sqm", f"${price_per_sqm:,.0f}")
        with col2:
            remaining_lease = 99 - flat_age
            st.metric("Remaining Lease", f"{remaining_lease} years")
        with col3:
            st.metric("Property Age", f"{flat_age} years")
        
        # Show insights
        st.markdown("### 📋 Property Analysis")
        st.markdown(insights)
        
        # Show chart
        if chart:
            st.markdown("### 📈 Market Comparison")
            st.plotly_chart(chart, use_container_width=True)

# Add information section
with st.expander("ℹ️ About This Predictor"):
    st.markdown("""
    **How it works:**
    - This tool uses a machine learning model (XGBoost) trained on historical HDB resale data
    - The model considers factors like location, flat type, size, age, and floor level
    - Predictions are estimates based on historical patterns and market trends
    
    **Important Notes:**
    - Predictions are for reference only and not financial advice
    - Actual prices may vary based on market conditions and property condition
    - Always consult with real estate professionals for accurate valuations
    """)

# Add footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.9rem;'>
    <p>Built with ❤️ using Streamlit | HDB Resale Price Predictor</p>
    <p>Predictions are estimates only | © 2024</p>
</div>
""", unsafe_allow_html=True)
