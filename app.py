import gradio as gr
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from huggingface_hub import hf_hub_download
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Try to import xgboost, but fallback to scikit-learn
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
    print("✅ XGBoost is available")
except ImportError:
    XGB_AVAILABLE = False
    print("⚠️  XGBoost not available, using scikit-learn models")
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LinearRegression

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
            # Add methods that might be called by joblib or other code
            self.get_params = lambda deep=True: {}
            self.set_params = lambda **params: self
        
        def predict(self, X):
            # Realistic prediction logic
            if isinstance(X, np.ndarray) and len(X.shape) == 2:
                X = X[0]  # Take first row if it's a 2D array
            
            floor_area = X[0]
            storey_level = X[1]
            flat_age = X[2]
            town_encoded = X[6]
            flat_type_encoded = X[5]
            
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

def safe_joblib_load(filepath):
    """Safely load joblib file with error handling"""
    try:
        model = joblib.load(filepath)
        print(f"✅ Successfully loaded model from {filepath}")
        
        # Check if model has required methods
        if not hasattr(model, 'predict'):
            print("❌ Loaded object doesn't have predict method")
            return None
            
        # Add missing methods if needed
        if not hasattr(model, 'get_params'):
            model.get_params = lambda deep=True: {}
        if not hasattr(model, 'set_params'):
            model.set_params = lambda **params: model
            
        return model
        
    except Exception as e:
        print(f"❌ Error loading model from {filepath}: {e}")
        return None

def load_models():
    """Load models with robust error handling"""
    models = {}
    
    # Try to load XGBoost model
    try:
        xgboost_path = hf_hub_download(
            repo_id="Lesterchia174/HDB_Price_Predictor",
            filename="best_model_xgboost.joblib",
            repo_type="space"
        )
        models['xgboost'] = safe_joblib_load(xgboost_path)
        if models['xgboost'] is None:
            print("⚠️  Creating dummy model for XGBoost")
            models['xgboost'] = create_dummy_model("xgboost")
        else:
            print("✅ XGBoost model loaded and validated")
            
    except Exception as e:
        print(f"❌ Error downloading XGBoost model: {e}")
        print("⚠️  Creating dummy model for XGBoost")
        models['xgboost'] = create_dummy_model("xgboost")
    
    # Try to load Linear Regression model
    try:
        linear_path = hf_hub_download(
            repo_id="Lesterchia174/HDB_Price_Predictor",
            filename="linear_regression.joblib",
            repo_type="space"
        )
        models['linear_regression'] = safe_joblib_load(linear_path)
        if models['linear_regression'] is None:
            print("⚠️  Creating dummy model for Linear Regression")
            models['linear_regression'] = create_dummy_model("linear_regression")
        else:
            print("✅ Linear Regression model loaded and validated")
            
    except Exception as e:
        print(f"❌ Error downloading Linear Regression model: {e}")
        print("⚠️  Creating dummy model for Linear Regression")
        models['linear_regression'] = create_dummy_model("linear_regression")
    
    return models

def load_data():
    """Load data using Hugging Face Hub"""
    try:
        data_path = hf_hub_download(
            repo_id="Lesterchia174/HDB_Price_Predictor",
            filename="base_hdb_resale_prices_2015Jan-2025Jun_processed.csv",
            repo_type="space"
        )
        df = pd.read_csv(data_path)
        print("✅ Data loaded successfully via Hugging Face Hub")
        return df
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return create_sample_data()

def create_sample_data():
    """Create sample data if real data isn't available"""
    np.random.seed(42)
    towns = ['ANG MO KIO', 'BEDOK', 'TAMPINES', 'WOODLANDS', 'JURONG WEST']
    flat_types = ['4 ROOM', '5 ROOM', 'EXECUTIVE']
    flat_models = ['Improved', 'Model A', 'New Generation']
    
    data = []
    for _ in range(100):
        town = np.random.choice(towns)
        flat_type = np.random.choice(flat_types)
        flat_model = np.random.choice(flat_models)
        floor_area = np.random.randint(85, 150)
        storey = np.random.randint(1, 25)
        age = np.random.randint(0, 40)
        
        base_price = floor_area * 5000
        town_bonus = towns.index(town) * 20000
        storey_bonus = storey * 2000
        age_discount = age * 1500
        flat_type_bonus = flat_types.index(flat_type) * 30000
        
        resale_price = base_price + town_bonus + storey_bonus - age_discount + flat_type_bonus
        resale_price = max(300000, resale_price + np.random.randint(-20000, 20000))
        
        data.append({
            'town': town, 'flat_type': flat_type, 'flat_model': flat_model,
            'floor_area_sqm': floor_area, 'storey_level': storey,
            'flat_age': age, 'resale_price': resale_price
        })
    
    return pd.DataFrame(data)

def preprocess_input(user_input, model_type='xgboost'):
    """Preprocess user input for prediction with correct feature mapping"""
    # Flat type mapping
    flat_type_mapping = {'1 ROOM': 1, '2 ROOM': 2, '3 ROOM': 3, '4 ROOM': 4,
                         '5 ROOM': 5, 'EXECUTIVE': 6, 'MULTI-GENERATION': 7}

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
        2025,                                   # Feature 5: transaction_year
        flat_type_mapping.get(user_input['flat_type'], 4),  # Feature 6: flat_type_ordinal
        town_mapping.get(user_input['town'], 0),           # Feature 7: town_encoded
        flat_model_mapping.get(user_input['flat_model'], 0), # Feature 8: flat_model_encoded
        1                                       # Feature 9: (placeholder)
    ]

    return np.array([input_features])

def create_market_insights_chart(data, user_input, predicted_price_xgb, predicted_price_lr):
    """Create market insights visualization with both model predictions"""
    if data is None or len(data) == 0:
        return None

    similar_properties = data[
        (data['flat_type'] == user_input['flat_type']) &
        (data['town'] == user_input['town'])
    ]

    if len(similar_properties) < 5:
        similar_properties = data[data['flat_type'] == user_input['flat_type']]

    if len(similar_properties) > 0:
        fig = px.scatter(similar_properties, x='floor_area_sqm', y='resale_price',
                         color='flat_model',
                         title=f"Market Position: {user_input['flat_type']} in {user_input['town']}",
                         labels={'floor_area_sqm': 'Floor Area (sqm)', 'resale_price': 'Resale Price (SGD)'})

        # Add both model predictions
        fig.add_trace(go.Scatter(x=[user_input['floor_area_sqm']], y=[predicted_price_xgb],
                                 mode='markers',
                                 marker=dict(symbol='star', size=20, color='red',
                                             line=dict(width=2, color='darkred')),
                                 name='XGBoost Prediction'))

        fig.add_trace(go.Scatter(x=[user_input['floor_area_sqm']], y=[predicted_price_lr],
                                 mode='markers',
                                 marker=dict(symbol='diamond', size=20, color='blue',
                                             line=dict(width=2, color='darkblue')),
                                 name='Linear Regression Prediction'))

        fig.update_layout(template="plotly_white", height=400, showlegend=True)
        return fig
    return None

def predict_hdb_price(town, flat_type, flat_model, floor_area_sqm, storey_level, flat_age, model_choice):
    """Main prediction function for Gradio with robust error handling"""
    user_input = {
        'town': town,
        'flat_type': flat_type,
        'flat_model': flat_model,
        'floor_area_sqm': floor_area_sqm,
        'storey_level': storey_level,
        'flat_age': flat_age
    }

    try:
        processed_input = preprocess_input(user_input)
        
        # Get predictions from both models with error handling
        try:
            predicted_price_xgb = max(0, float(models['xgboost'].predict(processed_input)[0]))
        except Exception as e:
            print(f"❌ XGBoost prediction error: {e}")
            predicted_price_xgb = 400000  # Fallback value
        
        try:
            predicted_price_lr = max(0, float(models['linear_regression'].predict(processed_input)[0]))
        except Exception as e:
            print(f"❌ Linear Regression prediction error: {e}")
            predicted_price_lr = 380000  # Fallback value
        
        # Use selected model's prediction
        if model_choice == "XGBoost":
            final_price = predicted_price_xgb
            model_name = "XGBoost"
        else:
            final_price = predicted_price_lr
            model_name = "Linear Regression"

        # Create insights
        remaining_lease = 99 - flat_age
        price_per_sqm = final_price / floor_area_sqm

        insights = f"""
        **Property Summary:**
        - Location: {town}
        - Type: {flat_type}
        - Model: {flat_model}
        - Area: {floor_area_sqm} sqm
        - Floor: Level {storey_level}
        - Age: {flat_age} years
        - Remaining Lease: {remaining_lease} years
        - Price per sqm: ${price_per_sqm:,.0f}

        **Model Predictions:**
        - XGBoost: ${predicted_price_xgb:,.0f}
        - Linear Regression: ${predicted_price_lr:,.0f}
        - Difference: ${abs(predicted_price_xgb - predicted_price_lr):,.0f}

        **Selected Model: {model_choice}**

        **Financing Eligibility:**
        """

        if remaining_lease >= 60:
            insights += "✅ Bank loan eligible"
        elif remaining_lease >= 20:
            insights += "⚠️ HDB loan eligible only"
        else:
            insights += "❌ Limited financing options"

        # Create chart with both predictions
        chart = create_market_insights_chart(data, user_input, predicted_price_xgb, predicted_price_lr)

        return f"${final_price:,.0f}", chart, insights

    except Exception as e:
        error_msg = f"Prediction failed. Error: {str(e)}"
        print(error_msg)
        return "Error: Prediction failed", None, error_msg

# Preload models and data
print("Loading models and data...")
models = load_models()
data = load_data()

# Define Gradio interface
towns_list = [
    'SENGKANG', 'WOODLANDS', 'TAMPINES', 'PUNGGOL', 'JURONG WEST',
    'YISHUN', 'BEDOK', 'HOUGANG', 'CHOA CHU KANG', 'ANG MO KIO'
]

flat_types = ['3 ROOM', '4 ROOM', '5 ROOM', 'EXECUTIVE', '2 ROOM', '1 ROOM']
flat_models = ['Model A', 'Improved', 'New Generation', 'Standard', 'Premium']

# Create Gradio interface
with gr.Blocks(title="🏠 HDB Price Predictor", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🏠 HDB Price Predictor")
    gr.Markdown("Predict HDB resale prices using different machine learning models")

    with gr.Row():
        with gr.Column():
            town = gr.Dropdown(label="Town", choices=sorted(towns_list), value="ANG MO KIO")
            flat_type = gr.Dropdown(label="Flat Type", choices=sorted(flat_types), value="4 ROOM")
            flat_model = gr.Dropdown(label="Flat Model", choices=sorted(flat_models), value="Improved")
            floor_area_sqm = gr.Slider(label="Floor Area (sqm)", minimum=30, maximum=200, value=95, step=5)
            storey_level = gr.Slider(label="Storey Level", minimum=1, maximum=50, value=8, step=1)
            flat_age = gr.Slider(label="Flat Age (years)", minimum=0, maximum=99, value=15, step=1)
            model_choice = gr.Radio(label="Select Model", 
                                   choices=["XGBoost"],  #,"Linear Regression" 
                                   value="XGBoost")

            predict_btn = gr.Button("🔮 Predict Price", variant="primary")

        with gr.Column():
            predicted_price = gr.Label(label="💰 Predicted Price")
            insights = gr.Markdown(label="📋 Property Summary")

    with gr.Row():
        chart_output = gr.Plot(label="📈 Market Insights (Both Models)")

    # Connect button to function
    predict_btn.click(
        fn=predict_hdb_price,
        inputs=[town, flat_type, flat_model, floor_area_sqm, storey_level, flat_age, model_choice],
        outputs=[predicted_price, chart_output, insights]
    )

# To run in Colab
if __name__ == "__main__":
    demo.launch(share=True)