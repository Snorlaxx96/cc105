# game/views.py
import pickle
import numpy as np
import pandas as pd
from django.shortcuts import render, redirect
from django.contrib.auth import login # For signup view
from django.contrib.auth.decorators import login_required
# from django.http import HttpResponse # Can be used for quick debugging if needed
from .forms import GamePredictionForm, SignUpForm # Ensure SignUpForm is defined in your game/forms.py
from django.conf import settings
import os

# --- Global Variables for Model and Components ---
# These MUST be defined at the module level (outside any function)
MODEL_PATH = os.path.join(settings.BASE_DIR, 'game', 'game_popularity_model.pkl')

# Initialize all global model-related variables to None
LOADED_MODEL_PACKAGE = None
MODEL = None
PLATFORM_ENCODER = None
GENRE_ENCODER = None
PUBLISHER_ENCODER = None
FEATURE_NAMES = None
# This will store details about the model loading process for debugging
MODEL_LOADING_ERROR_DETAILS = f"Initial: Attempting to load model from: {str(MODEL_PATH)}. "

# --- Model Loading Try-Except Block (Executed once when Django starts/reloads views.py) ---
try:
    print(f"DEBUG: views.py top-level: Trying to load model from {MODEL_PATH}")
    if not os.path.exists(MODEL_PATH):
        MODEL_LOADING_ERROR_DETAILS += "FileDoesNotExist. "
        print(f"DEBUG: views.py top-level: File not found at path: {MODEL_PATH}")
        # All model-related globals will remain None as initialized above
    else:
        with open(MODEL_PATH, 'rb') as f:
            LOADED_MODEL_PACKAGE = pickle.load(f) # Assigns to the global LOADED_MODEL_PACKAGE
            MODEL_LOADING_ERROR_DETAILS += "File opened & unpickled. Checking contents... "
        
        if LOADED_MODEL_PACKAGE and isinstance(LOADED_MODEL_PACKAGE, dict):
            print("DEBUG: views.py top-level: LOADED_MODEL_PACKAGE is a dictionary.")
            MODEL = LOADED_MODEL_PACKAGE.get('model')
            encoders_dict = LOADED_MODEL_PACKAGE.get('encoders')
            FEATURE_NAMES = LOADED_MODEL_PACKAGE.get('feature_names')
            
            # Check for essential prediction components
            if encoders_dict and isinstance(encoders_dict, dict):
                PLATFORM_ENCODER = encoders_dict.get('Platform')
                GENRE_ENCODER = encoders_dict.get('Genre')
                PUBLISHER_ENCODER = encoders_dict.get('Publisher')
            else:
                MODEL_LOADING_ERROR_DETAILS += "'encoders' key missing or not a dict in package. "
                MODEL = None # Invalidate MODEL if essential components are missing

            if not all([MODEL, PLATFORM_ENCODER, GENRE_ENCODER, PUBLISHER_ENCODER, FEATURE_NAMES]):
                MODEL_LOADING_ERROR_DETAILS += "One or more core prediction components (model, specific encoder, feature_names) missing or None after attempting to load from package. "
                MODEL = None # Ensure model is None if setup is incomplete for prediction
                print(f"DEBUG: views.py top-level: Core prediction components missing or failed to load. MODEL set to None. Details: {MODEL_LOADING_ERROR_DETAILS}")
            else:
                MODEL_LOADING_ERROR_DETAILS += "All prediction components look OK. "
                print("DEBUG: views.py top-level: All prediction components loaded from package.")

            # Check for dashboard-specific keys (these don't invalidate MODEL for prediction if missing, but dashboard view will note it)
            if 'accuracy' not in LOADED_MODEL_PACKAGE:
                 MODEL_LOADING_ERROR_DETAILS += "'accuracy' key missing from package for dashboard. "
            if 'classification_report_dict' not in LOADED_MODEL_PACKAGE:
                 MODEL_LOADING_ERROR_DETAILS += "'classification_report_dict' key missing from package for dashboard. "
        else:
            MODEL_LOADING_ERROR_DETAILS += "LOADED_MODEL_PACKAGE is None or not a dictionary after unpickling. "
            MODEL = None 
            print("DEBUG: views.py top-level: LOADED_MODEL_PACKAGE is None or not a dict after unpickling.")

except FileNotFoundError: # This path is less likely if os.path.exists() is used above, but good for safety
    MODEL_LOADING_ERROR_DETAILS += f"Explicit FileNotFoundError was raised for {MODEL_PATH}. "
    print(f"DEBUG: views.py top-level: Caught explicit FileNotFoundError. Path: {MODEL_PATH}")
    LOADED_MODEL_PACKAGE = None 
    MODEL = None
    PLATFORM_ENCODER, GENRE_ENCODER, PUBLISHER_ENCODER, FEATURE_NAMES = [None] * 4
except Exception as e:
    MODEL_LOADING_ERROR_DETAILS += f"General Exception during global model load: {type(e).__name__} - {str(e)}. "
    print(f"DEBUG: views.py top-level: Caught General Exception during model load: {type(e).__name__} - {e}")
    LOADED_MODEL_PACKAGE = None 
    MODEL = None
    PLATFORM_ENCODER, GENRE_ENCODER, PUBLISHER_ENCODER, FEATURE_NAMES = [None] * 4

# Final confirmation print when server starts/reloads for the core prediction MODEL
if MODEL:
    print(f"DEBUG: views.py top-level FINAL CHECK: Core prediction MODEL is loaded. Current Load Details: {MODEL_LOADING_ERROR_DETAILS}")
else:
    print(f"DEBUG: views.py top-level FINAL CHECK: Core prediction MODEL is NOT loaded. Current Load Details: {MODEL_LOADING_ERROR_DETAILS}")


# --- Views ---

@login_required
def predict_game_popularity(request):
    print("DEBUG: Entered predict_game_popularity view.")
    form = GamePredictionForm()
    prediction_text = None
    current_request_error_message = None 

    if MODEL is None: 
        print("DEBUG: predict_game_popularity: MODEL is None.")
        current_request_error_message = "Machine learning model for prediction is not available. "
        if MODEL_LOADING_ERROR_DETAILS: 
            current_request_error_message += f"Startup Debug Info: {MODEL_LOADING_ERROR_DETAILS}"
        else:
            current_request_error_message += "No specific loading error details captured."

    if request.method == 'POST':
        print("DEBUG: predict_game_popularity: Is POST request.")
        if MODEL is None: 
            print("DEBUG: predict_game_popularity: POST request but MODEL is None.")
        elif not all([PLATFORM_ENCODER, GENRE_ENCODER, PUBLISHER_ENCODER, FEATURE_NAMES]): 
            current_request_error_message = "Model components (encoders/feature names) not loaded. Cannot predict. " + MODEL_LOADING_ERROR_DETAILS
            print("DEBUG: predict_game_popularity: POST request but encoders/features are None.")
        else: 
            print("DEBUG: predict_game_popularity: POST request, MODEL and components are available.")
            form = GamePredictionForm(request.POST)
            if form.is_valid():
                print("DEBUG: predict_game_popularity: Form is valid.")
                try:
                    platform = form.cleaned_data['platform']
                    genre = form.cleaned_data['genre']
                    publisher = form.cleaned_data['publisher']
                    year = form.cleaned_data['year']
                    na_sales = form.cleaned_data['na_sales']
                    eu_sales = form.cleaned_data['eu_sales']
                    jp_sales = form.cleaned_data['jp_sales']
                    other_sales = form.cleaned_data['other_sales']
                    print("DEBUG: predict_game_popularity: Form data extracted.")

                    platform_encoded = PLATFORM_ENCODER.transform([platform])[0]
                    genre_encoded = GENRE_ENCODER.transform([genre])[0]
                    publisher_encoded = PUBLISHER_ENCODER.transform([publisher])[0]
                    print("DEBUG: predict_game_popularity: Data encoded.")
                    
                    input_features_df = pd.DataFrame([[
                        platform_encoded, genre_encoded, publisher_encoded,
                        year, na_sales, eu_sales, jp_sales, other_sales
                    ]], columns=FEATURE_NAMES)

                    prediction_result = MODEL.predict(input_features_df)[0]
                    prediction_text = "Popular" if prediction_result == 1 else "Not Popular"
                    print(f"DEBUG: predict_game_popularity: Prediction made: {prediction_text}")
                
                except ValueError as ve:
                     current_request_error_message = f"Error processing input: {ve}. Please ensure selected values are known to the model."
                     print(f"DEBUG: predict_game_popularity: ValueError during prediction: {ve}")
                except Exception as e:
                    current_request_error_message = f"An error occurred during prediction: {type(e).__name__} - {e}"
                    print(f"DEBUG: predict_game_popularity: Exception during prediction: {e}")
            else: 
                print(f"DEBUG: predict_game_popularity: Form is NOT valid. Errors: {form.errors}")
    else: 
        print("DEBUG: predict_game_popularity: Is GET request.")

    context = {
        'form': form,
        'prediction_text': prediction_text,
        'error_message': current_request_error_message
    }
    print(f"DEBUG: predict_game_popularity: Preparing to render. Context error: {current_request_error_message}")
    response = render(request, 'game/input_form.html', context)
    print(f"DEBUG: predict_game_popularity: Rendered response. Type: {type(response)}")
    return response


@login_required
def dashboard_view(request):
    print("DEBUG: Entered dashboard_view.")
    # Explicitly state we are referring to the module-level (global) variables
    # This is mainly for clarity and to reinforce which variables are being accessed.
    # Python's default scope resolution should find these module-level variables anyway.
    global LOADED_MODEL_PACKAGE, MODEL, MODEL_LOADING_ERROR_DETAILS 

    context = {
        'model_is_available_for_dashboard': False,
        'dataset_available': False, 
        'dataset_stats': {}, 
        'model_performance': {}, 
        'chart_data': {}
    }

    # 1. Dataset Statistics
    try:
        dataset_path = os.path.join(settings.BASE_DIR, 'video games sales.csv')
        if os.path.exists(dataset_path):
            df = pd.read_csv(dataset_path)
            df_cleaned = df.dropna(subset=['Year', 'Genre', 'Publisher', 'Global_Sales']).copy()

            context['dataset_stats']['num_records_raw'] = len(df)
            context['dataset_stats']['num_records_cleaned'] = len(df_cleaned)
            
            numeric_cols_summary = ['Year', 'NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales', 'Global_Sales']
            existing_numeric_cols = [col for col in numeric_cols_summary if col in df_cleaned.columns and pd.api.types.is_numeric_dtype(df_cleaned[col])]
            if existing_numeric_cols:
                context['dataset_stats']['feature_summary_html'] = df_cleaned[existing_numeric_cols].describe().to_html(classes='table table-striped table-sm')
            
            if 'Global_Sales' in df_cleaned.columns and pd.api.types.is_numeric_dtype(df_cleaned['Global_Sales']):
                median_global_sales = df_cleaned['Global_Sales'].median()
                df_cleaned.loc[:, 'Popularity_Calculated'] = (df_cleaned['Global_Sales'] >= median_global_sales).astype(int)
                target_dist = df_cleaned['Popularity_Calculated'].value_counts().rename(index={0: 'Not Popular', 1: 'Popular'})
                context['dataset_stats']['target_distribution_dict'] = target_dist.to_dict()
                context['chart_data']['popularity_labels'] = target_dist.index.tolist()
                context['chart_data']['popularity_values'] = target_dist.values.tolist()

            if 'Genre' in df_cleaned.columns:
                genre_counts = df_cleaned['Genre'].value_counts().nlargest(10)
                context['chart_data']['genre_labels'] = genre_counts.index.tolist()
                context['chart_data']['genre_values'] = genre_counts.values.tolist()
            
            context['dataset_available'] = True
            print("DEBUG: Dashboard - Dataset statistics loaded.")
        else:
            context['dataset_stats']['error'] = f"Dataset CSV file not found at {dataset_path}"
            print(f"DEBUG: Dashboard - Dataset CSV file not found at {dataset_path}")
    except Exception as e:
        context['dataset_stats']['error'] = f"Error processing dataset for dashboard: {str(e)}"
        print(f"DEBUG: Dashboard - Error processing dataset: {e}")

    # 2. Model Performance
    # This is where the NameError for LOADED_MODEL_PACKAGE was occurring
    print(f"DEBUG: Dashboard - About to check MODEL. MODEL is None: {MODEL is None}")
    print(f"DEBUG: Dashboard - About to check LOADED_MODEL_PACKAGE. LOADED_MODEL_PACKAGE is None: {LOADED_MODEL_PACKAGE is None}")
    
    if MODEL and LOADED_MODEL_PACKAGE and isinstance(LOADED_MODEL_PACKAGE, dict):
        print("DEBUG: Dashboard - MODEL and LOADED_MODEL_PACKAGE are available for metrics.")
        context['model_is_available_for_dashboard'] = True
        context['model_performance']['accuracy'] = LOADED_MODEL_PACKAGE.get('accuracy')
        
        cr_dict = LOADED_MODEL_PACKAGE.get('classification_report_dict', {})
        report_list = []
        for label, metrics_dict in cr_dict.items(): # renamed 'metrics' to 'metrics_dict' to avoid conflict if label is 'metrics'
            if isinstance(metrics_dict, dict): 
                report_list.append({
                    'label': label, 
                    'precision': metrics_dict.get('precision'), 
                    'recall': metrics_dict.get('recall'), 
                    'f1_score': metrics_dict.get('f1-score'), 
                    'support': metrics_dict.get('support')
                })
            else: # For overall accuracy if it's stored directly under a simple key in report
                 report_list.append({'label': label, 'value': metrics_dict, 'is_overall_metric': True})
        context['model_performance']['classification_report_list'] = report_list
        print("DEBUG: Dashboard - Model performance metrics extracted.")
    else:
        print("DEBUG: Dashboard - MODEL or LOADED_MODEL_PACKAGE not suitable for metrics.")
        context['model_performance']['error'] = "Model performance data not available. Model/Package might not be loaded or is incomplete."
        context['model_performance']['debug_info'] = MODEL_LOADING_ERROR_DETAILS
            
    print("DEBUG: Dashboard - Preparing to render.")
    return render(request, 'game/dashboard.html', context)


def signup(request):
    print("DEBUG: Entered signup view.")
    if request.method == 'POST':
        form = SignUpForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user) 
            print("DEBUG: signup: User created and logged in. Redirecting to 'predict_game_popularity'.")
            return redirect('predict_game_popularity') 
    else:
        form = SignUpForm()
    print("DEBUG: signup: Rendering signup form.")
    return render(request, 'game/signup.html', {'form': form})