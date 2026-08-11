import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def get_feature_groups(feature_names, importances):
    """Aggregates one-hot encoded feature importances back to original groups."""
    importance_dict = {}
    for name, imp in zip(feature_names, importances):
        # Determine the group name
        if name.startswith('City_'):
            group = 'City'
        elif name.startswith('Cuisines_'):
            group = 'Cuisines'
        else:
            group = name
            
        # Aggregate the importance
        if group in importance_dict:
            importance_dict[group] += imp
        else:
            importance_dict[group] = imp
            
    # Convert to sorted list of tuples
    sorted_importances = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
    return sorted_importances

def main():
    print("Loading cleaned dataset...")
    df = pd.read_csv("data/cleaned_restaurant_data.csv")
    
    target_col = 'Aggregate rating'
    y = df[target_col]
    X = df.drop(columns=[target_col])
    
    numerical_features = [
        'Country Code', 'Longitude', 'Latitude', 'Average Cost for two', 
        'Price range', 'Votes', 'Has Table booking', 'Has Online delivery', 
        'Is delivering now'
    ]
    categorical_features = ['City', 'Cuisines']
    
    numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # --- PART A: Baseline & Hyperparameter tuning ---
    print("\nTraining Baseline Random Forest...")
    baseline_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(random_state=42, n_jobs=-1))
    ])
    baseline_pipeline.fit(X_train, y_train)
    y_pred_base = baseline_pipeline.predict(X_test)
    base_mae = mean_absolute_error(y_test, y_pred_base)
    base_mse = mean_squared_error(y_test, y_pred_base)
    base_rmse = np.sqrt(base_mse)
    base_r2 = r2_score(y_test, y_pred_base)
    
    print("\nStarting Hyperparameter Tuning (this may take a minute)...")
    # Define parameters with 'regressor__' prefix since the model is inside a Pipeline
    param_dist = {
        'regressor__n_estimators': [100, 200, 300],
        'regressor__max_depth': [None, 10, 20, 30],
        'regressor__min_samples_split': [2, 5, 10],
        'regressor__min_samples_leaf': [1, 2, 4],
        'regressor__max_features': ['sqrt', 'log2', 0.5]
    }
    
    pipeline_for_tuning = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(random_state=42))
    ])
    
    random_search = RandomizedSearchCV(
        estimator=pipeline_for_tuning,
        param_distributions=param_dist,
        n_iter=10,
        cv=3,
        scoring='r2',
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
    
    random_search.fit(X_train, y_train)
    best_pipeline = random_search.best_estimator_
    
    # Evaluate tuned model
    y_pred_tuned = best_pipeline.predict(X_test)
    tuned_mae = mean_absolute_error(y_test, y_pred_tuned)
    tuned_mse = mean_squared_error(y_test, y_pred_tuned)
    tuned_rmse = np.sqrt(tuned_mse)
    tuned_r2 = r2_score(y_test, y_pred_tuned)
    
    # --- PART B: Feature Importance ---
    best_rf = best_pipeline.named_steps['regressor']
    raw_importances = best_rf.feature_importances_
    
    # Get feature names after one-hot encoding
    cat_feature_names = best_pipeline.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_features)
    all_feature_names = numerical_features + list(cat_feature_names)
    
    # Aggregate importance
    grouped_importances = get_feature_groups(all_feature_names, raw_importances)
    top_10_features = grouped_importances[:10]
    
    # Plot feature importance
    os.makedirs('plots', exist_ok=True)
    features, importance_vals = zip(*top_10_features)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=list(importance_vals), y=list(features), palette='viridis')
    plt.title('Top Features Influencing Restaurant Ratings')
    plt.xlabel('Aggregated Feature Importance')
    plt.ylabel('Feature Group')
    plt.savefig('plots/feature_importance.png', bbox_inches='tight')
    plt.close()
    
    # --- PART C: Save Model ---
    os.makedirs('models', exist_ok=True)
    model_path = 'models/restaurant_rating_model.pkl'
    joblib.dump(best_pipeline, model_path)
    
    # --- PRINT OUTPUTS ---
    print("\n" + "="*50)
    print("1. Best Hyperparameters:")
    # Strip the 'regressor__' prefix for cleaner display
    best_params = {k.replace('regressor__', ''): v for k, v in random_search.best_params_.items()}
    for k, v in best_params.items():
        print(f"   - {k}: {v}")
        
    print("\n2. Baseline Random Forest Metrics:")
    print(f"   MAE:  {base_mae:.4f}")
    print(f"   MSE:  {base_mse:.4f}")
    print(f"   RMSE: {base_rmse:.4f}")
    print(f"   R²:   {base_r2:.4f}")
    
    print("\n3. Tuned Random Forest Metrics:")
    print(f"   MAE:  {tuned_mae:.4f}")
    print(f"   MSE:  {tuned_mse:.4f}")
    print(f"   RMSE: {tuned_rmse:.4f}")
    print(f"   R²:   {tuned_r2:.4f}")
    
    print("\n4. Top 10 Feature Importances (Aggregated):")
    for rank, (feat, imp) in enumerate(top_10_features, 1):
        print(f"   {rank}. {feat} ({imp:.4f})")
        
    print(f"\n5. Saved model path: {model_path}")
    print("="*50)

if __name__ == "__main__":
    main()
