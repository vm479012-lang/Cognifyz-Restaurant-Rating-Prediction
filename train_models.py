import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def main():
    print("Loading cleaned dataset...")
    df = pd.read_csv("data/cleaned_restaurant_data.csv")
    
    # 1. Separate the target (y) and input features (X)
    target_col = 'Aggregate rating'
    y = df[target_col]
    X = df.drop(columns=[target_col])
    
    # 2. Identify numerical and categorical features
    numerical_features = [
        'Country Code', 'Longitude', 'Latitude', 'Average Cost for two', 
        'Price range', 'Votes', 'Has Table booking', 
        'Has Online delivery', 'Is delivering now'
    ]
    categorical_features = ['City', 'Cuisines']
    
    # 3. Create preprocessing steps
    numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    # 4. Split the data (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # 5. Define the models to evaluate
    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree Regressor": DecisionTreeRegressor(random_state=42),
        "Random Forest Regressor": RandomForestRegressor(random_state=42, n_jobs=-1)
    }
    
    # Dictionary to hold the evaluation results
    results = []
    
    print("\nTraining and evaluating models...\n")
    
    # 6. Train and evaluate each model
    for name, model in models.items():
        # Create a complete pipeline: Preprocessing -> Model
        # This guarantees preprocessing is fit ONLY on the training data
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', model)
        ])
        
        # Train the model on the training set
        pipeline.fit(X_train, y_train)
        
        # Predict on the test set
        y_pred = pipeline.predict(X_test)
        
        # Calculate metrics
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        # Save results
        results.append({
            "Model": name,
            "MAE": mae,
            "MSE": mse,
            "RMSE": rmse,
            "R²": r2
        })
    
    # 7. Sort and display the results
    # Convert results to a DataFrame for easy sorting and display
    results_df = pd.DataFrame(results)
    
    # Sort by R² score from highest to lowest
    results_df = results_df.sort_values(by="R²", ascending=False).reset_index(drop=True)
    
    # Format floats for a cleaner table display
    pd.options.display.float_format = '{:.4f}'.format
    
    print("--- Model Comparison Table ---")
    print(results_df.to_string(index=False))
    print("\n" + "="*60 + "\n")
    
    # 8. Print Conclusion
    best_model = results_df.iloc[0]
    print("--- Conclusion ---")
    print(f"The best performing model is the {best_model['Model']} with an R² score of {best_model['R²']:.4f}.")
    print("It explains the highest percentage of variance in restaurant ratings while keeping errors (MAE, RMSE) the lowest among the three models tested.")

if __name__ == "__main__":
    main()
