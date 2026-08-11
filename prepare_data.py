import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def main():
    print("Loading cleaned dataset...")
    df = pd.read_csv("data/cleaned_restaurant_data.csv")
    
    # 1. Separate the target (y) and input features (X)
    target_col = 'Aggregate rating'
    y = df[target_col]
    X = df.drop(columns=[target_col])
    
    # 2. Identify numerical and categorical features
    numerical_features = [
        'Country Code', 
        'Longitude', 
        'Latitude', 
        'Average Cost for two', 
        'Price range', 
        'Votes', 
        'Has Table booking', 
        'Has Online delivery', 
        'Is delivering now'
    ]
    
    categorical_features = ['City', 'Cuisines']
    
    # 3. Create preprocessing pipelines
    # For numerical features, we apply StandardScaler to scale the data (mean=0, std=1)
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    # For categorical features, we apply OneHotEncoder
    # handle_unknown='ignore' ensures the model doesn't crash if it sees 
    # a new City or Cuisine in the test set that wasn't in the training set
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    # Combine both transformers into a single ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    # 4. Split the data into 80% training and 20% testing
    # random_state=42 ensures reproducibility of the split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print("\nApplying preprocessing pipeline...")
    # 5. IMPORTANT: Fit preprocessing ONLY on the training data to avoid data leakage
    # We learn the scaling parameters (mean/std) and one-hot categories from X_train
    X_train_processed = preprocessor.fit_transform(X_train)
    
    # Then we apply those learned transformations to the test set
    X_test_processed = preprocessor.transform(X_test)
    
    # 6. Print requested metrics
    print("\n--- Data Shapes ---")
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_test shape: {y_test.shape}")
    
    print("\n--- Preprocessing Results ---")
    # Because OneHotEncoder creates a new column for every unique category,
    # the number of features expands significantly.
    print(f"Number of features after preprocessing: {X_train_processed.shape[1]}")

if __name__ == "__main__":
    main()
