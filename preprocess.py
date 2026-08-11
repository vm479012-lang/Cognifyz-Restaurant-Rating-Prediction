import os
import pandas as pd

def main():
    print("Loading original dataset...")
    df = pd.read_csv("Dataset .csv")
    
    original_count = len(df)
    
    # 1. Remove rows where Aggregate rating == 0 ("Not rated")
    # This prevents the model from predicting artificially low ratings
    # when the rating is actually just missing.
    df_cleaned = df[df['Aggregate rating'] != 0.0].copy()
    
    removed_count = original_count - len(df_cleaned)
    final_count = len(df_cleaned)
    
    # 2. Define the columns to drop based on instructions
    columns_to_drop = [
        "Restaurant ID",
        "Restaurant Name",
        "Address",
        "Locality",
        "Locality Verbose",
        "Currency",
        "Switch to order menu",
        "Rating color",
        "Rating text"
    ]
    
    # Drop the unnecessary columns
    df_cleaned = df_cleaned.drop(columns=columns_to_drop)

    # 3. Handle missing values in the 'Cuisines' column
    df_cleaned['Cuisines'] = df_cleaned['Cuisines'].fillna('Unknown')

    # 4. Convert Yes/No categorical columns into numerical values (1 for Yes, 0 for No)
    yes_no_mapping = {'Yes': 1, 'No': 0}
    binary_columns = ['Has Table booking', 'Has Online delivery', 'Is delivering now']
    
    for col in binary_columns:
        df_cleaned[col] = df_cleaned[col].map(yes_no_mapping)

    # 5. Separate features (X) and target variable (y)
    # City and Cuisines are kept as object type for scikit-learn preprocessing later
    target_col = 'Aggregate rating'
    y = df_cleaned[target_col]
    X = df_cleaned.drop(columns=[target_col])

    # 6. Save the cleaned dataset to 'data' folder
    os.makedirs('data', exist_ok=True)
    save_path = 'data/cleaned_restaurant_data.csv'
    df_cleaned.to_csv(save_path, index=False)

    # 7. Print all requested metrics
    print(f"\n--- Data Filtering Summary ---")
    print(f"Original row count: {original_count}")
    print(f"Rows removed (Rating == 0): {removed_count}")
    print(f"Final row count: {final_count}")
    
    print(f"\n--- Dataset Shapes ---")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    
    print(f"\n--- Missing Values Check ---")
    print(df_cleaned.isnull().sum().to_dict())
    
    print(f"\n--- Target Rating Range ---")
    print(f"Min rating: {y.min()}")
    print(f"Max rating: {y.max()}")
    
    print(f"\nCleaned dataset successfully saved to: {save_path}")

if __name__ == "__main__":
    main()
