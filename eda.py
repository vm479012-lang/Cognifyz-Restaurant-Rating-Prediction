import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def get_cleaned_data():
    """Applies the cleaning steps from preprocess.py"""
    df = pd.read_csv("Dataset .csv")
    columns_to_drop = [
        "Restaurant ID", "Restaurant Name", "Address", "Locality", 
        "Locality Verbose", "Currency", "Switch to order menu", 
        "Rating color", "Rating text"
    ]
    df_cleaned = df.drop(columns=columns_to_drop)
    df_cleaned['Cuisines'] = df_cleaned['Cuisines'].fillna('Unknown')
    
    yes_no_mapping = {'Yes': 1, 'No': 0}
    for col in ['Has Table booking', 'Has Online delivery', 'Is delivering now']:
        df_cleaned[col] = df_cleaned[col].map(yes_no_mapping)
        
    return df_cleaned

def main():
    # Load cleaned data
    df = get_cleaned_data()
    
    # Create the 'plots' folder if it doesn't exist
    if not os.path.exists('plots'):
        os.makedirs('plots')
        
    print("--- 1. Summary Statistics for Numerical Columns ---")
    # Including Aggregate rating as well as the features
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
    print(df[numerical_cols].describe())
    print("\n" + "="*50 + "\n")
    
    print("--- 2. Plotting Distribution of Aggregate Rating ---")
    plt.figure(figsize=(8, 5))
    sns.histplot(df['Aggregate rating'], bins=30, kde=True, color='skyblue')
    plt.title('Distribution of Aggregate Ratings')
    plt.xlabel('Aggregate Rating')
    plt.ylabel('Frequency')
    plt.savefig('plots/distribution_rating.png', bbox_inches='tight')
    plt.close()
    print("Saved to plots/distribution_rating.png")
    
    print("--- 3. Scatter plot of Votes vs Aggregate rating ---")
    plt.figure(figsize=(8, 5))
    sns.scatterplot(x='Votes', y='Aggregate rating', data=df, alpha=0.5, color='coral')
    plt.title('Votes vs Aggregate Rating')
    plt.xlabel('Number of Votes')
    plt.ylabel('Aggregate Rating')
    plt.savefig('plots/scatter_votes_rating.png', bbox_inches='tight')
    plt.close()
    print("Saved to plots/scatter_votes_rating.png")
    
    print("--- 4. Box plot of Price range vs Aggregate rating ---")
    plt.figure(figsize=(8, 5))
    sns.boxplot(x='Price range', y='Aggregate rating', data=df, palette='Set2')
    plt.title('Price Range vs Aggregate Rating')
    plt.xlabel('Price Range (1 to 4)')
    plt.ylabel('Aggregate Rating')
    plt.savefig('plots/boxplot_price_rating.png', bbox_inches='tight')
    plt.close()
    print("Saved to plots/boxplot_price_rating.png")
    
    print("\n--- 5. Compare Average Ratings for Binary Features ---")
    for feature in ['Has Table booking', 'Has Online delivery', 'Is delivering now']:
        # Group by the binary feature and calculate mean of Aggregate rating
        avg_rating = df.groupby(feature)['Aggregate rating'].mean()
        # Note: 1 = Yes, 0 = No
        print(f"\nAverage Rating by '{feature}':")
        print(f"  No (0): {avg_rating.get(0, 0):.2f}")
        print(f"  Yes (1): {avg_rating.get(1, 0):.2f}")
    
    print("\n" + "="*50 + "\n")
    
    print("--- 6. Top 10 Cities by Average Restaurant Rating (min 20 restaurants) ---")
    # Group by city, count restaurants, and calculate average rating
    city_stats = df.groupby('City').agg(
        num_restaurants=('City', 'count'),
        avg_rating=('Aggregate rating', 'mean')
    )
    # Filter for cities with at least 20 restaurants
    valid_cities = city_stats[city_stats['num_restaurants'] >= 20]
    # Sort by avg rating descending and get top 10
    top_10_cities = valid_cities.sort_values(by='avg_rating', ascending=False).head(10)
    print(top_10_cities)
    
    print("\n" + "="*50 + "\n")
    
    print("--- 7. Correlation Matrix for Numerical Variables ---")
    corr_matrix = df[numerical_cols].corr()
    print(corr_matrix)
    
    print("\n--- 8. Heatmap of Correlation Matrix ---")
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title('Correlation Matrix of Numerical Variables')
    plt.savefig('plots/heatmap_correlation.png', bbox_inches='tight')
    plt.close()
    print("Saved to plots/heatmap_correlation.png")

if __name__ == "__main__":
    main()
