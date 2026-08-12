# 🍽️ Restaurant Rating Prediction

## 🚀 Live Demo

👉 [Open Restaurant Rating Predictor](https://restaurant-rating-ml.streamlit.app/)

A Machine Learning application that predicts restaurant aggregate ratings using a Random Forest Regressor.
# Restaurant Rating Prediction

**Cognifyz Machine Learning Internship - Task 1**

## 1. Project Overview
This project focuses on predicting the aggregate rating of a restaurant based on various features such as location, cuisines, cost, and available services (like online delivery and table booking). The project covers the entire machine learning lifecycle, from data preprocessing and exploratory data analysis to model training, hyperparameter tuning, and final deployment via a Streamlit web application.

## 2. Objective
The main objective of this task is to build a regression model capable of accurately predicting a restaurant's user rating (on a scale of roughly 1.8 to 4.9). 

## 3. Dataset
The original dataset provided for this task contained **9,551 rows and 21 columns**. It included a mix of numerical, categorical, and geographical features for restaurants across various cities and countries.

## 4. Data Preprocessing
To prepare the dataset for machine learning, the following steps were taken:
- Removed features that act as unique identifiers (e.g., `Restaurant ID`, `Restaurant Name`).
- Removed features causing data leakage (e.g., `Rating color`, `Rating text`).
- Handled missing values (filled 9 missing `Cuisines` entries with "Unknown").
- Converted binary textual columns (`Yes`/`No`) into numerical values (`1`/`0`).
- **Target Variable Handling:** Found that **2,148 rows** had an `Aggregate rating` of `0.0`. Upon investigation, these explicitly correlated with the label "Not rated". To prevent the regression model from artificially predicting zero for missing reviews, these 2,148 rows were removed.
- **Final modeling dataset:** **7,403 rows**.

## 5. Exploratory Data Analysis
During the EDA phase, several visualizations were created to understand the distribution of ratings, the impact of price ranges, the correlation of numerical features, and the average ratings across binary features. The plots generated are stored in the `plots/` directory.

## 6. Features Used
The final input features used for predicting the `Aggregate rating` are:
1. `Country Code`
2. `City`
3. `Longitude`
4. `Latitude`
5. `Cuisines`
6. `Average Cost for two`
7. `Has Table booking`
8. `Has Online delivery`
9. `Is delivering now`
10. `Price range`
11. `Votes`

*Categorical features (`City` and `Cuisines`) were processed using scikit-learn's `OneHotEncoder` within the machine learning pipeline, preventing data leakage and cleanly handling unknown test variables.*

## 7. Models Compared
Three regression models were trained and evaluated on an 80/20 train/test split:
- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor

## 8. Model Evaluation
The Random Forest Regressor proved to be the most capable model by a wide margin. A hyperparameter tuning search (`RandomizedSearchCV`) was performed to see if the default parameters could be improved. 

**Baseline Random Forest:**
- **MAE:** 0.2425
- **MSE:** 0.1097
- **RMSE:** 0.3312
- **R²:** 0.6454

**Tuned Random Forest:**
- **MAE:** 0.2452
- **MSE:** 0.1102
- **RMSE:** 0.3319
- **R²:** 0.6437

## 9. Final Model
Because the `Tuned Random Forest` performed fractionally worse than the original model, the default scikit-learn parameters were deemed highly optimal. Therefore, the **Baseline Random Forest Regressor** was selected as the final model for deployment.

## 10. Feature Importance
By aggregating the hundreds of one-hot encoded columns back into their original feature groups, we identified the following top 10 most influential features driving the model's predictions:

1. **Votes** = 41.71%
2. **Cuisines** = 15.22%
3. **Longitude** = 14.30%
4. **Latitude** = 9.19%
5. **Country Code** = 6.75%
6. **Average Cost for two** = 5.28%
7. **Price range** = 3.45%
8. **City** = 2.94%
9. **Has Online delivery** = 0.74%
10. **Has Table booking** = 0.34%

*Note: Feature importance indicates predictive importance for the model, **not strict causation**. For example, restaurants receiving high ratings naturally accumulate more votes; the model heavily relies on this strong correlation, but simply adding "Votes" to a bad restaurant will not make it better.*

## 11. Streamlit Application
An interactive web application was built using Streamlit (`app.py`). It allows users to input the 11 required features and receive an instant rating prediction. 
- The app uses the saved `restaurant_rating_model.pkl` pipeline to automatically encode and scale user input. 
- `City` is enforced as a strict dropdown using valid training cities to prevent erroneous free-text entries.
- `Cuisines` whitespace is normalized to guarantee smooth processing.
- The output is bounded realistically to the 1.8 - 4.9 rating scale.

## 12. Project Structure
```text
.
├── Dataset .csv                    # Original unmodified dataset
├── data/
│   └── cleaned_restaurant_data.csv # Processed dataset (no 0 ratings)
├── models/
│   └── restaurant_rating_model.pkl # Saved ML Pipeline
├── plots/                          # Saved EDA & Feature Importance charts
├── preprocess.py                   # Data cleaning script
├── eda.py                          # Exploratory data analysis script
├── prepare_data.py                 # Pipeline preparation script
├── train_models.py                 # Model training & comparison script
├── tune_model.py                   # Hyperparameter tuning script
├── app.py                          # Streamlit application
└── README.md                       # Project documentation
```

## 13. Installation
Ensure you have Python installed, then install the required dependencies:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn streamlit joblib
```

## 14. How to Run
To use the final application, simply launch the Streamlit server from your terminal:

```bash
streamlit run app.py
```
This will open the interactive predictor in your default web browser.

## 15. Results
The final Random Forest model accounts for approximately **64.5% of the variance** in user ratings. With a **Mean Absolute Error (MAE) of just ~0.24**, the model is extremely accurate—on a 5-point rating scale, its predictions are generally off by less than a quarter of a point.

## 16. Limitations
- The model treats the number of `Votes` as a primary predictor, which makes predicting the rating for a brand new restaurant (with 0 votes) inherently difficult. 
- Geography (Longitude/Latitude/Country/City) plays an immense role in the model, indicating the model may struggle to generalize if tested on a dataset consisting entirely of a new, unseen country.

## 17. Future Improvements
- Implement advanced NLP techniques (e.g., TF-IDF or Word2Vec) for the `Cuisines` feature instead of OneHotEncoding to find similarities between complex food profiles.
- Collect and engineer demographic data for the specific `City` or `Longitude/Latitude` coordinates to give the model better contextual clues about the location's wealth or dining culture.
- Exclude `Votes` during training to build a strict "Cold Start" model that predicts ratings based entirely on intrinsic features (Price, Location, Cuisine, Amenities).
