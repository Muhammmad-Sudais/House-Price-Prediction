# House-Price-Prediction
House price prediction is a common machine learning task that uses historical data and property features to estimate the market value of a home. It is primarily used by real estate agents, investors, and home buyers to make informed financial decisions.

# Task Objective
The task objective is to build a House Price Prediction machine learning application:
Generate synthetic data (generate_data.py) with features like square footage, bedrooms, bathrooms, age, and location.
Train and evaluate models (main.py) using Linear Regression and Gradient Boosting Regressor, then save the best model.
Create a Streamlit web app (app.py) for real-time price predictions in PKR based on user inputs.

# Dataset Used
The dataset used is house_prices.csv, a synthetically generated CSV file with 1,000 samples. It includes the following columns:
SquareFootage: Integer (e.g., 2248 sq ft)
Bedrooms: Integer (1-5)
Bathrooms: Integer (1-3)
Age: Integer (0-49 years)
Location: Categorical (Downtown, Suburban, Rural)
Price: Float (target variable, in USD, e.g., 640955.84)

# Model Applied
The models applied are:
Linear Regression: A simple linear model for baseline predictions.
Gradient Boosting Regressor: An ensemble method (using scikit-learn's GradientBoostingRegressor with random_state=42) for improved accuracy.
Both are trained in a pipeline with preprocessing: StandardScaler for numerical features (SquareFootage, Bedrooms, Bathrooms, Age) and OneHotEncoder for categorical features (Location).

# Key Results and Findings:
Dataset Overview: 1,000 samples with no missing values. Features are well-distributed; target (Price) ranges from ~300K to ~700K USD.
Model Performance (on 20% test set):
Linear Regression: MAE = 15,716.80, RMSE = 20,217.12
Gradient Boosting Regressor: MAE = 13,500.89, RMSE = 17,142.37
Findings: Gradient Boosting outperforms Linear Regression (lower errors), likely due to handling non-linear relationships and feature interactions better. It was selected as the best model and saved for deployment.
Visualization: Scatter plots of actual vs. predicted prices saved as prediction_results.png, showing reasonable fit with some outliers.
Deployment: Model integrated into Streamlit app for real-time predictions, converting USD to PKR (~278 rate).

# How to Run the Project
Clone the Repository
Make sure you have Git installed on your system. Open your terminal and run:
```
https://github.com/Muhammmad-Sudais/House-Price-Prediction.git
```
```
pip install -r requirements.txt
```
```
streamlit run app.py
```

