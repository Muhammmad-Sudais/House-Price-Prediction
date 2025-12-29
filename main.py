import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib

def load_data(filepath):
    """Loads the dataset from a CSV file."""
    try:
        df = pd.read_csv(filepath)
        print(f"Data loaded successfully from {filepath}")
        return df
    except FileNotFoundError:
        print(f"Error: File {filepath} not found.")
        return None

def preprocess_data(df):
    """Preprocesses the data: handles missing values, encodes categorical vars, scales features."""
    
    # Separate features and target
    X = df.drop('Price', axis=1)
    y = df['Price']
    
    # Identify numerical and categorical columns
    numeric_features = ['SquareFootage', 'Bedrooms', 'Bathrooms', 'Age']
    categorical_features = ['Location']
    
    # Create transformers
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    # Combine transformers
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
        
    return X, y, preprocessor

def train_models(X_train, y_train, preprocessor):
    """Trains Linear Regression and Gradient Boosting models."""
    
    models = {
        'Linear Regression': Pipeline(steps=[('preprocessor', preprocessor),
                                             ('regressor', LinearRegression())]),
        'Gradient Boosting': Pipeline(steps=[('preprocessor', preprocessor),
                                             ('regressor', GradientBoostingRegressor(random_state=42))])
    }
    
    trained_models = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        trained_models[name] = model
        print(f"{name} trained.")
        
    return trained_models

def evaluate_model(models, X_test, y_test):
    """Evaluates models using MAE and RMSE."""
    results = {}
    for name, model in models.items():
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        results[name] = {'MAE': mae, 'RMSE': rmse, 'y_pred': y_pred}
        print(f"\n{name} Performance:")
        print(f"MAE: {mae:,.2f}")
        print(f"RMSE: {rmse:,.2f}")
    return results

def visualize_results(y_test, results):
    """Visualizes predicted vs actual prices."""
    plt.figure(figsize=(14, 6))
    
    for i, (name, result) in enumerate(results.items()):
        y_pred = result['y_pred']
        
        plt.subplot(1, 2, i+1)
        sns.scatterplot(x=y_test, y=y_pred, alpha=0.6)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        plt.xlabel('Actual Price')
        plt.ylabel('Predicted Price')
        plt.title(f'{name}: Actual vs Predicted')
        
    plt.tight_layout()
    plt.savefig('prediction_results.png')
    print("\nVisualization saved as 'prediction_results.png'")

def main():
    filepath = 'house_prices.csv'
    df = load_data(filepath)
    
    if df is not None:
        # Basic EDA
        print("\nDataset Head:")
        print(df.head())
        print("\nDataset Info:")
        print(df.info())
        
        X, y, preprocessor = preprocess_data(df)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        trained_models = train_models(X_train, y_train, preprocessor)
        results = evaluate_model(trained_models, X_test, y_test)
        visualize_results(y_test, results)
        
        # Save the best model (Gradient Boosting)
        best_model = trained_models['Gradient Boosting']
        joblib.dump(best_model, 'model.joblib')
        print("\nModel saved as 'model.joblib'")

if __name__ == "__main__":
    main()
