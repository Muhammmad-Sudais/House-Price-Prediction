import pandas as pd
import numpy as np

def generate_data(num_samples=1000, filename='house_prices.csv'):
    np.random.seed(42)
    
    # Features
    square_footage = np.random.normal(2000, 500, num_samples).astype(int)
    bedrooms = np.random.randint(1, 6, num_samples)
    bathrooms = np.random.randint(1, 4, num_samples)
    age = np.random.randint(0, 50, num_samples)
    
    # Location (Categorical)
    locations = ['Downtown', 'Suburban', 'Rural']
    location_data = np.random.choice(locations, num_samples, p=[0.2, 0.5, 0.3])
    
    # Base price calculation (simplified formula for synthetic data)
    # Price = Base + (SqFt * Coeff) + (Bed * Coeff) + (Bath * Coeff) - (Age * Coeff) + Location_Bonus + Noise
    
    price = 50000 + (square_footage * 150) + (bedrooms * 10000) + (bathrooms * 5000) - (age * 500)
    
    # Add location effect
    location_multipliers = {'Downtown': 1.5, 'Suburban': 1.2, 'Rural': 1.0}
    location_factor = np.array([location_multipliers[loc] for loc in location_data])
    
    price = price * location_factor
    
    # Add random noise
    noise = np.random.normal(0, 15000, num_samples)
    price = price + noise
    
    df = pd.DataFrame({
        'SquareFootage': square_footage,
        'Bedrooms': bedrooms,
        'Bathrooms': bathrooms,
        'Age': age,
        'Location': location_data,
        'Price': price
    })
    
    df.to_csv(filename, index=False)
    print(f"Generated {filename} with {num_samples} samples.")

if __name__ == "__main__":
    generate_data()
