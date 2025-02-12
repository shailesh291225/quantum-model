import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib  # To save the trained model

# Load dataset
file_path = "Data.csv"
df = pd.read_csv(file_path)

# Drop unnecessary columns
df = df[['Ro', 'Su', 'Bhn', 'E']]  # Keeping only relevant columns

# Handle missing values (replace NaN with column mean)
df.fillna(df.mean(), inplace=True)

# Features (input) and target variables (output)
X = df[['Ro']]  # Mass density as input
y = df[['Su', 'Bhn', 'E']]  # Predicting Strength (Su), Hardness (Bhn), and Elasticity (E)

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Save the trained model
joblib.dump(model, "material_model.pkl")

# Function to predict material properties
def predict_properties(mass_density):
    model = joblib.load("material_model.pkl")
    input_data = np.array([[mass_density]])
    prediction = model.predict(input_data)
    return prediction[0]  # Return predicted values

# Take user input for mass density and predict properties
if __name__ == "__main__":
    mass_density = float(input("\nEnter mass density (Ro): "))
    predicted_values = predict_properties(mass_density)
    
    print("\n🔹 Predicted Properties:")
    print(f"  - Ultimate Strength (Su): {predicted_values[0]:.2f} MPa")
    print(f"  - Brinell Hardness (Bhn): {predicted_values[1]:.2f}")
    print(f"  - Elastic Modulus (E): {predicted_values[2]:.2f} MPa")

