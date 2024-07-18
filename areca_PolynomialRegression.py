import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
import joblib

# Read the CSV file
df = pd.read_csv("Agmar.csv")

# Preprocess the data
df['Price Date'] = pd.to_datetime(df['Price Date'])
df['Year'] = df['Price Date'].dt.year
df['Month'] = df['Price Date'].dt.month
df = df[['Year', 'Month', 'Modal Price (Rs./Quintal)']]

# Split into features and target
X = df[['Year', 'Month']]
y = df['Modal Price (Rs./Quintal)']

# Scale the features and target
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(np.array(y).reshape(-1, 1))

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

# Build and train the polynomial regression model
degree = 2  # Degree of the polynomial
model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
model.fit(X_train, y_train)

# Save the model and scalers
joblib.dump(model, 'model_poly.pkl')
joblib.dump(scaler_X, 'scaler_X_poly.pkl')
joblib.dump(scaler_y, 'scaler_y_poly.pkl')

# Function to predict the price for a specific date
def predict_price(year, month):
    date_features = np.array([[year, month]])
    date_features_scaled = scaler_X.transform(date_features)
    predicted_price_scaled = model.predict(date_features_scaled)
    predicted_price = scaler_y.inverse_transform(predicted_price_scaled)
    return predicted_price[0][0]

# Ask the user for a specific date
input_date = input("Enter the date (YYYY-MM) to predict the arecanut price: ")
input_year, input_month = map(int, input_date.split('-'))

# Predict and print the price
predicted_price = predict_price(input_year, input_month)
print(f"The predicted arecanut price for {input_year}-{input_month} is: {predicted_price} Rs./Quintal")
