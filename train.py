import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import joblib

# Load dataset
df = pd.read_csv("houseprice_dataset.csv")

# Features and target
X = df[['Bedrooms', 'Age', 'Area_SqFt']]
y = df['Price']

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Feature scaling
scaler = StandardScaler()

x_train_scaled = scaler.fit_transform(x_train)

# Create model
model = LinearRegression()

# Train model
model.fit(x_train_scaled, y_train)

# Save model and scaler
joblib.dump(model, "house_price_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("Model trained successfully")
print("Files created:")
print("1. house_price_model.pkl")
print("2. scaler.pkl")