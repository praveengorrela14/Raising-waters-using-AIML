import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

# 1. Generate Synthetic Historical Weather Data
print("Generating sample historical weather data...")
np.random.seed(42)
num_samples = 2000

# Features: Annual Rainfall (mm), Cloud Visibility (%), Seasonal Rainfall Pattern Score (0-100)
annual_rainfall = np.random.uniform(500, 3500, num_samples)
cloud_visibility = np.random.uniform(10, 100, num_samples)
seasonal_pattern = np.random.uniform(0, 100, num_samples)

# Logic to determine flood likelihood (1 = Flood, 0 = No Flood)
# Floods are highly likely when rainfall is high and visibility/seasonal patterns spike risk
flood_risk = (annual_rainfall * 0.4 + seasonal_pattern * 0.3 - cloud_visibility * 0.1)
flood_label = np.where(flood_risk > 700, 1, 0)

# Create DataFrame
df = pd.DataFrame({
    'annual_rainfall': annual_rainfall,
    'cloud_visibility': cloud_visibility,
    'seasonal_pattern': seasonal_pattern,
    'flood': flood_label
})

# 2. Split Data into Train and Test Sets
X = df[['annual_rainfall', 'cloud_visibility', 'seasonal_pattern']]
y = df['flood']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Scale Features (Crucial for KNN)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Define and Train Algorithms
models = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
}

best_accuracy = 0
best_model_name = ""
best_model = None

print("\nEvaluating Model Performance:")
for name, model in models.items():
    # Use scaled data for KNN, unscaled for trees (XGBoost handles both well, but scaled works for consistency)
    model.fit(X_train_scaled, y_train)
    predictions = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, predictions)
    print(f"- {name} Accuracy: {acc * 100:.2f}%")
    
    if acc > best_accuracy:
        best_accuracy = acc
        best_model_name = name
        best_model = model

print(f"\nBest Model: {best_model_name} with {best_accuracy * 100:.2f}% accuracy.")

# 5. Save the Best Model and Scaler
joblib.dump(best_model, 'best_flood_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("Model and Scaler successfully saved!")
