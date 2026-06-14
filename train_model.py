import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv("dataset/indian_roads_dataset.csv")

# Risk category
def risk_category(score):
    if score <= 0.33:
        return "Low"
    elif score <= 0.66:
        return "Medium"
    else:
        return "High"

data["risk_level"] = data["risk_score"].apply(risk_category)

features = [
    "road_type",
    "traffic_signal",
    "weather",
    "visibility",
    "temperature",
    "traffic_density",
    "lanes",
    "is_peak_hour"
]

X = data[features].copy()
y = data["risk_level"]

# Encoders
encoders = {}

for col in ["road_type", "weather", "visibility", "traffic_density"]:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    encoders[col] = le

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# Train Model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# Accuracy
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print("Accuracy:", round(accuracy * 100, 2), "%")

# Save Model
joblib.dump(model, "model/risk_model.pkl")

# Save Encoders
joblib.dump(encoders, "model/encoders.pkl")

print("Model Saved Successfully!")