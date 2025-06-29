import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

# Load CSV
df = pd.read_csv("data/iris.csv")

# Split features and target
X = df.drop(columns=["species"])
y = df["species"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Save model and label classes
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/model.joblib")
joblib.dump(model.classes_, "model/label_encoder.joblib")
