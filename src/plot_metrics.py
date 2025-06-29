import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Load data
df = pd.read_csv("data/iris.csv")
X = df.drop(columns=["species"])
y = df["species"]

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Load model
model = joblib.load("model/model.joblib")
y_pred = model.predict(X_test)

# Plot confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
disp.plot()
plt.savefig("metrics.png")

# Write markdown report
with open("report.md", "w") as f:
    f.write("# 🧪 Model Performance Report\n\n")
    f.write("### Confusion Matrix\n")
    f.write("![Confusion Matrix](metrics.png)\n")
