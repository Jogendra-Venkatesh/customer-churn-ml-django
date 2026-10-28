
from pathlib import Path
import joblib
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

# Generate sample data for learning
X, y = make_classification(
    n_samples=1000,
    n_features=3,
    n_informative=3,
    n_redundant=0,
    n_classes=2,
    random_state=42
)

# Train the model
model = RandomForestClassifier(random_state=42)
model.fit(X, y)

# Save the trained model
model_path = Path(__file__).resolve().parent / "churn_model.joblib"
joblib.dump(model, model_path)

print(f"Model saved to: {model_path}")
