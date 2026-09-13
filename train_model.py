import joblib
import pandas as pd

from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error


# Load dataset
df = pd.read_csv("f1_race_data.csv")

print("\nMissing values:")
print(df.isna().sum())

print("\nDataset shape:")
print(df.shape)


# Chronological split
train_df = df[df["year"] <= 2023].copy()
validation_df = df[df["year"] == 2024].copy()
test_df = df[df["year"] == 2025].copy()


# Separate features and target
X_train = train_df.drop(["finish_pos", "year"], axis=1)
y_train = train_df["finish_pos"]

X_validation = validation_df.drop(["finish_pos", "year"], axis=1)
y_validation = validation_df["finish_pos"]

X_test = test_df.drop(["finish_pos", "year"], axis=1)
y_test = test_df["finish_pos"]


# Convert categorical variables into numbers
X_train = pd.get_dummies(X_train, columns=["driver", "team", "track"])
X_validation = pd.get_dummies(X_validation, columns=["driver", "team", "track"])
X_test = pd.get_dummies(X_test, columns=["driver", "team", "track"])


# Make validation and test columns match training
X_validation = X_validation.reindex(columns=X_train.columns, fill_value=0)
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)


# Model
model = make_pipeline(
    SimpleImputer(strategy="median"),
    GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    )
)


# Train
model.fit(X_train, y_train)


# Validation predictions
validation_pred = model.predict(X_validation)

validation_mae = mean_absolute_error(
    y_validation,
    validation_pred
)

print(f"\nValidation MAE: {validation_mae:.2f}")


# Test predictions
test_pred = model.predict(X_test)

test_mae = mean_absolute_error(y_test,test_pred)

print(f"Test MAE: {test_mae:.2f}")

# Save model
joblib.dump(model, "f1_model.pkl")

# Save feature names
joblib.dump(
    X_train.columns.tolist(),
    "model_features.pkl"
)

# Save test predictions
predictions = test_df[["year", "track", "driver", "finish_pos"]].copy()

predictions["predicted_finish"] = test_pred

predictions.to_csv(
    "model_predictions.csv",
    index=False
)


print("\nSaved:")
print("f1_model.pkl")
print("model_features.pkl")
print("model_predictions.csv")