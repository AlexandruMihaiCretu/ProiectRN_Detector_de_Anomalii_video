import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent

TRAIN_PATH = ROOT_DIR / "data" / "train" / "train.csv"
VAL_PATH = ROOT_DIR / "data" / "validation" / "val.csv"
TEST_PATH = ROOT_DIR / "data" / "test" / "test.csv"

MODEL_DIR = ROOT_DIR / "models"
MODEL_FILE = MODEL_DIR / "bouncer_model.pkl"

train_df = pd.read_csv(TRAIN_PATH)
val_df = pd.read_csv(VAL_PATH)
test_df = pd.read_csv(TEST_PATH)

# 2. Select the "Behavioral DNA" features for the brain to look at
# We exclude 'timestamp' and 'obj_id' because they don't help predict if a detection is a glitch
features = ['class_id', 'confidence', 'age_frames', 'velocity_px', 'aspect_ratio']
target = 'is_real'

X_train = train_df[features]
y_train = train_df[target]

X_val = val_df[features]
y_val = val_df[target]

X_test = test_df[features]
y_test = test_df[target]

print(f"Training 'The Bouncer' on {len(X_train)} rows...")

# 3. Build and Train the Random Forest
# n_estimators=100 means 100 small "decision trees" voting together
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 4. Evaluate the brain
y_pred = model.predict(X_test)
print("\n--- Model Performance ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2%}")
print("\nDetailed Report:")
print(classification_report(y_test, y_pred))

# 5. Save the Brain
# This .pkl file is what we will load into the main detection script
MODEL_DIR.mkdir(parents=True, exist_ok=True)
joblib.dump(model, MODEL_FILE)
print("\n Success! The Bouncer brain is saved in 'models/bouncer_model.pkl'")