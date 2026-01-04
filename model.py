import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
import os

# Load dataset
df = pd.read_csv("DiseaseAndSymptoms.csv")

# Separate target
y = df["Disease"]
X_raw = df.drop("Disease", axis=1)

# Convert text symptoms to binary features
all_symptoms = set()
for col in X_raw.columns:
    all_symptoms.update(X_raw[col].dropna().unique())

all_symptoms = sorted(all_symptoms)

X = pd.DataFrame(0, index=df.index, columns=all_symptoms)

for i, row in X_raw.iterrows():
    for symptom in row:
        if pd.notna(symptom):
            X.at[i, symptom] = 1

# Encode disease labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Check if model exists, if not train and save
model_file = "disease_model.pkl"
encoder_file = "label_encoder.pkl"
symptoms_file = "symptoms_list.pkl"

if os.path.exists(model_file) and os.path.exists(encoder_file) and os.path.exists(symptoms_file):
    # Load existing model
    model = joblib.load(model_file)
    le = joblib.load(encoder_file)
    symptom_columns = joblib.load(symptoms_file)
else:
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42
    )

    # Train model
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    # Save model and encoders
    joblib.dump(model, model_file)
    joblib.dump(le, encoder_file)
    joblib.dump(X.columns.tolist(), symptoms_file)
    symptom_columns = X.columns.tolist()

# Prediction function
def predict_top3(input_df):
    probs = model.predict_proba(input_df)[0]
    top3_idx = np.argsort(probs)[-3:][::-1]
    diseases = le.inverse_transform(top3_idx)
    scores = probs[top3_idx]
    return list(zip(diseases, scores))
