"""
Disease Prediction ML Training Script
======================================
This script trains a RandomForest classifier on the Disease Prediction dataset
and generates comprehensive evaluation metrics for research paper publication.

Dataset: Disease Prediction Using Machine Learning (Kaggle)
Model: RandomForestClassifier
Output: Trained model + evaluation metrics + confusion matrix heatmap
"""

import pandas as pd
import numpy as np
import pickle
import json
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Configuration
RANDOM_STATE = 42
TEST_SIZE = 0.2
N_ESTIMATORS = 100

print("=" * 70)
print("DISEASE PREDICTION ML MODEL TRAINING")
print("=" * 70)
print(f"Training Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()

# Step 1: Load Dataset
print("Step 1: Loading Dataset...")
print("-" * 70)

# Note: Download dataset from Kaggle first:
# https://www.kaggle.com/datasets/kaushil268/disease-prediction-using-machine-learning
# Expected files: Training.csv and Testing.csv

try:
    # Load training data
    df_train = pd.read_csv('Training.csv')
    print(f"✓ Training data loaded: {df_train.shape[0]} samples, {df_train.shape[1]} features")

    # Load testing data (if available)
    try:
        df_test = pd.read_csv('Testing.csv')
        print(f"✓ Testing data loaded: {df_test.shape[0]} samples")
        # Combine for full dataset
        df = pd.concat([df_train, df_test], ignore_index=True)
        print(f"✓ Combined dataset: {df.shape[0]} samples")
    except FileNotFoundError:
        df = df_train
        print("! Testing.csv not found, using Training.csv only")

    print(f"\nDataset Overview:")
    print(f"  - Total Samples: {len(df)}")
    print(f"  - Total Features: {len(df.columns) - 1}")  # Excluding target column
    print(f"  - Unique Diseases: {df['prognosis'].nunique()}")
    print(f"  - Disease Distribution:")
    print(df['prognosis'].value_counts().head(10))

except FileNotFoundError:
    print("\n" + "!" * 70)
    print("ERROR: Dataset files not found!")
    print("!" * 70)
    print("\nPlease download the dataset from Kaggle:")
    print("https://www.kaggle.com/datasets/kaushil268/disease-prediction-using-machine-learning")
    print("\nExpected files in the same directory:")
    print("  - Training.csv")
    print("  - Testing.csv (optional)")
    print("\nAlternatively, use this synthetic dataset generator:")
    print("=" * 70)

    # Generate synthetic dataset for demonstration
    print("\nGenerating SYNTHETIC dataset for demonstration...")
    print("(Replace with real Kaggle data for production)")

    diseases = [
        'Fungal infection', 'Allergy', 'GERD', 'Chronic cholestasis', 'Drug Reaction',
        'Peptic ulcer disease', 'AIDS', 'Diabetes', 'Gastroenteritis', 'Bronchial Asthma',
        'Hypertension', 'Migraine', 'Cervical spondylosis', 'Paralysis (brain hemorrhage)',
        'Jaundice', 'Malaria', 'Chicken pox', 'Dengue', 'Typhoid', 'Hepatitis A',
        'Hepatitis B', 'Hepatitis C', 'Hepatitis D', 'Hepatitis E', 'Alcoholic hepatitis',
        'Tuberculosis', 'Common Cold', 'Pneumonia', 'Dimorphic hemmorhoids(piles)',
        'Heart attack', 'Varicose veins', 'Hypothyroidism', 'Hyperthyroidism', 'Hypoglycemia',
        'Osteoarthristis', 'Arthritis', 'Vertigo', 'Acne', 'Urinary tract infection',
        'Psoriasis', 'Impetigo'
    ]

    symptoms = [
        'itching', 'skin_rash', 'nodal_skin_eruptions', 'continuous_sneezing', 'shivering',
        'chills', 'joint_pain', 'stomach_pain', 'acidity', 'ulcers_on_tongue', 'muscle_wasting',
        'vomiting', 'burning_micturition', 'spotting_urination', 'fatigue', 'weight_gain',
        'anxiety', 'cold_hands_and_feets', 'mood_swings', 'weight_loss', 'restlessness',
        'lethargy', 'patches_in_throat', 'irregular_sugar_level', 'cough', 'high_fever',
        'sunken_eyes', 'breathlessness', 'sweating', 'dehydration', 'indigestion', 'headache',
        'yellowish_skin', 'dark_urine', 'nausea', 'loss_of_appetite', 'pain_behind_the_eyes',
        'back_pain', 'constipation', 'abdominal_pain', 'diarrhoea', 'mild_fever', 'yellow_urine',
        'yellowing_of_eyes', 'acute_liver_failure', 'fluid_overload', 'swelling_of_stomach',
        'swelled_lymph_nodes', 'malaise', 'blurred_and_distorted_vision', 'phlegm', 'throat_irritation',
        'redness_of_eyes', 'sinus_pressure', 'runny_nose', 'congestion', 'chest_pain',
        'weakness_in_limbs', 'fast_heart_rate', 'pain_during_bowel_movements',
        'pain_in_anal_region', 'bloody_stool', 'irritation_in_anus', 'neck_pain', 'dizziness',
        'cramps', 'bruising', 'obesity', 'swollen_legs', 'swollen_blood_vessels',
        'puffy_face_and_eyes', 'enlarged_thyroid', 'brittle_nails', 'swollen_extremeties',
        'excessive_hunger', 'extra_marital_contacts', 'drying_and_tingling_lips', 'slurred_speech',
        'knee_pain', 'hip_joint_pain', 'muscle_weakness', 'stiff_neck', 'swelling_joints',
        'movement_stiffness', 'spinning_movements', 'loss_of_balance', 'unsteadiness',
        'weakness_of_one_body_side', 'loss_of_smell', 'bladder_discomfort',
        'foul_smell_of_urine', 'continuous_feel_of_urine', 'passage_of_gases', 'internal_itching',
        'toxic_look_(typhos)', 'depression', 'irritability', 'muscle_pain', 'altered_sensorium',
        'red_spots_over_body', 'belly_pain', 'abnormal_menstruation', 'dischromic_patches',
        'watering_from_eyes', 'increased_appetite', 'polyuria', 'family_history', 'mucoid_sputum',
        'rusty_sputum', 'lack_of_concentration', 'visual_disturbances', 'receiving_blood_transfusion',
        'receiving_unsterile_injections', 'coma', 'stomach_bleeding', 'distention_of_abdomen',
        'history_of_alcohol_consumption', 'fluid_overload', 'blood_in_sputum', 'prominent_veins_on_calf',
        'palpitations', 'painful_walking', 'pus_filled_pimples', 'blackheads', 'scurring',
        'skin_peeling', 'silver_like_dusting', 'small_dents_in_nails', 'inflammatory_nails',
        'blister', 'red_sore_around_nose', 'yellow_crust_ooze'
    ]

    # Generate synthetic data
    np.random.seed(RANDOM_STATE)
    n_samples = 4920

    data = {}
    for symptom in symptoms:
        data[symptom] = np.random.choice([0, 1], size=n_samples, p=[0.85, 0.15])

    data['prognosis'] = np.random.choice(diseases, size=n_samples)
    df = pd.DataFrame(data)

    print(f"✓ Synthetic dataset generated: {len(df)} samples, {len(symptoms)} symptoms")

# Step 2: Data Preprocessing
print("\n" + "=" * 70)
print("Step 2: Data Preprocessing")
print("-" * 70)

# Separate features and target
X = df.drop('prognosis', axis=1)
y = df['prognosis']

print(f"Features (X): {X.shape}")
print(f"Target (y): {y.shape}")

# Handle missing values
missing_count = X.isnull().sum().sum()
if missing_count > 0:
    print(f"! Found {missing_count} missing values - filling with 0")
    X = X.fillna(0)
else:
    print("✓ No missing values found")

# Encode target labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
print(f"✓ Encoded {len(label_encoder.classes_)} disease classes")

# Store disease classes for later use
disease_classes = label_encoder.classes_.tolist()

# Step 3: Train-Test Split
print("\n" + "=" * 70)
print("Step 3: Train-Test Split")
print("-" * 70)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_encoded
)

print(f"Training set: {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)")
print(f"Testing set: {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)")

# Step 4: Model Training
print("\n" + "=" * 70)
print("Step 4: Training RandomForest Model")
print("-" * 70)

model = RandomForestClassifier(
    n_estimators=N_ESTIMATORS,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbose=1
)

print(f"Model Configuration:")
print(f"  - Algorithm: RandomForestClassifier")
print(f"  - N Estimators: {N_ESTIMATORS}")
print(f"  - Random State: {RANDOM_STATE}")
print()

model.fit(X_train, y_train)
print("✓ Model training completed")

# Step 5: Model Evaluation
print("\n" + "=" * 70)
print("Step 5: Model Evaluation")
print("-" * 70)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)

print("\n📊 OVERALL METRICS:")
print("=" * 70)
print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"Precision: {precision:.4f} ({precision*100:.2f}%)")
print(f"Recall:    {recall:.4f} ({recall*100:.2f}%)")
print(f"F1-Score:  {f1:.4f} ({f1*100:.2f}%)")
print("=" * 70)

# Classification Report
print("\n📋 CLASSIFICATION REPORT:")
print("=" * 70)
report = classification_report(y_test, y_pred, target_names=disease_classes, zero_division=0)
print(report)

# Classification report as dict
report_dict = classification_report(
    y_test, y_pred,
    target_names=disease_classes,
    output_dict=True,
    zero_division=0
)

# Confusion Matrix
print("\n🔥 CONFUSION MATRIX:")
print("=" * 70)
cm = confusion_matrix(y_test, y_pred)
print(f"Shape: {cm.shape}")
print(f"Total predictions: {cm.sum()}")

# Step 6: Confusion Matrix Visualization
print("\n" + "=" * 70)
print("Step 6: Generating Confusion Matrix Heatmap")
print("-" * 70)

plt.figure(figsize=(16, 14))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=disease_classes,
    yticklabels=disease_classes,
    cbar_kws={'label': 'Count'}
)
plt.title('Confusion Matrix - Disease Prediction Model', fontsize=16, fontweight='bold')
plt.xlabel('Predicted Disease', fontsize=12)
plt.ylabel('True Disease', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()

heatmap_path = 'confusion_matrix_heatmap.png'
plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
print(f"✓ Heatmap saved: {heatmap_path}")
plt.close()

# Step 7: Save Model and Artifacts
print("\n" + "=" * 70)
print("Step 7: Saving Model and Artifacts")
print("-" * 70)

# Save trained model
model_path = 'disease_model.pkl'
with open(model_path, 'wb') as f:
    pickle.dump(model, f)
print(f"✓ Model saved: {model_path}")

# Save label encoder
encoder_path = 'label_encoder.pkl'
with open(encoder_path, 'wb') as f:
    pickle.dump(label_encoder, f)
print(f"✓ Label encoder saved: {encoder_path}")

# Save feature names
feature_names_path = 'feature_names.json'
with open(feature_names_path, 'w') as f:
    json.dump(X.columns.tolist(), f)
print(f"✓ Feature names saved: {feature_names_path}")

# Save disease classes
disease_classes_path = 'disease_classes.json'
with open(disease_classes_path, 'w') as f:
    json.dump(disease_classes, f)
print(f"✓ Disease classes saved: {disease_classes_path}")

# Save evaluation metrics
metrics = {
    'model_name': 'RandomForestClassifier',
    'version': '1.0',
    'training_date': datetime.now().isoformat(),
    'dataset_info': {
        'total_samples': len(df),
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'n_features': len(X.columns),
        'n_diseases': len(disease_classes),
        'test_split': TEST_SIZE
    },
    'model_params': {
        'n_estimators': N_ESTIMATORS,
        'random_state': RANDOM_STATE
    },
    'metrics': {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1)
    },
    'confusion_matrix': cm.tolist(),
    'classification_report': report_dict
}

metrics_path = 'model_evaluation_metrics.json'
with open(metrics_path, 'w') as f:
    json.dump(metrics, f, indent=2)
print(f"✓ Metrics saved: {metrics_path}")

# Step 8: Feature Importance
print("\n" + "=" * 70)
print("Step 8: Feature Importance Analysis")
print("-" * 70)

feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 20 Most Important Features:")
print(feature_importance.head(20).to_string(index=False))

feature_importance_path = 'feature_importance.csv'
feature_importance.to_csv(feature_importance_path, index=False)
print(f"\n✓ Feature importance saved: {feature_importance_path}")

# Step 9: Sample Predictions
print("\n" + "=" * 70)
print("Step 9: Testing Sample Predictions")
print("-" * 70)

sample_indices = np.random.choice(len(X_test), size=5, replace=False)
for i, idx in enumerate(sample_indices, 1):
    sample = X_test.iloc[idx:idx+1]
    true_label = disease_classes[y_test[idx]]
    pred_label = disease_classes[model.predict(sample)[0]]
    confidence = model.predict_proba(sample)[0].max()

    print(f"\nSample {i}:")
    print(f"  True Disease: {true_label}")
    print(f"  Predicted: {pred_label}")
    print(f"  Confidence: {confidence*100:.2f}%")
    print(f"  Match: {'✓' if true_label == pred_label else '✗'}")

# Final Summary
print("\n" + "=" * 70)
print("TRAINING COMPLETE - FILES GENERATED")
print("=" * 70)
print(f"\n✓ disease_model.pkl - Trained RandomForest model")
print(f"✓ label_encoder.pkl - Label encoder for diseases")
print(f"✓ feature_names.json - List of symptom features")
print(f"✓ disease_classes.json - List of disease classes")
print(f"✓ model_evaluation_metrics.json - Complete metrics")
print(f"✓ confusion_matrix_heatmap.png - Visual heatmap")
print(f"✓ feature_importance.csv - Feature rankings")

print("\n" + "=" * 70)
print("NEXT STEPS:")
print("=" * 70)
print("1. Review the confusion_matrix_heatmap.png")
print("2. Check model_evaluation_metrics.json for paper metrics")
print("3. Integrate these files into your Supabase Edge Functions")
print("4. Upload metrics to your database")
print("\n" + "=" * 70)
print(f"Training End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 70)
