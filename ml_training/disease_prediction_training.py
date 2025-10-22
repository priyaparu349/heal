"""
Disease Prediction ML Training Script with Multiple Models
===========================================================
This script trains multiple ML classifiers on the real Kaggle Disease Prediction dataset
and performs comprehensive model comparison with hyperparameter tuning.

Dataset: Disease Prediction Using Machine Learning (Kaggle)
https://www.kaggle.com/datasets/kaushil268/disease-prediction-using-machine-learning

Models Evaluated:
- RandomForestClassifier
- DecisionTreeClassifier
- LogisticRegression
- Support Vector Machine (SVM)

Goal: Achieve at least 90% accuracy on real data
"""

import pandas as pd
import numpy as np
import pickle
import json
import os
from pathlib import Path
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
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
import warnings
warnings.filterwarnings('ignore')

# Configuration
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

print("=" * 80)
print("DISEASE PREDICTION ML MODEL TRAINING - MULTI-MODEL COMPARISON")
print("=" * 80)
print(f"Training Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()

# ============================================================================
# STEP 1: LOAD REAL KAGGLE DATASET
# ============================================================================
print("Step 1: Loading Real Kaggle Dataset...")
print("-" * 80)

# Get script directory
script_dir = Path(__file__).parent.absolute()
print(f"Working Directory: {script_dir}")

# Try multiple possible locations for CSV files
possible_paths = [
    script_dir,
    Path.cwd(),
    script_dir.parent / 'data',
    Path.cwd() / 'data'
]

training_csv = None
testing_csv = None

for path in possible_paths:
    training_candidate = path / 'Training.csv'
    testing_candidate = path / 'Testing.csv'

    if training_candidate.exists():
        training_csv = training_candidate
        if testing_candidate.exists():
            testing_csv = testing_candidate
        break

if training_csv is None:
    print("\n" + "!" * 80)
    print("ERROR: Dataset files not found!")
    print("!" * 80)
    print(f"\nSearched in the following locations:")
    for path in possible_paths:
        print(f"  - {path}")
    print("\n" + "=" * 80)
    print("MANUAL SETUP REQUIRED")
    print("=" * 80)
    print("\n1. Download the Kaggle dataset from:")
    print("   https://www.kaggle.com/datasets/kaushil268/disease-prediction-using-machine-learning")
    print("\n2. Extract and place these files in THIS directory:")
    print(f"   {script_dir}")
    print("\n   Required files:")
    print("   - Training.csv")
    print("   - Testing.csv")
    print("\n3. Verify the files are in place:")
    print(f"   Training.csv should be at: {script_dir / 'Training.csv'}")
    print(f"   Testing.csv should be at: {script_dir / 'Testing.csv'}")
    print("\n4. Run this script again:")
    print("   python disease_prediction_training.py")
    print("\n" + "!" * 80)
    exit(1)

try:
    # Load training data
    print(f"\nLoading from: {training_csv}")
    df_train = pd.read_csv(training_csv)
    print(f"✓ Training.csv loaded: {df_train.shape[0]} samples, {df_train.shape[1]} features")

    # Load testing data
    if testing_csv and testing_csv.exists():
        print(f"Loading from: {testing_csv}")
        df_test = pd.read_csv(testing_csv)
        print(f"✓ Testing.csv loaded: {df_test.shape[0]} samples")
        # Combine for full dataset to ensure proper splitting
        df = pd.concat([df_train, df_test], ignore_index=True)
        print(f"✓ Combined dataset: {df.shape[0]} samples")
    else:
        df = df_train
        print("! Testing.csv not found, using Training.csv only")

    print(f"\nDataset Overview:")
    print(f"  - Total Samples: {len(df)}")
    print(f"  - Total Features: {len(df.columns) - 1}")  # Excluding target column
    print(f"  - Unique Diseases: {df['prognosis'].nunique()}")

    # Check for 'prognosis' column
    if 'prognosis' not in df.columns:
        raise ValueError("'prognosis' column not found in dataset!")

    print(f"\nDisease Distribution (Top 10):")
    disease_counts = df['prognosis'].value_counts()
    for disease, count in disease_counts.head(10).items():
        print(f"  - {disease}: {count} samples")

except Exception as e:
    print("\n" + "!" * 80)
    print("ERROR: Failed to load dataset!")
    print("!" * 80)
    print(f"\n{str(e)}")
    print("\nPlease verify:")
    print("  1. The CSV files are not corrupted")
    print("  2. The files contain the 'prognosis' column")
    print("  3. The files are properly formatted CSV files")
    print("\n" + "!" * 80)
    exit(1)

# ============================================================================
# STEP 2: DATA PREPROCESSING
# ============================================================================
print("\n" + "=" * 80)
print("Step 2: Data Preprocessing")
print("-" * 80)

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

# Ensure all feature values are numeric
X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
print("✓ All features converted to numeric")

# Encode target labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
print(f"✓ Encoded {len(label_encoder.classes_)} disease classes")

# Store disease classes
disease_classes = label_encoder.classes_.tolist()

# ============================================================================
# STEP 3: TRAIN-TEST SPLIT
# ============================================================================
print("\n" + "=" * 80)
print("Step 3: Train-Test Split")
print("-" * 80)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y_encoded
)

print(f"Training set: {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)")
print(f"Testing set: {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)")

# ============================================================================
# STEP 4: DEFINE MODELS AND HYPERPARAMETER GRIDS
# ============================================================================
print("\n" + "=" * 80)
print("Step 4: Configuring Models and Hyperparameter Grids")
print("-" * 80)

models_config = {
    'RandomForest': {
        'model': RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
        'params': {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
    },
    'DecisionTree': {
        'model': DecisionTreeClassifier(random_state=RANDOM_STATE),
        'params': {
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10, 20],
            'min_samples_leaf': [1, 2, 4, 8],
            'criterion': ['gini', 'entropy']
        }
    },
    'LogisticRegression': {
        'model': LogisticRegression(random_state=RANDOM_STATE, max_iter=1000, n_jobs=-1),
        'params': {
            'C': [0.01, 0.1, 1, 10, 100],
            'solver': ['lbfgs', 'liblinear'],
            'penalty': ['l2']
        }
    },
    'SVM': {
        'model': SVC(random_state=RANDOM_STATE, probability=True),
        'params': {
            'C': [0.1, 1, 10, 100],
            'kernel': ['linear', 'rbf'],
            'gamma': ['scale', 'auto']
        }
    }
}

print(f"✓ Configured {len(models_config)} models:")
for model_name in models_config.keys():
    print(f"  - {model_name}")

# ============================================================================
# STEP 5: TRAIN AND EVALUATE MODELS WITH GRIDSEARCHCV
# ============================================================================
print("\n" + "=" * 80)
print("Step 5: Training Models with GridSearchCV (5-Fold CV)")
print("=" * 80)

results = {}

for model_name, config in models_config.items():
    print(f"\n{'='*80}")
    print(f"Training {model_name}...")
    print(f"{'='*80}")

    # GridSearchCV with 5-fold cross-validation
    grid_search = GridSearchCV(
        estimator=config['model'],
        param_grid=config['params'],
        cv=CV_FOLDS,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )

    print(f"Searching hyperparameters...")
    grid_search.fit(X_train, y_train)

    # Get best model
    best_model = grid_search.best_estimator_

    # Predictions
    y_pred = best_model.predict(X_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)

    # Cross-validation score
    cv_scores = cross_val_score(best_model, X_train, y_train, cv=CV_FOLDS, scoring='accuracy')

    # Store results
    results[model_name] = {
        'model': best_model,
        'grid_search': grid_search,
        'best_params': grid_search.best_params_,
        'cv_score_mean': cv_scores.mean(),
        'cv_score_std': cv_scores.std(),
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'y_pred': y_pred,
        'confusion_matrix': confusion_matrix(y_test, y_pred)
    }

    print(f"\n✓ {model_name} Results:")
    print(f"  Best Parameters: {grid_search.best_params_}")
    print(f"  CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    print(f"  Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1-Score: {f1:.4f}")

# ============================================================================
# STEP 6: MODEL COMPARISON TABLE
# ============================================================================
print("\n" + "=" * 80)
print("STEP 6: MODEL COMPARISON TABLE")
print("=" * 80)

comparison_data = []
for model_name, result in results.items():
    comparison_data.append({
        'Model': model_name,
        'CV Score': f"{result['cv_score_mean']:.4f} ± {result['cv_score_std']:.4f}",
        'Test Accuracy': f"{result['accuracy']:.4f}",
        'Precision': f"{result['precision']:.4f}",
        'Recall': f"{result['recall']:.4f}",
        'F1-Score': f"{result['f1_score']:.4f}"
    })

comparison_df = pd.DataFrame(comparison_data)
comparison_df = comparison_df.sort_values('Test Accuracy', ascending=False)

print("\n📊 MODEL PERFORMANCE COMPARISON:")
print("=" * 80)
print(comparison_df.to_string(index=False))
print("=" * 80)

# ============================================================================
# STEP 7: SELECT BEST MODEL
# ============================================================================
print("\n" + "=" * 80)
print("Step 7: Selecting Best Model")
print("-" * 80)

# Find best model by accuracy
best_model_name = max(results.items(), key=lambda x: x[1]['accuracy'])[0]
best_result = results[best_model_name]
best_model = best_result['model']

print(f"\n🏆 BEST MODEL: {best_model_name}")
print(f"  Accuracy: {best_result['accuracy']:.4f} ({best_result['accuracy']*100:.2f}%)")
print(f"  Precision: {best_result['precision']:.4f}")
print(f"  Recall: {best_result['recall']:.4f}")
print(f"  F1-Score: {best_result['f1_score']:.4f}")

if best_result['accuracy'] >= 0.90:
    print(f"\n✓ SUCCESS: Model achieved {best_result['accuracy']*100:.2f}% accuracy (≥ 90% target)")
else:
    print(f"\n⚠ WARNING: Model achieved {best_result['accuracy']*100:.2f}% accuracy (< 90% target)")
    print("  Consider: More data, feature engineering, or different algorithms")

# ============================================================================
# STEP 8: DETAILED METRICS FOR BEST MODEL
# ============================================================================
print("\n" + "=" * 80)
print(f"Step 8: Detailed Evaluation - {best_model_name}")
print("=" * 80)

print("\n📋 CLASSIFICATION REPORT:")
print("-" * 80)
report = classification_report(y_test, best_result['y_pred'], target_names=disease_classes, zero_division=0)
print(report)

report_dict = classification_report(
    y_test, best_result['y_pred'],
    target_names=disease_classes,
    output_dict=True,
    zero_division=0
)

# ============================================================================
# STEP 9: CONFUSION MATRIX VISUALIZATION
# ============================================================================
print("\n" + "=" * 80)
print("Step 9: Generating Confusion Matrix Heatmap")
print("-" * 80)

cm = best_result['confusion_matrix']

plt.figure(figsize=(20, 16))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=disease_classes,
    yticklabels=disease_classes,
    cbar_kws={'label': 'Count'},
    square=True
)
plt.title(f'Confusion Matrix - {best_model_name} Model\nAccuracy: {best_result["accuracy"]*100:.2f}%',
          fontsize=18, fontweight='bold', pad=20)
plt.xlabel('Predicted Disease', fontsize=14, fontweight='bold')
plt.ylabel('True Disease', fontsize=14, fontweight='bold')
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.yticks(rotation=0, fontsize=10)
plt.tight_layout()

heatmap_path = 'confusion_matrix_heatmap.png'
plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
print(f"✓ Confusion matrix heatmap saved: {heatmap_path}")
plt.close()

# ============================================================================
# STEP 10: FEATURE IMPORTANCE (for tree-based models)
# ============================================================================
print("\n" + "=" * 80)
print("Step 10: Feature Importance Analysis")
print("-" * 80)

if hasattr(best_model, 'feature_importances_'):
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)

    print("\n🔝 TOP 10 MOST IMPORTANT SYMPTOMS:")
    print("=" * 80)
    top_10 = feature_importance.head(10)
    for idx, row in top_10.iterrows():
        print(f"  {row['feature']:30s} : {row['importance']:.6f}")
    print("=" * 80)

    feature_importance_path = 'feature_importance.csv'
    feature_importance.to_csv(feature_importance_path, index=False)
    print(f"\n✓ Feature importance saved: {feature_importance_path}")
else:
    print(f"⚠ {best_model_name} does not provide feature importance")
    feature_importance = None

# ============================================================================
# STEP 11: SAVE MODEL AND ARTIFACTS
# ============================================================================
print("\n" + "=" * 80)
print("Step 11: Saving Best Model and Artifacts")
print("-" * 80)

# Save best model
model_path = 'best_model.pkl'
with open(model_path, 'wb') as f:
    pickle.dump(best_model, f)
print(f"✓ Best model saved: {model_path}")

# Save label encoder
encoder_path = 'label_encoder.pkl'
with open(encoder_path, 'wb') as f:
    pickle.dump(label_encoder, f)
print(f"✓ Label encoder saved: {encoder_path}")

# Save feature names
feature_names_path = 'feature_names.json'
with open(feature_names_path, 'w') as f:
    json.dump(X.columns.tolist(), f, indent=2)
print(f"✓ Feature names saved: {feature_names_path}")

# Save disease classes
disease_classes_path = 'disease_classes.json'
with open(disease_classes_path, 'w') as f:
    json.dump(disease_classes, f, indent=2)
print(f"✓ Disease classes saved: {disease_classes_path}")

# Save comprehensive evaluation metrics
metrics = {
    'best_model': best_model_name,
    'version': '2.0',
    'training_date': datetime.now().isoformat(),
    'dataset_info': {
        'total_samples': len(df),
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'n_features': len(X.columns),
        'n_diseases': len(disease_classes),
        'test_split': TEST_SIZE,
        'cv_folds': CV_FOLDS
    },
    'all_models_comparison': {
        model_name: {
            'best_params': result['best_params'],
            'cv_score_mean': float(result['cv_score_mean']),
            'cv_score_std': float(result['cv_score_std']),
            'accuracy': float(result['accuracy']),
            'precision': float(result['precision']),
            'recall': float(result['recall']),
            'f1_score': float(result['f1_score'])
        }
        for model_name, result in results.items()
    },
    'best_model_details': {
        'model_name': best_model_name,
        'best_params': best_result['best_params'],
        'metrics': {
            'accuracy': float(best_result['accuracy']),
            'precision': float(best_result['precision']),
            'recall': float(best_result['recall']),
            'f1_score': float(best_result['f1_score']),
            'cv_score_mean': float(best_result['cv_score_mean']),
            'cv_score_std': float(best_result['cv_score_std'])
        },
        'confusion_matrix': cm.tolist(),
        'classification_report': report_dict
    },
    'feature_importance': feature_importance.to_dict('records') if feature_importance is not None else None
}

metrics_path = 'model_evaluation_metrics.json'
with open(metrics_path, 'w') as f:
    json.dump(metrics, f, indent=2)
print(f"✓ Evaluation metrics saved: {metrics_path}")

# ============================================================================
# STEP 12: SAMPLE PREDICTIONS
# ============================================================================
print("\n" + "=" * 80)
print("Step 12: Testing Sample Predictions")
print("-" * 80)

sample_indices = np.random.choice(len(X_test), size=min(5, len(X_test)), replace=False)
correct_predictions = 0

for i, idx in enumerate(sample_indices, 1):
    sample = X_test.iloc[idx:idx+1]
    true_label = disease_classes[y_test.iloc[idx]]
    pred_label = disease_classes[best_model.predict(sample)[0]]

    if hasattr(best_model, 'predict_proba'):
        confidence = best_model.predict_proba(sample)[0].max()
        confidence_str = f"{confidence*100:.2f}%"
    else:
        confidence_str = "N/A"

    match = true_label == pred_label
    if match:
        correct_predictions += 1

    print(f"\nSample {i}:")
    print(f"  True Disease: {true_label}")
    print(f"  Predicted: {pred_label}")
    print(f"  Confidence: {confidence_str}")
    print(f"  Match: {'✓ Correct' if match else '✗ Wrong'}")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("TRAINING COMPLETE - ARTIFACTS GENERATED")
print("=" * 80)

print(f"\n📦 Generated Files:")
print(f"  ✓ best_model.pkl - Trained {best_model_name} model ({best_result['accuracy']*100:.2f}% accuracy)")
print(f"  ✓ label_encoder.pkl - Label encoder for {len(disease_classes)} diseases")
print(f"  ✓ feature_names.json - List of {len(X.columns)} symptom features")
print(f"  ✓ disease_classes.json - List of {len(disease_classes)} disease classes")
print(f"  ✓ model_evaluation_metrics.json - Comprehensive metrics for all models")
print(f"  ✓ confusion_matrix_heatmap.png - Visual confusion matrix")
if feature_importance is not None:
    print(f"  ✓ feature_importance.csv - Symptom importance rankings")

print("\n" + "=" * 80)
print("SUMMARY STATISTICS:")
print("=" * 80)
print(f"Best Model: {best_model_name}")
print(f"Test Accuracy: {best_result['accuracy']*100:.2f}%")
print(f"Total Diseases: {len(disease_classes)}")
print(f"Total Symptoms: {len(X.columns)}")
print(f"Training Samples: {len(X_train)}")
print(f"Testing Samples: {len(X_test)}")

print("\n" + "=" * 80)
print("NEXT STEPS:")
print("=" * 80)
print("1. Review confusion_matrix_heatmap.png for model performance")
print("2. Check model_evaluation_metrics.json for detailed metrics")
print("3. Integrate best_model.pkl into Supabase Edge Functions")
print("4. Upload metrics to Supabase database for tracking")

print("\n" + "=" * 80)
print(f"Training End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 80)
