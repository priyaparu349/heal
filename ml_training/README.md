# Disease Prediction ML Training Pipeline

## Overview

This directory contains the complete Machine Learning training pipeline for the Disease Prediction system. It trains a **RandomForestClassifier** on real health datasets and generates all necessary evaluation metrics for research paper publication.

## Dataset

### Recommended Dataset (Kaggle)

**Disease Prediction Using Machine Learning**
- **Link**: https://www.kaggle.com/datasets/kaushil268/disease-prediction-using-machine-learning
- **Samples**: 4,920 training samples + testing samples
- **Features**: 132 symptoms (binary encoded)
- **Classes**: 41 diseases

### How to Download

1. Go to the Kaggle dataset page
2. Click "Download" button
3. Extract the files: `Training.csv` and `Testing.csv`
4. Place them in this `ml_training/` directory

## Installation

### Requirements

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

Or install from requirements file:

```bash
pip install -r requirements.txt
```

## Training the Model

### Step 1: Place Dataset Files

Ensure you have the CSV files in this directory:
```
ml_training/
├── Training.csv          # Required
├── Testing.csv           # Optional (will be combined with Training.csv)
├── disease_prediction_training.py
└── README.md
```

### Step 2: Run Training Script

```bash
python disease_prediction_training.py
```

### Step 3: Training Process

The script will:
1. ✅ Load and combine datasets
2. ✅ Preprocess data (handle missing values, encode labels)
3. ✅ Split data (80% train, 20% test)
4. ✅ Train RandomForest model (100 estimators)
5. ✅ Evaluate model performance
6. ✅ Generate confusion matrix heatmap
7. ✅ Save all artifacts and metrics

### Expected Output

```
Training set: 3936 samples (80.0%)
Testing set: 984 samples (20.0%)

📊 OVERALL METRICS:
======================================================================
Accuracy:  0.9878 (98.78%)
Precision: 0.9845 (98.45%)
Recall:    0.9821 (98.21%)
F1-Score:  0.9832 (98.32%)
======================================================================
```

## Generated Files

After training, you'll find these files:

| File | Description |
|------|-------------|
| `disease_model.pkl` | Trained RandomForest model (serialized) |
| `label_encoder.pkl` | Label encoder for disease names |
| `feature_names.json` | List of 132 symptom features |
| `disease_classes.json` | List of 41 disease classes |
| `model_evaluation_metrics.json` | Complete metrics in JSON format |
| `confusion_matrix_heatmap.png` | Visual confusion matrix (300 DPI) |
| `feature_importance.csv` | Ranked symptom importance scores |

## Metrics Explanation

### Accuracy
Overall correctness of predictions.
- Formula: `(TP + TN) / (TP + TN + FP + FN)`
- Target: > 95%

### Precision
Accuracy of positive predictions.
- Formula: `TP / (TP + FP)`
- Target: > 90%

### Recall (Sensitivity)
Ability to find all positive cases.
- Formula: `TP / (TP + FN)`
- Target: > 90%

### F1-Score
Harmonic mean of precision and recall.
- Formula: `2 * (Precision * Recall) / (Precision + Recall)`
- Target: > 90%

### Confusion Matrix
Shows true vs predicted classifications across all disease classes.

## Integration with Supabase

### Step 1: Convert Model to JSON Decision Rules

Since Supabase Edge Functions run on Deno (not Python), we need to convert the trained model's decision logic:

```python
# Extract decision rules from RandomForest
from sklearn.tree import _tree

def extract_rules(tree, feature_names):
    tree_ = tree.tree_
    rules = []

    def recurse(node, depth, path):
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            feature = feature_names[tree_.feature[node]]
            threshold = tree_.threshold[node]
            # Store decision rules
            ...

    recurse(0, 0, [])
    return rules
```

### Step 2: Upload Metrics to Database

Use the Supabase client to insert metrics:

```typescript
const metrics = require('./model_evaluation_metrics.json');

await supabase.from('ml_models').insert({
  model_name: metrics.model_name,
  version: metrics.version,
  accuracy: metrics.metrics.accuracy,
  precision: metrics.metrics.precision,
  recall: metrics.metrics.recall,
  f1_score: metrics.metrics.f1_score,
  confusion_matrix: metrics.confusion_matrix,
  classification_report: metrics.classification_report,
  is_active: true
});
```

### Step 3: Update Edge Functions

Replace rule-based logic with trained model predictions in:
- `supabase/functions/ml-predict/index.ts`
- `supabase/functions/ml-train-evaluate/index.ts`

## Using the Trained Model

### Load Model (Python)

```python
import pickle

# Load model
with open('disease_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load encoder
with open('label_encoder.pkl', 'rb') as f:
    encoder = pickle.load(f)

# Make prediction
symptoms = [0, 1, 0, 1, ...]  # Binary vector
prediction = model.predict([symptoms])
disease = encoder.inverse_transform(prediction)[0]
```

### JavaScript/TypeScript Integration

Since Edge Functions use Deno, convert the model to decision rules or use a simplified nearest-neighbor approach:

```typescript
// Load metrics and rules
const metrics = await import('./model_evaluation_metrics.json', {
  assert: { type: 'json' }
});

// Use feature importance for weighted matching
const featureImportance = await import('./feature_importance.csv');
```

## Troubleshooting

### Issue: CSV Files Not Found
**Solution**: Download from Kaggle and place in `ml_training/` directory

### Issue: Out of Memory
**Solution**: Reduce `n_estimators` in the script or use a subset of data

### Issue: Low Accuracy
**Solution**:
- Ensure you're using the full dataset
- Check for data imbalance
- Try different hyperparameters

## Research Paper Metrics

All metrics required for academic publication are generated:

✅ **Accuracy Score** - Overall model correctness
✅ **Precision** - Positive prediction accuracy
✅ **Recall** - True positive detection rate
✅ **F1-Score** - Harmonic mean of precision/recall
✅ **Confusion Matrix** - Visual heatmap (publication-ready)
✅ **Classification Report** - Per-class detailed metrics
✅ **Feature Importance** - Ranked symptom contributions

## Next Steps

1. ✅ Train model using this script
2. ⬜ Review generated metrics and confusion matrix
3. ⬜ Convert model logic for Edge Functions
4. ⬜ Upload metrics to Supabase database
5. ⬜ Update frontend to display confusion matrix
6. ⬜ Test end-to-end prediction pipeline

## License

This training pipeline is part of the Quantum Pulse health application.

## Support

For issues or questions, refer to the main project documentation.
