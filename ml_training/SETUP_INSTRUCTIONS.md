# Disease Prediction ML Training - Setup Instructions

## Overview

This training script trains multiple machine learning models on the real Kaggle Disease Prediction dataset and automatically selects the best performing model (targeting ≥90% accuracy).

## Prerequisites

1. **Python 3.7+** installed on your system
2. **Required Python packages** (see requirements.txt)

## Step-by-Step Setup

### Step 1: Install Python Dependencies

Open a terminal/command prompt in this directory and run:

```bash
pip install -r requirements.txt
```

This will install:
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn

### Step 2: Download the Kaggle Dataset

1. Go to: https://www.kaggle.com/datasets/kaushil268/disease-prediction-using-machine-learning

2. Click "Download" to get the dataset (you may need to create a free Kaggle account)

3. Extract the ZIP file - you should get two CSV files:
   - `Training.csv`
   - `Testing.csv`

### Step 3: Place CSV Files in This Directory

**IMPORTANT:** The CSV files MUST be placed in the same directory as `disease_prediction_training.py`

Your directory structure should look like this:

```
ml_training/
├── disease_prediction_training.py
├── Training.csv              ← Place here
├── Testing.csv               ← Place here
├── requirements.txt
└── SETUP_INSTRUCTIONS.md
```

**To verify the correct location:**

#### Windows:
```powershell
dir *.csv
```

#### Mac/Linux:
```bash
ls -la *.csv
```

You should see both `Training.csv` and `Testing.csv` listed.

### Step 4: Run the Training Script

```bash
python disease_prediction_training.py
```

## What the Script Does

The script will:

1. **Load the real Kaggle dataset** (Training.csv + Testing.csv)
2. **Train 4 different models** with hyperparameter tuning:
   - RandomForest
   - DecisionTree
   - LogisticRegression
   - Support Vector Machine (SVM)
3. **Compare all models** using 5-fold cross-validation
4. **Select the best model** (highest accuracy)
5. **Generate comprehensive artifacts**:
   - `best_model.pkl` - Best trained model
   - `label_encoder.pkl` - Disease label encoder
   - `feature_names.json` - List of all symptoms
   - `disease_classes.json` - List of all diseases
   - `model_evaluation_metrics.json` - Detailed metrics for all models
   - `confusion_matrix_heatmap.png` - Visual confusion matrix
   - `feature_importance.csv` - Top symptoms ranked by importance
6. **Display results**:
   - Model comparison table
   - Top 10 most important symptoms
   - Sample predictions

## Expected Runtime

- **Small dataset (< 5000 samples):** 5-10 minutes
- **Medium dataset (5000-10000 samples):** 10-20 minutes
- **Large dataset (> 10000 samples):** 20-40 minutes

The script uses GridSearchCV with 5-fold cross-validation, which is computationally intensive but ensures optimal model performance.

## Output

### Console Output

The script prints detailed progress information:
- Dataset statistics
- Training progress for each model
- Model comparison table
- Best model selection
- Feature importance rankings
- Sample predictions

### Generated Files

After successful completion, you'll find these files in the same directory:

1. **best_model.pkl** - The trained model ready for deployment
2. **label_encoder.pkl** - Encoder for disease labels
3. **feature_names.json** - All symptom features
4. **disease_classes.json** - All disease classes
5. **model_evaluation_metrics.json** - Complete metrics and comparison
6. **confusion_matrix_heatmap.png** - Visual performance matrix
7. **feature_importance.csv** - Symptom importance rankings

## Troubleshooting

### Error: "Dataset files not found!"

**Solution:** Make sure `Training.csv` and `Testing.csv` are in the SAME directory as `disease_prediction_training.py`

The script will show you the exact path where it's looking for the files.

### Error: "ModuleNotFoundError: No module named 'pandas'"

**Solution:** Install the required packages:
```bash
pip install -r requirements.txt
```

### Error: "'prognosis' column not found"

**Solution:** Your CSV files might be corrupted or from a different dataset. Re-download from the Kaggle link above.

### Low Accuracy (< 90%)

If the best model achieves less than 90% accuracy, try:
1. Ensure you're using the complete dataset (both Training.csv and Testing.csv)
2. Check if the CSV files are complete and not truncated
3. Increase the hyperparameter search space in the script

## Integration with Supabase Edge Functions

After training completes successfully:

1. The generated artifacts can be converted to JSON format using `convert_model_to_json.py`
2. Upload the metrics to Supabase using `upload_metrics_to_supabase.js`
3. Deploy the model to Supabase Edge Functions for real-time predictions

## Questions or Issues?

If you encounter any issues:
1. Check that all files are in the correct location
2. Verify Python version: `python --version` (should be 3.7+)
3. Verify all packages are installed: `pip list | grep -E "pandas|numpy|scikit|matplotlib|seaborn"`
4. Check the error message carefully - it usually indicates the exact problem

## Dataset Information

- **Source:** Kaggle - Disease Prediction Using Machine Learning
- **URL:** https://www.kaggle.com/datasets/kaushil268/disease-prediction-using-machine-learning
- **Format:** CSV files with symptom columns and disease labels
- **Expected Columns:** 130+ symptom features + 1 'prognosis' column
- **Expected Diseases:** 40+ disease classes
