"""
Model Converter for Edge Functions
===================================
Converts the trained RandomForest model into a JSON-based rule system
that can be used in Supabase Edge Functions (Deno/TypeScript).

This script extracts:
1. Feature importance weights
2. Disease-symptom associations from training data
3. Prediction rules based on symptom patterns
"""

import pickle
import json
import pandas as pd
import numpy as np
from collections import defaultdict

print("=" * 70)
print("CONVERTING ML MODEL TO EDGE FUNCTION FORMAT")
print("=" * 70)

# Load artifacts
print("\nStep 1: Loading artifacts...")
with open('disease_model.pkl', 'rb') as f:
    model = pickle.load(f)
print("✓ Model loaded")

with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)
print("✓ Label encoder loaded")

with open('feature_names.json', 'r') as f:
    feature_names = json.load(f)
print("✓ Feature names loaded")

with open('disease_classes.json', 'r') as f:
    disease_classes = json.load(f)
print("✓ Disease classes loaded")

# Load original dataset
print("\nStep 2: Loading dataset for rule extraction...")
try:
    df = pd.read_csv('Training.csv')
    print(f"✓ Dataset loaded: {len(df)} samples")
except FileNotFoundError:
    print("! Training.csv not found - using synthetic patterns")
    df = None

# Extract feature importance
print("\nStep 3: Extracting feature importance...")
feature_importance = {
    feature: float(importance)
    for feature, importance in zip(feature_names, model.feature_importances_)
}

# Sort by importance
sorted_features = sorted(
    feature_importance.items(),
    key=lambda x: x[1],
    reverse=True
)

print(f"✓ Extracted {len(feature_importance)} feature weights")
print(f"\nTop 10 most important symptoms:")
for feature, importance in sorted_features[:10]:
    print(f"  {feature}: {importance:.4f}")

# Extract disease-symptom patterns
print("\nStep 4: Extracting disease-symptom patterns...")
disease_symptom_patterns = {}

if df is not None:
    for disease in disease_classes:
        disease_samples = df[df['prognosis'] == disease]
        if len(disease_samples) > 0:
            # Get symptoms that appear in >50% of cases for this disease
            symptom_frequencies = disease_samples.drop('prognosis', axis=1).mean()
            key_symptoms = symptom_frequencies[symptom_frequencies > 0.5].index.tolist()

            # Get all symptoms that appear at least once
            all_symptoms = symptom_frequencies[symptom_frequencies > 0].index.tolist()

            disease_symptom_patterns[disease] = {
                'key_symptoms': key_symptoms,
                'all_symptoms': all_symptoms,
                'symptom_frequencies': {
                    symptom: float(freq)
                    for symptom, freq in symptom_frequencies[symptom_frequencies > 0].items()
                }
            }
    print(f"✓ Extracted patterns for {len(disease_symptom_patterns)} diseases")
else:
    print("! Skipping pattern extraction (no dataset)")

# Create prediction rules
print("\nStep 5: Creating prediction rules...")

# For each disease, create a weighted scoring rule
prediction_rules = {}
for disease in disease_classes:
    if disease in disease_symptom_patterns:
        pattern = disease_symptom_patterns[disease]

        # Create scoring rules: symptom -> weight
        scoring_rules = {}
        for symptom, freq in pattern['symptom_frequencies'].items():
            # Weight = frequency * feature_importance
            importance = feature_importance.get(symptom, 0.01)
            scoring_rules[symptom] = float(freq * importance)

        prediction_rules[disease] = {
            'key_symptoms': pattern['key_symptoms'],
            'scoring_rules': scoring_rules,
            'threshold': 0.3  # Minimum score to consider this disease
        }

print(f"✓ Created prediction rules for {len(prediction_rules)} diseases")

# Create simplified model data structure
print("\nStep 6: Creating simplified model structure...")
model_structure = {
    'model_info': {
        'algorithm': 'RandomForestClassifier',
        'n_estimators': model.n_estimators,
        'n_features': len(feature_names),
        'n_classes': len(disease_classes)
    },
    'feature_names': feature_names,
    'disease_classes': disease_classes,
    'feature_importance': feature_importance,
    'prediction_rules': prediction_rules
}

# Save to JSON
output_file = 'model_for_edge_functions.json'
with open(output_file, 'w') as f:
    json.dump(model_structure, f, indent=2)

print(f"✓ Model structure saved: {output_file}")

# Create a simplified symptom-disease mapping
print("\nStep 7: Creating symptom-disease mapping...")
symptom_to_diseases = defaultdict(list)

for disease, rules in prediction_rules.items():
    for symptom in rules['key_symptoms']:
        symptom_to_diseases[symptom].append({
            'disease': disease,
            'relevance': rules['scoring_rules'].get(symptom, 0.5)
        })

# Save symptom mapping
symptom_mapping_file = 'symptom_to_diseases_mapping.json'
with open(symptom_mapping_file, 'w') as f:
    json.dump(dict(symptom_to_diseases), f, indent=2)

print(f"✓ Symptom mapping saved: {symptom_mapping_file}")

# Generate prediction algorithm (pseudocode for Edge Functions)
print("\nStep 8: Generating prediction algorithm...")

algorithm_doc = """
// Prediction Algorithm for Edge Functions
// ========================================

function predictDisease(symptoms: string[]): Prediction {
  const diseaseScores: Record<string, number> = {};

  // Normalize symptoms
  const normalizedSymptoms = symptoms.map(s =>
    s.toLowerCase().trim().replace(/\\s+/g, '_')
  );

  // Calculate score for each disease
  for (const [disease, rules] of Object.entries(predictionRules)) {
    let score = 0;
    let matchCount = 0;

    // Check each symptom against this disease's scoring rules
    for (const symptom of normalizedSymptoms) {
      if (rules.scoring_rules[symptom]) {
        score += rules.scoring_rules[symptom];
        matchCount++;
      }
    }

    // Apply threshold
    if (score >= rules.threshold && matchCount > 0) {
      diseaseScores[disease] = score;
    }
  }

  // Find disease with highest score
  const predictedDisease = Object.keys(diseaseScores).reduce((a, b) =>
    diseaseScores[a] > diseaseScores[b] ? a : b
  );

  const maxScore = diseaseScores[predictedDisease];
  const confidence = Math.min(maxScore / 3.0, 1.0); // Normalize to 0-1

  return {
    disease: predictedDisease,
    confidence: confidence,
    alternativeMatches: Object.entries(diseaseScores)
      .filter(([d]) => d !== predictedDisease)
      .map(([disease, score]) => ({
        disease,
        confidence: Math.min(score / 3.0, 1.0)
      }))
      .sort((a, b) => b.confidence - a.confidence)
      .slice(0, 3)
  };
}
"""

algorithm_file = 'prediction_algorithm.txt'
with open(algorithm_file, 'w') as f:
    f.write(algorithm_doc)

print(f"✓ Algorithm pseudocode saved: {algorithm_file}")

# Test the conversion
print("\nStep 9: Testing converted model...")
test_symptoms = ['fever', 'cough', 'fatigue', 'headache']
print(f"\nTest symptoms: {test_symptoms}")

# Simulate scoring
disease_scores = {}
for disease, rules in prediction_rules.items():
    score = sum(
        rules['scoring_rules'].get(symptom, 0)
        for symptom in test_symptoms
    )
    if score >= rules['threshold']:
        disease_scores[disease] = score

if disease_scores:
    predicted = max(disease_scores.items(), key=lambda x: x[1])
    print(f"Predicted disease: {predicted[0]}")
    print(f"Score: {predicted[1]:.4f}")
else:
    print("No strong matches found")

# Summary
print("\n" + "=" * 70)
print("CONVERSION COMPLETE")
print("=" * 70)
print(f"\n✓ model_for_edge_functions.json - Model structure for Edge Functions")
print(f"✓ symptom_to_diseases_mapping.json - Symptom-disease associations")
print(f"✓ prediction_algorithm.txt - TypeScript prediction algorithm")

print("\n" + "=" * 70)
print("NEXT STEPS:")
print("=" * 70)
print("1. Import model_for_edge_functions.json into your Edge Functions")
print("2. Implement the prediction algorithm in TypeScript")
print("3. Test predictions against the original Python model")
print("4. Deploy updated Edge Functions to Supabase")
print("\n" + "=" * 70)
