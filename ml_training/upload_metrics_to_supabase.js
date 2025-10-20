#!/usr/bin/env node

/**
 * Upload Trained Model Metrics to Supabase
 * =========================================
 * This script uploads the trained ML model's metrics and confusion matrix
 * to your Supabase database for display in the app.
 *
 * Usage:
 *   node upload_metrics_to_supabase.js
 *
 * Prerequisites:
 *   - npm install @supabase/supabase-js
 *   - Set environment variables: VITE_SUPABASE_URL, VITE_SUPABASE_SERVICE_ROLE_KEY
 *   - Run disease_prediction_training.py first
 */

const fs = require('fs');
const path = require('path');

console.log('=' .repeat(70));
console.log('UPLOADING ML METRICS TO SUPABASE');
console.log('=' .repeat(70));

// Check for required files
const requiredFiles = [
  'model_evaluation_metrics.json',
  'confusion_matrix_heatmap.png',
  'disease_classes.json',
  'feature_importance.csv'
];

console.log('\nStep 1: Checking for required files...');
for (const file of requiredFiles) {
  if (fs.existsSync(file)) {
    console.log(`  ✓ ${file}`);
  } else {
    console.error(`  ✗ ${file} - NOT FOUND!`);
    console.error('\nPlease run disease_prediction_training.py first.');
    process.exit(1);
  }
}

// Load Supabase client
console.log('\nStep 2: Loading Supabase client...');

let supabase;
try {
  const { createClient } = require('@supabase/supabase-js');

  const supabaseUrl = process.env.VITE_SUPABASE_URL;
  const supabaseKey = process.env.VITE_SUPABASE_SERVICE_ROLE_KEY;

  if (!supabaseUrl || !supabaseKey) {
    throw new Error('Missing environment variables. Set VITE_SUPABASE_URL and VITE_SUPABASE_SERVICE_ROLE_KEY');
  }

  supabase = createClient(supabaseUrl, supabaseKey);
  console.log('  ✓ Supabase client initialized');
} catch (error) {
  console.error('  ✗ Failed to load Supabase client');
  console.error('  Error:', error.message);
  console.error('\nInstall @supabase/supabase-js:');
  console.error('  npm install @supabase/supabase-js');
  process.exit(1);
}

// Load metrics
console.log('\nStep 3: Loading metrics...');
const metrics = JSON.parse(fs.readFileSync('model_evaluation_metrics.json', 'utf8'));
console.log('  ✓ Loaded evaluation metrics');
console.log(`    - Accuracy: ${(metrics.metrics.accuracy * 100).toFixed(2)}%`);
console.log(`    - Precision: ${(metrics.metrics.precision * 100).toFixed(2)}%`);
console.log(`    - Recall: ${(metrics.metrics.recall * 100).toFixed(2)}%`);
console.log(`    - F1-Score: ${(metrics.metrics.f1_score * 100).toFixed(2)}%`);

// Convert confusion matrix image to base64
console.log('\nStep 4: Converting confusion matrix to base64...');
const imageBuffer = fs.readFileSync('confusion_matrix_heatmap.png');
const imageBase64 = `data:image/png;base64,${imageBuffer.toString('base64')}`;
console.log(`  ✓ Image converted (${(imageBase64.length / 1024).toFixed(2)} KB)`);

// Upload to Supabase
console.log('\nStep 5: Uploading to Supabase...');

async function uploadMetrics() {
  try {
    // Deactivate old models
    console.log('  - Deactivating old models...');
    await supabase
      .from('ml_models')
      .update({ is_active: false })
      .eq('is_active', true);

    // Insert new model
    console.log('  - Inserting new model...');
    const { data: modelData, error: modelError } = await supabase
      .from('ml_models')
      .insert({
        model_name: metrics.model_name,
        version: metrics.version,
        accuracy: metrics.metrics.accuracy,
        precision: metrics.metrics.precision,
        recall: metrics.metrics.recall,
        f1_score: metrics.metrics.f1_score,
        training_date: metrics.training_date,
        dataset_info: metrics.dataset_info,
        model_params: metrics.model_params,
        confusion_matrix: metrics.confusion_matrix,
        classification_report: metrics.classification_report,
        confusion_matrix_image: imageBase64,
        is_active: true,
      })
      .select()
      .single();

    if (modelError) {
      throw modelError;
    }

    console.log('  ✓ Model inserted successfully');
    console.log(`    - Model ID: ${modelData.id}`);

    // Upload confusion matrix as artifact
    console.log('  - Uploading confusion matrix artifact...');
    const { error: artifactError } = await supabase
      .from('model_artifacts')
      .insert({
        model_id: modelData.id,
        artifact_type: 'confusion_matrix',
        file_name: 'confusion_matrix_heatmap.png',
        file_data: imageBase64,
        mime_type: 'image/png',
      });

    if (artifactError) {
      console.warn('  ⚠ Warning: Could not upload artifact:', artifactError.message);
    } else {
      console.log('  ✓ Confusion matrix artifact uploaded');
    }

    // Load disease classes and feature importance
    const diseaseClasses = JSON.parse(fs.readFileSync('disease_classes.json', 'utf8'));
    const featureImportanceRaw = fs.readFileSync('feature_importance.csv', 'utf8');

    // Parse feature importance CSV
    const featureImportanceLines = featureImportanceRaw.split('\n').slice(1); // Skip header
    const featureImportanceMap = {};

    for (const line of featureImportanceLines) {
      if (!line.trim()) continue;
      const [feature, importance] = line.split(',');
      if (feature && importance) {
        featureImportanceMap[feature.trim()] = parseFloat(importance.trim());
      }
    }

    console.log(`  - Loaded ${diseaseClasses.length} diseases and ${Object.keys(featureImportanceMap).length} features`);

    // Upload disease-symptom patterns (if we have training data)
    if (fs.existsSync('../ml_training/Training.csv')) {
      console.log('  - Analyzing disease-symptom patterns...');

      // Read CSV (simplified - in production use proper CSV parser)
      const csv = fs.readFileSync('../ml_training/Training.csv', 'utf8');
      const lines = csv.split('\n');
      const headers = lines[0].split(',').map(h => h.trim());
      const prognosisIndex = headers.indexOf('prognosis');

      const diseaseSymptomFreq = {};

      for (let i = 1; i < lines.length; i++) {
        if (!lines[i].trim()) continue;

        const values = lines[i].split(',');
        const disease = values[prognosisIndex]?.trim();

        if (!disease || !diseaseClasses.includes(disease)) continue;

        if (!diseaseSymptomFreq[disease]) {
          diseaseSymptomFreq[disease] = {};
        }

        for (let j = 0; j < headers.length; j++) {
          if (j === prognosisIndex) continue;

          const symptom = headers[j];
          const value = parseInt(values[j]);

          if (value === 1) {
            diseaseSymptomFreq[disease][symptom] = (diseaseSymptomFreq[disease][symptom] || 0) + 1;
          }
        }
      }

      // Calculate frequencies and insert into database
      console.log('  - Uploading disease-symptom patterns...');
      const patterns = [];

      for (const [disease, symptoms] of Object.entries(diseaseSymptomFreq)) {
        const totalCases = Object.values(symptoms).reduce((sum, count) => Math.max(sum, count), 1);

        for (const [symptom, count] of Object.entries(symptoms)) {
          const frequency = count / totalCases;
          const importance = featureImportanceMap[symptom] || 0.01;

          patterns.push({
            disease_name: disease,
            symptom_name: symptom,
            frequency,
            importance,
          });
        }
      }

      if (patterns.length > 0) {
        const { error: patternsError } = await supabase
          .from('disease_symptom_patterns')
          .upsert(patterns, { onConflict: 'disease_name,symptom_name' });

        if (patternsError) {
          console.warn('  ⚠ Warning: Could not upload patterns:', patternsError.message);
        } else {
          console.log(`  ✓ Uploaded ${patterns.length} disease-symptom patterns`);
        }
      }
    }

    console.log('\n' + '='.repeat(70));
    console.log('UPLOAD COMPLETE');
    console.log('='.repeat(70));
    console.log('\n✓ Model metrics successfully uploaded to Supabase');
    console.log('✓ Confusion matrix stored as base64');
    console.log('✓ Ready to view in ML Evaluation page');

    console.log('\nNext steps:');
    console.log('1. Navigate to /ml-evaluation in your app');
    console.log('2. Click "Load Pre-trained Model"');
    console.log('3. View all metrics and confusion matrix');
    console.log('4. Test predictions on /symptom-checker page');

    console.log('\n' + '='.repeat(70));

  } catch (error) {
    console.error('\n' + '!'.repeat(70));
    console.error('UPLOAD FAILED');
    console.error('!'.repeat(70));
    console.error('\nError:', error.message);
    console.error('\nDetails:', error);
    process.exit(1);
  }
}

// Run upload
uploadMetrics().then(() => {
  console.log('\nDone!');
  process.exit(0);
}).catch((error) => {
  console.error('\nFatal error:', error);
  process.exit(1);
});
