import { createClient } from 'npm:@supabase/supabase-js@2';

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Methods': 'GET, POST, PUT, DELETE, OPTIONS',
  'Access-Control-Allow-Headers': 'Content-Type, Authorization, X-Client-Info, Apikey',
};

interface TrainingData {
  symptoms: string[];
  disease: string;
}

interface ModelMetrics {
  accuracy: number;
  precision: number;
  recall: number;
  f1Score: number;
  confusionMatrix: number[][];
  classificationReport: Record<string, any>;
}

Deno.serve(async (req: Request) => {
  if (req.method === 'OPTIONS') {
    return new Response(null, {
      status: 200,
      headers: corsHeaders,
    });
  }

  try {
    const supabase = createClient(
      Deno.env.get('SUPABASE_URL') ?? '',
      Deno.env.get('SUPABASE_SERVICE_ROLE_KEY') ?? ''
    );

    // Simulated training dataset (in production, this would load from Kaggle CSV)
    const trainingData: TrainingData[] = [
      // Migraine patterns
      { symptoms: ['headache', 'nausea', 'light_sensitivity', 'visual_disturbances'], disease: 'Migraine' },
      { symptoms: ['severe_headache', 'vomiting', 'light_sensitivity', 'throbbing_pain'], disease: 'Migraine' },
      { symptoms: ['headache', 'nausea', 'sound_sensitivity', 'aura'], disease: 'Migraine' },
      { symptoms: ['pulsating_headache', 'nausea', 'light_sensitivity', 'neck_pain'], disease: 'Migraine' },
      { symptoms: ['headache', 'vomiting', 'visual_disturbances', 'fatigue'], disease: 'Migraine' },
      
      // Anxiety patterns
      { symptoms: ['worry', 'restlessness', 'rapid_heartbeat', 'sweating'], disease: 'Anxiety' },
      { symptoms: ['nervousness', 'tension', 'increased_heart_rate', 'trembling'], disease: 'Anxiety' },
      { symptoms: ['fear', 'panic', 'shortness_of_breath', 'dizziness'], disease: 'Anxiety' },
      { symptoms: ['worry', 'irritability', 'muscle_tension', 'sleep_problems'], disease: 'Anxiety' },
      { symptoms: ['nervousness', 'sweating', 'rapid_heartbeat', 'difficulty_concentrating'], disease: 'Anxiety' },
      
      // Insomnia patterns
      { symptoms: ['difficulty_sleeping', 'fatigue', 'irritability', 'poor_concentration'], disease: 'Insomnia' },
      { symptoms: ['trouble_falling_asleep', 'waking_up_frequently', 'daytime_sleepiness', 'mood_changes'], disease: 'Insomnia' },
      { symptoms: ['sleep_disturbance', 'tiredness', 'difficulty_concentrating', 'anxiety'], disease: 'Insomnia' },
      { symptoms: ['cant_sleep', 'fatigue', 'irritability', 'headache'], disease: 'Insomnia' },
      { symptoms: ['waking_too_early', 'daytime_fatigue', 'mood_disturbances', 'poor_focus'], disease: 'Insomnia' },
      
      // Stress patterns
      { symptoms: ['tension', 'headache', 'muscle_pain', 'fatigue'], disease: 'Stress' },
      { symptoms: ['overwhelmed', 'irritability', 'anxiety', 'sleep_problems'], disease: 'Stress' },
      { symptoms: ['pressure', 'worry', 'rapid_heartbeat', 'stomach_upset'], disease: 'Stress' },
      { symptoms: ['tension', 'difficulty_relaxing', 'headache', 'jaw_clenching'], disease: 'Stress' },
      { symptoms: ['overwhelmed', 'fatigue', 'concentration_problems', 'irritability'], disease: 'Stress' },
      
      // Fatigue patterns
      { symptoms: ['tiredness', 'weakness', 'lack_of_energy', 'difficulty_concentrating'], disease: 'Fatigue' },
      { symptoms: ['exhaustion', 'muscle_weakness', 'sleepiness', 'reduced_motivation'], disease: 'Fatigue' },
      { symptoms: ['low_energy', 'tired', 'sluggishness', 'mental_fog'], disease: 'Fatigue' },
      { symptoms: ['weakness', 'fatigue', 'lack_of_stamina', 'drowsiness'], disease: 'Fatigue' },
      { symptoms: ['exhaustion', 'body_aches', 'difficulty_staying_awake', 'poor_concentration'], disease: 'Fatigue' },
      
      // Depression patterns
      { symptoms: ['sadness', 'loss_of_interest', 'fatigue', 'sleep_changes'], disease: 'Depression' },
      { symptoms: ['hopelessness', 'lack_of_energy', 'appetite_changes', 'difficulty_concentrating'], disease: 'Depression' },
      { symptoms: ['low_mood', 'withdrawal', 'sleep_problems', 'worthlessness'], disease: 'Depression' },
      { symptoms: ['sadness', 'fatigue', 'loss_of_pleasure', 'suicidal_thoughts'], disease: 'Depression' },
      { symptoms: ['depression', 'irritability', 'sleep_disturbance', 'loss_of_appetite'], disease: 'Depression' },
      
      // Hypertension patterns
      { symptoms: ['high_blood_pressure', 'headache', 'dizziness', 'chest_pain'], disease: 'Hypertension' },
      { symptoms: ['elevated_bp', 'shortness_of_breath', 'nosebleeds', 'vision_problems'], disease: 'Hypertension' },
      { symptoms: ['high_bp', 'headache', 'fatigue', 'irregular_heartbeat'], disease: 'Hypertension' },
      { symptoms: ['hypertension', 'dizziness', 'chest_discomfort', 'anxiety'], disease: 'Hypertension' },
      { symptoms: ['high_blood_pressure', 'pounding_in_chest', 'severe_headache', 'confusion'], disease: 'Hypertension' },
      
      // Arthritis patterns
      { symptoms: ['joint_pain', 'stiffness', 'swelling', 'reduced_range_of_motion'], disease: 'Arthritis' },
      { symptoms: ['joint_inflammation', 'morning_stiffness', 'pain', 'warmth_in_joints'], disease: 'Arthritis' },
      { symptoms: ['joint_pain', 'swelling', 'difficulty_moving', 'tenderness'], disease: 'Arthritis' },
      { symptoms: ['stiff_joints', 'pain', 'decreased_flexibility', 'redness'], disease: 'Arthritis' },
      { symptoms: ['joint_ache', 'swelling', 'stiffness', 'fatigue'], disease: 'Arthritis' },
      
      // Asthma patterns
      { symptoms: ['wheezing', 'shortness_of_breath', 'coughing', 'chest_tightness'], disease: 'Asthma' },
      { symptoms: ['difficulty_breathing', 'wheezing', 'cough', 'rapid_breathing'], disease: 'Asthma' },
      { symptoms: ['breathlessness', 'chest_constriction', 'coughing', 'wheezing'], disease: 'Asthma' },
      { symptoms: ['asthma_attack', 'shortness_of_breath', 'wheezing', 'panic'], disease: 'Asthma' },
      { symptoms: ['breathing_difficulty', 'chest_tightness', 'cough', 'fatigue'], disease: 'Asthma' },
      
      // Diabetes patterns
      { symptoms: ['excessive_thirst', 'frequent_urination', 'fatigue', 'blurred_vision'], disease: 'Diabetes' },
      { symptoms: ['increased_hunger', 'weight_loss', 'frequent_urination', 'slow_healing'], disease: 'Diabetes' },
      { symptoms: ['high_blood_sugar', 'thirst', 'frequent_urination', 'numbness'], disease: 'Diabetes' },
      { symptoms: ['excessive_thirst', 'fatigue', 'blurred_vision', 'infections'], disease: 'Diabetes' },
      { symptoms: ['polyuria', 'polydipsia', 'weight_loss', 'fatigue'], disease: 'Diabetes' },
    ];

    // Extract unique symptoms and diseases
    const uniqueSymptoms = Array.from(new Set(trainingData.flatMap(d => d.symptoms))).sort();
    const uniqueDiseases = Array.from(new Set(trainingData.map(d => d.disease))).sort();

    console.log(`Training with ${trainingData.length} samples`);
    console.log(`Unique symptoms: ${uniqueSymptoms.length}`);
    console.log(`Unique diseases: ${uniqueDiseases.length}`);

    // Encode data (symptoms as binary vectors)
    const encodedData = trainingData.map(sample => {
      const vector = uniqueSymptoms.map(symptom => 
        sample.symptoms.includes(symptom) ? 1 : 0
      );
      return {
        features: vector,
        label: uniqueDiseases.indexOf(sample.disease)
      };
    });

    // Split data (80% train, 20% test)
    const shuffled = encodedData.sort(() => Math.random() - 0.5);
    const splitIndex = Math.floor(shuffled.length * 0.8);
    const trainData = shuffled.slice(0, splitIndex);
    const testData = shuffled.slice(splitIndex);

    console.log(`Training samples: ${trainData.length}`);
    console.log(`Testing samples: ${testData.length}`);

    // Simple Decision Tree implementation (for demonstration)
    class SimpleDecisionTree {
      private tree: Map<string, number> = new Map();

      train(data: Array<{features: number[], label: number}>) {
        // Store feature patterns mapped to most common label
        data.forEach(sample => {
          const key = sample.features.join(',');
          this.tree.set(key, sample.label);
        });
      }

      predict(features: number[]): number {
        const key = features.join(',');
        // Exact match
        if (this.tree.has(key)) {
          return this.tree.get(key)!;
        }
        // Find nearest match (Hamming distance)
        let minDistance = Infinity;
        let bestLabel = 0;
        for (const [patternKey, label] of this.tree.entries()) {
          const pattern = patternKey.split(',').map(Number);
          const distance = features.reduce((sum, val, idx) => 
            sum + Math.abs(val - pattern[idx]), 0
          );
          if (distance < minDistance) {
            minDistance = distance;
            bestLabel = label;
          }
        }
        return bestLabel;
      }
    }

    // Train model
    const model = new SimpleDecisionTree();
    model.train(trainData);

    // Evaluate model
    const predictions = testData.map(sample => model.predict(sample.features));
    const trueLabels = testData.map(sample => sample.label);

    // Calculate metrics
    const correct = predictions.filter((pred, idx) => pred === trueLabels[idx]).length;
    const accuracy = correct / predictions.length;

    // Confusion matrix
    const confusionMatrix = Array(uniqueDiseases.length).fill(0).map(() => 
      Array(uniqueDiseases.length).fill(0)
    );
    predictions.forEach((pred, idx) => {
      confusionMatrix[trueLabels[idx]][pred]++;
    });

    // Per-class metrics
    const classMetrics = uniqueDiseases.map((disease, classIdx) => {
      const tp = confusionMatrix[classIdx][classIdx];
      const fp = confusionMatrix.reduce((sum, row, i) => 
        sum + (i !== classIdx ? row[classIdx] : 0), 0
      );
      const fn = confusionMatrix[classIdx].reduce((sum, val, i) => 
        sum + (i !== classIdx ? val : 0), 0
      );
      const tn = confusionMatrix.reduce((sum, row, i) => 
        sum + row.reduce((s, val, j) => 
          s + (i !== classIdx && j !== classIdx ? val : 0), 0
        ), 0
      );

      const precision = tp + fp > 0 ? tp / (tp + fp) : 0;
      const recall = tp + fn > 0 ? tp / (tp + fn) : 0;
      const f1 = precision + recall > 0 ? 2 * (precision * recall) / (precision + recall) : 0;

      return { disease, precision, recall, f1, support: tp + fn };
    });

    const avgPrecision = classMetrics.reduce((sum, m) => sum + m.precision, 0) / classMetrics.length;
    const avgRecall = classMetrics.reduce((sum, m) => sum + m.recall, 0) / classMetrics.length;
    const avgF1 = classMetrics.reduce((sum, m) => sum + m.f1, 0) / classMetrics.length;

    const metrics: ModelMetrics = {
      accuracy,
      precision: avgPrecision,
      recall: avgRecall,
      f1Score: avgF1,
      confusionMatrix,
      classificationReport: {
        classes: classMetrics,
        macro_avg: { precision: avgPrecision, recall: avgRecall, f1: avgF1 },
      }
    };

    console.log('\n=== MODEL EVALUATION RESULTS ===');
    console.log(`Accuracy: ${(accuracy * 100).toFixed(2)}%`);
    console.log(`Precision: ${(avgPrecision * 100).toFixed(2)}%`);
    console.log(`Recall: ${(avgRecall * 100).toFixed(2)}%`);
    console.log(`F1-Score: ${(avgF1 * 100).toFixed(2)}%`);
    console.log('\n=== Classification Report ===');
    classMetrics.forEach(m => {
      console.log(`${m.disease}: Precision=${(m.precision * 100).toFixed(2)}%, Recall=${(m.recall * 100).toFixed(2)}%, F1=${(m.f1 * 100).toFixed(2)}%`);
    });

    // Save model metadata to database
    const { data: modelData, error: modelError } = await supabase
      .from('ml_models')
      .insert({
        model_name: 'SimpleDecisionTree',
        version: '1.0',
        accuracy,
        precision: avgPrecision,
        recall: avgRecall,
        f1_score: avgF1,
        dataset_info: {
          total_samples: trainingData.length,
          train_samples: trainData.length,
          test_samples: testData.length,
          unique_symptoms: uniqueSymptoms.length,
          unique_diseases: uniqueDiseases.length,
        },
        model_params: {
          algorithm: 'DecisionTree',
          feature_encoding: 'binary',
          train_test_split: 0.8,
        },
        confusion_matrix: confusionMatrix,
        classification_report: metrics.classificationReport,
        is_active: true,
      })
      .select()
      .single();

    if (modelError) {
      console.error('Error saving model:', modelError);
    } else {
      console.log('Model saved successfully:', modelData.id);
    }

    // Fetch disease-to-frequency mappings
    const { data: frequencyMappings } = await supabase
      .from('disease_frequency_mapping')
      .select('disease_name, frequency');

    const diseaseFrequencyMap = Object.fromEntries(
      (frequencyMappings || []).map(m => [m.disease_name, m.frequency])
    );

    console.log('\n=== Disease → Healing Frequency Mapping ===');
    console.log(JSON.stringify(diseaseFrequencyMap, null, 2));

    // Generate sample predictions
    const samplePredictions = [
      { symptoms: ['headache', 'nausea', 'light_sensitivity'], expected: 'Migraine' },
      { symptoms: ['worry', 'restlessness', 'rapid_heartbeat'], expected: 'Anxiety' },
      { symptoms: ['difficulty_sleeping', 'fatigue', 'irritability'], expected: 'Insomnia' },
    ].map(sample => {
      const features = uniqueSymptoms.map(s => sample.symptoms.includes(s) ? 1 : 0);
      const prediction = model.predict(features);
      const predictedDisease = uniqueDiseases[prediction];
      return {
        symptoms: sample.symptoms,
        predicted: predictedDisease,
        frequency: diseaseFrequencyMap[predictedDisease],
        expected: sample.expected,
      };
    });

    console.log('\n=== Sample Predictions ===');
    samplePredictions.forEach(p => {
      console.log(`Symptoms: ${p.symptoms.join(', ')}`);
      console.log(`Predicted: ${p.predicted} → ${p.frequency}`);
      console.log(`Expected: ${p.expected}`);
      console.log('---');
    });

    return new Response(
      JSON.stringify({
        success: true,
        metrics,
        confusion_matrix: confusionMatrix,
        classification_report: metrics.classificationReport,
        disease_labels: uniqueDiseases,
        symptom_features: uniqueSymptoms,
        disease_frequency_mapping: diseaseFrequencyMap,
        sample_predictions: samplePredictions,
        model_id: modelData?.id,
      }),
      {
        headers: {
          ...corsHeaders,
          'Content-Type': 'application/json',
        },
      }
    );
  } catch (error) {
    console.error('Training error:', error);
    return new Response(
      JSON.stringify({ 
        success: false, 
        error: error instanceof Error ? error.message : 'Unknown error' 
      }),
      {
        status: 500,
        headers: {
          ...corsHeaders,
          'Content-Type': 'application/json',
        },
      }
    );
  }
});