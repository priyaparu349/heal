import { createClient } from 'npm:@supabase/supabase-js@2';

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Methods': 'GET, POST, PUT, DELETE, OPTIONS',
  'Access-Control-Allow-Headers': 'Content-Type, Authorization, X-Client-Info, Apikey',
};

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

    const { action } = await req.json().catch(() => ({ action: 'train' }));

    if (action === 'load_pretrained') {
      console.log('Loading pre-trained model metrics from database...');

      const { data: activeModel, error: modelError } = await supabase
        .from('ml_models')
        .select('*')
        .eq('is_active', true)
        .order('created_at', { ascending: false })
        .limit(1)
        .single();

      if (modelError || !activeModel) {
        console.log('No pre-trained model found, using default metrics');
        return new Response(
          JSON.stringify({
            success: false,
            error: 'No pre-trained model found. Please train a model first.'
          }),
          {
            status: 404,
            headers: { ...corsHeaders, 'Content-Type': 'application/json' },
          }
        );
      }

      const { data: frequencyMappings } = await supabase
        .from('disease_frequency_mapping')
        .select('disease_name, frequency');

      const diseaseFrequencyMap = Object.fromEntries(
        (frequencyMappings || []).map(m => [m.disease_name, m.frequency])
      );

      const { data: confusionMatrixImage } = await supabase
        .from('model_artifacts')
        .select('file_url, file_data')
        .eq('model_id', activeModel.id)
        .eq('artifact_type', 'confusion_matrix')
        .single();

      return new Response(
        JSON.stringify({
          success: true,
          metrics: {
            accuracy: activeModel.accuracy,
            precision: activeModel.precision,
            recall: activeModel.recall,
            f1Score: activeModel.f1_score,
            confusionMatrix: activeModel.confusion_matrix,
            classificationReport: activeModel.classification_report,
          },
          confusion_matrix: activeModel.confusion_matrix,
          confusion_matrix_image: confusionMatrixImage?.file_url || confusionMatrixImage?.file_data,
          classification_report: activeModel.classification_report,
          disease_labels: Object.keys(diseaseFrequencyMap),
          disease_frequency_mapping: diseaseFrequencyMap,
          model_info: {
            id: activeModel.id,
            model_name: activeModel.model_name,
            version: activeModel.version,
            training_date: activeModel.training_date,
            dataset_info: activeModel.dataset_info,
          },
          sample_predictions: [],
          model_id: activeModel.id,
        }),
        {
          headers: { ...corsHeaders, 'Content-Type': 'application/json' },
        }
      );
    }

    console.log('Training enhanced decision tree model...');

    const trainingData = [
      { symptoms: ['headache', 'nausea', 'light_sensitivity', 'visual_disturbances'], disease: 'Migraine' },
      { symptoms: ['severe_headache', 'vomiting', 'light_sensitivity', 'throbbing_pain'], disease: 'Migraine' },
      { symptoms: ['headache', 'nausea', 'sound_sensitivity', 'aura'], disease: 'Migraine' },
      { symptoms: ['pulsating_headache', 'nausea', 'light_sensitivity', 'neck_pain'], disease: 'Migraine' },
      { symptoms: ['headache', 'vomiting', 'visual_disturbances', 'fatigue'], disease: 'Migraine' },
      { symptoms: ['worry', 'restlessness', 'rapid_heartbeat', 'sweating'], disease: 'Anxiety' },
      { symptoms: ['nervousness', 'tension', 'increased_heart_rate', 'trembling'], disease: 'Anxiety' },
      { symptoms: ['fear', 'panic', 'shortness_of_breath', 'dizziness'], disease: 'Anxiety' },
      { symptoms: ['worry', 'irritability', 'muscle_tension', 'sleep_problems'], disease: 'Anxiety' },
      { symptoms: ['nervousness', 'sweating', 'rapid_heartbeat', 'difficulty_concentrating'], disease: 'Anxiety' },
      { symptoms: ['difficulty_sleeping', 'fatigue', 'irritability', 'poor_concentration'], disease: 'Insomnia' },
      { symptoms: ['trouble_falling_asleep', 'waking_up_frequently', 'daytime_sleepiness', 'mood_changes'], disease: 'Insomnia' },
      { symptoms: ['sleep_disturbance', 'tiredness', 'difficulty_concentrating', 'anxiety'], disease: 'Insomnia' },
      { symptoms: ['cant_sleep', 'fatigue', 'irritability', 'headache'], disease: 'Insomnia' },
      { symptoms: ['waking_too_early', 'daytime_fatigue', 'mood_disturbances', 'poor_focus'], disease: 'Insomnia' },
      { symptoms: ['tension', 'headache', 'muscle_pain', 'fatigue'], disease: 'Stress' },
      { symptoms: ['overwhelmed', 'irritability', 'anxiety', 'sleep_problems'], disease: 'Stress' },
      { symptoms: ['pressure', 'worry', 'rapid_heartbeat', 'stomach_upset'], disease: 'Stress' },
      { symptoms: ['tension', 'difficulty_relaxing', 'headache', 'jaw_clenching'], disease: 'Stress' },
      { symptoms: ['overwhelmed', 'fatigue', 'concentration_problems', 'irritability'], disease: 'Stress' },
      { symptoms: ['tiredness', 'weakness', 'lack_of_energy', 'difficulty_concentrating'], disease: 'Fatigue' },
      { symptoms: ['exhaustion', 'muscle_weakness', 'sleepiness', 'reduced_motivation'], disease: 'Fatigue' },
      { symptoms: ['low_energy', 'tired', 'sluggishness', 'mental_fog'], disease: 'Fatigue' },
      { symptoms: ['weakness', 'fatigue', 'lack_of_stamina', 'drowsiness'], disease: 'Fatigue' },
      { symptoms: ['exhaustion', 'body_aches', 'difficulty_staying_awake', 'poor_concentration'], disease: 'Fatigue' },
      { symptoms: ['sadness', 'loss_of_interest', 'fatigue', 'sleep_changes'], disease: 'Depression' },
      { symptoms: ['hopelessness', 'lack_of_energy', 'appetite_changes', 'difficulty_concentrating'], disease: 'Depression' },
      { symptoms: ['low_mood', 'withdrawal', 'sleep_problems', 'worthlessness'], disease: 'Depression' },
      { symptoms: ['sadness', 'fatigue', 'loss_of_pleasure', 'suicidal_thoughts'], disease: 'Depression' },
      { symptoms: ['depression', 'irritability', 'sleep_disturbance', 'loss_of_appetite'], disease: 'Depression' },
      { symptoms: ['high_blood_pressure', 'headache', 'dizziness', 'chest_pain'], disease: 'Hypertension' },
      { symptoms: ['elevated_bp', 'shortness_of_breath', 'nosebleeds', 'vision_problems'], disease: 'Hypertension' },
      { symptoms: ['high_bp', 'headache', 'fatigue', 'irregular_heartbeat'], disease: 'Hypertension' },
      { symptoms: ['hypertension', 'dizziness', 'chest_discomfort', 'anxiety'], disease: 'Hypertension' },
      { symptoms: ['high_blood_pressure', 'pounding_in_chest', 'severe_headache', 'confusion'], disease: 'Hypertension' },
      { symptoms: ['joint_pain', 'stiffness', 'swelling', 'reduced_range_of_motion'], disease: 'Arthritis' },
      { symptoms: ['joint_inflammation', 'morning_stiffness', 'pain', 'warmth_in_joints'], disease: 'Arthritis' },
      { symptoms: ['joint_pain', 'swelling', 'difficulty_moving', 'tenderness'], disease: 'Arthritis' },
      { symptoms: ['stiff_joints', 'pain', 'decreased_flexibility', 'redness'], disease: 'Arthritis' },
      { symptoms: ['joint_ache', 'swelling', 'stiffness', 'fatigue'], disease: 'Arthritis' },
      { symptoms: ['wheezing', 'shortness_of_breath', 'coughing', 'chest_tightness'], disease: 'Asthma' },
      { symptoms: ['difficulty_breathing', 'wheezing', 'cough', 'rapid_breathing'], disease: 'Asthma' },
      { symptoms: ['breathlessness', 'chest_constriction', 'coughing', 'wheezing'], disease: 'Asthma' },
      { symptoms: ['asthma_attack', 'shortness_of_breath', 'wheezing', 'panic'], disease: 'Asthma' },
      { symptoms: ['breathing_difficulty', 'chest_tightness', 'cough', 'fatigue'], disease: 'Asthma' },
      { symptoms: ['excessive_thirst', 'frequent_urination', 'fatigue', 'blurred_vision'], disease: 'Diabetes' },
      { symptoms: ['increased_hunger', 'weight_loss', 'frequent_urination', 'slow_healing'], disease: 'Diabetes' },
      { symptoms: ['high_blood_sugar', 'thirst', 'frequent_urination', 'numbness'], disease: 'Diabetes' },
      { symptoms: ['excessive_thirst', 'fatigue', 'blurred_vision', 'infections'], disease: 'Diabetes' },
      { symptoms: ['polyuria', 'polydipsia', 'weight_loss', 'fatigue'], disease: 'Diabetes' },
    ];

    const uniqueSymptoms = Array.from(new Set(trainingData.flatMap(d => d.symptoms))).sort();
    const uniqueDiseases = Array.from(new Set(trainingData.map(d => d.disease))).sort();

    const encodedData = trainingData.map(sample => ({
      features: uniqueSymptoms.map(symptom => sample.symptoms.includes(symptom) ? 1 : 0),
      label: uniqueDiseases.indexOf(sample.disease)
    }));

    const shuffled = encodedData.sort(() => Math.random() - 0.5);
    const splitIndex = Math.floor(shuffled.length * 0.8);
    const trainData = shuffled.slice(0, splitIndex);
    const testData = shuffled.slice(splitIndex);

    class SimpleDecisionTree {
      private tree: Map<string, number> = new Map();

      train(data: Array<{features: number[], label: number}>) {
        data.forEach(sample => {
          this.tree.set(sample.features.join(','), sample.label);
        });
      }

      predict(features: number[]): number {
        const key = features.join(',');
        if (this.tree.has(key)) return this.tree.get(key)!;

        let minDistance = Infinity;
        let bestLabel = 0;
        for (const [patternKey, label] of this.tree.entries()) {
          const pattern = patternKey.split(',').map(Number);
          const distance = features.reduce((sum, val, idx) => sum + Math.abs(val - pattern[idx]), 0);
          if (distance < minDistance) {
            minDistance = distance;
            bestLabel = label;
          }
        }
        return bestLabel;
      }
    }

    const model = new SimpleDecisionTree();
    model.train(trainData);

    const predictions = testData.map(sample => model.predict(sample.features));
    const trueLabels = testData.map(sample => sample.label);

    const correct = predictions.filter((pred, idx) => pred === trueLabels[idx]).length;
    const accuracy = correct / predictions.length;

    const confusionMatrix = Array(uniqueDiseases.length).fill(0).map(() => Array(uniqueDiseases.length).fill(0));
    predictions.forEach((pred, idx) => {
      confusionMatrix[trueLabels[idx]][pred]++;
    });

    const classMetrics = uniqueDiseases.map((disease, classIdx) => {
      const tp = confusionMatrix[classIdx][classIdx];
      const fp = confusionMatrix.reduce((sum, row, i) => sum + (i !== classIdx ? row[classIdx] : 0), 0);
      const fn = confusionMatrix[classIdx].reduce((sum, val, i) => sum + (i !== classIdx ? val : 0), 0);

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

    await supabase.from('ml_models').update({ is_active: false }).eq('is_active', true);

    const { data: modelData } = await supabase
      .from('ml_models')
      .insert({
        model_name: 'SimpleDecisionTree',
        version: '2.0',
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
        model_params: { algorithm: 'DecisionTree', feature_encoding: 'binary', train_test_split: 0.8 },
        confusion_matrix: confusionMatrix,
        classification_report: metrics.classificationReport,
        is_active: true,
      })
      .select()
      .single();

    const { data: frequencyMappings } = await supabase
      .from('disease_frequency_mapping')
      .select('disease_name, frequency');

    const diseaseFrequencyMap = Object.fromEntries(
      (frequencyMappings || []).map(m => [m.disease_name, m.frequency])
    );

    return new Response(
      JSON.stringify({
        success: true,
        metrics,
        confusion_matrix: confusionMatrix,
        classification_report: metrics.classificationReport,
        disease_labels: uniqueDiseases,
        symptom_features: uniqueSymptoms,
        disease_frequency_mapping: diseaseFrequencyMap,
        sample_predictions: [],
        model_id: modelData?.id,
      }),
      {
        headers: { ...corsHeaders, 'Content-Type': 'application/json' },
      }
    );
  } catch (error) {
    console.error('Training error:', error);
    return new Response(
      JSON.stringify({ success: false, error: error instanceof Error ? error.message : 'Unknown error' }),
      {
        status: 500,
        headers: { ...corsHeaders, 'Content-Type': 'application/json' },
      }
    );
  }
});
