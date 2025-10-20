import { createClient } from 'npm:@supabase/supabase-js@2';

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Methods': 'GET, POST, PUT, DELETE, OPTIONS',
  'Access-Control-Allow-Headers': 'Content-Type, Authorization, X-Client-Info, Apikey',
};

interface PredictionRequest {
  symptoms: string[];
  user_id?: string;
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
      Deno.env.get('SUPABASE_ANON_KEY') ?? ''
    );

    const { symptoms, user_id }: PredictionRequest = await req.json();

    if (!symptoms || !Array.isArray(symptoms) || symptoms.length === 0) {
      return new Response(
        JSON.stringify({ error: 'Symptoms array is required' }),
        {
          status: 400,
          headers: { ...corsHeaders, 'Content-Type': 'application/json' },
        }
      );
    }

    // Normalize symptoms to lowercase and replace spaces with underscores
    const normalizedSymptoms = symptoms.map(s => 
      s.toLowerCase().trim().replace(/\s+/g, '_')
    );

    console.log('Received symptoms:', normalizedSymptoms);

    // Define symptom-to-disease mapping (simplified rule-based system)
    const diseasePatterns = [
      {
        disease: 'Migraine',
        keywords: ['headache', 'severe_headache', 'pulsating_headache', 'nausea', 'vomiting', 'light_sensitivity', 'visual_disturbances', 'aura', 'throbbing_pain'],
        threshold: 2,
      },
      {
        disease: 'Anxiety',
        keywords: ['worry', 'nervousness', 'fear', 'panic', 'restlessness', 'rapid_heartbeat', 'increased_heart_rate', 'sweating', 'trembling', 'shortness_of_breath', 'dizziness', 'tension'],
        threshold: 2,
      },
      {
        disease: 'Insomnia',
        keywords: ['difficulty_sleeping', 'trouble_falling_asleep', 'cant_sleep', 'sleep_disturbance', 'waking_up_frequently', 'waking_too_early', 'fatigue', 'tiredness', 'daytime_sleepiness', 'poor_concentration'],
        threshold: 2,
      },
      {
        disease: 'Stress',
        keywords: ['tension', 'overwhelmed', 'pressure', 'headache', 'muscle_pain', 'irritability', 'anxiety', 'worry', 'difficulty_relaxing', 'jaw_clenching', 'stomach_upset'],
        threshold: 2,
      },
      {
        disease: 'Fatigue',
        keywords: ['tiredness', 'exhaustion', 'weakness', 'lack_of_energy', 'low_energy', 'tired', 'sleepiness', 'sluggishness', 'drowsiness', 'body_aches', 'mental_fog', 'lack_of_stamina'],
        threshold: 2,
      },
      {
        disease: 'Depression',
        keywords: ['sadness', 'hopelessness', 'low_mood', 'depression', 'loss_of_interest', 'loss_of_pleasure', 'worthlessness', 'withdrawal', 'suicidal_thoughts', 'appetite_changes', 'sleep_changes'],
        threshold: 2,
      },
      {
        disease: 'Hypertension',
        keywords: ['high_blood_pressure', 'elevated_bp', 'high_bp', 'hypertension', 'chest_pain', 'chest_discomfort', 'pounding_in_chest', 'dizziness', 'shortness_of_breath', 'nosebleeds', 'severe_headache', 'confusion', 'vision_problems'],
        threshold: 2,
      },
      {
        disease: 'Arthritis',
        keywords: ['joint_pain', 'joint_inflammation', 'joint_ache', 'stiffness', 'stiff_joints', 'morning_stiffness', 'swelling', 'reduced_range_of_motion', 'difficulty_moving', 'tenderness', 'warmth_in_joints', 'redness'],
        threshold: 2,
      },
      {
        disease: 'Asthma',
        keywords: ['wheezing', 'shortness_of_breath', 'breathlessness', 'difficulty_breathing', 'breathing_difficulty', 'coughing', 'cough', 'chest_tightness', 'chest_constriction', 'rapid_breathing', 'asthma_attack', 'panic'],
        threshold: 2,
      },
      {
        disease: 'Diabetes',
        keywords: ['excessive_thirst', 'polyuria', 'polydipsia', 'frequent_urination', 'increased_hunger', 'weight_loss', 'fatigue', 'blurred_vision', 'high_blood_sugar', 'slow_healing', 'numbness', 'infections'],
        threshold: 2,
      },
    ];

    // Calculate match scores for each disease
    const diseaseScores = diseasePatterns.map(pattern => {
      const matches = normalizedSymptoms.filter(symptom => 
        pattern.keywords.some(keyword => 
          symptom.includes(keyword) || keyword.includes(symptom)
        )
      ).length;
      const score = matches / pattern.keywords.length;
      return { disease: pattern.disease, score, matches, meetsThreshold: matches >= pattern.threshold };
    });

    // Find best match
    const validPredictions = diseaseScores.filter(d => d.meetsThreshold);
    const bestMatch = validPredictions.length > 0 
      ? validPredictions.reduce((best, current) => 
          current.score > best.score ? current : best
        )
      : diseaseScores.reduce((best, current) => 
          current.matches > best.matches ? current : best
        );

    const predictedDisease = bestMatch.disease;
    const confidenceScore = Math.min(bestMatch.score * 1.5, 1); // Scale confidence

    console.log('Disease scores:', diseaseScores);
    console.log('Predicted:', predictedDisease, 'Confidence:', confidenceScore);

    // Fetch frequency mapping for predicted disease
    const { data: frequencyData } = await supabase
      .from('disease_frequency_mapping')
      .select('*')
      .eq('disease_name', predictedDisease)
      .single();

    const suggestedFrequency = frequencyData?.frequency || '432 Hz';
    const description = frequencyData?.description || 'Natural healing frequency';
    const benefits = frequencyData?.benefits || [];

    // Get active model info
    const { data: activeModel } = await supabase
      .from('ml_models')
      .select('id')
      .eq('is_active', true)
      .order('created_at', { ascending: false })
      .limit(1)
      .single();

    // Save prediction to database
    const { data: predictionData, error: predictionError } = await supabase
      .from('symptom_predictions')
      .insert({
        user_id: user_id || null,
        symptoms: normalizedSymptoms,
        predicted_disease: predictedDisease,
        confidence_score: confidenceScore,
        suggested_frequency: suggestedFrequency,
        model_id: activeModel?.id,
      })
      .select()
      .single();

    if (predictionError) {
      console.error('Error saving prediction:', predictionError);
    }

    return new Response(
      JSON.stringify({
        success: true,
        prediction: {
          disease: predictedDisease,
          confidence: confidenceScore,
          healing_frequency: suggestedFrequency,
          description,
          benefits,
          prediction_id: predictionData?.id,
        },
        alternative_matches: diseaseScores
          .filter(d => d.disease !== predictedDisease && d.matches > 0)
          .sort((a, b) => b.score - a.score)
          .slice(0, 3)
          .map(d => ({ disease: d.disease, confidence: Math.min(d.score * 1.5, 1) })),
      }),
      {
        headers: {
          ...corsHeaders,
          'Content-Type': 'application/json',
        },
      }
    );
  } catch (error) {
    console.error('Prediction error:', error);
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