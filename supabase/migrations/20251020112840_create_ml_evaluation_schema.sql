/*
  # ML Evaluation Schema for Quantum Pulse

  1. New Tables
    - `ml_models`
      - `id` (uuid, primary key) - Unique model identifier
      - `model_name` (text) - Name of the ML model (e.g., "RandomForest", "DecisionTree")
      - `version` (text) - Model version
      - `accuracy` (numeric) - Model accuracy score
      - `precision` (numeric) - Precision score
      - `recall` (numeric) - Recall score
      - `f1_score` (numeric) - F1 score
      - `training_date` (timestamptz) - When model was trained
      - `dataset_info` (jsonb) - Information about training dataset
      - `model_params` (jsonb) - Model hyperparameters
      - `confusion_matrix` (jsonb) - Confusion matrix data
      - `classification_report` (jsonb) - Full classification report
      - `is_active` (boolean) - Whether this is the currently active model
      - `created_at` (timestamptz) - Record creation timestamp
      - `updated_at` (timestamptz) - Record update timestamp

    - `symptom_predictions`
      - `id` (uuid, primary key) - Unique prediction identifier
      - `user_id` (uuid, nullable) - User who requested prediction (if authenticated)
      - `symptoms` (jsonb) - Array of symptoms provided
      - `predicted_disease` (text) - Predicted disease name
      - `confidence_score` (numeric) - Prediction confidence (0-1)
      - `suggested_frequency` (text) - Healing frequency (e.g., "528 Hz")
      - `model_id` (uuid) - Reference to ml_models table
      - `created_at` (timestamptz) - Prediction timestamp

    - `disease_frequency_mapping`
      - `id` (uuid, primary key) - Unique mapping identifier
      - `disease_name` (text, unique) - Disease name
      - `frequency` (text) - Healing frequency in Hz
      - `description` (text) - Description of healing properties
      - `benefits` (jsonb) - Array of benefits
      - `created_at` (timestamptz) - Record creation timestamp

  2. Security
    - Enable RLS on all tables
    - Public read access for disease_frequency_mapping
    - Authenticated users can create predictions
    - Only service role can manage ml_models
*/

-- Create ml_models table
CREATE TABLE IF NOT EXISTS ml_models (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  model_name text NOT NULL,
  version text NOT NULL DEFAULT '1.0',
  accuracy numeric CHECK (accuracy >= 0 AND accuracy <= 1),
  precision numeric CHECK (precision >= 0 AND precision <= 1),
  recall numeric CHECK (recall >= 0 AND recall <= 1),
  f1_score numeric CHECK (f1_score >= 0 AND f1_score <= 1),
  training_date timestamptz DEFAULT now(),
  dataset_info jsonb DEFAULT '{}',
  model_params jsonb DEFAULT '{}',
  confusion_matrix jsonb DEFAULT '{}',
  classification_report jsonb DEFAULT '{}',
  is_active boolean DEFAULT false,
  created_at timestamptz DEFAULT now(),
  updated_at timestamptz DEFAULT now()
);

-- Create symptom_predictions table
CREATE TABLE IF NOT EXISTS symptom_predictions (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id uuid,
  symptoms jsonb NOT NULL DEFAULT '[]',
  predicted_disease text NOT NULL,
  confidence_score numeric CHECK (confidence_score >= 0 AND confidence_score <= 1),
  suggested_frequency text,
  model_id uuid REFERENCES ml_models(id),
  created_at timestamptz DEFAULT now()
);

-- Create disease_frequency_mapping table
CREATE TABLE IF NOT EXISTS disease_frequency_mapping (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  disease_name text UNIQUE NOT NULL,
  frequency text NOT NULL,
  description text DEFAULT '',
  benefits jsonb DEFAULT '[]',
  created_at timestamptz DEFAULT now()
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_ml_models_active ON ml_models(is_active) WHERE is_active = true;
CREATE INDEX IF NOT EXISTS idx_symptom_predictions_user ON symptom_predictions(user_id);
CREATE INDEX IF NOT EXISTS idx_symptom_predictions_created ON symptom_predictions(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_disease_frequency_mapping_name ON disease_frequency_mapping(disease_name);

-- Enable Row Level Security
ALTER TABLE ml_models ENABLE ROW LEVEL SECURITY;
ALTER TABLE symptom_predictions ENABLE ROW LEVEL SECURITY;
ALTER TABLE disease_frequency_mapping ENABLE ROW LEVEL SECURITY;

-- RLS Policies for ml_models (admin/service role only for writes)
CREATE POLICY "Anyone can view ML models"
  ON ml_models FOR SELECT
  TO public
  USING (true);

CREATE POLICY "Service role can insert ML models"
  ON ml_models FOR INSERT
  TO service_role
  WITH CHECK (true);

CREATE POLICY "Service role can update ML models"
  ON ml_models FOR UPDATE
  TO service_role
  USING (true)
  WITH CHECK (true);

-- RLS Policies for symptom_predictions
CREATE POLICY "Users can view their own predictions"
  ON symptom_predictions FOR SELECT
  TO authenticated
  USING (auth.uid() = user_id OR user_id IS NULL);

CREATE POLICY "Anyone can create predictions"
  ON symptom_predictions FOR INSERT
  TO public
  WITH CHECK (true);

-- RLS Policies for disease_frequency_mapping
CREATE POLICY "Anyone can view disease frequency mappings"
  ON disease_frequency_mapping FOR SELECT
  TO public
  USING (true);

CREATE POLICY "Service role can manage disease frequency mappings"
  ON disease_frequency_mapping FOR ALL
  TO service_role
  USING (true)
  WITH CHECK (true);

-- Insert default disease-to-frequency mappings
INSERT INTO disease_frequency_mapping (disease_name, frequency, description, benefits) VALUES
  ('Migraine', '528 Hz', 'DNA repair and transformation frequency for headache relief', '["Pain reduction", "Stress relief", "Mental clarity", "Cellular healing"]'),
  ('Anxiety', '396 Hz', 'Liberation from fear and guilt, root chakra healing', '["Reduces anxiety", "Emotional balance", "Fear release", "Grounding"]'),
  ('Insomnia', '639 Hz', 'Heart chakra frequency for sleep and relaxation', '["Better sleep", "Emotional healing", "Relationship harmony", "Deep rest"]'),
  ('Stress', '741 Hz', 'Awakening intuition and reducing stress', '["Stress reduction", "Mental clarity", "Detoxification", "Problem solving"]'),
  ('Fatigue', '852 Hz', 'Spiritual awakening and energy restoration', '["Increased energy", "Spiritual connection", "Mental alertness", "Inner strength"]'),
  ('Depression', '417 Hz', 'Facilitating change and removing negative energy', '["Mood improvement", "Energy flow", "Emotional release", "Positive transformation"]'),
  ('Hypertension', '432 Hz', 'Natural healing frequency for blood pressure regulation', '["Blood pressure regulation", "Cardiovascular health", "Relaxation", "Natural balance"]'),
  ('Arthritis', '285 Hz', 'Tissue regeneration and pain relief', '["Joint pain relief", "Tissue healing", "Inflammation reduction", "Physical healing"]'),
  ('Asthma', '174 Hz', 'Pain reduction and breathing support', '["Breathing improvement", "Pain relief", "Foundation healing", "Respiratory support"]'),
  ('Diabetes', '963 Hz', 'Crown chakra activation for metabolic balance', '["Metabolic balance", "Spiritual healing", "Cellular communication", "Energy regulation"]')
ON CONFLICT (disease_name) DO NOTHING;
