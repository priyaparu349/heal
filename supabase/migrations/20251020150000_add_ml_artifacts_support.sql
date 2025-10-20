/*
  # Add ML Artifacts Support

  1. Schema Updates
    - Add confusion_matrix_image column to ml_models (base64 or URL)
    - Add model_artifacts table for storing large binary data
    - Add feature_importance column
    - Add disease_symptom_patterns table

  2. Security
    - Maintain existing RLS policies
    - Add policies for new tables
*/

-- Add confusion matrix image support to ml_models
DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM information_schema.columns
    WHERE table_name = 'ml_models' AND column_name = 'confusion_matrix_image'
  ) THEN
    ALTER TABLE ml_models ADD COLUMN confusion_matrix_image text;
  END IF;
END $$;

-- Add feature importance
DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM information_schema.columns
    WHERE table_name = 'ml_models' AND column_name = 'feature_importance'
  ) THEN
    ALTER TABLE ml_models ADD COLUMN feature_importance jsonb DEFAULT '{}';
  END IF;
END $$;

-- Create model_artifacts table for storing large files
CREATE TABLE IF NOT EXISTS model_artifacts (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  model_id uuid REFERENCES ml_models(id) ON DELETE CASCADE,
  artifact_type text NOT NULL CHECK (artifact_type IN ('confusion_matrix', 'roc_curve', 'feature_importance', 'model_file')),
  file_name text NOT NULL,
  file_url text,
  file_data text,
  mime_type text DEFAULT 'image/png',
  created_at timestamptz DEFAULT now()
);

-- Create disease_symptom_patterns table
CREATE TABLE IF NOT EXISTS disease_symptom_patterns (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  disease_name text NOT NULL,
  symptom_name text NOT NULL,
  frequency numeric CHECK (frequency >= 0 AND frequency <= 1),
  importance numeric CHECK (importance >= 0 AND importance <= 1),
  created_at timestamptz DEFAULT now(),
  UNIQUE(disease_name, symptom_name)
);

-- Create indexes
CREATE INDEX IF NOT EXISTS idx_model_artifacts_model_id ON model_artifacts(model_id);
CREATE INDEX IF NOT EXISTS idx_model_artifacts_type ON model_artifacts(artifact_type);
CREATE INDEX IF NOT EXISTS idx_disease_symptom_patterns_disease ON disease_symptom_patterns(disease_name);
CREATE INDEX IF NOT EXISTS idx_disease_symptom_patterns_symptom ON disease_symptom_patterns(symptom_name);

-- Enable RLS
ALTER TABLE model_artifacts ENABLE ROW LEVEL SECURITY;
ALTER TABLE disease_symptom_patterns ENABLE ROW LEVEL SECURITY;

-- RLS Policies for model_artifacts
CREATE POLICY "Anyone can view model artifacts"
  ON model_artifacts FOR SELECT
  TO public
  USING (true);

CREATE POLICY "Service role can manage model artifacts"
  ON model_artifacts FOR ALL
  TO service_role
  USING (true)
  WITH CHECK (true);

-- RLS Policies for disease_symptom_patterns
CREATE POLICY "Anyone can view disease symptom patterns"
  ON disease_symptom_patterns FOR SELECT
  TO public
  USING (true);

CREATE POLICY "Service role can manage disease symptom patterns"
  ON disease_symptom_patterns FOR ALL
  TO service_role
  USING (true)
  WITH CHECK (true);
