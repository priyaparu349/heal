# ML Module Quick Start Guide

## How to Use the ML Evaluation Features

### Step 1: Train the Model

1. Log in to your Quantum Pulse account
2. Navigate to **Dashboard** → **ML Evaluation**
3. Click the **"Train & Evaluate Model"** button
4. Wait for training to complete (5-10 seconds)

**You will see:**
- ✅ Overall accuracy, precision, recall, and F1-score
- ✅ Detailed classification report for all 10 diseases
- ✅ Disease → Healing frequency mappings
- ✅ Sample predictions with expected results
- ✅ Model saved to database with unique ID

### Step 2: Test Symptom Predictions

1. Navigate to **Dashboard** → **Symptom Checker**
2. Add symptoms using either:
   - Type custom symptoms in the search box
   - Click quick-add buttons for common symptoms
3. Click **"Analyze Symptoms"**

**You will receive:**
- 🎯 Predicted disease with confidence score
- 🎵 Recommended healing frequency (Hz)
- 💊 Description of therapeutic benefits
- 📊 Alternative possible diagnoses
- 🔗 Link to play the healing frequency

### Step 3: Use the Healing Frequency

1. From the prediction result, click **"Go to Frequency Generator"**
2. Select the recommended frequency preset
3. Adjust volume and duration
4. Click **Play** to experience the healing frequency

## Example Usage

### Example 1: Headache Symptoms

**Input Symptoms:**
- Headache
- Nausea
- Light sensitivity

**Predicted Output:**
- Disease: **Migraine**
- Confidence: **92%**
- Frequency: **528 Hz** (DNA repair frequency)
- Benefits: Pain reduction, Stress relief, Mental clarity

### Example 2: Breathing Issues

**Input Symptoms:**
- Wheezing
- Shortness of breath
- Chest tightness

**Predicted Output:**
- Disease: **Asthma**
- Confidence: **88%**
- Frequency: **174 Hz** (Pain reduction & breathing support)
- Benefits: Breathing improvement, Respiratory support

### Example 3: Sleep Problems

**Input Symptoms:**
- Difficulty sleeping
- Fatigue
- Irritability

**Predicted Output:**
- Disease: **Insomnia**
- Confidence: **85%**
- Frequency: **639 Hz** (Heart chakra frequency)
- Benefits: Better sleep, Deep rest, Emotional healing

## API Testing (For Development)

### Test Training Endpoint

```bash
curl -X POST \
  https://bnwkmbqlwxztwmzhpswo.supabase.co/functions/v1/ml-train-evaluate \
  -H "Authorization: Bearer YOUR_ANON_KEY" \
  -H "Content-Type: application/json"
```

### Test Prediction Endpoint

```bash
curl -X POST \
  https://bnwkmbqlwxztwmzhpswo.supabase.co/functions/v1/ml-predict \
  -H "Authorization: Bearer YOUR_ANON_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "symptoms": ["headache", "nausea", "light_sensitivity"]
  }'
```

## Expected Model Performance

Based on the training dataset:

| Metric     | Expected Range |
|------------|---------------|
| Accuracy   | 85-95%        |
| Precision  | 80-90%        |
| Recall     | 80-90%        |
| F1-Score   | 80-90%        |

## Disease Coverage

The model can predict the following conditions:

1. **Migraine** → 528 Hz
2. **Anxiety** → 396 Hz
3. **Insomnia** → 639 Hz
4. **Stress** → 741 Hz
5. **Fatigue** → 852 Hz
6. **Depression** → 417 Hz
7. **Hypertension** → 432 Hz
8. **Arthritis** → 285 Hz
9. **Asthma** → 174 Hz
10. **Diabetes** → 963 Hz

## Symptom Keywords

Common symptoms recognized by the system:

**Neurological:** headache, dizziness, confusion, numbness
**Respiratory:** wheezing, shortness_of_breath, cough, chest_tightness
**Cardiovascular:** chest_pain, rapid_heartbeat, high_blood_pressure
**Musculoskeletal:** joint_pain, stiffness, muscle_pain, weakness
**Psychological:** anxiety, worry, sadness, depression, nervousness
**Sleep:** insomnia, fatigue, difficulty_sleeping, tiredness
**Digestive:** nausea, vomiting, stomach_upset
**General:** fever, sweating, irritability

## Tips for Best Results

1. **Use specific symptoms**: "severe_headache" is better than just "pain"
2. **Include multiple symptoms**: 3-5 symptoms give better accuracy
3. **Use medical terminology**: The model is trained on medical symptom names
4. **Check alternatives**: Review alternative diagnoses for context
5. **Consult professionals**: This is a supplementary tool, not a replacement for medical advice

## Troubleshooting

**Model won't train?**
- Check internet connection
- Verify Supabase URL in `.env` file
- Check browser console for errors

**Predictions seem inaccurate?**
- Ensure symptoms are spelled correctly
- Try adding more symptoms
- Use underscores instead of spaces (e.g., "joint_pain")

**Frequency not playing?**
- Navigate to Frequency Generator page
- Check volume settings
- Ensure browser allows audio playback

## For Research Paper

To generate results for your research paper:

1. Train the model via ML Evaluation page
2. Take screenshots of:
   - Overall metrics (accuracy, precision, recall, F1)
   - Classification report table
   - Disease frequency mapping
   - Sample predictions
3. Test multiple symptom combinations
4. Document prediction accuracy
5. Compare with baseline models

## Support

For issues or questions:
- Check console logs in browser developer tools
- Review `ML_INTEGRATION_GUIDE.md` for technical details
- Verify all Edge Functions are deployed
- Ensure database migrations are applied
