import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Search, Plus, X, Sparkles, Activity, AlertCircle, Loader } from 'lucide-react';
import { useNavigate } from 'react-router-dom';

interface PredictionResult {
  disease: string;
  confidence: number;
  healing_frequency: string;
  description: string;
  benefits: string[];
  prediction_id: string;
}

interface AlternativeMatch {
  disease: string;
  confidence: number;
}

const SymptomChecker = () => {
  const navigate = useNavigate();
  const [symptomInput, setSymptomInput] = useState('');
  const [symptoms, setSymptoms] = useState<string[]>([]);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [prediction, setPrediction] = useState<PredictionResult | null>(null);
  const [alternatives, setAlternatives] = useState<AlternativeMatch[]>([]);
  const [error, setError] = useState<string | null>(null);

  const commonSymptoms = [
    'Headache', 'Fatigue', 'Nausea', 'Dizziness', 'Fever',
    'Cough', 'Shortness of breath', 'Chest pain', 'Anxiety',
    'Insomnia', 'Joint pain', 'Muscle pain', 'Weakness',
  ];

  const handleAddSymptom = () => {
    if (symptomInput.trim() && !symptoms.includes(symptomInput.trim())) {
      setSymptoms([...symptoms, symptomInput.trim()]);
      setSymptomInput('');
    }
  };

  const handleRemoveSymptom = (symptom: string) => {
    setSymptoms(symptoms.filter(s => s !== symptom));
  };

  const handleQuickAdd = (symptom: string) => {
    if (!symptoms.includes(symptom)) {
      setSymptoms([...symptoms, symptom]);
    }
  };

  const handleAnalyze = async () => {
    if (symptoms.length === 0) {
      setError('Please add at least one symptom');
      return;
    }

    setIsAnalyzing(true);
    setError(null);
    setPrediction(null);
    setAlternatives([]);

    try {
      const apiUrl = `${import.meta.env.VITE_SUPABASE_URL}/functions/v1/ml-predict`;
      const headers = {
        'Authorization': `Bearer ${import.meta.env.VITE_SUPABASE_ANON_KEY}`,
        'Content-Type': 'application/json',
      };

      const response = await fetch(apiUrl, {
        method: 'POST',
        headers,
        body: JSON.stringify({ symptoms }),
      });

      if (!response.ok) {
        throw new Error('Prediction failed');
      }

      const result = await response.json();
      if (result.success) {
        setPrediction(result.prediction);
        setAlternatives(result.alternative_matches || []);
      } else {
        throw new Error(result.error || 'Prediction failed');
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Analysis failed');
      console.error('Analysis error:', err);
    } finally {
      setIsAnalyzing(false);
    }
  };

  const handlePlayFrequency = () => {
    if (prediction) {
      navigate('/frequency-generator');
    }
  };

  return (
    <div className="min-h-screen py-8 px-4 sm:px-6 lg:px-8">
      <div className="max-w-4xl mx-auto">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-center mb-8"
        >
          <div className="flex justify-center mb-4">
            <div className="bg-gradient-to-r from-teal-600 to-blue-600 w-16 h-16 rounded-full flex items-center justify-center">
              <Activity className="w-8 h-8 text-white" />
            </div>
          </div>
          <h1 className="text-3xl sm:text-4xl font-bold text-gray-900 dark:text-white mb-2">
            Intelligent Symptom Checker
          </h1>
          <p className="text-lg text-gray-600 dark:text-gray-400">
            AI-powered health assessment with personalized healing frequencies
          </p>
        </motion.div>

        {/* Symptom Input */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="bg-white dark:bg-gray-800 rounded-2xl p-6 shadow-lg border border-gray-100 dark:border-gray-700 mb-6"
        >
          <h2 className="text-xl font-bold text-gray-900 dark:text-white mb-4">
            Add Your Symptoms
          </h2>

          <div className="flex space-x-2 mb-4">
            <div className="flex-1 relative">
              <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-5 h-5 text-gray-400" />
              <input
                type="text"
                value={symptomInput}
                onChange={(e) => setSymptomInput(e.target.value)}
                onKeyPress={(e) => e.key === 'Enter' && handleAddSymptom()}
                placeholder="Enter a symptom..."
                className="w-full pl-10 pr-4 py-3 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white placeholder-gray-500 dark:placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>
            <motion.button
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              onClick={handleAddSymptom}
              className="px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors flex items-center space-x-2"
            >
              <Plus className="w-5 h-5" />
              <span>Add</span>
            </motion.button>
          </div>

          {/* Current Symptoms */}
          {symptoms.length > 0 && (
            <div className="mb-4">
              <h3 className="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-2">
                Current Symptoms ({symptoms.length})
              </h3>
              <div className="flex flex-wrap gap-2">
                {symptoms.map((symptom) => (
                  <motion.div
                    key={symptom}
                    initial={{ opacity: 0, scale: 0.8 }}
                    animate={{ opacity: 1, scale: 1 }}
                    exit={{ opacity: 0, scale: 0.8 }}
                    className="flex items-center space-x-2 bg-blue-100 dark:bg-blue-900/30 text-blue-800 dark:text-blue-200 px-3 py-2 rounded-lg"
                  >
                    <span className="text-sm font-medium">{symptom}</span>
                    <button
                      onClick={() => handleRemoveSymptom(symptom)}
                      className="hover:bg-blue-200 dark:hover:bg-blue-900/50 rounded-full p-1 transition-colors"
                    >
                      <X className="w-4 h-4" />
                    </button>
                  </motion.div>
                ))}
              </div>
            </div>
          )}

          {/* Quick Add Symptoms */}
          <div>
            <h3 className="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-2">
              Common Symptoms
            </h3>
            <div className="flex flex-wrap gap-2">
              {commonSymptoms.map((symptom) => (
                <button
                  key={symptom}
                  onClick={() => handleQuickAdd(symptom)}
                  disabled={symptoms.includes(symptom)}
                  className="px-3 py-2 text-sm bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 rounded-lg hover:bg-gray-200 dark:hover:bg-gray-600 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {symptom}
                </button>
              ))}
            </div>
          </div>

          {/* Analyze Button */}
          <motion.button
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
            onClick={handleAnalyze}
            disabled={symptoms.length === 0 || isAnalyzing}
            className="w-full mt-6 py-4 bg-gradient-to-r from-teal-600 to-blue-600 text-white rounded-xl font-semibold text-lg shadow-lg hover:shadow-xl transition-all duration-300 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center space-x-3"
          >
            {isAnalyzing ? (
              <>
                <Loader className="w-6 h-6 animate-spin" />
                <span>Analyzing...</span>
              </>
            ) : (
              <>
                <Sparkles className="w-6 h-6" />
                <span>Analyze Symptoms</span>
              </>
            )}
          </motion.button>
        </motion.div>

        {/* Error Message */}
        <AnimatePresence>
          {error && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              className="mb-6 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-xl p-4"
            >
              <div className="flex items-center space-x-3">
                <AlertCircle className="w-6 h-6 text-red-600 dark:text-red-400" />
                <p className="text-red-700 dark:text-red-300">{error}</p>
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Prediction Result */}
        <AnimatePresence>
          {prediction && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              className="space-y-6"
            >
              {/* Main Prediction */}
              <div className="bg-gradient-to-r from-teal-50 to-blue-50 dark:from-teal-900/20 dark:to-blue-900/20 rounded-2xl p-6 shadow-lg border border-teal-200 dark:border-teal-800">
                <div className="flex items-center justify-between mb-4">
                  <h2 className="text-2xl font-bold text-gray-900 dark:text-white">
                    Prediction Result
                  </h2>
                  <div className="bg-white dark:bg-gray-800 px-4 py-2 rounded-lg">
                    <p className="text-sm text-gray-600 dark:text-gray-400">Confidence</p>
                    <p className="text-xl font-bold text-teal-600 dark:text-teal-400">
                      {(prediction.confidence * 100).toFixed(1)}%
                    </p>
                  </div>
                </div>

                <div className="bg-white dark:bg-gray-800 rounded-xl p-6 mb-4">
                  <h3 className="text-3xl font-bold text-gray-900 dark:text-white mb-2">
                    {prediction.disease}
                  </h3>
                  <p className="text-gray-600 dark:text-gray-400 mb-4">
                    {prediction.description}
                  </p>

                  <div className="bg-gradient-to-r from-blue-100 to-teal-100 dark:from-blue-900/30 dark:to-teal-900/30 rounded-lg p-4 mb-4">
                    <h4 className="font-semibold text-gray-900 dark:text-white mb-2">
                      Recommended Healing Frequency
                    </h4>
                    <p className="text-3xl font-bold text-blue-600 dark:text-blue-400">
                      {prediction.healing_frequency}
                    </p>
                  </div>

                  {prediction.benefits.length > 0 && (
                    <div>
                      <h4 className="font-semibold text-gray-900 dark:text-white mb-2">Benefits</h4>
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
                        {prediction.benefits.map((benefit, idx) => (
                          <div key={idx} className="flex items-center space-x-2">
                            <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                            <span className="text-sm text-gray-700 dark:text-gray-300">{benefit}</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>

                <motion.button
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.98 }}
                  onClick={handlePlayFrequency}
                  className="w-full py-3 bg-gradient-to-r from-blue-600 to-teal-600 text-white rounded-lg font-semibold hover:shadow-lg transition-all duration-300"
                >
                  Go to Frequency Generator
                </motion.button>
              </div>

              {/* Alternative Matches */}
              {alternatives.length > 0 && (
                <div className="bg-white dark:bg-gray-800 rounded-2xl p-6 shadow-lg border border-gray-100 dark:border-gray-700">
                  <h3 className="text-lg font-bold text-gray-900 dark:text-white mb-4">
                    Alternative Possibilities
                  </h3>
                  <div className="space-y-3">
                    {alternatives.map((alt, idx) => (
                      <div
                        key={idx}
                        className="flex items-center justify-between bg-gray-50 dark:bg-gray-700 rounded-lg p-4"
                      >
                        <span className="font-medium text-gray-900 dark:text-white">{alt.disease}</span>
                        <span className="text-sm text-gray-600 dark:text-gray-400">
                          {(alt.confidence * 100).toFixed(1)}% match
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </div>
  );
};

export default SymptomChecker;
