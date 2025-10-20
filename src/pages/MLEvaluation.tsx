import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { Brain, Activity, TrendingUp, CheckCircle, AlertCircle, Loader, Play, BarChart3 } from 'lucide-react';

interface ModelMetrics {
  accuracy: number;
  precision: number;
  recall: number;
  f1Score: number;
  confusionMatrix: number[][];
  classificationReport: {
    classes: Array<{
      disease: string;
      precision: number;
      recall: number;
      f1: number;
      support: number;
    }>;
  };
}

interface TrainingResult {
  success: boolean;
  metrics: ModelMetrics;
  disease_labels: string[];
  disease_frequency_mapping: Record<string, string>;
  sample_predictions: Array<{
    symptoms: string[];
    predicted: string;
    frequency: string;
    expected: string;
  }>;
  model_id: string;
}

const MLEvaluation = () => {
  const [isTraining, setIsTraining] = useState(false);
  const [trainingResult, setTrainingResult] = useState<TrainingResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleTrainModel = async () => {
    setIsTraining(true);
    setError(null);

    try {
      const apiUrl = `${import.meta.env.VITE_SUPABASE_URL}/functions/v1/ml-train-evaluate`;
      const headers = {
        'Authorization': `Bearer ${import.meta.env.VITE_SUPABASE_ANON_KEY}`,
        'Content-Type': 'application/json',
      };

      const response = await fetch(apiUrl, {
        method: 'POST',
        headers
      });

      if (!response.ok) {
        throw new Error('Training failed');
      }

      const result = await response.json();
      setTrainingResult(result);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Training failed');
      console.error('Training error:', err);
    } finally {
      setIsTraining(false);
    }
  };

  const formatPercentage = (value: number) => `${(value * 100).toFixed(2)}%`;

  return (
    <div className="min-h-screen py-8 px-4 sm:px-6 lg:px-8">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-center mb-8"
        >
          <div className="flex justify-center mb-4">
            <div className="bg-gradient-to-r from-blue-600 to-teal-600 w-16 h-16 rounded-full flex items-center justify-center">
              <Brain className="w-8 h-8 text-white" />
            </div>
          </div>
          <h1 className="text-3xl sm:text-4xl font-bold text-gray-900 dark:text-white mb-2">
            ML Model Evaluation
          </h1>
          <p className="text-lg text-gray-600 dark:text-gray-400">
            Train and evaluate the intelligent health prediction system
          </p>
        </motion.div>

        {/* Train Button */}
        {!trainingResult && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="text-center mb-8"
          >
            <motion.button
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              onClick={handleTrainModel}
              disabled={isTraining}
              className="px-8 py-4 bg-gradient-to-r from-blue-600 to-teal-600 text-white rounded-xl font-semibold text-lg shadow-lg hover:shadow-xl transition-all duration-300 disabled:opacity-50 disabled:cursor-not-allowed flex items-center space-x-3 mx-auto"
            >
              {isTraining ? (
                <>
                  <Loader className="w-6 h-6 animate-spin" />
                  <span>Training Model...</span>
                </>
              ) : (
                <>
                  <Play className="w-6 h-6" />
                  <span>Train & Evaluate Model</span>
                </>
              )}
            </motion.button>
          </motion.div>
        )}

        {/* Error Message */}
        {error && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="mb-8 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-xl p-6"
          >
            <div className="flex items-center space-x-3">
              <AlertCircle className="w-6 h-6 text-red-600 dark:text-red-400" />
              <div>
                <h3 className="font-semibold text-red-900 dark:text-red-100">Training Error</h3>
                <p className="text-red-700 dark:text-red-300">{error}</p>
              </div>
            </div>
          </motion.div>
        )}

        {/* Results */}
        {trainingResult && (
          <div className="space-y-8">
            {/* Metrics Overview */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6"
            >
              <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-gray-100 dark:border-gray-700">
                <div className="flex items-center justify-between mb-2">
                  <h3 className="text-sm font-medium text-gray-600 dark:text-gray-400">Accuracy</h3>
                  <Activity className="w-5 h-5 text-blue-600" />
                </div>
                <p className="text-3xl font-bold text-gray-900 dark:text-white">
                  {formatPercentage(trainingResult.metrics.accuracy)}
                </p>
              </div>

              <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-gray-100 dark:border-gray-700">
                <div className="flex items-center justify-between mb-2">
                  <h3 className="text-sm font-medium text-gray-600 dark:text-gray-400">Precision</h3>
                  <TrendingUp className="w-5 h-5 text-green-600" />
                </div>
                <p className="text-3xl font-bold text-gray-900 dark:text-white">
                  {formatPercentage(trainingResult.metrics.precision)}
                </p>
              </div>

              <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-gray-100 dark:border-gray-700">
                <div className="flex items-center justify-between mb-2">
                  <h3 className="text-sm font-medium text-gray-600 dark:text-gray-400">Recall</h3>
                  <CheckCircle className="w-5 h-5 text-teal-600" />
                </div>
                <p className="text-3xl font-bold text-gray-900 dark:text-white">
                  {formatPercentage(trainingResult.metrics.recall)}
                </p>
              </div>

              <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-gray-100 dark:border-gray-700">
                <div className="flex items-center justify-between mb-2">
                  <h3 className="text-sm font-medium text-gray-600 dark:text-gray-400">F1 Score</h3>
                  <BarChart3 className="w-5 h-5 text-purple-600" />
                </div>
                <p className="text-3xl font-bold text-gray-900 dark:text-white">
                  {formatPercentage(trainingResult.metrics.f1Score)}
                </p>
              </div>
            </motion.div>

            {/* Classification Report */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.1 }}
              className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-gray-100 dark:border-gray-700"
            >
              <h2 className="text-xl font-bold text-gray-900 dark:text-white mb-4">
                Classification Report
              </h2>
              <div className="overflow-x-auto">
                <table className="w-full">
                  <thead>
                    <tr className="border-b border-gray-200 dark:border-gray-700">
                      <th className="text-left py-3 px-4 text-sm font-semibold text-gray-700 dark:text-gray-300">Disease</th>
                      <th className="text-right py-3 px-4 text-sm font-semibold text-gray-700 dark:text-gray-300">Precision</th>
                      <th className="text-right py-3 px-4 text-sm font-semibold text-gray-700 dark:text-gray-300">Recall</th>
                      <th className="text-right py-3 px-4 text-sm font-semibold text-gray-700 dark:text-gray-300">F1-Score</th>
                    </tr>
                  </thead>
                  <tbody>
                    {trainingResult.metrics.classificationReport.classes.map((cls, idx) => (
                      <tr key={idx} className="border-b border-gray-100 dark:border-gray-700 hover:bg-gray-50 dark:hover:bg-gray-700/50">
                        <td className="py-3 px-4 text-sm font-medium text-gray-900 dark:text-white">{cls.disease}</td>
                        <td className="py-3 px-4 text-sm text-right text-gray-600 dark:text-gray-400">{formatPercentage(cls.precision)}</td>
                        <td className="py-3 px-4 text-sm text-right text-gray-600 dark:text-gray-400">{formatPercentage(cls.recall)}</td>
                        <td className="py-3 px-4 text-sm text-right text-gray-600 dark:text-gray-400">{formatPercentage(cls.f1)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </motion.div>

            {/* Disease to Frequency Mapping */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.2 }}
              className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-gray-100 dark:border-gray-700"
            >
              <h2 className="text-xl font-bold text-gray-900 dark:text-white mb-4">
                Disease → Healing Frequency Mapping
              </h2>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {Object.entries(trainingResult.disease_frequency_mapping).map(([disease, frequency]) => (
                  <div key={disease} className="bg-gradient-to-r from-blue-50 to-teal-50 dark:from-blue-900/20 dark:to-teal-900/20 rounded-lg p-4 border border-blue-200 dark:border-blue-800">
                    <h3 className="font-semibold text-gray-900 dark:text-white mb-1">{disease}</h3>
                    <p className="text-lg font-bold text-blue-600 dark:text-blue-400">{frequency}</p>
                  </div>
                ))}
              </div>
            </motion.div>

            {/* Sample Predictions */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.3 }}
              className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-gray-100 dark:border-gray-700"
            >
              <h2 className="text-xl font-bold text-gray-900 dark:text-white mb-4">
                Sample Predictions
              </h2>
              <div className="space-y-4">
                {trainingResult.sample_predictions.map((pred, idx) => (
                  <div key={idx} className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4">
                    <div className="flex items-start justify-between mb-2">
                      <div>
                        <p className="text-sm font-medium text-gray-600 dark:text-gray-400 mb-1">Symptoms:</p>
                        <p className="text-sm text-gray-900 dark:text-white">{pred.symptoms.join(', ')}</p>
                      </div>
                      {pred.predicted === pred.expected && (
                        <CheckCircle className="w-5 h-5 text-green-600" />
                      )}
                    </div>
                    <div className="mt-3 pt-3 border-t border-gray-200 dark:border-gray-600">
                      <div className="flex items-center justify-between">
                        <div>
                          <p className="text-xs text-gray-500 dark:text-gray-400">Predicted:</p>
                          <p className="font-semibold text-gray-900 dark:text-white">{pred.predicted}</p>
                        </div>
                        <div className="text-right">
                          <p className="text-xs text-gray-500 dark:text-gray-400">Frequency:</p>
                          <p className="font-semibold text-blue-600 dark:text-blue-400">{pred.frequency}</p>
                        </div>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </motion.div>

            {/* Retrain Button */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.4 }}
              className="text-center"
            >
              <motion.button
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                onClick={handleTrainModel}
                disabled={isTraining}
                className="px-6 py-3 bg-gray-600 dark:bg-gray-700 text-white rounded-lg font-medium hover:bg-gray-700 dark:hover:bg-gray-600 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              >
                Retrain Model
              </motion.button>
            </motion.div>
          </div>
        )}
      </div>
    </div>
  );
};

export default MLEvaluation;
