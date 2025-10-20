import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { Play, Pause, Volume2, VolumeX, Music, Timer, Zap, Heart, Brain, Sparkles } from 'lucide-react';
import { useAudio } from '../contexts/AudioContext';

interface FrequencyPreset {
  id: string;
  name: string;
  frequency: number;
  description: string;
  benefits: string[];
  color: string;
  icon: React.ComponentType<any>;
  category: string;
  duration: number; // in seconds
}

const FrequencyGenerator = () => {
  const { isPlaying, currentFrequency, playFrequency, stopFrequency, volume, setVolume } = useAudio();
  const [selectedPreset, setSelectedPreset] = useState<FrequencyPreset | null>(null);
  const [customFrequency, setCustomFrequency] = useState(432);
  const [sessionTime, setSessionTime] = useState(0);
  const [isCustomMode, setIsCustomMode] = useState(false);

  const frequencyPresets: FrequencyPreset[] = [
    {
      id: 'stress-relief',
      name: 'Stress Relief',
      frequency: 432,
      description: 'Natural healing frequency that promotes relaxation and stress reduction',
      benefits: ['Reduces anxiety', 'Promotes calmness', 'Balances nervous system', 'Improves focus'],
      color: 'from-blue-500 to-cyan-500',
      icon: Heart,
      category: 'Wellness',
      duration: 600, // 10 minutes
    },
    {
      id: 'dna-repair',
      name: 'DNA Repair',
      frequency: 528,
      description: 'The "Love Frequency" known for healing and DNA repair',
      benefits: ['DNA healing', 'Cellular regeneration', 'Emotional healing', 'Increased energy'],
      color: 'from-green-500 to-emerald-500',
      icon: Zap,
      category: 'Healing',
      duration: 900, // 15 minutes
    },
    {
      id: 'deep-sleep',
      name: 'Deep Sleep',
      frequency: 110,
      description: 'Low gamma waves that promote deep, restorative sleep',
      benefits: ['Deeper sleep', 'Better dreams', 'Faster sleep onset', 'Morning refreshment'],
      color: 'from-purple-500 to-indigo-500',
      icon: Brain,
      category: 'Sleep',
      duration: 1800, // 30 minutes
    },
    {
      id: 'meditation',
      name: 'Meditation',
      frequency: 136,
      description: 'Alpha waves that enhance meditation and mindfulness',
      benefits: ['Enhanced meditation', 'Increased awareness', 'Mental clarity', 'Spiritual connection'],
      color: 'from-orange-500 to-yellow-500',
      icon: Sparkles,
      category: 'Spiritual',
      duration: 1200, // 20 minutes
    },
    {
      id: 'pain-relief',
      name: 'Pain Relief',
      frequency: 285,
      description: 'Natural anesthetic frequency for pain management',
      benefits: ['Reduces pain', 'Muscle relaxation', 'Anti-inflammatory', 'Physical healing'],
      color: 'from-red-500 to-pink-500',
      icon: Heart,
      category: 'Healing',
      duration: 900, // 15 minutes
    },
    {
      id: 'focus-clarity',
      name: 'Focus & Clarity',
      frequency: 963,
      description: 'Gamma waves that enhance cognitive function and concentration',
      benefits: ['Improved focus', 'Mental clarity', 'Enhanced learning', 'Better memory'],
      color: 'from-teal-500 to-blue-500',
      icon: Brain,
      category: 'Cognitive',
      duration: 1500, // 25 minutes
    },
  ];

  // Session timer
  useEffect(() => {
    let interval: NodeJS.Timeout;
    if (isPlaying) {
      interval = setInterval(() => {
        setSessionTime(prev => prev + 1);
      }, 1000);
    } else {
      setSessionTime(0);
    }
    return () => clearInterval(interval);
  }, [isPlaying]);

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  };

  const handlePresetSelect = (preset: FrequencyPreset) => {
    setSelectedPreset(preset);
    setIsCustomMode(false);
    if (isPlaying) {
      stopFrequency();
    }
  };

  const handlePlayPreset = (preset: FrequencyPreset) => {
    if (isPlaying && currentFrequency === preset.frequency) {
      stopFrequency();
    } else {
      playFrequency(preset.frequency, preset.duration);
      setSelectedPreset(preset);
    }
  };

  const handleCustomPlay = () => {
    if (isPlaying && currentFrequency === customFrequency) {
      stopFrequency();
    } else {
      playFrequency(customFrequency);
      setSelectedPreset(null);
    }
  };

  const categories = [...new Set(frequencyPresets.map(preset => preset.category))];

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
            <div className="bg-gradient-to-r from-purple-600 to-blue-600 w-16 h-16 rounded-full flex items-center justify-center">
              <Music className="w-8 h-8 text-white" />
            </div>
          </div>
          <h1 className="text-3xl sm:text-4xl font-bold text-gray-900 dark:text-white mb-2">
            Frequency Healing Generator
          </h1>
          <p className="text-lg text-gray-600 dark:text-gray-400">
            Experience therapeutic sound frequencies tailored to your wellness needs
          </p>
        </motion.div>

        {/* Current Session Status */}
        {(isPlaying || selectedPreset) && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="mb-8 bg-white dark:bg-gray-800 rounded-2xl p-6 shadow-lg border border-gray-100 dark:border-gray-700"
          >
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center space-x-4">
                <div className={`w-12 h-12 rounded-full bg-gradient-to-r ${selectedPreset?.color || 'from-blue-600 to-purple-600'} flex items-center justify-center`}>
                  {selectedPreset ? (
                    <selectedPreset.icon className="w-6 h-6 text-white" />
                  ) : (
                    <Music className="w-6 h-6 text-white" />
                  )}
                </div>
                <div>
                  <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
                    {selectedPreset?.name || `Custom ${customFrequency} Hz`}
                  </h3>
                  <p className="text-sm text-gray-600 dark:text-gray-400">
                    {currentFrequency} Hz • {formatTime(sessionTime)}
                  </p>
                </div>
              </div>
              <div className="flex items-center space-x-4">
                <div className="flex items-center space-x-2">
                  <VolumeX className="w-4 h-4 text-gray-400" />
                  <input
                    type="range"
                    min="0"
                    max="1"
                    step="0.1"
                    value={volume}
                    onChange={(e) => setVolume(parseFloat(e.target.value))}
                    className="w-20"
                  />
                  <Volume2 className="w-4 h-4 text-gray-400" />
                </div>
                <motion.button
                  whileHover={{ scale: 1.1 }}
                  whileTap={{ scale: 0.9 }}
                  onClick={stopFrequency}
                  className="p-2 bg-red-100 dark:bg-red-900/20 text-red-600 dark:text-red-400 rounded-full hover:bg-red-200 dark:hover:bg-red-900/40 transition-colors"
                >
                  <Pause className="w-5 h-5" />
                </motion.button>
              </div>
            </div>
            
            {selectedPreset && (
              <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4">
                <p className="text-sm text-gray-700 dark:text-gray-300 mb-2">
                  {selectedPreset.description}
                </p>
                <div className="flex flex-wrap gap-2">
                  {selectedPreset.benefits.map((benefit, index) => (
                    <span
                      key={index}
                      className="px-2 py-1 bg-white dark:bg-gray-600 text-xs text-gray-600 dark:text-gray-300 rounded-full"
                    >
                      {benefit}
                    </span>
                  ))}
                </div>
              </div>
            )}
          </motion.div>
        )}

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Frequency Presets */}
          <div className="lg:col-span-2">
            <div className="flex items-center justify-between mb-6">
              <h2 className="text-2xl font-bold text-gray-900 dark:text-white">
                Healing Frequencies
              </h2>
              <div className="flex space-x-2">
                <button
                  onClick={() => setIsCustomMode(false)}
                  className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                    !isCustomMode
                      ? 'bg-blue-600 text-white'
                      : 'bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-600'
                  }`}
                >
                  Presets
                </button>
                <button
                  onClick={() => setIsCustomMode(true)}
                  className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                    isCustomMode
                      ? 'bg-blue-600 text-white'
                      : 'bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-600'
                  }`}
                >
                  Custom
                </button>
              </div>
            </div>

            {!isCustomMode ? (
              <div className="space-y-6">
                {categories.map(category => (
                  <div key={category}>
                    <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
                      {category}
                    </h3>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      {frequencyPresets
                        .filter(preset => preset.category === category)
                        .map((preset, index) => (
                          <motion.div
                            key={preset.id}
                            initial={{ opacity: 0, y: 30 }}
                            animate={{ opacity: 1, y: 0 }}
                            transition={{ delay: index * 0.1 }}
                            whileHover={{ y: -2 }}
                            className={`bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border transition-all duration-300 cursor-pointer ${
                              selectedPreset?.id === preset.id
                                ? 'border-blue-500 ring-2 ring-blue-200 dark:ring-blue-800'
                                : 'border-gray-100 dark:border-gray-700 hover:shadow-xl'
                            }`}
                            onClick={() => handlePresetSelect(preset)}
                          >
                            <div className="flex items-center justify-between mb-4">
                              <div className={`w-12 h-12 rounded-full bg-gradient-to-r ${preset.color} flex items-center justify-center`}>
                                <preset.icon className="w-6 h-6 text-white" />
                              </div>
                              <div className="flex items-center space-x-2">
                                <Timer className="w-4 h-4 text-gray-400" />
                                <span className="text-sm text-gray-500 dark:text-gray-400">
                                  {Math.floor(preset.duration / 60)}min
                                </span>
                              </div>
                            </div>
                            
                            <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-2">
                              {preset.name}
                            </h3>
                            <p className="text-sm text-gray-600 dark:text-gray-400 mb-3">
                              {preset.description}
                            </p>
                            <p className="text-lg font-bold text-blue-600 dark:text-blue-400 mb-4">
                              {preset.frequency} Hz
                            </p>
                            
                            <motion.button
                              whileHover={{ scale: 1.05 }}
                              whileTap={{ scale: 0.95 }}
                              onClick={(e) => {
                                e.stopPropagation();
                                handlePlayPreset(preset);
                              }}
                              className={`w-full py-2 px-4 rounded-lg font-medium transition-all duration-300 ${
                                isPlaying && currentFrequency === preset.frequency
                                  ? 'bg-red-100 dark:bg-red-900/20 text-red-600 dark:text-red-400 hover:bg-red-200 dark:hover:bg-red-900/40'
                                  : `bg-gradient-to-r ${preset.color} text-white hover:shadow-lg`
                              }`}
                            >
                              <div className="flex items-center justify-center space-x-2">
                                {isPlaying && currentFrequency === preset.frequency ? (
                                  <>
                                    <Pause className="w-4 h-4" />
                                    <span>Stop</span>
                                  </>
                                ) : (
                                  <>
                                    <Play className="w-4 h-4" />
                                    <span>Play</span>
                                  </>
                                )}
                              </div>
                            </motion.button>
                          </motion.div>
                        ))}
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-gray-100 dark:border-gray-700"
              >
                <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
                  Custom Frequency Generator
                </h3>
                <div className="space-y-6">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                      Frequency (Hz)
                    </label>
                    <div className="flex items-center space-x-4">
                      <input
                        type="range"
                        min="1"
                        max="1000"
                        value={customFrequency}
                        onChange={(e) => setCustomFrequency(parseInt(e.target.value))}
                        className="flex-1"
                      />
                      <input
                        type="number"
                        min="1"
                        max="20000"
                        value={customFrequency}
                        onChange={(e) => setCustomFrequency(parseInt(e.target.value) || 432)}
                        className="w-24 px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white text-center"
                      />
                      <span className="text-sm text-gray-500 dark:text-gray-400">Hz</span>
                    </div>
                  </div>
                  
                  <motion.button
                    whileHover={{ scale: 1.02 }}
                    whileTap={{ scale: 0.98 }}
                    onClick={handleCustomPlay}
                    className={`w-full py-3 px-6 rounded-lg font-medium transition-all duration-300 ${
                      isPlaying && currentFrequency === customFrequency
                        ? 'bg-red-100 dark:bg-red-900/20 text-red-600 dark:text-red-400 hover:bg-red-200 dark:hover:bg-red-900/40'
                        : 'bg-gradient-to-r from-purple-600 to-blue-600 text-white hover:from-purple-700 hover:to-blue-700'
                    }`}
                  >
                    <div className="flex items-center justify-center space-x-2">
                      {isPlaying && currentFrequency === customFrequency ? (
                        <>
                          <Pause className="w-5 h-5" />
                          <span>Stop Custom Frequency</span>
                        </>
                      ) : (
                        <>
                          <Play className="w-5 h-5" />
                          <span>Play Custom Frequency</span>
                        </>
                      )}
                    </div>
                  </motion.button>
                </div>
              </motion.div>
            )}
          </div>

          {/* Information Panel */}
          <div className="space-y-6">
            <motion.div
              initial={{ opacity: 0, x: 30 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.3 }}
              className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-gray-100 dark:border-gray-700"
            >
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
                How Sound Healing Works
              </h3>
              <div className="space-y-4 text-sm text-gray-600 dark:text-gray-400">
                <p>
                  Sound healing uses specific frequencies to promote physical and emotional well-being through resonance and entrainment.
                </p>
                <ul className="list-disc list-inside space-y-2">
                  <li>Different frequencies target different aspects of health</li>
                  <li>Binaural beats can synchronize brainwaves</li>
                  <li>Regular sessions may improve overall wellness</li>
                  <li>Safe, non-invasive therapeutic approach</li>
                </ul>
              </div>
            </motion.div>

            <motion.div
              initial={{ opacity: 0, x: 30 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.4 }}
              className="bg-gradient-to-r from-blue-50 to-purple-50 dark:from-blue-900/20 dark:to-purple-900/20 rounded-xl p-6 border border-blue-200 dark:border-blue-800"
            >
              <h3 className="text-lg font-semibold text-blue-900 dark:text-blue-100 mb-4">
                Safety Guidelines
              </h3>
              <div className="space-y-2 text-sm text-blue-800 dark:text-blue-200">
                <p>• Start with low volume and gradually increase</p>
                <p>• Take breaks during longer sessions</p>
                <p>• Stop if you experience discomfort</p>
                <p>• Consult healthcare providers for medical conditions</p>
                <p>• Use headphones for best binaural effect</p>
              </div>
            </motion.div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default FrequencyGenerator;