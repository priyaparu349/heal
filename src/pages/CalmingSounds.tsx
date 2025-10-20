import React, { useState, useRef, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import { Play, Pause, Volume2, VolumeX, Search, ExternalLink, ArrowLeft, Timer, Repeat } from 'lucide-react';

interface CalmingSound {
  id: string;
  name: string;
  description: string;
  benefits: string[];
  color: string;
  audioUrl: string;
  duration: string;
}

const CalmingSounds = () => {
  const { soundId } = useParams();
  const navigate = useNavigate();
  const [isPlaying, setIsPlaying] = useState(false);
  const [volume, setVolume] = useState(0.5);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [isLooping, setIsLooping] = useState(true);
  const [customSearch, setCustomSearch] = useState('');
  const audioRef = useRef<HTMLAudioElement>(null);

  const calmingSounds: CalmingSound[] = [
    {
      id: 'birds-chirping',
      name: 'Birds Chirping',
      description: 'Peaceful morning bird songs to reduce stress and anxiety',
      benefits: ['Reduces stress', 'Improves mood', 'Enhances focus', 'Natural relaxation'],
      color: 'from-green-500 to-blue-500',
      audioUrl: 'https://myckworfnrmkkhzpagto.supabase.co/storage/v1/object/public/audios/Bird-Sounds-Birds-Chirping-Sounds-.mp3',
      duration: '10:00',
    },
    {
      id: 'wind-gushes',
      name: 'Wind Gushes',
      description: 'Gentle wind sounds for meditation and deep relaxation',
      benefits: ['Deep relaxation', 'Meditation aid', 'Stress relief', 'Mental clarity'],
      color: 'from-gray-400 to-blue-400',
      audioUrl: 'https://myckworfnrmkkhzpagto.supabase.co/storage/v1/object/public/audios/wind-sound-strong-howling-2-hours-swaying-spruce-trees_k8OCNL4D.mp3',
      duration: '15:00',
    },
    {
      id: 'sea-waves',
      name: 'Sea Waves',
      description: 'Rhythmic ocean waves for sleep and tranquility',
      benefits: ['Better sleep', 'Anxiety reduction', 'Blood pressure regulation', 'Peaceful mind'],
      color: 'from-blue-500 to-cyan-500',
      audioUrl: 'https://myckworfnrmkkhzpagto.supabase.co/storage/v1/object/public/audios/Beach-Sea-Waves_SOUND-EFFECT-128kbps.m4a.webm',
      duration: '20:00',
    },
    {
      id: 'om-chanting',
      name: 'Om Chanting',
      description: 'Sacred Om vibrations for spiritual healing and meditation',
      benefits: ['Spiritual connection', 'Chakra alignment', 'Deep meditation', 'Inner peace'],
      color: 'from-purple-500 to-pink-500',
      audioUrl: 'https://myckworfnrmkkhzpagto.supabase.co/storage/v1/object/public/audios/Om%20Chanting%20Sound.mp3',
      duration: '30:00',
    },
    // {
    //   id: 'ocean-waves',
    //   name: 'Ocean Waves',
    //   description: 'Powerful ocean waves for deep relaxation and focus',
    //   benefits: ['Enhanced focus', 'Stress reduction', 'Natural white noise', 'Improved concentration'],
    //   color: 'from-blue-600 to-teal-600',
    //   audioUrl: 'https://cdn.pixabay.com/audio/2022/05/27/audio_1808fbf07a.mp3',
    //   duration: '25:00',
    // },
    {
      id: 'rainfall',
      name: 'Rainfall',
      description: 'Gentle rain sounds for sleep and relaxation',
      benefits: ['Better sleep quality', 'Anxiety relief', 'Mood improvement', 'Mental calmness'],
      color: 'from-gray-500 to-blue-500',
      audioUrl: 'https://myckworfnrmkkhzpagto.supabase.co/storage/v1/object/public/audios/Rain%20Sound.mp4',
      duration: '45:00',
    },
    {
      id: 'babbling-brooks',
      name: 'Babbling Brooks',
      description: 'Peaceful stream sounds for meditation and stress relief',
      benefits: ['Stress relief', 'Meditation enhancement', 'Natural healing', 'Mental clarity'],
      color: 'from-blue-400 to-green-400',
      audioUrl: 'https://myckworfnrmkkhzpagto.supabase.co/storage/v1/object/public/audios/Brook%20sound.mp3',
      duration: '35:00',
    },
    // {
    //   id: 'forest-rivers',
    //   name: 'Forest Rivers',
    //   description: 'Flowing river sounds in a peaceful forest setting',
    //   benefits: ['Deep relaxation', 'Nature connection', 'Stress reduction', 'Peaceful sleep'],
    //   color: 'from-green-600 to-blue-600',
    //   audioUrl: 'https://cdn.pixabay.com/audio/2023/09/21/audio_4be4813c47.mp3',
    //   duration: '40:00',
    // },
    {
      id: 'Bells',
      name: 'Bells',
      description: 'Powerful Bells sounds for energy and focus',
      benefits: ['Increased energy', 'Enhanced focus', 'Natural white noise', 'Mood elevation'],
      color: 'from-cyan-500 to-blue-500',
      audioUrl: 'https://myckworfnrmkkhzpagto.supabase.co/storage/v1/object/public/audios/Temple%20Bell%20Sound.mp3',
      duration: '30:00',
    },
    {
      id: 'night-ambience',
      name: 'Night Ambience',
      description: 'Peaceful night sounds with crickets and gentle breeze',
      benefits: ['Better sleep', 'Relaxation', 'Anxiety reduction', 'Peaceful dreams'],
      color: 'from-blue-600 to-teal-600',
      audioUrl: 'https://myckworfnrmkkhzpagto.supabase.co/storage/v1/object/public/audios/Night%20ambience.mp3',
      duration: '60:00',
    },
  ];

  const currentSound = calmingSounds.find(sound => sound.id === soundId);
  const isCustomMode = soundId === 'custom';

  useEffect(() => {
    if (audioRef.current) {
      audioRef.current.volume = volume;
      audioRef.current.loop = isLooping;
    }
  }, [volume, isLooping]);

  useEffect(() => {
    const audio = audioRef.current;
    if (!audio) return;

    const updateTime = () => setCurrentTime(audio.currentTime);
    const updateDuration = () => setDuration(audio.duration);
    const handleEnded = () => setIsPlaying(false);

    audio.addEventListener('timeupdate', updateTime);
    audio.addEventListener('loadedmetadata', updateDuration);
    audio.addEventListener('ended', handleEnded);

    return () => {
      audio.removeEventListener('timeupdate', updateTime);
      audio.removeEventListener('loadedmetadata', updateDuration);
      audio.removeEventListener('ended', handleEnded);
    };
  }, [currentSound]);

  const togglePlayPause = () => {
    if (!audioRef.current) return;

    if (isPlaying) {
      audioRef.current.pause();
    } else {
      audioRef.current.play();
    }
    setIsPlaying(!isPlaying);
  };

  const handleSeek = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (!audioRef.current) return;
    const newTime = parseFloat(e.target.value);
    audioRef.current.currentTime = newTime;
    setCurrentTime(newTime);
  };

  const formatTime = (time: number) => {
    const minutes = Math.floor(time / 60);
    const seconds = Math.floor(time % 60);
    return `${minutes}:${seconds.toString().padStart(2, '0')}`;
  };

  const handleCustomSearch = () => {
    if (customSearch.trim()) {
      const searchQuery = encodeURIComponent(customSearch.trim() + ' relaxing sounds');
      window.open(`https://www.youtube.com/results?search_query=${searchQuery}`, '_blank');
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      handleCustomSearch();
    }
  };

  if (isCustomMode) {
    return (
      <div className="min-h-screen py-8 px-4 sm:px-6 lg:px-8">
        <div className="max-w-4xl mx-auto">
          {/* Back Button */}
          <motion.button
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            onClick={() => navigate(-1)}
            className="flex items-center space-x-2 text-gray-600 dark:text-gray-400 hover:text-blue-600 dark:hover:text-blue-400 mb-6 transition-colors"
          >
            <ArrowLeft className="w-5 h-5" />
            <span>Back</span>
          </motion.button>

          {/* Custom Search */}
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            className="bg-white dark:bg-gray-800 rounded-2xl shadow-xl p-8 border border-gray-100 dark:border-gray-700"
          >
            <div className="text-center mb-8">
              <div className="w-16 h-16 bg-gradient-to-r from-purple-600 to-blue-600 rounded-full flex items-center justify-center mx-auto mb-4">
                <Search className="w-8 h-8 text-white" />
              </div>
              <h1 className="text-3xl font-bold text-gray-900 dark:text-white mb-2">
                Custom Sound Search
              </h1>
              <p className="text-gray-600 dark:text-gray-400">
                Search for any calming sound and we'll find it on YouTube
              </p>
            </div>

            <div className="max-w-md mx-auto">
              <div className="relative mb-6">
                <input
                  type="text"
                  value={customSearch}
                  onChange={(e) => setCustomSearch(e.target.value)}
                  onKeyPress={handleKeyPress}
                  placeholder="Enter sound name (e.g., thunderstorm, fireplace, white noise)"
                  className="w-full px-4 py-3 pl-12 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white placeholder-gray-500 dark:placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                />
                <Search className="absolute left-4 top-1/2 transform -translate-y-1/2 w-5 h-5 text-gray-400" />
              </div>

              <motion.button
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
                onClick={handleCustomSearch}
                disabled={!customSearch.trim()}
                className="w-full py-3 px-6 bg-gradient-to-r from-purple-600 to-blue-600 text-white rounded-lg hover:from-purple-700 hover:to-blue-700 transition-all duration-300 flex items-center justify-center space-x-2 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                <ExternalLink className="w-5 h-5" />
                <span>Search on YouTube</span>
              </motion.button>
            </div>

            <div className="mt-8 p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg border border-blue-200 dark:border-blue-800">
              <p className="text-sm text-blue-700 dark:text-blue-300 text-center">
                <strong>Tip:</strong> Try searching for sounds like "thunderstorm", "fireplace crackling", 
                "white noise", "brown noise", or "forest ambience"
              </p>
            </div>
          </motion.div>
        </div>
      </div>
    );
  }

  if (!currentSound) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-center">
          <h1 className="text-2xl font-bold text-gray-900 dark:text-white mb-4">
            Sound Not Found
          </h1>
          <button
            onClick={() => navigate('/dashboard')}
            className="px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
          >
            Back to Dashboard
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen py-8 px-4 sm:px-6 lg:px-8">
      <div className="max-w-4xl mx-auto">
        {/* Back Button */}
        <motion.button
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          onClick={() => navigate(-1)}
          className="flex items-center space-x-2 text-gray-600 dark:text-gray-400 hover:text-blue-600 dark:hover:text-blue-400 mb-6 transition-colors"
        >
          <ArrowLeft className="w-5 h-5" />
          <span>Back</span>
        </motion.button>

        {/* Sound Player */}
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          className="bg-white dark:bg-gray-800 rounded-2xl shadow-xl overflow-hidden border border-gray-100 dark:border-gray-700"
        >
          {/* Header */}
          <div className={`bg-gradient-to-r ${currentSound.color} p-8 text-white`}>
            <div className="text-center">
              <h1 className="text-3xl font-bold mb-2">{currentSound.name}</h1>
              <p className="text-white/90 mb-4">{currentSound.description}</p>
              <div className="flex justify-center space-x-4 text-sm">
                <span className="flex items-center space-x-1">
                  <Timer className="w-4 h-4" />
                  <span>{currentSound.duration}</span>
                </span>
              </div>
            </div>
          </div>

          {/* Audio Player */}
          <div className="p-8">
            <audio
              ref={audioRef}
              src={currentSound.audioUrl}
              preload="metadata"
            />

            {/* Controls */}
            <div className="flex items-center justify-center space-x-6 mb-6">
              <motion.button
                whileHover={{ scale: 1.1 }}
                whileTap={{ scale: 0.9 }}
                onClick={togglePlayPause}
                className={`w-16 h-16 rounded-full bg-gradient-to-r ${currentSound.color} text-white flex items-center justify-center shadow-lg hover:shadow-xl transition-all duration-300`}
              >
                {isPlaying ? (
                  <Pause className="w-8 h-8" />
                ) : (
                  <Play className="w-8 h-8 ml-1" />
                )}
              </motion.button>

              <button
                onClick={() => setIsLooping(!isLooping)}
                className={`p-3 rounded-full transition-colors ${
                  isLooping
                    ? 'bg-blue-100 dark:bg-blue-900/20 text-blue-600 dark:text-blue-400'
                    : 'bg-gray-100 dark:bg-gray-700 text-gray-600 dark:text-gray-400'
                }`}
              >
                <Repeat className="w-5 h-5" />
              </button>
            </div>

            {/* Progress Bar */}
            <div className="mb-6">
              <input
                type="range"
                min="0"
                max={duration || 0}
                value={currentTime}
                onChange={handleSeek}
                className="w-full h-2 bg-gray-200 dark:bg-gray-700 rounded-lg appearance-none cursor-pointer"
              />
              <div className="flex justify-between text-sm text-gray-500 dark:text-gray-400 mt-2">
                <span>{formatTime(currentTime)}</span>
                <span>{formatTime(duration)}</span>
              </div>
            </div>

            {/* Volume Control */}
            <div className="flex items-center justify-center space-x-4 mb-8">
              <VolumeX className="w-5 h-5 text-gray-400" />
              <input
                type="range"
                min="0"
                max="1"
                step="0.1"
                value={volume}
                onChange={(e) => setVolume(parseFloat(e.target.value))}
                className="w-32 h-2 bg-gray-200 dark:bg-gray-700 rounded-lg appearance-none cursor-pointer"
              />
              <Volume2 className="w-5 h-5 text-gray-400" />
            </div>

            {/* Benefits */}
            <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-6">
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
                Benefits of {currentSound.name}
              </h3>
              <div className="grid grid-cols-2 gap-3">
                {currentSound.benefits.map((benefit, index) => (
                  <div
                    key={index}
                    className="flex items-center space-x-2 text-sm text-gray-600 dark:text-gray-400"
                  >
                    <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                    <span>{benefit}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </motion.div>

        {/* Usage Tips */}
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3 }}
          className="mt-8 bg-gradient-to-r from-blue-50 to-purple-50 dark:from-blue-900/20 dark:to-purple-900/20 rounded-xl p-6 border border-blue-200 dark:border-blue-800"
        >
          <h3 className="text-lg font-semibold text-blue-900 dark:text-blue-100 mb-4">
            Usage Tips
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm text-blue-800 dark:text-blue-200">
            <div>• Use headphones for the best experience</div>
            <div>• Start with lower volume and adjust as needed</div>
            <div>• Create a comfortable environment</div>
            <div>• Focus on your breathing while listening</div>
          </div>
        </motion.div>
      </div>
    </div>
  );
};

export default CalmingSounds;