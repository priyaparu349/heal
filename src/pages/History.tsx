import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { Calendar, Download, Filter, MessageCircle, Music, Stethoscope, Clock, TrendingUp, BarChart3, Activity } from 'lucide-react';
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, PieChart, Pie, Cell } from 'recharts';

interface HistoryItem {
  id: string;
  type: 'chat' | 'frequency' | 'doctor';
  title: string;
  date: Date;
  duration?: number;
  details: any;
}

const History = () => {
  const [selectedFilter, setSelectedFilter] = useState('all');
  const [selectedTimeRange, setSelectedTimeRange] = useState('30days');

  const mockHistory: HistoryItem[] = [
    {
      id: '1',
      type: 'frequency',
      title: 'Stress Relief Session',
      date: new Date(2025, 0, 15, 14, 30),
      duration: 15,
      details: {
        frequency: 432,
        preset: 'Stress Relief',
        volume: 0.3,
        completed: true,
      },
    },
    {
      id: '2',
      type: 'chat',
      title: 'Sleep Issues Consultation',
      date: new Date(2025, 0, 14, 20, 15),
      duration: 8,
      details: {
        messages: 12,
        topic: 'Sleep improvement',
        aiSuggestions: ['Sleep hygiene', 'Relaxation techniques', '432 Hz frequency'],
      },
    },
    {
      id: '3',
      type: 'doctor',
      title: 'Dr. Sarah Chen - Holistic Medicine',
      date: new Date(2025, 0, 12, 10, 0),
      details: {
        doctor: 'Dr. Sarah Chen',
        specialty: 'Holistic Medicine',
        reason: 'Sound therapy consultation',
        status: 'Completed',
      },
    },
    {
      id: '4',
      type: 'frequency',
      title: 'DNA Repair Session',
      date: new Date(2025, 0, 10, 16, 45),
      duration: 20,
      details: {
        frequency: 528,
        preset: 'DNA Repair',
        volume: 0.4,
        completed: true,
      },
    },
    {
      id: '5',
      type: 'chat',
      title: 'Anxiety Management',
      date: new Date(2025, 0, 8, 19, 30),
      duration: 12,
      details: {
        messages: 18,
        topic: 'Anxiety relief',
        aiSuggestions: ['Breathing exercises', 'Meditation', '40 Hz frequency'],
      },
    },
  ];

  // Calculate weekly activity from actual history data
  const calculateWeeklyActivity = () => {
    const today = new Date();
    const weekDays = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'];
    const weeklyData = [];

    // Get last 7 days
    for (let i = 6; i >= 0; i--) {
      const date = new Date(today);
      date.setDate(today.getDate() - i);
      const dayName = weekDays[date.getDay()];
      
      // Filter sessions for this specific day
      const dayHistory = mockHistory.filter(item => {
        const itemDate = new Date(item.date);
        return itemDate.toDateString() === date.toDateString();
      });
      
      // Count sessions by type
      const frequencySessions = dayHistory.filter(item => item.type === 'frequency').length;
      const chatSessions = dayHistory.filter(item => item.type === 'chat').length;
      const doctorSessions = dayHistory.filter(item => item.type === 'doctor').length;
      const total = frequencySessions + chatSessions + doctorSessions;
      
      weeklyData.push({
        day: dayName,
        frequency: frequencySessions,
        chat: chatSessions,
        doctor: doctorSessions,
        total: total,
        date: date.toISOString().split('T')[0] // For tooltip
      });
    }
    
    return weeklyData;
  };

  const weeklySessionsData = calculateWeeklyActivity();
  // Calculate session type distribution from actual data
  const calculateSessionTypeData = () => {
    const frequencyCount = mockHistory.filter(item => item.type === 'frequency').length;
    const chatCount = mockHistory.filter(item => item.type === 'chat').length;
    const doctorCount = mockHistory.filter(item => item.type === 'doctor').length;
    const total = frequencyCount + chatCount + doctorCount;
    
    return [
      { 
        name: 'Frequency Sessions', 
        value: total > 0 ? Math.round((frequencyCount / total) * 100) : 0, 
        count: frequencyCount,
        color: '#8B5CF6' 
      },
      { 
        name: 'AI Consultations', 
        value: total > 0 ? Math.round((chatCount / total) * 100) : 0, 
        count: chatCount,
        color: '#3B82F6' 
      },
      { 
        name: 'Doctor Visits', 
        value: total > 0 ? Math.round((doctorCount / total) * 100) : 0, 
        count: doctorCount,
        color: '#10B981' 
      },
    ];
  };

  const sessionTypeData = calculateSessionTypeData();
  const filteredHistory = mockHistory.filter(item => {
    if (selectedFilter === 'all') return true;
    return item.type === selectedFilter;
  });

  const getTypeIcon = (type: string) => {
    switch (type) {
      case 'chat':
        return MessageCircle;
      case 'frequency':
        return Music;
      case 'doctor':
        return Stethoscope;
      default:
        return Clock;
    }
  };

  const getTypeColor = (type: string) => {
    switch (type) {
      case 'chat':
        return 'from-blue-600 to-blue-700';
      case 'frequency':
        return 'from-purple-600 to-purple-700';
      case 'doctor':
        return 'from-green-600 to-green-700';
      default:
        return 'from-gray-600 to-gray-700';
    }
  };

  const exportHistory = () => {
    const dataStr = JSON.stringify(filteredHistory, null, 2);
    const dataUri = 'data:application/json;charset=utf-8,'+ encodeURIComponent(dataStr);
    
    const exportFileDefaultName = `soundheal-history-${new Date().toISOString().split('T')[0]}.json`;
    
    const linkElement = document.createElement('a');
    linkElement.setAttribute('href', dataUri);
    linkElement.setAttribute('download', exportFileDefaultName);
    linkElement.click();
  };

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
              <Activity className="w-8 h-8 text-white" />
            </div>
          </div>
          <h1 className="text-3xl sm:text-4xl font-bold text-gray-900 dark:text-white mb-2">
            Wellness History
          </h1>
          <p className="text-lg text-gray-600 dark:text-gray-400">
            Track your healing journey and monitor your progress
          </p>
        </motion.div>

        {/* Stats Overview */}
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6 mb-8"
        >
          <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-gray-100 dark:border-gray-700">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-gray-600 dark:text-gray-400">Total Sessions</p>
                <p className="text-2xl font-bold text-gray-900 dark:text-white">127</p>
              </div>
              <div className="w-12 h-12 bg-purple-100 dark:bg-purple-900/20 rounded-lg flex items-center justify-center">
                <BarChart3 className="w-6 h-6 text-purple-600 dark:text-purple-400" />
              </div>
            </div>
          </div>
          <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-gray-100 dark:border-gray-700">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-gray-600 dark:text-gray-400">This Week</p>
                <p className="text-2xl font-bold text-gray-900 dark:text-white">8</p>
              </div>
              <div className="w-12 h-12 bg-blue-100 dark:bg-blue-900/20 rounded-lg flex items-center justify-center">
                <TrendingUp className="w-6 h-6 text-blue-600 dark:text-blue-400" />
              </div>
            </div>
          </div>
          <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-gray-100 dark:border-gray-700">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-gray-600 dark:text-gray-400">Total Time</p>
                <p className="text-2xl font-bold text-gray-900 dark:text-white">24.5h</p>
              </div>
              <div className="w-12 h-12 bg-green-100 dark:bg-green-900/20 rounded-lg flex items-center justify-center">
                <Clock className="w-6 h-6 text-green-600 dark:text-green-400" />
              </div>
            </div>
          </div>
          <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-gray-100 dark:border-gray-700">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-gray-600 dark:text-gray-400">Streak</p>
                <p className="text-2xl font-bold text-gray-900 dark:text-white">12 days</p>
              </div>
              <div className="w-12 h-12 bg-orange-100 dark:bg-orange-900/20 rounded-lg flex items-center justify-center">
                <Calendar className="w-6 h-6 text-orange-600 dark:text-orange-400" />
              </div>
            </div>
          </div>
        </motion.div>

        {/* Charts Section */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-8">
          {/* Weekly Sessions Chart */}
          <motion.div
            initial={{ opacity: 0, x: -30 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.4 }}
            className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-gray-100 dark:border-gray-700"
          >
            <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
              Weekly Activity
            </h3>
            <ResponsiveContainer width="100%" height={200}>
              <AreaChart data={weeklySessionsData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="day" />
                <YAxis />
                <Tooltip />
                <Area type="monotone" dataKey="frequency" stackId="1" stroke="#8B5CF6" fill="#8B5CF6" fillOpacity={0.6} />
                <Area type="monotone" dataKey="chat" stackId="1" stroke="#3B82F6" fill="#3B82F6" fillOpacity={0.6} />
                <Area type="monotone" dataKey="doctor" stackId="1" stroke="#10B981" fill="#10B981" fillOpacity={0.6} />
              </AreaChart>
            </ResponsiveContainer>
          </motion.div>

          {/* Session Types Chart */}
          <motion.div
            initial={{ opacity: 0, x: 30 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.6 }}
            className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-gray-100 dark:border-gray-700"
          >
            <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
              Session Distribution
            </h3>
            <ResponsiveContainer width="100%" height={200}>
              <PieChart>
                <Pie
                  data={sessionTypeData}
                  cx="50%"
                  cy="50%"
                  outerRadius={80}
                  fill="#8884d8"
                  dataKey="value"
                  label={({ name, value, count }) => `${name}: ${count}`}
                >
                  {sessionTypeData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </motion.div>
        </div>

        {/* Filters and Controls */}
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.8 }}
          className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-gray-100 dark:border-gray-700 mb-8"
        >
          <div className="flex flex-col sm:flex-row items-center justify-between gap-4">
            <div className="flex items-center space-x-4">
              <Filter className="w-5 h-5 text-gray-400" />
              <select
                value={selectedFilter}
                onChange={(e) => setSelectedFilter(e.target.value)}
                className="px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                <option value="all">All Activities</option>
                <option value="frequency">Frequency Sessions</option>
                <option value="chat">AI Consultations</option>
                <option value="doctor">Doctor Visits</option>
              </select>
              <select
                value={selectedTimeRange}
                onChange={(e) => setSelectedTimeRange(e.target.value)}
                className="px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                <option value="7days">Last 7 Days</option>
                <option value="30days">Last 30 Days</option>
                <option value="90days">Last 3 Months</option>
                <option value="all">All Time</option>
              </select>
            </div>
            <motion.button
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              onClick={exportHistory}
              className="px-4 py-2 bg-gradient-to-r from-blue-600 to-purple-600 text-white rounded-lg hover:from-blue-700 hover:to-purple-700 transition-all duration-300 flex items-center space-x-2"
            >
              <Download className="w-4 h-4" />
              <span>Export History</span>
            </motion.button>
          </div>
        </motion.div>

        {/* History List */}
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 1.0 }}
          className="bg-white dark:bg-gray-800 rounded-xl shadow-lg border border-gray-100 dark:border-gray-700"
        >
          <div className="p-6">
            <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-6">
              Recent Activity
            </h3>
            <div className="space-y-4">
              {filteredHistory.map((item, index) => {
                const IconComponent = getTypeIcon(item.type);
                const colorClasses = getTypeColor(item.type);
                
                return (
                  <motion.div
                    key={item.id}
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: index * 0.1 }}
                    className="flex items-center justify-between p-4 bg-gray-50 dark:bg-gray-700 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-600 transition-colors"
                  >
                    <div className="flex items-center space-x-4">
                      <div className={`w-10 h-10 bg-gradient-to-r ${colorClasses} rounded-full flex items-center justify-center`}>
                        <IconComponent className="w-5 h-5 text-white" />
                      </div>
                      <div>
                        <h4 className="font-medium text-gray-900 dark:text-white">
                          {item.title}
                        </h4>
                        <div className="flex items-center space-x-4 text-sm text-gray-600 dark:text-gray-400">
                          <span>{item.date.toLocaleDateString()}</span>
                          <span>{item.date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</span>
                          {item.duration && (
                            <span>{item.duration} min</span>
                          )}
                        </div>
                        
                        {/* Type-specific details */}
                        {item.type === 'frequency' && (
                          <div className="flex items-center space-x-2 mt-1">
                            <span className="px-2 py-1 bg-purple-100 dark:bg-purple-900/20 text-purple-700 dark:text-purple-400 text-xs rounded-full">
                              {item.details.frequency} Hz
                            </span>
                            <span className="px-2 py-1 bg-gray-100 dark:bg-gray-600 text-gray-700 dark:text-gray-300 text-xs rounded-full">
                              {item.details.preset}
                            </span>
                          </div>
                        )}
                        
                        {item.type === 'chat' && (
                          <div className="flex items-center space-x-2 mt-1">
                            <span className="px-2 py-1 bg-blue-100 dark:bg-blue-900/20 text-blue-700 dark:text-blue-400 text-xs rounded-full">
                              {item.details.messages} messages
                            </span>
                            <span className="px-2 py-1 bg-gray-100 dark:bg-gray-600 text-gray-700 dark:text-gray-300 text-xs rounded-full">
                              {item.details.topic}
                            </span>
                          </div>
                        )}
                        
                        {item.type === 'doctor' && (
                          <div className="flex items-center space-x-2 mt-1">
                            <span className="px-2 py-1 bg-green-100 dark:bg-green-900/20 text-green-700 dark:text-green-400 text-xs rounded-full">
                              {item.details.specialty}
                            </span>
                            <span className="px-2 py-1 bg-gray-100 dark:bg-gray-600 text-gray-700 dark:text-gray-300 text-xs rounded-full">
                              {item.details.status}
                            </span>
                          </div>
                        )}
                      </div>
                    </div>
                  </motion.div>
                );
              })}
            </div>
          </div>
        </motion.div>
      </div>
    </div>
  );
};

export default History;