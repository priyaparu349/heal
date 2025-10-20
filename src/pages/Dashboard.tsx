import React from 'react';
import { Link } from 'react-router-dom';
import { motion } from 'framer-motion';
import { MessageCircle, Music, Stethoscope, History, TrendingUp, Clock, Heart, Brain, Activity, BarChart3 } from 'lucide-react';
import { useAuth } from '../contexts/AuthContext';

const Dashboard = () => {
  const { user } = useAuth();

  const quickActions = [
    {
      icon: Activity,
      title: 'Symptom Checker',
      description: 'AI-powered health assessment',
      path: '/symptom-checker',
      color: 'from-teal-600 to-teal-700',
      iconBg: 'bg-teal-100 dark:bg-teal-900/20',
      iconColor: 'text-teal-600 dark:text-teal-400',
    },
    {
      icon: MessageCircle,
      title: 'AI Health Chat',
      description: 'Get personalized health guidance',
      path: '/chat',
      color: 'from-blue-600 to-blue-700',
      iconBg: 'bg-blue-100 dark:bg-blue-900/20',
      iconColor: 'text-blue-600 dark:text-blue-400',
    },
    {
      icon: Music,
      title: 'Frequency Healing',
      description: 'Experience therapeutic sounds',
      path: '/frequency',
      color: 'from-purple-600 to-purple-700',
      iconBg: 'bg-purple-100 dark:bg-purple-900/20',
      iconColor: 'text-purple-600 dark:text-purple-400',
    },
    {
      icon: BarChart3,
      title: 'ML Evaluation',
      description: 'View model performance metrics',
      path: '/ml-evaluation',
      color: 'from-cyan-600 to-cyan-700',
      iconBg: 'bg-cyan-100 dark:bg-cyan-900/20',
      iconColor: 'text-cyan-600 dark:text-cyan-400',
    },
    {
      icon: Stethoscope,
      title: 'Find Doctors',
      description: 'Connect with healthcare professionals',
      path: '/doctors',
      color: 'from-green-600 to-green-700',
      iconBg: 'bg-green-100 dark:bg-green-900/20',
      iconColor: 'text-green-600 dark:text-green-400',
    },
    {
      icon: History,
      title: 'View History',
      description: 'Track your wellness journey',
      path: '/history',
      color: 'from-orange-600 to-orange-700',
      iconBg: 'bg-orange-100 dark:bg-orange-900/20',
      iconColor: 'text-orange-600 dark:text-orange-400',
    },
  ];

  const stats = [
    {
      icon: Heart,
      label: 'Healing Sessions',
      value: '24',
      change: '+3 this week',
      color: 'text-red-600 dark:text-red-400',
      bgColor: 'bg-red-100 dark:bg-red-900/20',
    },
    {
      icon: Brain,
      label: 'AI Health Chats',
      value: '18',
      change: '+2 today',
      color: 'text-blue-600 dark:text-blue-400',
      bgColor: 'bg-blue-100 dark:bg-blue-900/20',
    },
    {
      icon: Clock,
      label: 'Therapy Time',
      value: '12.5h',
      change: '+2.5h this week',
      color: 'text-purple-600 dark:text-purple-400',
      bgColor: 'bg-purple-100 dark:bg-purple-900/20',
    },
    {
      icon: TrendingUp,
      label: 'Active Sessions',
      value: '3',
      change: '+2 today',
      color: 'text-green-600 dark:text-green-400',
      bgColor: 'bg-green-100 dark:bg-green-900/20',
    },
  ];

  const recentActivity = [
    {
      type: 'frequency',
      title: 'Stress Relief Session',
      time: '2 hours ago',
      frequency: '432 Hz',
      duration: '15 min',
    },
    {
      type: 'chat',
      title: 'AI Health Consultation',
      time: '5 hours ago',
      topic: 'Sleep improvement',
      messages: 8,
    },
    {
      type: 'doctor',
      title: 'Dr. Sarah Chen',
      time: '1 day ago',
      specialty: 'Holistic Medicine',
      location: 'Downtown Medical',
    },
  ];

  return (
    <div className="min-h-screen py-8 px-4 sm:px-6 lg:px-8">
      <div className="max-w-7xl mx-auto">
        {/* Welcome Section */}
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
          className="mb-8"
        >
          <h1 className="text-3xl sm:text-4xl font-bold text-gray-900 dark:text-white mb-2">
            Welcome back, {user?.name}!
          </h1>
          <p className="text-lg text-gray-600 dark:text-gray-400">
            Ready to continue your wellness journey? Here's your personalized dashboard.
          </p>
        </motion.div>

        {/* Stats Grid */}
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.2 }}
          className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6 mb-8"
        >
          {stats.map((stat, index) => (
            <motion.div
              key={stat.label}
              whileHover={{ scale: 1.02, y: -2 }}
              className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-gray-100 dark:border-gray-700"
            >
              <div className="flex items-center justify-between mb-4">
                <div className={`${stat.bgColor} w-12 h-12 rounded-lg flex items-center justify-center`}>
                  <stat.icon className={`w-6 h-6 ${stat.color}`} />
                </div>
                <span className="text-sm text-green-600 dark:text-green-400 font-medium">
                  {stat.change}
                </span>
              </div>
              <h3 className="text-2xl font-bold text-gray-900 dark:text-white mb-1">
                {stat.value}
              </h3>
              <p className="text-gray-600 dark:text-gray-400 text-sm">
                {stat.label}
              </p>
            </motion.div>
          ))}
        </motion.div>

        {/* Quick Actions */}
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.4 }}
          className="mb-8"
        >
          <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-6">
            Quick Actions
          </h2>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6">
            {quickActions.map((action, index) => (
              <motion.div
                key={action.title}
                whileHover={{ scale: 1.05, y: -5 }}
                whileTap={{ scale: 0.95 }}
              >
                <Link
                  to={action.path}
                  className="block bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg hover:shadow-xl transition-all duration-300 border border-gray-100 dark:border-gray-700"
                >
                  <div className={`${action.iconBg} w-14 h-14 rounded-lg flex items-center justify-center mb-4`}>
                    <action.icon className={`w-7 h-7 ${action.iconColor}`} />
                  </div>
                  <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-2">
                    {action.title}
                  </h3>
                  <p className="text-gray-600 dark:text-gray-400 text-sm">
                    {action.description}
                  </p>
                </Link>
              </motion.div>
            ))}
          </div>
        </motion.div>

        {/* Recent Activity */}
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.6 }}
        >
          <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-6">
            Recent Activity
          </h2>
          <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg border border-gray-100 dark:border-gray-700">
            {recentActivity.map((activity, index) => (
              <div
                key={index}
                className={`p-6 ${index !== recentActivity.length - 1 ? 'border-b border-gray-100 dark:border-gray-700' : ''}`}
              >
                <div className="flex items-center justify-between">
                  <div>
                    <h3 className="font-semibold text-gray-900 dark:text-white mb-1">
                      {activity.title}
                    </h3>
                    <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">
                      {activity.time}
                    </p>
                    {activity.type === 'frequency' && (
                      <div className="flex items-center space-x-4 text-sm">
                        <span className="text-purple-600 dark:text-purple-400">
                          {activity.frequency}
                        </span>
                        <span className="text-gray-500">•</span>
                        <span className="text-gray-600 dark:text-gray-400">
                          {activity.duration}
                        </span>
                      </div>
                    )}
                    {activity.type === 'chat' && (
                      <div className="flex items-center space-x-4 text-sm">
                        <span className="text-blue-600 dark:text-blue-400">
                          {activity.topic}
                        </span>
                        <span className="text-gray-500">•</span>
                        <span className="text-gray-600 dark:text-gray-400">
                          {activity.messages} messages
                        </span>
                      </div>
                    )}
                    {activity.type === 'doctor' && (
                      <div className="flex items-center space-x-4 text-sm">
                        <span className="text-green-600 dark:text-green-400">
                          {activity.specialty}
                        </span>
                        <span className="text-gray-500">•</span>
                        <span className="text-gray-600 dark:text-gray-400">
                          {activity.location}
                        </span>
                      </div>
                    )}
                  </div>
                  <div className="flex-shrink-0">
                    {activity.type === 'frequency' && (
                      <div className="w-10 h-10 bg-purple-100 dark:bg-purple-900/20 rounded-full flex items-center justify-center">
                        <Music className="w-5 h-5 text-purple-600 dark:text-purple-400" />
                      </div>
                    )}
                    {activity.type === 'chat' && (
                      <div className="w-10 h-10 bg-blue-100 dark:bg-blue-900/20 rounded-full flex items-center justify-center">
                        <MessageCircle className="w-5 h-5 text-blue-600 dark:text-blue-400" />
                      </div>
                    )}
                    {activity.type === 'doctor' && (
                      <div className="w-10 h-10 bg-green-100 dark:bg-green-900/20 rounded-full flex items-center justify-center">
                        <Stethoscope className="w-5 h-5 text-green-600 dark:text-green-400" />
                      </div>
                    )}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </motion.div>
      </div>
    </div>
  );
};

export default Dashboard;