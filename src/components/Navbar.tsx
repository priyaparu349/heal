import React from 'react';
import { Link, useNavigate, useLocation } from 'react-router-dom';
import { motion } from 'framer-motion';
import { Sun, Moon, User, LogOut, Heart, Menu, X, ChevronDown, Volume2 } from 'lucide-react';
import { useTheme } from '../contexts/ThemeContext';
import { useAuth } from '../contexts/AuthContext';
import { useState } from 'react';

const Navbar = () => {
  const { isDark, toggleTheme } = useTheme();
  const { user, logout } = useAuth();
  const navigate = useNavigate();
  const location = useLocation();
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);
  const [isCalmingDropdownOpen, setIsCalmingDropdownOpen] = useState(false);

  const handleLogout = () => {
    logout();
    navigate('/');
    setIsMobileMenuOpen(false);
  };

  const calmingSounds = [
    { name: 'Birds Chirping', id: 'birds-chirping' },
    { name: 'Wind Gushes', id: 'wind-gushes' },
    { name: 'Sea Waves', id: 'sea-waves' },
    { name: 'Om Chanting', id: 'om-chanting' },
    // { name: 'Ocean Waves', id: 'ocean-waves' },
    { name: 'Rainfall', id: 'rainfall' },
    { name: 'Babbling Brooks', id: 'babbling-brooks' },
    // { name: 'Forest Rivers', id: 'forest-rivers' },
    { name: 'Bells', id: 'Bells' },
    { name: 'Night Ambience', id: 'night-ambience' },
  ];

  const handleCalmingSoundSelect = (soundId: string) => {
    navigate(`/calming/${soundId}`);
    setIsCalmingDropdownOpen(false);
    setIsMobileMenuOpen(false);
  };

  const handleCustomSoundSelect = () => {
    navigate('/calming/custom');
    setIsCalmingDropdownOpen(false);
    setIsMobileMenuOpen(false);
  };
  const navItems = user ? [
    { path: '/dashboard', label: 'Dashboard' },
    { path: '/chat', label: 'AI Chat' },
    { path: '/frequency', label: 'Frequency' },
    { path: '/doctors', label: 'Doctors' },
    { path: '/history', label: 'History' },
  ] : [];

  const isActive = (path: string) => location.pathname === path;

  return (
    <nav className="bg-white/80 dark:bg-gray-900/80 backdrop-blur-md border-b border-gray-200 dark:border-gray-700 sticky top-0 z-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between items-center h-16">
          {/* Logo */}
          <Link to="/" className="flex items-center space-x-2">
            <motion.div
              whileHover={{ scale: 1.1 }}
              whileTap={{ scale: 0.95 }}
              className="flex items-center space-x-2"
            >
              <Heart className="w-8 h-8 text-blue-600 dark:text-blue-400" />
              <span className="text-xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
                SoundHeal
              </span>
            </motion.div>
          </Link>

          {/* Desktop Navigation */}
          <div className="hidden md:flex items-center space-x-6">
            {navItems.map((item) => (
              <Link
                key={item.path}
                to={item.path}
                className={`px-3 py-2 rounded-md text-sm font-medium transition-colors ${
                  isActive(item.path)
                    ? 'text-blue-600 dark:text-blue-400 bg-blue-50 dark:bg-blue-900/20'
                    : 'text-gray-600 dark:text-gray-300 hover:text-blue-600 dark:hover:text-blue-400'
                }`}
              >
                {item.label}
              </Link>
            ))}
            
            {/* Calming Frequencies Dropdown */}
            {user && (
              <div className="relative">
                <motion.button
                  whileHover={{ scale: 1.05 }}
                  onClick={() => setIsCalmingDropdownOpen(!isCalmingDropdownOpen)}
                  className="flex items-center space-x-1 px-3 py-2 rounded-md text-sm font-medium text-gray-600 dark:text-gray-300 hover:text-blue-600 dark:hover:text-blue-400 transition-colors"
                >
                  <Volume2 className="w-4 h-4" />
                  <span>Calming Frequencies</span>
                  <ChevronDown className={`w-4 h-4 transition-transform ${isCalmingDropdownOpen ? 'rotate-180' : ''}`} />
                </motion.button>
                
                {isCalmingDropdownOpen && (
                  <motion.div
                    initial={{ opacity: 0, y: -10 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="absolute right-0 mt-2 w-48 bg-white dark:bg-gray-800 rounded-lg shadow-lg border border-gray-200 dark:border-gray-700 py-2 z-50"
                  >
                    {calmingSounds.map((sound) => (
                      <button
                        key={sound.id}
                        onClick={() => handleCalmingSoundSelect(sound.id)}
                        className="w-full text-left px-4 py-2 text-sm text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors"
                      >
                        {sound.name}
                      </button>
                    ))}
                    <hr className="my-2 border-gray-200 dark:border-gray-600" />
                    <button
                      onClick={handleCustomSoundSelect}
                      className="w-full text-left px-4 py-2 text-sm text-blue-600 dark:text-blue-400 hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors font-medium"
                    >
                      Custom Sound
                    </button>
                  </motion.div>
                )}
              </div>
            )}
          </div>

          {/* Right side controls */}
          <div className="flex items-center space-x-4">
            {/* Theme toggle */}
            <motion.button
              whileHover={{ scale: 1.1 }}
              whileTap={{ scale: 0.95 }}
              onClick={toggleTheme}
              className="p-2 rounded-md text-gray-600 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors"
            >
              {isDark ? <Sun className="w-5 h-5" /> : <Moon className="w-5 h-5" />}
            </motion.button>

            {/* User menu */}
            {user ? (
              <div className="flex items-center space-x-3">
                <div className="hidden sm:flex items-center space-x-2">
                  <User className="w-5 h-5 text-gray-600 dark:text-gray-300" />
                  <span className="text-sm text-gray-700 dark:text-gray-300">
                    {user.name}
                  </span>
                </div>
                <motion.button
                  whileHover={{ scale: 1.1 }}
                  whileTap={{ scale: 0.95 }}
                  onClick={handleLogout}
                  className="p-2 rounded-md text-red-600 hover:bg-red-50 dark:hover:bg-red-900/20 transition-colors"
                >
                  <LogOut className="w-5 h-5" />
                </motion.button>
              </div>
            ) : (
              <div className="hidden sm:flex items-center space-x-3">
                <Link
                  to="/login"
                  className="px-4 py-2 text-sm font-medium text-blue-600 hover:text-blue-700 dark:text-blue-400 dark:hover:text-blue-300 transition-colors"
                >
                  Login
                </Link>
                <Link
                  to="/signup"
                  className="px-4 py-2 text-sm font-medium bg-blue-600 text-white rounded-md hover:bg-blue-700 transition-colors"
                >
                  Sign Up
                </Link>
              </div>
            )}

            {/* Mobile menu button */}
            <div className="md:hidden">
              <motion.button
                whileHover={{ scale: 1.1 }}
                whileTap={{ scale: 0.95 }}
                onClick={() => setIsMobileMenuOpen(!isMobileMenuOpen)}
                className="p-2 rounded-md text-gray-600 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors"
              >
                {isMobileMenuOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
              </motion.button>
            </div>
          </div>
        </div>

        {/* Mobile menu */}
        {isMobileMenuOpen && (
          <motion.div
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            className="md:hidden py-4 border-t border-gray-200 dark:border-gray-700"
          >
            <div className="flex flex-col space-y-2">
              {navItems.map((item) => (
                <Link
                  key={item.path}
                  to={item.path}
                  onClick={() => setIsMobileMenuOpen(false)}
                  className={`px-3 py-2 rounded-md text-sm font-medium transition-colors ${
                    isActive(item.path)
                      ? 'text-blue-600 dark:text-blue-400 bg-blue-50 dark:bg-blue-900/20'
                      : 'text-gray-600 dark:text-gray-300 hover:text-blue-600 dark:hover:text-blue-400'
                  }`}
                >
                  {item.label}
                </Link>
              ))}
              
              {/* Mobile Calming Frequencies */}
              {user && (
                <div className="pt-2 border-t border-gray-200 dark:border-gray-700">
                  <div className="px-3 py-2 text-sm font-medium text-gray-900 dark:text-white">
                    Calming Frequencies
                  </div>
                  {calmingSounds.map((sound) => (
                    <button
                      key={sound.id}
                      onClick={() => handleCalmingSoundSelect(sound.id)}
                      className="w-full text-left px-6 py-2 text-sm text-gray-600 dark:text-gray-300 hover:text-blue-600 dark:hover:text-blue-400 transition-colors"
                    >
                      {sound.name}
                    </button>
                  ))}
                  <button
                    onClick={handleCustomSoundSelect}
                    className="w-full text-left px-6 py-2 text-sm text-blue-600 dark:text-blue-400 font-medium transition-colors"
                  >
                    Custom Sound
                  </button>
                </div>
              )}
              {!user && (
                <div className="flex flex-col space-y-2 pt-2 border-t border-gray-200 dark:border-gray-700">
                  <Link
                    to="/login"
                    onClick={() => setIsMobileMenuOpen(false)}
                    className="px-3 py-2 text-sm font-medium text-blue-600 hover:text-blue-700 dark:text-blue-400 dark:hover:text-blue-300 transition-colors"
                  >
                    Login
                  </Link>
                  <Link
                    to="/signup"
                    onClick={() => setIsMobileMenuOpen(false)}
                    className="px-3 py-2 text-sm font-medium bg-blue-600 text-white rounded-md hover:bg-blue-700 transition-colors text-center"
                  >
                    Sign Up
                  </Link>
                </div>
              )}
            </div>
          </motion.div>
        )}
      </div>
    </nav>
  );
};

export default Navbar;