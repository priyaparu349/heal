import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { ThemeProvider } from './contexts/ThemeContext';
import { AuthProvider } from './contexts/AuthContext';
import { AudioProvider } from './contexts/AudioContext';
import Navbar from './components/Navbar';
import Home from './pages/Home';
import Login from './pages/Login';
import Signup from './pages/Signup';
import Dashboard from './pages/Dashboard';
import Chat from './pages/Chat';
import FrequencyGenerator from './pages/FrequencyGenerator';
import DoctorFinder from './pages/DoctorFinder';
import History from './pages/History';
import CalmingSounds from './pages/CalmingSounds';
import MLEvaluation from './pages/MLEvaluation';
import SymptomChecker from './pages/SymptomChecker';
import ProtectedRoute from './components/ProtectedRoute';

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      refetchOnWindowFocus: false,
      retry: 1,
    },
  },
});

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <ThemeProvider>
        <AuthProvider>
          <AudioProvider>
            <Router>
              <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-purple-50 dark:from-gray-900 dark:via-gray-800 dark:to-purple-900 transition-colors duration-300">
                <Navbar />
                <Routes>
                  <Route path="/" element={<Home />} />
                  <Route path="/login" element={<Login />} />
                  <Route path="/signup" element={<Signup />} />
                  <Route path="/dashboard" element={
                    <ProtectedRoute>
                      <Dashboard />
                    </ProtectedRoute>
                  } />
                  <Route path="/chat" element={
                    <ProtectedRoute>
                      <Chat />
                    </ProtectedRoute>
                  } />
                  <Route path="/frequency" element={
                    <ProtectedRoute>
                      <FrequencyGenerator />
                    </ProtectedRoute>
                  } />
                  <Route path="/doctors" element={
                    <ProtectedRoute>
                      <DoctorFinder />
                    </ProtectedRoute>
                  } />
                  <Route path="/history" element={
                    <ProtectedRoute>
                      <History />
                    </ProtectedRoute>
                  } />
                  <Route path="/calming/:soundId" element={
                    <ProtectedRoute>
                      <CalmingSounds />
                    </ProtectedRoute>
                  } />
                  <Route path="/ml-evaluation" element={
                    <ProtectedRoute>
                      <MLEvaluation />
                    </ProtectedRoute>
                  } />
                  <Route path="/symptom-checker" element={
                    <ProtectedRoute>
                      <SymptomChecker />
                    </ProtectedRoute>
                  } />
                </Routes>
              </div>
            </Router>
          </AudioProvider>
        </AuthProvider>
      </ThemeProvider>
    </QueryClientProvider>
  );
}

export default App;