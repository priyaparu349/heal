import React, { createContext, useContext, useState, useRef } from 'react';

// 🔥 NO IMPORTS NEEDED FOR WEB AUDIO API! 
// It's built into the browser as a native JavaScript API

// Web Audio API Implementation for Therapeutic Frequency Generation
// This uses browser-native audio synthesis - no external sound files needed

interface AudioContextType {
  isPlaying: boolean;
  currentFrequency: number | null;
  playFrequency: (frequency: number, duration?: number) => void;
  stopFrequency: () => void;
  volume: number;
  setVolume: (volume: number) => void;
}

const AudioContext = createContext<AudioContextType | undefined>(undefined);

export const useAudio = () => {
  const context = useContext(AudioContext);
  if (!context) {
    throw new Error('useAudio must be used within an AudioProvider');
  }
  return context;
};

export const AudioProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentFrequency, setCurrentFrequency] = useState<number | null>(null);
  const [volume, setVolume] = useState(0.3);
  
  const audioContextRef = useRef<AudioContext | null>(null);
  const oscillatorRef = useRef<OscillatorNode | null>(null);
  const gainNodeRef = useRef<GainNode | null>(null);

  const playFrequency = (frequency: number, duration?: number) => {
    stopFrequency(); // Stop any current frequency
    
    try {
      // 1. CREATE AUDIO CONTEXT (Browser's audio engine)
      // Create or resume audio context
      if (!audioContextRef.current || audioContextRef.current.state === 'closed') {
        // 👆 THIS IS THE WEB AUDIO API - Built into the browser!
        // window.AudioContext is a native browser constructor
        audioContextRef.current = new window.AudioContext();
      }
      
      if (audioContextRef.current.state === 'suspended') {
        audioContextRef.current.resume();
      }

      const audioContext = audioContextRef.current;
      
      // 2. CREATE AUDIO NODES
      // Oscillator = Sound wave generator
      // Gain = Volume controller
      // Create oscillator and gain nodes
      // 👇 These are also native Web Audio API methods
      const oscillator = audioContext.createOscillator();
      const gainNode = audioContext.createGain();
      
      // 3. CONFIGURE OSCILLATOR (Sound wave properties)
      // Configure oscillator
      oscillator.type = 'sine';
      oscillator.frequency.setValueAtTime(frequency, audioContext.currentTime);
      
      // 4. CONFIGURE VOLUME WITH SMOOTH FADE-IN
      // Configure gain (volume)
      gainNode.gain.setValueAtTime(0, audioContext.currentTime);
      gainNode.gain.linearRampToValueAtTime(volume, audioContext.currentTime + 0.1);
      
      // 5. CONNECT AUDIO PIPELINE
      // Oscillator → Volume Control → Speakers
      // Connect nodes
      oscillator.connect(gainNode);
      gainNode.connect(audioContext.destination);
      
      // 6. START PLAYING
      // Start oscillator
      oscillator.start();
      
      // Store references
      oscillatorRef.current = oscillator;
      gainNodeRef.current = gainNode;
      
      setIsPlaying(true);
      setCurrentFrequency(frequency);

      // Auto-stop after duration if specified
      if (duration) {
        setTimeout(() => {
          stopFrequency();
        }, duration * 1000);
      }

      // Handle oscillator end
      oscillator.onended = () => {
        setIsPlaying(false);
        setCurrentFrequency(null);
      };

    } catch (error) {
      console.error('Error playing frequency:', error);
    }
  };

  const stopFrequency = () => {
    if (oscillatorRef.current && gainNodeRef.current) {
      try {
        // Fade out
        const audioContext = audioContextRef.current!;
        gainNodeRef.current.gain.linearRampToValueAtTime(0, audioContext.currentTime + 0.1);
        
        // Stop after fade
        setTimeout(() => {
          if (oscillatorRef.current) {
            oscillatorRef.current.stop();
            oscillatorRef.current = null;
            gainNodeRef.current = null;
          }
        }, 100);
        
        setIsPlaying(false);
        setCurrentFrequency(null);
      } catch (error) {
        console.error('Error stopping frequency:', error);
      }
    }
  };

  const handleVolumeChange = (newVolume: number) => {
    setVolume(newVolume);
    if (gainNodeRef.current && audioContextRef.current) {
      gainNodeRef.current.gain.setValueAtTime(newVolume, audioContextRef.current.currentTime);
    }
  };

  return (
    <AudioContext.Provider value={{
      isPlaying,
      currentFrequency,
      playFrequency,
      stopFrequency,
      volume,
      setVolume: handleVolumeChange
    }}>
      {children}
    </AudioContext.Provider>
  );
};