import React, { useState, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Send, Bot, User, Loader2, Lightbulb, Heart, Brain, AlertTriangle } from 'lucide-react';
import { useAuth } from '../contexts/AuthContext';

interface Message {
  id: string;
  content: string;
  sender: 'user' | 'ai';
  timestamp: Date;
  type?: 'text' | 'suggestion';
}

const Chat = () => {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: '1',
      content: 'Hello! I\'m your AI health companion powered by Google Gemini. I\'m here to help you with health-related questions, provide guidance on symptoms, and suggest therapeutic approaches. How can I assist you with your health today?',
      sender: 'ai',
      timestamp: new Date(),
    }
  ]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isTyping, setIsTyping] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const { user } = useAuth();

  const suggestions = [
    'I have trouble sleeping',
    'I feel stressed lately',
    'I have a headache',
    'I want to improve my mood',
    'I have digestive issues',
    'I need help with anxiety'
  ];

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isTyping]);

  const isHealthRelated = (message: string): boolean => {
    const healthKeywords = [
      'health', 'medical', 'doctor', 'symptom', 'pain', 'ache', 'sick', 'illness', 'disease',
      'medicine', 'medication', 'treatment', 'therapy', 'healing', 'wellness', 'fitness',
      'diet', 'nutrition', 'exercise', 'sleep', 'stress', 'anxiety', 'depression', 'mental',
      'physical', 'body', 'mind', 'heart', 'blood', 'pressure', 'diabetes', 'cancer',
      'infection', 'virus', 'bacteria', 'fever', 'cold', 'flu', 'cough', 'headache',
      'stomach', 'digestive', 'nausea', 'fatigue', 'tired', 'energy', 'weight', 'skin',
      'allergy', 'asthma', 'breathing', 'chest', 'back', 'joint', 'muscle', 'bone',
      'injury', 'wound', 'cut', 'burn', 'rash', 'swelling', 'inflammation', 'chronic',
      'acute', 'diagnosis', 'prognosis', 'recovery', 'rehabilitation', 'prevention',
      'vaccine', 'immunization', 'checkup', 'screening', 'test', 'lab', 'x-ray',
      'mri', 'ct', 'scan', 'surgery', 'operation', 'hospital', 'clinic', 'emergency',
      'first aid', 'wound care', 'bandage', 'prescription', 'dosage', 'side effect',
      'contraindication', 'allergy', 'reaction', 'overdose', 'addiction', 'withdrawal',
      'rehabilitation', 'recovery', 'relapse', 'remission', 'cure', 'heal', 'better',
      'worse', 'improve', 'deteriorate', 'stable', 'critical', 'serious', 'mild',
      'severe', 'moderate', 'chronic', 'acute', 'sudden', 'gradual', 'persistent',
      'intermittent', 'frequent', 'rare', 'common', 'unusual', 'normal', 'abnormal'
    ];
    
    const lowerMessage = message.toLowerCase();
    return healthKeywords.some(keyword => lowerMessage.includes(keyword));
  };

  const generateAIResponse = async (userMessage: string): Promise<string> => {
    // Check if the message is health-related
    if (!isHealthRelated(userMessage)) {
      return `I'm specifically designed to help with health and wellness-related questions. Please ask me about:

🏥 **Health Concerns**: Symptoms, conditions, or medical questions
💊 **Wellness**: Nutrition, exercise, sleep, or stress management  
🧠 **Mental Health**: Anxiety, depression, or emotional wellbeing
🎵 **Sound Therapy**: How frequencies can support your healing journey

How can I help you with your health today?`;
    }

    try {
      const response = await fetch('https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key=AIzaSyCcquOR2BVTRdPZitzsY7sRMkiIeEAhEro', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          contents: [{
            parts: [{
              text: `You are a professional AI health assistant for a sound frequency healing app. The user asked: "${userMessage}".

Please analyze this health query and provide a structured response in the following JSON format:

{
  "symptoms_detected": ["list of symptoms mentioned or implied"],
  "possible_causes": ["potential underlying causes or conditions"],
  "treatment_suggestions": ["general wellness recommendations and self-care tips"],
  "healing_frequency": "specific frequency in Hz that may help (e.g., 432, 528, 174, 285, 396, 417, 528, 639, 741, 852, 963)",
  "disclaimer": "This is not a medical diagnosis. Please consult a healthcare professional for proper treatment."
}

Guidelines:
1. Be empathetic and supportive in your analysis
2. Only suggest general wellness practices, not specific medical treatments
3. Always include the safety disclaimer exactly as shown
4. Choose healing frequencies based on sound therapy research:
   - 174 Hz: Pain relief, natural anesthetic
   - 285 Hz: Tissue healing, cellular repair
   - 396 Hz: Fear release, guilt removal
   - 417 Hz: Change facilitation, trauma healing
   - 432 Hz: Stress relief, natural healing
   - 528 Hz: DNA repair, love frequency
   - 639 Hz: Relationship harmony, heart healing
   - 741 Hz: Expression, problem solving
   - 852 Hz: Spiritual awakening, intuition
   - 963 Hz: Crown chakra, higher consciousness
5. If the condition seems serious, emphasize seeking immediate medical attention
6. Keep each field concise but informative

Respond ONLY with the JSON structure, no additional text.`
            }]
          }],
          generationConfig: {
            temperature: 0.7,
            topK: 40,
            topP: 0.95,
            maxOutputTokens: 1024,
          },
          safetySettings: [
            {
              category: "HARM_CATEGORY_HARASSMENT",
              threshold: "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
              category: "HARM_CATEGORY_HATE_SPEECH", 
              threshold: "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
              category: "HARM_CATEGORY_SEXUALLY_EXPLICIT",
              threshold: "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
              category: "HARM_CATEGORY_DANGEROUS_CONTENT",
              threshold: "BLOCK_MEDIUM_AND_ABOVE"
            }
          ]
        })
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      
      if (data.candidates && data.candidates[0] && data.candidates[0].content) {
        const rawResponse = data.candidates[0].content.parts[0].text;
        
        try {
          // Strip markdown code block delimiters before parsing
          const cleanedResponse = rawResponse
            .replace(/```json\s*/g, '')
            .replace(/```\s*/g, '')
            .trim();
          
          // Try to parse the cleaned JSON response
          const structuredResponse = JSON.parse(cleanedResponse);
          
          // Format the structured response for display
          return formatStructuredResponse(structuredResponse);
        } catch (parseError) {
          console.error('Error parsing structured response:', parseError);
          // Fallback to raw response if JSON parsing fails
          return rawResponse;
        }
      } else {
        throw new Error('Invalid response format from Gemini API');
      }
    } catch (error) {
      console.error('Error calling Gemini API:', error);
      
      // Fallback response
      return `I apologize, but I'm having trouble connecting to my knowledge base right now. Here's some general guidance:

🏥 **For your health concern**: Consider consulting with a healthcare professional for proper evaluation and treatment.

🎵 **Sound Therapy Support**: While you wait, you might find our frequency healing sessions helpful for relaxation and stress relief. Try:
- 432 Hz for general stress relief
- 528 Hz for emotional healing
- 174 Hz for pain relief

⚠️ **Important**: If you're experiencing severe symptoms or this is an emergency, please seek immediate medical attention.

Please try asking your question again, or contact our support team if the issue persists.`;
    }
  };

  const formatStructuredResponse = (response: any) => {
    const {
      symptoms_detected = [],
      possible_causes = [],
      treatment_suggestions = [],
      healing_frequency = '432',
      disclaimer = 'This is not a medical diagnosis. Please consult a healthcare professional for proper treatment.'
    } = response;

    return `## 🔍 **Analysis Summary**

### **Symptoms Detected:**
${symptoms_detected.map((symptom: string) => `• ${symptom}`).join('\n')}

### **Possible Causes:**
${possible_causes.map((cause: string) => `• ${cause}`).join('\n')}

### **Wellness Recommendations:**
${treatment_suggestions.map((suggestion: string) => `• ${suggestion}`).join('\n')}

### **🎵 Recommended Healing Frequency:**
**${healing_frequency} Hz** - Try this frequency in our Frequency Generator for potential therapeutic benefits.

### **⚠️ Important Disclaimer:**
${disclaimer}

---

💡 **Next Steps:**
1. Visit our **Frequency Generator** to try the recommended ${healing_frequency} Hz frequency
2. Consider consulting with a healthcare professional for proper evaluation
3. If symptoms are severe or worsening, seek immediate medical attention`;
  };

  const handleSendMessage = async (content: string) => {
    if (!content.trim()) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      content: content.trim(),
      sender: 'user',
      timestamp: new Date(),
    };

    setMessages(prev => [...prev, userMessage]);
    setInputValue('');
    setIsLoading(true);
    setIsTyping(true);

    try {
      const aiResponse = await generateAIResponse(content);
      
      const aiMessage: Message = {
        id: (Date.now() + 1).toString(),
        content: aiResponse,
        sender: 'ai',
        timestamp: new Date(),
      };

      setIsTyping(false);
      setMessages(prev => [...prev, aiMessage]);
    } catch (error) {
      console.error('Error generating AI response:', error);
      setIsTyping(false);
      
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        content: 'I apologize, but I encountered an error while processing your message. Please try again or contact support if the issue persists.',
        sender: 'ai',
        timestamp: new Date(),
      };
      
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    handleSendMessage(inputValue);
  };

  const handleSuggestionClick = (suggestion: string) => {
    handleSendMessage(suggestion);
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
            <div className="bg-gradient-to-r from-blue-600 to-purple-600 w-16 h-16 rounded-full flex items-center justify-center">
              <Brain className="w-8 h-8 text-white" />
            </div>
          </div>
          <h1 className="text-3xl font-bold text-gray-900 dark:text-white mb-2">
            AI Health Companion
          </h1>
          <p className="text-gray-600 dark:text-gray-400">
            Powered by Google Gemini • Get personalized health guidance and wellness support
          </p>
        </motion.div>

        {/* Chat Container */}
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          className="bg-white dark:bg-gray-800 rounded-2xl shadow-xl border border-gray-100 dark:border-gray-700 overflow-hidden"
        >
          {/* Messages Area */}
          <div className="h-96 sm:h-[500px] overflow-y-auto p-6 space-y-4">
            <AnimatePresence>
              {messages.map((message) => (
                <motion.div
                  key={message.id}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.3 }}
                  className={`flex items-start space-x-3 ${
                    message.sender === 'user' ? 'flex-row-reverse space-x-reverse' : ''
                  }`}
                >
                  {/* Avatar */}
                  <div className={`flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center ${
                    message.sender === 'user' 
                      ? 'bg-blue-600' 
                      : 'bg-gradient-to-r from-purple-600 to-blue-600'
                  }`}>
                    {message.sender === 'user' ? (
                      <User className="w-5 h-5 text-white" />
                    ) : (
                      <Bot className="w-5 h-5 text-white" />
                    )}
                  </div>

                  {/* Message Content */}
                  <div className={`max-w-xs sm:max-w-sm md:max-w-md lg:max-w-lg xl:max-w-xl ${
                    message.sender === 'user' ? 'text-right' : ''
                  }`}>
                    <div className={`inline-block p-4 rounded-2xl ${
                      message.sender === 'user'
                        ? 'bg-blue-600 text-white'
                        : 'bg-gray-100 dark:bg-gray-700 text-gray-900 dark:text-white'
                    }`}>
                      <div className="whitespace-pre-wrap text-sm leading-relaxed">
                        {message.content}
                      </div>
                    </div>
                    <div className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                      {message.timestamp.toLocaleTimeString([], { 
                        hour: '2-digit', 
                        minute: '2-digit' 
                      })}
                    </div>
                  </div>
                </motion.div>
              ))}
            </AnimatePresence>

            {/* Typing Indicator */}
            {isTyping && (
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                className="flex items-start space-x-3"
              >
                <div className="flex-shrink-0 w-8 h-8 rounded-full bg-gradient-to-r from-purple-600 to-blue-600 flex items-center justify-center">
                  <Bot className="w-5 h-5 text-white" />
                </div>
                <div className="bg-gray-100 dark:bg-gray-700 p-4 rounded-2xl">
                  <div className="flex space-x-1">
                    <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"></div>
                    <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0.1s' }}></div>
                    <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></div>
                  </div>
                </div>
              </motion.div>
            )}

            <div ref={messagesEndRef} />
          </div>

          {/* Suggestions */}
          {messages.length === 1 && (
            <div className="px-6 pb-4">
              <div className="flex items-center space-x-2 mb-3">
                <Lightbulb className="w-4 h-4 text-yellow-500" />
                <span className="text-sm text-gray-600 dark:text-gray-400">
                  Quick suggestions:
                </span>
              </div>
              <div className="flex flex-wrap gap-2">
                {suggestions.map((suggestion, index) => (
                  <motion.button
                    key={index}
                    whileHover={{ scale: 1.05 }}
                    whileTap={{ scale: 0.95 }}
                    onClick={() => handleSuggestionClick(suggestion)}
                    className="px-3 py-2 bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 rounded-full text-sm hover:bg-gray-200 dark:hover:bg-gray-600 transition-colors"
                  >
                    {suggestion}
                  </motion.button>
                ))}
              </div>
            </div>
          )}

          {/* Input Area */}
          <div className="border-t border-gray-100 dark:border-gray-700 p-4">
            <form onSubmit={handleSubmit} className="flex space-x-3">
              <div className="flex-1 relative">
                <input
                  type="text"
                  value={inputValue}
                  onChange={(e) => setInputValue(e.target.value)}
                  placeholder="Ask me about your health concerns..."
                  disabled={isLoading}
                  className="w-full px-4 py-3 border border-gray-300 dark:border-gray-600 rounded-xl bg-white dark:bg-gray-700 text-gray-900 dark:text-white placeholder-gray-500 dark:placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent disabled:opacity-50 disabled:cursor-not-allowed"
                />
              </div>
              <motion.button
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                type="submit"
                disabled={isLoading || !inputValue.trim()}
                className="px-6 py-3 bg-gradient-to-r from-blue-600 to-purple-600 text-white rounded-xl hover:from-blue-700 hover:to-purple-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-300 flex items-center"
              >
                {isLoading ? (
                  <Loader2 className="w-5 h-5 animate-spin" />
                ) : (
                  <Send className="w-5 h-5" />
                )}
              </motion.button>
            </form>
          </div>
        </motion.div>

        {/* Disclaimer */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.4 }}
          className="mt-6 p-4 bg-yellow-50 dark:bg-yellow-900/20 border border-yellow-200 dark:border-yellow-800 rounded-lg"
        >
          <div className="flex items-start space-x-3">
            <AlertTriangle className="w-5 h-5 text-yellow-600 dark:text-yellow-400 flex-shrink-0 mt-0.5" />
            <div className="text-sm text-yellow-700 dark:text-yellow-300">
              <strong>Medical Disclaimer:</strong> This AI assistant provides general wellness information and is not a substitute for professional medical advice. Always consult with qualified healthcare professionals for medical concerns. In case of emergency, contact emergency services immediately.
            </div>
          </div>
        </motion.div>

        {/* Features Info */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.6 }}
          className="mt-6 bg-gradient-to-r from-blue-50 to-purple-50 dark:from-blue-900/20 dark:to-purple-900/20 rounded-lg p-6 border border-blue-200 dark:border-blue-800"
        >
          <div className="text-center">
            <Brain className="w-8 h-8 text-blue-600 dark:text-blue-400 mx-auto mb-3" />
            <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-2">
              Powered by Google Gemini AI
            </h3>
            <p className="text-sm text-gray-600 dark:text-gray-400 mb-4">
              Advanced AI technology focused specifically on health and wellness guidance
            </p>
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 text-center">
              <div>
                <Heart className="w-6 h-6 text-red-500 mx-auto mb-2" />
                <p className="text-xs text-gray-600 dark:text-gray-400">Health-Focused Responses</p>
              </div>
              <div>
                <Brain className="w-6 h-6 text-blue-500 mx-auto mb-2" />
                <p className="text-xs text-gray-600 dark:text-gray-400">Evidence-Based Information</p>
              </div>
              <div>
                <Lightbulb className="w-6 h-6 text-yellow-500 mx-auto mb-2" />
                <p className="text-xs text-gray-600 dark:text-gray-400">Personalized Guidance</p>
              </div>
            </div>
          </div>
        </motion.div>
      </div>
    </div>
  );
};

export default Chat;