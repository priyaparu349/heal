import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { Search, MapPin, Phone, Star, Clock, Filter, User, Stethoscope, Heart, Award, Loader2, AlertCircle } from 'lucide-react';

interface Doctor {
  id: string;
  name: string;
  specialty: string;
  rating?: number;
  reviewCount?: number;
  distance: string;
  address: string;
  phone?: string;
  availability?: string;
  lat: number;
  lon: number;
  tags: any;
}

const DoctorFinder = () => {
  const [searchRadius, setSearchRadius] = useState(5000);
  const [doctors, setDoctors] = useState<Doctor[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [userLocation, setUserLocation] = useState<{lat: number, lon: number} | null>(null);
  const [selectedDoctor, setSelectedDoctor] = useState<Doctor | null>(null);

  // Get user's current location
  useEffect(() => {
    if (navigator.geolocation) {
      navigator.geolocation.getCurrentPosition(
        (position) => {
          setUserLocation({
            lat: position.coords.latitude,
            lon: position.coords.longitude
          });
          // Automatically search for doctors when location is obtained
          searchNearbyDoctors(position.coords.latitude, position.coords.longitude, searchRadius);
        },
        (error) => {
          console.error('Error getting location:', error);
          setError('Unable to get your location. Please enable location services.');
          // Fallback to a default location (e.g., Hyderabad, India)
          const fallbackLocation = { lat: 17.385044, lon: 78.486671 };
          setUserLocation(fallbackLocation);
          searchNearbyDoctors(fallbackLocation.lat, fallbackLocation.lon, searchRadius);
        }
      );
    } else {
      setError('Geolocation is not supported by this browser.');
    }
  }, []);

  const calculateDistance = (lat1: number, lon1: number, lat2: number, lon2: number) => {
    const R = 6371; // Radius of the Earth in kilometers
    const dLat = (lat2 - lat1) * Math.PI / 180;
    const dLon = (lon2 - lon1) * Math.PI / 180;
    const a = 
      Math.sin(dLat/2) * Math.sin(dLat/2) +
      Math.cos(lat1 * Math.PI / 180) * Math.cos(lat2 * Math.PI / 180) * 
      Math.sin(dLon/2) * Math.sin(dLon/2);
    const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1-a));
    const distance = R * c;
    return distance;
  };

  const searchNearbyDoctors = async (lat: number, lon: number, radius: number) => {
    setLoading(true);
    setError('');
    
    try {
      const query = `[out:json];
        (
          node["amenity"="doctors"](around:${radius},${lat},${lon});
          node["amenity"="clinic"](around:${radius},${lat},${lon});
          node["amenity"="hospital"](around:${radius},${lat},${lon});
          node["healthcare"="doctor"](around:${radius},${lat},${lon});
          node["healthcare"="clinic"](around:${radius},${lat},${lon});
        );
        out;`;

      const response = await fetch('https://overpass-api.de/api/interpreter', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/x-www-form-urlencoded',
        },
        body: `data=${encodeURIComponent(query)}`
      });

      if (!response.ok) {
        throw new Error('Failed to fetch doctors data');
      }

      const data = await response.json();
      
      const doctorsData: Doctor[] = data.elements
        .filter((element: any) => element.tags && (element.tags.name || element.tags['addr:street']))
        .map((element: any) => {
          const distance = calculateDistance(lat, lon, element.lat, element.lon);
          
          return {
            id: element.id.toString(),
            name: element.tags.name || element.tags['healthcare:speciality'] || 'Medical Facility',
            specialty: element.tags['healthcare:speciality'] || 
                      element.tags.amenity === 'hospital' ? 'Hospital' :
                      element.tags.amenity === 'clinic' ? 'Clinic' : 'General Practice',
            distance: `${distance.toFixed(1)} km`,
            address: [
              element.tags['addr:street'],
              element.tags['addr:city'],
              element.tags['addr:postcode']
            ].filter(Boolean).join(', ') || 'Address not available',
            phone: element.tags.phone || element.tags['contact:phone'],
            lat: element.lat,
            lon: element.lon,
            tags: element.tags,
            rating: Math.random() * 2 + 3, // Mock rating between 3-5
            reviewCount: Math.floor(Math.random() * 200) + 10, // Mock review count
            availability: ['Available Today', 'Available Tomorrow', 'Next Week'][Math.floor(Math.random() * 3)]
          };
        })
        .sort((a, b) => parseFloat(a.distance) - parseFloat(b.distance))
        .slice(0, 20); // Limit to 20 results

      setDoctors(doctorsData);
    } catch (err) {
      console.error('Error fetching doctors:', err);
      setError('Failed to load nearby doctors. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const handleSearch = () => {
    if (userLocation) {
      searchNearbyDoctors(userLocation.lat, userLocation.lon, searchRadius);
    }
  };

  const getAvailabilityColor = (availability?: string) => {
    switch (availability) {
      case 'Available Today':
        return 'text-green-600 dark:text-green-400 bg-green-100 dark:bg-green-900/20';
      case 'Available Tomorrow':
        return 'text-blue-600 dark:text-blue-400 bg-blue-100 dark:bg-blue-900/20';
      case 'Next Week':
        return 'text-orange-600 dark:text-orange-400 bg-orange-100 dark:bg-orange-900/20';
      default:
        return 'text-gray-600 dark:text-gray-400 bg-gray-100 dark:bg-gray-900/20';
    }
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
            <div className="bg-gradient-to-r from-green-600 to-blue-600 w-16 h-16 rounded-full flex items-center justify-center">
              <Stethoscope className="w-8 h-8 text-white" />
            </div>
          </div>
          <h1 className="text-3xl sm:text-4xl font-bold text-gray-900 dark:text-white mb-2">
            Find Nearby Healthcare Professionals
          </h1>
          <p className="text-lg text-gray-600 dark:text-gray-400">
            Discover qualified doctors and medical facilities in your area
          </p>
        </motion.div>

        {/* Search Controls */}
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          className="mb-8 bg-white dark:bg-gray-800 rounded-2xl shadow-lg p-6 border border-gray-100 dark:border-gray-700"
        >
          <div className="flex flex-col sm:flex-row gap-4 items-center">
            {/* Location Status */}
            <div className="flex items-center space-x-2 text-sm text-gray-600 dark:text-gray-400">
              <MapPin className="w-4 h-4" />
              <span>
                {userLocation 
                  ? `Location: ${userLocation.lat.toFixed(4)}, ${userLocation.lon.toFixed(4)}`
                  : 'Getting location...'
                }
              </span>
            </div>

            {/* Search Radius */}
            <div className="flex items-center space-x-2">
              <label className="text-sm font-medium text-gray-700 dark:text-gray-300">
                Radius:
              </label>
              <select
                value={searchRadius}
                onChange={(e) => setSearchRadius(parseInt(e.target.value))}
                className="px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                <option value={1000}>1 km</option>
                <option value={2000}>2 km</option>
                <option value={5000}>5 km</option>
                <option value={10000}>10 km</option>
                <option value={20000}>20 km</option>
              </select>
            </div>

            {/* Search Button */}
            <motion.button
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              onClick={handleSearch}
              disabled={loading || !userLocation}
              className="px-6 py-2 bg-gradient-to-r from-blue-600 to-green-600 text-white rounded-lg hover:from-blue-700 hover:to-green-700 transition-all duration-300 flex items-center space-x-2 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {loading ? (
                <>
                  <Loader2 className="w-4 h-4 animate-spin" />
                  <span>Searching...</span>
                </>
              ) : (
                <>
                  <Search className="w-4 h-4" />
                  <span>Search</span>
                </>
              )}
            </motion.button>
          </div>
        </motion.div>

        {/* Error Message */}
        {error && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="mb-6 p-4 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg flex items-center space-x-3"
          >
            <AlertCircle className="w-5 h-5 text-red-600 dark:text-red-400" />
            <p className="text-red-700 dark:text-red-300">{error}</p>
          </motion.div>
        )}

        {/* Results */}
        {loading ? (
          <div className="flex justify-center items-center py-12">
            <Loader2 className="w-8 h-8 animate-spin text-blue-600" />
            <span className="ml-2 text-gray-600 dark:text-gray-400">Finding nearby doctors...</span>
          </div>
        ) : doctors.length > 0 ? (
          <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-6">
            {doctors.map((doctor, index) => (
              <motion.div
                key={doctor.id}
                initial={{ opacity: 0, y: 30 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: index * 0.1 }}
                whileHover={{ y: -5 }}
                className="bg-white dark:bg-gray-800 rounded-2xl shadow-lg hover:shadow-xl transition-all duration-300 border border-gray-100 dark:border-gray-700 overflow-hidden cursor-pointer"
                onClick={() => setSelectedDoctor(doctor)}
              >
                <div className="p-6">
                  {/* Doctor Header */}
                  <div className="flex items-start justify-between mb-4">
                    <div className="flex items-center space-x-3">
                      <div className="w-12 h-12 bg-gradient-to-r from-blue-600 to-green-600 rounded-full flex items-center justify-center">
                        <User className="w-6 h-6 text-white" />
                      </div>
                      <div>
                        <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
                          {doctor.name}
                        </h3>
                        <p className="text-sm text-gray-600 dark:text-gray-400">
                          {doctor.specialty}
                        </p>
                      </div>
                    </div>
                    {doctor.rating && (
                      <div className="text-right">
                        <div className="flex items-center space-x-1 mb-1">
                          <Star className="w-4 h-4 text-yellow-500 fill-current" />
                          <span className="text-sm font-medium text-gray-900 dark:text-white">
                            {doctor.rating.toFixed(1)}
                          </span>
                        </div>
                        <p className="text-xs text-gray-500 dark:text-gray-400">
                          {doctor.reviewCount} reviews
                        </p>
                      </div>
                    )}
                  </div>

                  {/* Distance and Availability */}
                  <div className="flex items-center justify-between mb-4">
                    <div className="flex items-center space-x-2">
                      <MapPin className="w-4 h-4 text-gray-400" />
                      <span className="text-sm text-gray-600 dark:text-gray-400">
                        {doctor.distance}
                      </span>
                    </div>
                    {doctor.availability && (
                      <span className={`px-2 py-1 rounded-full text-xs font-medium ${getAvailabilityColor(doctor.availability)}`}>
                        {doctor.availability}
                      </span>
                    )}
                  </div>

                  {/* Address */}
                  <p className="text-sm text-gray-600 dark:text-gray-400 mb-4">
                    {doctor.address}
                  </p>

                  {/* Contact Button */}
                  {doctor.phone && (
                    <motion.button
                      whileHover={{ scale: 1.02 }}
                      whileTap={{ scale: 0.98 }}
                      onClick={(e) => {
                        e.stopPropagation();
                        window.open(`tel:${doctor.phone}`, '_self');
                      }}
                      className="w-full py-2 px-4 bg-gradient-to-r from-blue-600 to-green-600 text-white rounded-lg hover:from-blue-700 hover:to-green-700 transition-all duration-300 flex items-center justify-center space-x-2"
                    >
                      <Phone className="w-4 h-4" />
                      <span>Call</span>
                    </motion.button>
                  )}
                </div>
              </motion.div>
            ))}
          </div>
        ) : !loading && (
          <div className="text-center py-12">
            <Stethoscope className="w-16 h-16 text-gray-400 mx-auto mb-4" />
            <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-2">
              No doctors found
            </h3>
            <p className="text-gray-600 dark:text-gray-400">
              Try increasing the search radius or check your location settings.
            </p>
          </div>
        )}

        {/* Doctor Detail Modal */}
        {selectedDoctor && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="fixed inset-0 bg-black bg-opacity-50 z-50 flex items-center justify-center p-4"
            onClick={() => setSelectedDoctor(null)}
          >
            <motion.div
              initial={{ scale: 0.9, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              className="bg-white dark:bg-gray-800 rounded-2xl max-w-2xl w-full max-h-[90vh] overflow-y-auto"
              onClick={(e) => e.stopPropagation()}
            >
              <div className="p-6">
                {/* Header */}
                <div className="flex items-start justify-between mb-6">
                  <div className="flex items-center space-x-4">
                    <div className="w-16 h-16 bg-gradient-to-r from-blue-600 to-green-600 rounded-full flex items-center justify-center">
                      <User className="w-8 h-8 text-white" />
                    </div>
                    <div>
                      <h2 className="text-2xl font-bold text-gray-900 dark:text-white">
                        {selectedDoctor.name}
                      </h2>
                      <p className="text-gray-600 dark:text-gray-400">
                        {selectedDoctor.specialty}
                      </p>
                      {selectedDoctor.rating && (
                        <div className="flex items-center space-x-2 mt-1">
                          <div className="flex items-center space-x-1">
                            <Star className="w-4 h-4 text-yellow-500 fill-current" />
                            <span className="text-sm font-medium">
                              {selectedDoctor.rating.toFixed(1)}
                            </span>
                            <span className="text-sm text-gray-500">
                              ({selectedDoctor.reviewCount} reviews)
                            </span>
                          </div>
                        </div>
                      )}
                    </div>
                  </div>
                  <button
                    onClick={() => setSelectedDoctor(null)}
                    className="text-gray-400 hover:text-gray-600 dark:hover:text-gray-300"
                  >
                    <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M6 18L18 6M6 6l12 12" />
                    </svg>
                  </button>
                </div>

                {/* Location and Contact */}
                <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4">
                  <div className="space-y-3">
                    <div className="flex items-start space-x-3">
                      <MapPin className="w-5 h-5 text-gray-400 mt-0.5" />
                      <div>
                        <p className="text-sm text-gray-900 dark:text-white font-medium">
                          {selectedDoctor.address}
                        </p>
                        <p className="text-sm text-gray-600 dark:text-gray-400">
                          {selectedDoctor.distance} away
                        </p>
                      </div>
                    </div>
                    {selectedDoctor.phone && (
                      <div className="flex items-center space-x-3">
                        <Phone className="w-5 h-5 text-gray-400" />
                        <p className="text-sm text-gray-900 dark:text-white">
                          {selectedDoctor.phone}
                        </p>
                      </div>
                    )}
                    {selectedDoctor.availability && (
                      <div className="flex items-center space-x-3">
                        <Clock className="w-5 h-5 text-gray-400" />
                        <span className={`px-2 py-1 rounded-full text-sm font-medium ${getAvailabilityColor(selectedDoctor.availability)}`}>
                          {selectedDoctor.availability}
                        </span>
                      </div>
                    )}
                  </div>
                </div>

                {/* Action Buttons */}
                <div className="flex flex-col sm:flex-row gap-3 mt-6">
                  {selectedDoctor.phone && (
                    <motion.button
                      whileHover={{ scale: 1.02 }}
                      whileTap={{ scale: 0.98 }}
                      onClick={() => window.open(`tel:${selectedDoctor.phone}`, '_self')}
                      className="flex-1 py-3 px-4 bg-gradient-to-r from-blue-600 to-green-600 text-white rounded-lg hover:from-blue-700 hover:to-green-700 transition-all duration-300 flex items-center justify-center space-x-2"
                    >
                      <Phone className="w-5 h-5" />
                      <span>Call Now</span>
                    </motion.button>
                  )}
                  <motion.button
                    whileHover={{ scale: 1.02 }}
                    whileTap={{ scale: 0.98 }}
                    onClick={() => window.open(`https://maps.google.com/?q=${selectedDoctor.lat},${selectedDoctor.lon}`, '_blank')}
                    className="flex-1 py-3 px-4 border border-gray-300 dark:border-gray-600 text-gray-700 dark:text-gray-300 rounded-lg hover:bg-gray-50 dark:hover:bg-gray-700 transition-all duration-300 flex items-center justify-center space-x-2"
                  >
                    <MapPin className="w-5 h-5" />
                    <span>Get Directions</span>
                  </motion.button>
                </div>
              </div>
            </motion.div>
          </motion.div>
        )}

        {/* Info Section */}
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.6 }}
          className="mt-12 bg-gradient-to-r from-blue-50 to-green-50 dark:from-blue-900/20 dark:to-green-900/20 rounded-2xl p-8 border border-blue-200 dark:border-blue-800"
        >
          <div className="text-center">
            <Award className="w-12 h-12 text-blue-600 dark:text-blue-400 mx-auto mb-4" />
            <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-4">
              Real-Time Doctor Discovery
            </h2>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6 text-center">
              <div>
                <h3 className="font-semibold text-gray-900 dark:text-white mb-2">Live Data</h3>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  Real-time data from OpenStreetMap showing actual medical facilities
                </p>
              </div>
              <div>
                <h3 className="font-semibold text-gray-900 dark:text-white mb-2">Location-Based</h3>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  Uses your current location to find the nearest healthcare providers
                </p>
              </div>
              <div>
                <h3 className="font-semibold text-gray-900 dark:text-white mb-2">Comprehensive</h3>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  Includes doctors, clinics, hospitals, and specialized healthcare facilities
                </p>
              </div>
            </div>
          </div>
        </motion.div>
      </div>
    </div>
  );
};

export default DoctorFinder;