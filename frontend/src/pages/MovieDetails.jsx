import { useParams, Link } from 'react-router-dom';
import { ChevronLeft } from 'lucide-react';
import { useState, useEffect } from 'react';
import axios from 'axios';
import { apiUrl } from '../config';

export default function MovieDetails() {
  const { id } = useParams();
  const [movie, setMovie] = useState(null);
  const [rating, setRating] = useState(0);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [showPlayer, setShowPlayer] = useState(false);
  const [watchPercentage, setWatchPercentage] = useState(0);
  const [syncStatus, setSyncStatus] = useState('');
  
  const user = JSON.parse(localStorage.getItem('user'));
  
  useEffect(() => {
    axios.get(apiUrl(`/api/movie/${id}`))
      .then(res => setMovie(res.data))
      .catch(err => console.error(err));
  }, [id]);

  const handleRate = (star) => {
    if (!user || !user.token) {
      alert("Please sign in to rate movies.");
      return;
    }
    setRating(star);
    setIsSubmitting(true);
    
    axios.post(apiUrl('/api/rate'), {
      movie_id: parseInt(id),
      rating: star
    }, {
      headers: { Authorization: `Bearer ${user.token}`, 'X-User-ID': user.id }
    }).then(() => {
      setTimeout(() => setIsSubmitting(false), 600);
    }).catch(err => {
      console.error(err);
      setIsSubmitting(false);
    });
  }

  const handleSimulateWatch = () => {
    if (!user) {
       alert("Please sign in to track your watch history.");
       return;
    }
    setShowPlayer(true);
  };

  const syncWatchProgress = (percentage) => {
    if (!user || !user.token) return;
    setSyncStatus('Syncing progress...');
    axios.post(apiUrl('/api/watch'), {
       movie_id: parseInt(id),
       watch_percentage: percentage
    }, {
       headers: { Authorization: `Bearer ${user.token}` }
    }).then(() => {
       setSyncStatus('Progress saved! (Validates ratings if >50%)');
       window.dispatchEvent(new Event('storage'));
       setTimeout(() => setSyncStatus(''), 3000);
    }).catch(err => {
       console.error(err);
       setSyncStatus('Failed to sync.');
    });
  };

  const handleClosePlayer = () => {
    setShowPlayer(false);
    if (watchPercentage > 0) {
       syncWatchProgress(watchPercentage);
    }
  };

  if (!movie) {
    return <div className="min-h-screen pt-32 text-center text-gray-500 bg-gray-50 text-xl font-medium">Loading details...</div>;
  }
  
  const genres = movie.genres ? movie.genres.split('|') : [];
  const yearMatch = movie.title.match(/\((\d{4})\)/);
  const year = yearMatch ? yearMatch[1] : "Unknown Year";
  const cleanTitle = movie.title.replace(/\(\d{4}\)/, '');
  
  return (
    <div className="bg-gray-50 min-h-screen text-gray-900 font-sans pb-16">
      <div className="max-w-6xl mx-auto px-4 py-8">
        
        <Link to="/" className="inline-flex items-center text-blue-600 hover:underline hover:text-blue-800 mb-6 font-medium">
           <ChevronLeft className="w-5 h-5 mr-1" />
           Back to Catalog
        </Link>
        
        <div className="bg-white rounded-xl shadow-md border border-gray-200 overflow-hidden flex flex-col md:flex-row">
            {/* Left Column: Image */}
            <div className="w-full md:w-1/3 aspect-[2/3] bg-gray-100 flex-shrink-0 border-b md:border-b-0 md:border-r border-gray-200 relative">
               <img 
                 src={movie.poster_url || "/static/no-image.png"}
                 alt={cleanTitle}
                 className="w-full h-full object-cover"
               />
               <div className="absolute top-4 left-4 bg-white/90 backdrop-blur-sm px-3 py-1 rounded shadow text-sm font-bold text-amber-600 flex items-center">
                  ★ {movie.averageRating ? movie.averageRating.toFixed(1) : "3.0"}
               </div>
            </div>
            
            {/* Right Column: Details */}
            <div className="w-full md:w-2/3 p-6 md:p-10 flex flex-col">
               <div className="flex items-center gap-3 mb-3">
                 <span className="bg-blue-100 text-blue-800 px-2.5 py-1 rounded text-xs font-bold uppercase tracking-wide">HD Movie</span>
                 <span className="text-gray-500 font-medium">{year}</span>
               </div>
               
               <h1 className="text-3xl md:text-5xl font-extrabold text-gray-900 mb-6 leading-tight">
                 {cleanTitle}
               </h1>
               
               <div className="mb-8">
                 <h2 className="text-sm font-bold text-gray-500 uppercase tracking-widest mb-2">Synopsis</h2>
                 <p className="text-gray-700 leading-relaxed text-lg">
                   {movie.overview || "No synopsis available for this title yet."}
                 </p>
               </div>
               
               <div className="grid grid-cols-2 gap-4 mb-8 bg-gray-50 p-4 rounded-lg border border-gray-100">
                  <div>
                    <span className="text-gray-500 font-bold text-xs uppercase block mb-1">Genres</span>
                    <div className="flex flex-wrap gap-1">
                      {genres.map(g => (
                         <span key={g} className="bg-white border border-gray-200 text-gray-700 px-2 py-0.5 rounded text-sm">{g}</span>
                      ))}
                    </div>
                  </div>
                  <div>
                    <span className="text-gray-500 font-bold text-xs uppercase block mb-1">Cast (Simulated)</span>
                    <span className="text-gray-800 text-sm">Lead Actor 1, Supporting Actor 2</span>
                  </div>
               </div>

               {/* Actions container */}
               <div className="mt-auto border-t border-gray-200 pt-6">
                 <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
                    <button 
                       onClick={handleSimulateWatch} 
                       className="bg-blue-600 hover:bg-blue-700 text-white font-bold py-3 px-8 rounded shadow-sm transition-colors w-full sm:w-auto text-center"
                    >
                       Simulate Watch
                    </button>
                    
                    <div className="bg-gray-50 border border-gray-200 p-3 rounded-lg w-full sm:w-auto">
                       <h3 className="text-xs font-bold text-gray-500 uppercase text-center mb-2">Rate This Title</h3>
                       {!user && <p className="text-xs text-red-500 text-center mb-2">Requires Sign In</p>}
                       <div className="flex justify-center gap-2">
                         {[1, 2, 3, 4, 5].map((star) => (
                           <button 
                             key={star} 
                             onClick={() => handleRate(star)}
                             disabled={isSubmitting || !user}
                             className={`w-8 h-8 flex items-center justify-center rounded-full border transition-colors font-bold text-sm
                               ${rating >= star ? 'bg-amber-500 text-white border-amber-500' : 'bg-white text-gray-400 border-gray-300 hover:border-amber-400 hover:text-amber-500'}
                               disabled:opacity-50 disabled:cursor-not-allowed
                             `}
                           >
                             {star}
                           </button>
                         ))}
                       </div>
                       {rating > 0 && !isSubmitting && (
                          <div className="text-center mt-2 text-xs text-green-600 font-medium">Recorded!</div>
                       )}
                    </div>
                 </div>
                 
                 {syncStatus && (
                    <div className="mt-4 p-3 bg-green-50 border border-green-200 text-green-700 rounded text-sm font-medium">
                       {syncStatus}
                    </div>
                 )}
               </div>
            </div>
        </div>

        {/* Video Player Modal */}
        {showPlayer && (
           <div className="fixed inset-0 z-50 bg-gray-900/90 backdrop-blur-sm flex flex-col items-center justify-center p-4">
               <div className="bg-white rounded-xl shadow-2xl p-8 max-w-2xl w-full">
                 <h2 className="text-2xl font-bold text-gray-900 mb-2">Simulated Video Player</h2>
                 <p className="text-gray-500 mb-6 text-sm">
                   Drag the timeline to simulate how much of the movie you watched. Only ratings paired with &gt;= 50% watch time are calculated in your recommendation profile.
                 </p>
                 
                 <div className="bg-gray-100 p-6 rounded-lg border border-gray-200">
                    <div className="flex justify-between text-gray-600 text-sm font-bold mb-3">
                       <span>0:00</span>
                       <span className="text-blue-600">{watchPercentage}% Watched</span>
                       <span>2:15:00</span>
                    </div>
                    <input 
                       type="range" 
                       min="0" 
                       max="100" 
                       value={watchPercentage}
                       onChange={(e) => setWatchPercentage(parseFloat(e.target.value))}
                       className="w-full h-2 bg-gray-300 rounded-lg appearance-none cursor-pointer accent-blue-600"
                    />
                 </div>
                 
                 <div className="mt-8 flex justify-end gap-3">
                    <button 
                      onClick={() => setShowPlayer(false)} 
                      className="px-4 py-2 text-gray-600 hover:text-gray-900 font-medium transition-colors"
                    >
                      Cancel
                    </button>
                    <button 
                      onClick={handleClosePlayer} 
                      className="bg-blue-600 text-white font-medium py-2 px-6 rounded hover:bg-blue-700 transition-colors shadow-sm"
                    >
                      Save Progress & Exit
                    </button>
                 </div>
               </div>
           </div>
        )}
        
      </div>
    </div>
  );
}
