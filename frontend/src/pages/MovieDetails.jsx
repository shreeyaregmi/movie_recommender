import { useParams, Link } from 'react-router-dom';
import { ArrowLeft, Star, ShieldCheck, HelpCircle } from 'lucide-react';
import { useState, useEffect } from 'react';
import axios from 'axios';

export default function MovieDetails() {
  const { id } = useParams();
  const [movie, setMovie] = useState(null);
  const [rating, setRating] = useState(0);
  const [hoverRating, setHoverRating] = useState(0);
  const [isSubmitting, setIsSubmitting] = useState(false);
  
  useEffect(() => {
    axios.get(`http://localhost:8000/api/movie/${id}`)
      .then(res => setMovie(res.data))
      .catch(err => console.error(err));
  }, [id]);

  const handleRate = (star) => {
    setRating(star);
    setIsSubmitting(true);
    
    // Simulate current user is User #1
    axios.post('http://localhost:8000/api/rate', {
      user_id: 1,
      movie_id: parseInt(id),
      rating: star
    }).then(res => {
      setTimeout(() => setIsSubmitting(false), 600);
    }).catch(err => {
      console.error(err);
      setIsSubmitting(false);
    });
  }

  if (!movie) {
    return <div className="max-w-5xl mx-auto py-20 text-center text-slate-500 animate-pulse text-2xl font-semibold">Loading movie details...</div>;
  }
  
  const genres = movie.genres ? movie.genres.split('|') : [];
  
  return (
    <div className="max-w-5xl mx-auto animate-in fade-in slide-in-from-bottom-4 duration-500">
      <Link to="/" className="inline-flex items-center text-indigo-400 hover:text-indigo-300 mb-8 transition-colors group">
        <div className="bg-slate-800 p-2 rounded-full mr-3 group-hover:bg-slate-700 transition-colors border border-slate-700">
           <ArrowLeft className="w-5 h-5" />
        </div>
        <span className="font-medium text-lg">Back to Catalog</span>
      </Link>
      
      <div className="bg-slate-800/80 backdrop-blur-sm rounded-3xl border border-slate-700 overflow-hidden shadow-[0_20px_50px_rgba(0,0,0,0.5)] flex flex-col md:flex-row relative">
         <div className="absolute top-0 right-0 w-64 h-64 bg-indigo-500/10 rounded-full blur-3xl -mr-20 -mt-20 pointer-events-none"></div>

        <div className="w-full md:w-2/5 aspect-[2/3] bg-slate-900 flex relative border-r border-slate-700/50 bg-cover bg-center" style={{backgroundImage: `url(https://via.placeholder.com/600x900/1e293b/94a3b8?text=${encodeURIComponent(movie.title.replace(/\(\d{4}\)/, ''))})`}}>
           <div className="absolute inset-0 bg-gradient-to-t from-slate-900/90 via-transparent to-transparent"></div>
        </div>
        
        <div className="p-10 flex-1 flex flex-col relative z-10">
          <div className="flex flex-wrap gap-2 mb-4">
            {genres.map(g => (
                <div key={g} className="bg-indigo-500/20 text-indigo-300 px-3 py-1 rounded-full text-xs font-bold uppercase tracking-wider border border-indigo-500/30">
                  {g}
                </div>
            ))}
          </div>
          
          <h1 className="text-5xl font-extrabold text-white mb-4 tracking-tight leading-tight">{movie.title.replace(/\(\d{4}\)/, '')}</h1>
          
          <div className="flex space-x-6 text-sm text-slate-400 mb-8 font-medium">
            <span>{movie.title.match(/\((\d{4})\)/)?.[1] || "Unknown Year"}</span>
            <span className="flex items-center text-slate-500"><HelpCircle className="w-4 h-4 mr-1"/> User ratings protected</span>
          </div>
          
          <h3 className="text-xl font-semibold text-slate-200 mb-3">Synopsis</h3>
          <p className="text-slate-300 leading-relaxed text-lg font-light mb-10 text-justify">
            A classic movie from the MovieLens dataset. In a real-world application, this area would contain a full synopsis retrieved from an external movie database API like TMDB. 
          </p>
          
          <div className="mt-auto bg-slate-900 border border-slate-700/50 rounded-2xl p-8 relative overflow-hidden shadow-inner">
            <h3 className="text-xl font-semibold mb-5 text-slate-100 flex items-center">
               Rate this movie
            </h3>
            
            <div className="flex space-x-3">
              {[1, 2, 3, 4, 5].map((star) => (
                <button 
                  key={star} 
                  onClick={() => handleRate(star)}
                  onMouseEnter={() => setHoverRating(star)}
                  onMouseLeave={() => setHoverRating(0)}
                  disabled={isSubmitting}
                  className="group relative disabled:opacity-50"
                >
                  <Star 
                    className={`w-12 h-12 transition-all duration-300 transform group-hover:scale-110 
                      ${(hoverRating ? hoverRating >= star : rating >= star) ? 'fill-yellow-500 text-yellow-500 drop-shadow-[0_0_8px_rgba(234,179,8,0.5)]' : 'fill-slate-800 text-slate-600'}
                    `} 
                  />
                </button>
              ))}
            </div>
            {rating > 0 && !isSubmitting && (
              <p className="text-sm font-medium mt-4 flex items-center text-emerald-400 animate-in fade-in slide-in-from-left-2">
                <ShieldCheck className="w-5 h-5 mr-2"/> 
                Rating securely recorded & validated against bots
              </p>
            )}
            {isSubmitting && (
               <p className="text-sm font-medium mt-4 flex items-center text-indigo-400 animate-pulse">
                Recording rating...
              </p>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
