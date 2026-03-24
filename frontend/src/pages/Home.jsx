import { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import axios from 'axios';
import { ShieldCheck } from 'lucide-react';

export default function Home() {
  const [trendingMovies, setTrendingMovies] = useState([]);
  const [recommendedMovies, setRecommendedMovies] = useState([]);
  const [userId, setUserId] = useState(1); // Default user for demo

  useEffect(() => {
    // Fetch generic trending
    axios.get('http://localhost:8000/api/movies/trending')
      .then(res => setTrendingMovies(res.data.movies))
      .catch(err => console.error(err));
      
    // Fetch personalized robust recommendations
    fetchRecommendations(userId);
  }, []);

  const fetchRecommendations = (id) => {
    setRecommendedMovies([]); // loading state
    axios.get(`http://localhost:8000/api/recommendations/${id}`)
      .then(res => setRecommendedMovies(res.data.recommendations))
      .catch(err => console.error(err));
  };

  const handleUserChange = (e) => {
    const newId = parseInt(e.target.value);
    setUserId(newId);
    fetchRecommendations(newId);
  };

  const MovieGrid = ({ movies }) => (
    <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-6">
      {movies.map((m) => (
        <Link to={`/movie/${m.movieId}`} key={m.movieId} className="group relative rounded-2xl overflow-hidden hover:-translate-y-2 transition-all duration-300 bg-slate-800 border border-slate-700/[0.5] hover:border-indigo-500 hover:shadow-[0_8px_30px_rgb(0,0,0,0.5)] flex flex-col">
          <div className="aspect-[2/3] bg-slate-700 flex bg-cover bg-center" style={{backgroundImage: `url(https://via.placeholder.com/300x450/1e293b/94a3b8?text=${encodeURIComponent(m.title.replace(/\(\d{4}\)/, ''))})`}}>
          </div>
          <div className="p-4 flex-grow flex flex-col justify-between bg-gradient-to-t from-slate-900 to-slate-800">
            <h3 className="font-semibold text-sm text-slate-100 group-hover:text-indigo-300 transition-colors line-clamp-2" title={m.title}>{m.title}</h3>
            <div className="flex justify-between items-center mt-3 pt-3 border-t border-slate-700/50">
                <span className="text-xs text-slate-400 bg-slate-900 px-2 py-1 rounded-md border border-slate-700 truncate max-w-[120px]">{m.genres?.split('|')[0]}</span>
            </div>
          </div>
        </Link>
      ))}
    </div>
  );

  return (
    <div className="space-y-12 animate-in fade-in duration-500 pb-12">
      <header className="text-center py-16 bg-gradient-to-br from-slate-800 to-slate-800/50 rounded-3xl border border-slate-700 shadow-xl relative overflow-hidden">
        <div className="absolute inset-0 bg-blue-500/5 mix-blend-overlay"></div>
        <h1 className="text-5xl font-extrabold text-white mb-6 tracking-tight relative z-10">Discover Your Next <span className="text-transparent bg-clip-text bg-gradient-to-r from-indigo-400 to-blue-400">Favorite</span> Movie</h1>
        <p className="text-xl text-slate-300 max-w-2xl mx-auto font-light leading-relaxed relative z-10">
          Our robust recommendation engine resists fake ratings and natural noise to give you the most accurate movie suggestions.
        </p>
      </header>
      
      <section className="bg-slate-900/50 p-8 rounded-3xl border border-indigo-500/20 shadow-lg relative overflow-hidden">
        <div className="absolute top-0 right-0 w-64 h-64 bg-indigo-500/10 rounded-full blur-3xl -mr-20 -mt-20 pointer-events-none"></div>
        
        <div className="flex justify-between items-end mb-8 relative z-10">
          <div>
            <h2 className="text-3xl font-bold flex items-center text-white mb-2">
              <ShieldCheck className="w-8 h-8 text-indigo-400 mr-3" />
              Your Robust Recommendations
            </h2>
            <p className="text-slate-400">Personalized for you, protected against bot manipulation.</p>
          </div>
          
          <div className="flex flex-col">
            <label className="text-xs text-slate-400 mb-1 font-semibold uppercase tracking-wider">Simulate User ID</label>
            <select 
              value={userId} 
              onChange={handleUserChange}
              className="bg-slate-800 border-slate-600 text-slate-200 rounded-lg focus:ring-indigo-500 focus:border-indigo-500 py-2 px-4 shadow-sm font-medium"
            >
              {[1, 2, 3, 4, 5, 10, 20].map(id => (
                <option key={id} value={id}>User #{id}</option>
              ))}
            </select>
          </div>
        </div>
        
        {recommendedMovies.length > 0 ? (
          <MovieGrid movies={recommendedMovies} />
        ) : (
          <div className="py-12 text-center text-slate-500 animate-pulse font-semibold">Loading robust recommendations...</div>
        )}
      </section>

      <section>
        <h2 className="text-2xl font-semibold mb-8 flex items-center text-slate-100">
          <span className="bg-indigo-500 w-1.5 rounded-full h-8 mr-3 shadow-[0_0_10px_theme('colors.indigo.500')]"></span>
          Trending Movies
        </h2>
        
        {trendingMovies.length > 0 ? (
          <MovieGrid movies={trendingMovies} />
        ) : (
          <div className="py-12 text-center text-slate-500 animate-pulse font-semibold">Loading trending movies...</div>
        )}
      </section>
    </div>
  );
}
