import { Link } from 'react-router-dom';
import { Clapperboard } from 'lucide-react';

export default function Navbar() {
  return (
    <nav className="bg-slate-800 border-b border-slate-700 shadow-lg sticky top-0 z-50">
      <div className="container mx-auto px-4 py-4 flex items-center justify-between">
        <Link to="/" className="flex items-center space-x-2 text-indigo-400 hover:text-indigo-300 transition-colors">
          <Clapperboard className="w-8 h-8" />
          <span className="text-2xl font-bold tracking-tight">RobustFlix</span>
        </Link>
        <div className="text-sm font-medium text-slate-400 hidden sm:block bg-slate-900/50 px-3 py-1.5 rounded-full border border-slate-700 shadow-inner">
          Secure, Noise-Resistant Recommendations
        </div>
      </div>
    </nav>
  );
}
