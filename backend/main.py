from fastapi import FastAPI, HTTPException, Depends, Header
from pydantic import BaseModel, EmailStr
import pandas as pd
import numpy as np
import pickle
import csv
from fastapi.middleware.cors import CORSMiddleware
import os

from dotenv import load_dotenv

_backend_dir = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(_backend_dir, ".env"))

from sqlalchemy.orm import Session
from passlib.context import CryptContext
from pydantic import EmailStr
from database import engine, Base, get_db, User, WatchHistory, Rating
from google.oauth2 import id_token
from google.auth.transport import requests as google_requests
import jwt
from datetime import datetime, timedelta
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import RedirectResponse
import random
import requests
from typing import Dict, List, Optional, Tuple

# How many top-ranked titles to shuffle within on each home refresh (still personalized).
HOME_REFRESH_POOL_FACTOR = 4

Base.metadata.create_all(bind=engine)

SECRET_KEY = "dummy-secret-reccox-key"
ALGORITHM = "HS256"
security = HTTPBearer()

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(days=7)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security), db: Session = Depends(get_db)):
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: int = payload.get("sub")
        if user_id is None:
            raise HTTPException(status_code=401, detail="Invalid auth token")
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid auth token")
        
    user = db.query(User).filter(User.id == user_id).first()
    if user is None:
        raise HTTPException(status_code=401, detail="User not found")
    return user

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

app = FastAPI(title="Robust Movie Recommender API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load data and models
movies_df = pd.DataFrame()
baseline_model = None
lightgcn_model = None
lightgcl_model = None
recommender_backend = "svd"
# movieId -> score penalty applied at recommendation time (graph backends only)
item_graph_attack_penalty: Dict[int, float] = {}


def _load_pickle_if_exists(path: str):
    if not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception as exc:
        print(f"Failed to load artifact {path}: {exc}")
        return None


def _graph_attack_defense_enabled() -> bool:
    v = os.getenv("RECOMMENDER_GRAPH_ATTACK_DEFENSE", "").strip().lower()
    return v in ("1", "true", "yes", "on")


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _compute_item_graph_attack_penalties(
    ratings_df: pd.DataFrame,
    *,
    quantile: float,
    max_penalty: float,
    strength: float,
) -> Dict[int, float]:
    """Penalize items that were mostly rated by high-activity (likely fake) users at serve time."""
    if ratings_df.empty or "userId" not in ratings_df.columns or "movieId" not in ratings_df.columns:
        return {}
    uc = ratings_df["userId"].value_counts()
    if len(uc) < 10:
        return {}
    thr = float(uc.quantile(quantile))
    suspicious = set(uc[uc > thr].index.astype(int).tolist())
    if not suspicious:
        return {}
    is_susp = ratings_df["userId"].isin(suspicious)
    tot = ratings_df.groupby("movieId").size()
    susp = ratings_df.loc[is_susp].groupby("movieId").size()
    aligned = susp / tot.reindex(susp.index).fillna(1.0).clip(lower=1e-6)
    frac = aligned.clip(0.0, 1.0)
    penalties: Dict[int, float] = {}
    for mid, f in frac.items():
        penalties[int(mid)] = float(min(max_penalty, strength * float(f)))
    return penalties


def _apply_graph_attack_penalties_to_scores(scores: np.ndarray, item_ids: list) -> np.ndarray:
    if not item_graph_attack_penalty:
        return scores
    s = np.asarray(scores, dtype=np.float64).copy()
    for i, mid in enumerate(item_ids):
        p = item_graph_attack_penalty.get(int(mid), 0.0)
        if p:
            s[i] -= p
    return s


def load_resources():
    global movies_df, baseline_model, lightgcn_model, lightgcl_model, recommender_backend
    global item_graph_attack_penalty

    item_graph_attack_penalty = {}

    movies_path = "processed_data/movies.csv"
    if os.path.exists(movies_path):
        movies_df = pd.read_csv(movies_path)
        
        # Load ratings to calculate average rating
        ratings_path = "processed_data/poisoned_ratings.csv"
        if os.path.exists(ratings_path):
            ratings_df = pd.read_csv(ratings_path)
            rating_stats = ratings_df.groupby('movieId').agg(
                averageRating=('rating', 'mean'),
                ratingCount=('rating', 'count'),
            ).reset_index()
            movies_df = pd.merge(movies_df, rating_stats, on='movieId', how='left')

            if _graph_attack_defense_enabled():
                q = _env_float("RECOMMENDER_ATTACK_QUANTILE", 0.99)
                q = min(0.999, max(0.5, q))
                mx = _env_float("RECOMMENDER_ATTACK_MAX_PENALTY", 5.0)
                st = _env_float("RECOMMENDER_ATTACK_STRENGTH", 8.0)
                item_graph_attack_penalty = _compute_item_graph_attack_penalties(
                    ratings_df,
                    quantile=q,
                    max_penalty=mx,
                    strength=st,
                )
                tid_raw = os.getenv("RECOMMENDER_ATTACK_TARGET_MOVIE_ID", "").strip()
                extra = _env_float("RECOMMENDER_ATTACK_TARGET_EXTRA_PENALTY", 0.0)
                if tid_raw and extra > 0.0:
                    tid = int(tid_raw)
                    item_graph_attack_penalty[tid] = item_graph_attack_penalty.get(tid, 0.0) + extra
                print(
                    f"Graph attack defense ON: {len(item_graph_attack_penalty)} items with penalty; "
                    f"quantile={q}, max_penalty={mx}, strength={st}"
                )
        
        # Fill missing ratings with a default value like 3.0
        if 'averageRating' not in movies_df.columns:
            movies_df['averageRating'] = 3.0
        else:
            movies_df['averageRating'] = movies_df['averageRating'].fillna(3.0)
        if 'ratingCount' not in movies_df.columns:
            movies_df['ratingCount'] = 0
        else:
            movies_df['ratingCount'] = movies_df['ratingCount'].fillna(0).astype(int)
        movies_df['popularityScore'] = (
            movies_df['ratingCount'].astype(float) * movies_df['averageRating'].astype(float)
        )
            
        # Simulate 'type' for TV Shows (20% deterministic via modulo 5)
        movies_df['type'] = np.where(movies_df['movieId'] % 5 == 0, 'tv', 'movie')
        
        # Determine genres list for frontend
        global unique_genres
        all_genres = set()
        for g_str in movies_df['genres'].dropna():
            for g in g_str.split('|'):
                if g != '(no genres listed)':
                    all_genres.add(g)
        unique_genres = sorted(list(all_genres))
    
    baseline_model = _load_pickle_if_exists("models/baseline_model.pkl")
    lightgcn_model = _load_pickle_if_exists("models/lightgcn_model.pkl")
    lightgcl_model = _load_pickle_if_exists("models/lightgcl_model.pkl")
    # Default: graph recommender (LightGCN) when artifacts exist; SVD remains explicit fallback.
    recommender_backend = os.getenv("RECOMMENDER_BACKEND", "lightgcn").strip().lower()
    allowed = {"svd", "lightgcn", "lightgcl", "graph", "ensemble"}
    if recommender_backend not in allowed:
        print(f"Invalid RECOMMENDER_BACKEND={recommender_backend}; defaulting to lightgcn")
        recommender_backend = "lightgcn"
    print(
        "Loaded recommender artifacts:",
        {
            "backend": recommender_backend,
            "svd_baseline": baseline_model is not None,
            "lightgcn": lightgcn_model is not None,
            "lightgcl": lightgcl_model is not None,
        },
    )


def get_active_model():
    """Return the active model based on RECOMMENDER_BACKEND env var, falling back to SVD if needed."""
    if recommender_backend == "ensemble" and lightgcn_model and lightgcl_model:
        return {
            "_ensemble": True,
            "lightgcn": lightgcn_model,
            "lightgcl": lightgcl_model,
        }, "ensemble", None

    if recommender_backend == "graph":
        if lightgcn_model:
            return lightgcn_model, "lightgcn", "models/lightgcn_model.pkl"
        if lightgcl_model:
            return lightgcl_model, "lightgcl", "models/lightgcl_model.pkl"
        if baseline_model:
            return baseline_model, "svd_baseline", "models/baseline_model.pkl"
        return None, "none", None

    if recommender_backend == "lightgcl" and lightgcl_model:
        return lightgcl_model, "lightgcl", "models/lightgcl_model.pkl"
    if recommender_backend == "lightgcn" and lightgcn_model:
        return lightgcn_model, "lightgcn", "models/lightgcn_model.pkl"
    if recommender_backend == "svd" and baseline_model:
        return baseline_model, "svd_baseline", "models/baseline_model.pkl"

    # Requested graph backend but artifact missing, or ensemble with only one model.
    if recommender_backend == "ensemble":
        if lightgcn_model:
            return lightgcn_model, "lightgcn_fallback_ensemble_incomplete", "models/lightgcn_model.pkl"
        if lightgcl_model:
            return lightgcl_model, "lightgcl_fallback_ensemble_incomplete", "models/lightgcl_model.pkl"

    # Safe fallback for missing graph artifact.
    if baseline_model:
        return baseline_model, "svd_fallback_baseline", "models/baseline_model.pkl"
    return None, "none", None


def get_writable_svd_model() -> Tuple[dict | None, str | None]:
    """Return the SVD model for in-memory rating updates (separate from graph recommender path)."""
    if baseline_model:
        return baseline_model, "models/baseline_model.pkl"
    return None, None


def _parse_genre_list(genres_val) -> List[str]:
    if genres_val is None or (isinstance(genres_val, float) and np.isnan(genres_val)):
        return []
    s = str(genres_val).strip()
    if not s:
        return []
    return [p.strip() for p in s.split("|") if p.strip()]


def _build_genre_affinity(db: Session, user_id: int, movies_indexed: pd.DataFrame) -> Dict[str, float]:
    """Build per-genre affinity weights from this user's rating and watch history."""
    aff: Dict[str, float] = {}
    uid = int(user_id)

    for r in db.query(Rating).filter(Rating.user_id == uid).all():
        w = max(-1.0, min(1.0, (float(r.rating) - 3.5) / 1.5))
        row = movies_indexed.loc[int(r.movie_id)] if int(r.movie_id) in movies_indexed.index else None
        if row is None:
            continue
        for g in _parse_genre_list(row.get("genres")):
            aff[g] = aff.get(g, 0.0) + w * 0.6

    for wrow in db.query(WatchHistory).filter(WatchHistory.user_id == uid).all():
        pct = float(wrow.watch_percentage or 0.0)
        if pct < 50.0:
            continue
        row = movies_indexed.loc[int(wrow.movie_id)] if int(wrow.movie_id) in movies_indexed.index else None
        if row is None:
            continue
        boost = 0.25 * (pct / 100.0)
        for g in _parse_genre_list(row.get("genres")):
            aff[g] = aff.get(g, 0.0) + boost

    return aff


def _seen_movie_ids(db: Session, user_id: int) -> set[int]:
    """Return movie IDs this user has already rated or watched past 50%."""
    uid = int(user_id)
    seen: set[int] = set()
    for r in db.query(Rating).filter(Rating.user_id == uid).all():
        seen.add(int(r.movie_id))
    for w in db.query(WatchHistory).filter(WatchHistory.user_id == uid).all():
        if float(w.watch_percentage or 0.0) >= 50.0:
            seen.add(int(w.movie_id))
    return seen


def _behavior_adjust_graph_scores(
    scores: np.ndarray,
    item_ids: list,
    user_id: int,
    db: Session,
    movies_indexed: pd.DataFrame,
    genre_aff: Dict[str, float],
    seen: set[int],
) -> np.ndarray:
    """Adjust graph model scores using live rating, watch, and genre signals from SQLite."""
    s = np.asarray(scores, dtype=np.float64).copy()
    mid_to_i = {int(mid): i for i, mid in enumerate(item_ids)}
    uid = int(user_id)

    for r in db.query(Rating).filter(Rating.user_id == uid).all():
        i = mid_to_i.get(int(r.movie_id))
        if i is None:
            continue
        s[i] += 0.35 * (float(r.rating) - 3.5)

    for w in db.query(WatchHistory).filter(WatchHistory.user_id == uid).all():
        i = mid_to_i.get(int(w.movie_id))
        if i is None:
            continue
        pct = float(w.watch_percentage or 0.0)
        if pct >= 50.0:
            s[i] += 0.2 * (pct / 100.0)
        elif pct >= 25.0:
            s[i] += 0.06 * (pct / 100.0)

    if genre_aff:
        g_w = 0.08
        for i, mid in enumerate(item_ids):
            mid = int(mid)
            row = movies_indexed.loc[mid] if mid in movies_indexed.index else None
            if row is None:
                continue
            parts = _parse_genre_list(row.get("genres"))
            if not parts:
                continue
            gb = sum(genre_aff.get(p, 0.0) for p in parts)
            s[i] += g_w * min(gb / max(len(parts), 1), 4.0)

    for i, mid in enumerate(item_ids):
        if int(mid) in seen:
            s[i] -= 5.0

    return s


def _cold_start_with_behavior(
    user_id: int,
    limit: int,
    db: Session,
    movies_df_local: pd.DataFrame,
) -> Tuple[List[dict], str]:
    """Fallback for users not in the graph model — ranks catalog from ratings, watch history, and genres."""
    if movies_df_local.empty:
        return [], "No catalog loaded."
    mid_idx = movies_df_local.set_index("movieId", drop=False)
    aff = _build_genre_affinity(db, user_id, mid_idx)
    seen = _seen_movie_ids(db, user_id)
    scored: List[Tuple[float, dict]] = []
    for _, row in movies_df_local.iterrows():
        mid = int(row["movieId"])
        if mid in seen:
            continue
        parts = _parse_genre_list(row.get("genres"))
        gb = sum(aff.get(p, 0.0) for p in parts) if parts else 0.0
        base = float(row.get("averageRating", 3.0) or 3.0)
        score = base + 0.12 * min(gb, 10.0)
        scored.append((score, row.to_dict()))
    scored.sort(key=lambda t: t[0], reverse=True)
    pool_n = min(len(scored), max(limit * HOME_REFRESH_POOL_FACTOR, limit + 8))
    out = [d for _, d in scored[:pool_n]]
    random.shuffle(out)
    out = out[:limit]
    if len(out) < limit:
        rest = movies_df_local[~movies_df_local["movieId"].isin([int(x["movieId"]) for x in out] + list(seen))]
        extra = rest.nlargest(limit - len(out), "averageRating", keep="all").to_dict(orient="records")
        out.extend(extra[: limit - len(out)])
    note = "Cold start: ranked from your ratings, watch progress, and genre taste (not in graph embedding)."
    return out[:limit], note


def _recommend_ensemble_with_behavior(
    user_id: int,
    limit: int,
    gcn: dict,
    gcl: dict,
    db: Session,
    movies_indexed: pd.DataFrame,
) -> Tuple[list[int], str | None]:
    u_gcn = gcn.get("user_id_to_index", {}).get(int(user_id))
    u_gcl = gcl.get("user_id_to_index", {}).get(int(user_id))
    if u_gcn is None or u_gcl is None:
        return [], "User not in both graph models (cold start)."

    ids_gcn = [int(x) for x in gcn["item_ids"]]
    ids_gcl = [int(x) for x in gcl["item_ids"]]
    idx_gcn = {mid: i for i, mid in enumerate(ids_gcn)}
    idx_gcl = {mid: i for i, mid in enumerate(ids_gcl)}
    common = set(idx_gcn.keys()) & set(idx_gcl.keys())
    if not common:
        return [], "No overlapping items between graph models."

    aff = _build_genre_affinity(db, user_id, movies_indexed)
    seen = _seen_movie_ids(db, user_id)

    s_gcn = gcn["item_embeddings"] @ gcn["user_embeddings"][u_gcn]
    s_gcl = gcl["item_embeddings"] @ gcl["user_embeddings"][u_gcl]
    s_gcn = _behavior_adjust_graph_scores(s_gcn, ids_gcn, user_id, db, movies_indexed, aff, seen)
    s_gcl = _behavior_adjust_graph_scores(s_gcl, ids_gcl, user_id, db, movies_indexed, aff, seen)
    s_gcn = _apply_graph_attack_penalties_to_scores(s_gcn, ids_gcn)
    s_gcl = _apply_graph_attack_penalties_to_scores(s_gcl, ids_gcl)

    scored: List[Tuple[int, float]] = []
    for mid in common:
        s = 0.5 * (float(s_gcn[idx_gcn[mid]]) + float(s_gcl[idx_gcl[mid]]))
        scored.append((mid, s))
    scored.sort(key=lambda t: t[1], reverse=True)
    pool_n = min(len(scored), max(limit * HOME_REFRESH_POOL_FACTOR, limit + 8))
    pool = [m for m, _ in scored[:pool_n]]
    random.shuffle(pool)
    return pool[:limit], None


@app.get("/")
def read_root():
    return {"message": "Welcome to the Robust Movie Recommender API"}

def enhance_movies_with_posters(movies_list):
    for m in movies_list:
        title = str(m.get('title', ''))
        clean_title = re.sub(r'\(\d{4}\)', '', title).strip()
        
        if clean_title in TMDB_CACHE:
            m['poster_url'] = TMDB_CACHE[clean_title]
        else:
            url = get_tmdb_poster_url(clean_title)
            TMDB_CACHE[clean_title] = url
            m['poster_url'] = url
    return movies_list

@app.get("/api/movies")
def get_movies(limit: int = 20):
    if movies_df.empty:
        return {"movies": []}
    sample = movies_df.head(limit).to_dict(orient="records")
    sample = enhance_movies_with_posters(sample)
    return {"movies": sample}

HOME_RAIL_GENRES = ["Action", "Adventure", "Comedy", "Drama", "Sci-Fi", "Horror", "Thriller"]


def _shuffle_take(items: List, limit: int, pool_size: Optional[int] = None) -> List:
    """Pick `limit` items from a shuffled pool of the best candidates (new order each request)."""
    if not items:
        return []
    pool_n = pool_size if pool_size is not None else max(limit * HOME_REFRESH_POOL_FACTOR, limit + 8)
    pool = list(items[: min(len(items), pool_n)])
    random.shuffle(pool)
    return pool[:limit]


def _sort_by_popularity(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    cols = []
    if "popularityScore" in df.columns:
        cols.append("popularityScore")
    elif "ratingCount" in df.columns:
        cols.append("ratingCount")
    if "averageRating" in df.columns:
        cols.append("averageRating")
    if not cols:
        return df
    ascending = [False] * len(cols)
    return df.sort_values(cols, ascending=ascending)


@app.get("/api/movies/trending")
def get_trending_movies(limit: int = 10):
    """Popular catalog pool, shuffled on each request so refresh reshuffles Trending Now."""
    if movies_df.empty:
        return {"movies": []}
    ranked = _sort_by_popularity(movies_df)
    pool_n = min(len(ranked), max(limit * 5, 30))
    pool = ranked.head(pool_n).to_dict(orient="records")
    sample = _shuffle_take(pool, limit, pool_size=pool_n)
    for i, m in enumerate(sample, start=1):
        m["trending_rank"] = i
    sample = enhance_movies_with_posters(sample)
    return {"movies": sample, "source": "global_popularity"}


@app.get("/api/watch/continue/{user_id}")
def get_continue_watching(user_id: int, limit: int = 12, db: Session = Depends(get_db)):
    """In-progress titles for resume (watch % > 0 and not completed)."""
    if user_id <= 0 or movies_df.empty:
        return {"movies": [], "source": "watch_history"}
    watches = (
        db.query(WatchHistory)
        .filter(
            WatchHistory.user_id == int(user_id),
            WatchHistory.watch_percentage > 0,
            WatchHistory.watch_percentage < 95,
        )
        .order_by(WatchHistory.watch_percentage.desc())
        .limit(limit)
        .all()
    )
    if not watches:
        return {"movies": [], "source": "watch_history"}
    pct_by_mid = {int(w.movie_id): float(w.watch_percentage or 0.0) for w in watches}
    mids = list(pct_by_mid.keys())
    subset = movies_df[movies_df["movieId"].isin(mids)].copy()
    rows = subset.to_dict(orient="records")
    for m in rows:
        m["watch_percentage"] = pct_by_mid.get(int(m["movieId"]), 0.0)
    rows.sort(key=lambda m: pct_by_mid.get(int(m["movieId"]), 0.0), reverse=True)
    rows = enhance_movies_with_posters(rows)
    return {"movies": rows, "source": "watch_history"}


@app.get("/api/home/genre-rails")
def get_genre_rails(limit_per_genre: int = 12):
    """Labeled horizontal rails: popular titles per genre."""
    if movies_df.empty:
        return {"rails": []}
    rails = []
    for genre in HOME_RAIL_GENRES:
        mask = movies_df["genres"].str.contains(genre, case=False, na=False)
        filtered = movies_df.loc[mask]
        if filtered.empty:
            continue
        pool_n = min(len(filtered), max(limit_per_genre * 4, 36))
        pool = _sort_by_popularity(filtered).head(pool_n).to_dict(orient="records")
        top = enhance_movies_with_posters(_shuffle_take(pool, limit_per_genre, pool_size=pool_n))
        rails.append({"genre": genre, "movies": top})
    return {"rails": rails, "source": "metadata_genre_filter"}

@app.get("/api/genres")
def get_genres():
    if 'unique_genres' in globals():
        return {"genres": unique_genres}
    return {"genres": []}

@app.get("/api/catalog")
def get_catalog(
    type: Optional[str] = None, 
    search: Optional[str] = None, 
    genre: Optional[str] = None, 
    page: int = 1, 
    limit: int = 50
):
    if movies_df.empty:
        return {"results": [], "total": 0, "page": page, "pages": 0}
        
    filtered_df = movies_df.copy()
    
    if type:
        filtered_df = filtered_df[filtered_df['type'] == type]
        
    if search:
        filtered_df = filtered_df[filtered_df['title'].str.contains(search, case=False, na=False)]
        
    if genre and genre != "All":
        filtered_df = filtered_df[filtered_df['genres'].str.contains(genre, case=False, na=False)]
        
    total_results = len(filtered_df)
    total_pages = int(np.ceil(total_results / limit))
    
    start_idx = (page - 1) * limit
    end_idx = start_idx + limit
    
    results = filtered_df.iloc[start_idx:end_idx].to_dict(orient="records")
    results = enhance_movies_with_posters(results)
    
    return {
        "results": results,
        "total": total_results,
        "page": page,
        "pages": total_pages
    }

@app.get("/api/movie/{movie_id}")
def get_movie(movie_id: int):
    if movies_df.empty:
        raise HTTPException(status_code=404, detail="No movies data.")
    
    movie = movies_df[movies_df['movieId'] == movie_id]
    if movie.empty:
        raise HTTPException(status_code=404, detail="Movie not found.")
    
    # Fill NaN values to avoid JSON serialization errors
    movie_dict = movie.iloc[0].where(pd.notna(movie.iloc[0]), None).to_dict()
    
    # Fetch rich info from TMDB
    title = str(movie_dict.get('title', ''))
    clean_title = re.sub(r'\(\d{4}\)', '', title).strip()
    
    if clean_title not in TMDB_INFO_CACHE:
        TMDB_INFO_CACHE[clean_title] = get_tmdb_info(clean_title)
        
    info = TMDB_INFO_CACHE[clean_title]
    if info:
        movie_dict['overview'] = info.get('overview', '')
        movie_dict['tmdb_id'] = info.get('id')
        p_path = info.get('poster_path')
        if p_path:
            movie_dict['poster_url'] = f"https://image.tmdb.org/t/p/w500{p_path}"
        
    return movie_dict

try:
    from backend.tmdb import get_tmdb_poster_url, get_tmdb_info
except ModuleNotFoundError:
    from tmdb import get_tmdb_poster_url, get_tmdb_info
import re

TMDB_CACHE = {}
TMDB_INFO_CACHE = {}

@app.get("/api/poster/{movie_id}")
def get_movie_poster(movie_id: int):
    # Gracefully handle missing data
    if movies_df.empty:
        return RedirectResponse(url="https://via.placeholder.com/500x750?text=No+Data")

    movie_row = movies_df[movies_df['movieId'] == movie_id]
    if movie_row.empty:
        return RedirectResponse(url="https://via.placeholder.com/500x750?text=No+Movie")

    title = movie_row.iloc[0].get('title')
    if pd.isna(title) or not title:
        return RedirectResponse(url="https://via.placeholder.com/500x750?text=No+Title")

    # Clean the title from the year e.g., "Toy Story (1995)" -> "Toy Story"
    clean_title = re.sub(r'\(\d{4}\)', '', str(title)).strip()

    if clean_title in TMDB_CACHE:
        if TMDB_CACHE[clean_title]:
            return RedirectResponse(url=TMDB_CACHE[clean_title])
        else:
            return RedirectResponse(url="https://via.placeholder.com/500x750?text=No+Poster")

    # Call TMDB search API via our tmdb module
    poster_url = get_tmdb_poster_url(clean_title)
    if poster_url:
        TMDB_CACHE[clean_title] = poster_url
        return RedirectResponse(url=poster_url)
    
    # Store negative cache to prevent repeated failing lookups
    TMDB_CACHE[clean_title] = None
    return RedirectResponse(url="https://via.placeholder.com/500x750?text=No+Poster")

class GoogleAuthRequest(BaseModel):
    token: str

class WatchRequest(BaseModel):
    movie_id: int
    watch_percentage: float

class RatingRequest(BaseModel):
    movie_id: int
    rating: float

CLIENT_ID = "YOUR_GOOGLE_CLIENT_ID" # Will be overridden in prod

@app.post("/api/auth/google")
def google_auth(req: GoogleAuthRequest, db: Session = Depends(get_db)):
    try:
        # idinfo = id_token.verify_oauth2_token(req.token, google_requests.Request(), CLIENT_ID)
        # Note: Bypassing strict CLIENT ID check for local dev simulation, but we absolutely verify signature
        idinfo = id_token.verify_oauth2_token(req.token, google_requests.Request())
        email = idinfo['email']
        
        user = db.query(User).filter(User.email == email).first()
        if not user:
            # Auto-provision
            user = User(email=email, hashed_password="google_oauth_dummy")
            db.add(user)
            db.commit()
            db.refresh(user)
            
        access_token = create_access_token(data={"sub": user.id, "email": user.email})
        return {"id": user.id, "email": user.email, "token": access_token}
    except ValueError as e:
        print("Token verification failed:", e)
        raise HTTPException(status_code=400, detail="Invalid Google token")

@app.post("/api/watch")
def record_watch(req: WatchRequest, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    # Check if a record already exists, if so update it if higher
    watch = db.query(WatchHistory).filter(WatchHistory.user_id == current_user.id, WatchHistory.movie_id == req.movie_id).first()
    if watch:
        if req.watch_percentage > watch.watch_percentage:
            watch.watch_percentage = req.watch_percentage
            db.commit()
    else:
        watch = WatchHistory(user_id=current_user.id, movie_id=req.movie_id, watch_percentage=req.watch_percentage)
        db.add(watch)
        db.commit()
    return {"status": "success", "message": f"Recorded {req.watch_percentage}% watch for movie {req.movie_id}"}


@app.post("/api/rate")
def rate_movie(
    req: RatingRequest,
    x_user_id: int | None = Header(default=None, alias="X-User-ID"),
    db: Session = Depends(get_db),
):
    if x_user_id is None:
        raise HTTPException(status_code=400, detail="Missing X-User-ID header")
    active_model, model_name, _ = get_active_model()
    mutable_model, model_path = get_writable_svd_model()
    if not active_model or not mutable_model or not model_path:
        raise HTTPException(status_code=503, detail="Model not loaded")

    user_ids = mutable_model["user_ids"]
    movie_ids = mutable_model["movie_ids"]
    user_factors = mutable_model["user_factors"]
    item_factors = mutable_model["item_factors"]

    if req.movie_id not in movie_ids:
        raise HTTPException(status_code=404, detail="Movie not found in recommendation model")

    movie_idx = movie_ids.index(req.movie_id)
    centered_rating = float(req.rating) - 2.5
    item_vector = item_factors[:, movie_idx]

    # Update/seed user vector in-memory so recommendation response changes immediately.
    if x_user_id in user_ids:
        user_idx = user_ids.index(x_user_id)
        learning_rate = 0.2
        user_factors[user_idx] = (1.0 - learning_rate) * user_factors[user_idx] + learning_rate * (item_vector * centered_rating)
    else:
        new_vector = item_vector * centered_rating
        mutable_model["user_ids"].append(x_user_id)
        mutable_model["user_factors"] = np.vstack([user_factors, new_vector])

    # Persist model updates immediately so ratings survive server restarts.
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump(mutable_model, f)

    ratings_path = "processed_data/poisoned_ratings.csv"
    os.makedirs(os.path.dirname(ratings_path), exist_ok=True)
    with open(ratings_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([x_user_id, req.movie_id, req.rating, int(datetime.utcnow().timestamp())])

    existing = db.query(Rating).filter(Rating.user_id == x_user_id, Rating.movie_id == req.movie_id).first()
    if existing:
        existing.rating = float(req.rating)
    else:
        db.add(Rating(user_id=x_user_id, movie_id=req.movie_id, rating=float(req.rating)))
    db.commit()

    return {
        "status": "success",
        "message": f"Rated movie {req.movie_id} with {req.rating}",
        "user_id": x_user_id,
        "active_model": model_name,
        "updated_model_path": model_path,
        "updated_in_memory_model": True,
        "persisted_to_disk": True,
        "rating_stored_for_graph_behavior": True,
    }

@app.get("/api/recommendations/{user_id}")
def get_recommendations(user_id: int, limit: int = 10, db: Session = Depends(get_db)):
    active_model, model_name, _ = get_active_model()
    if not active_model:
        return {"recommendations": []}

    mid_idx = movies_df.set_index("movieId", drop=False) if not movies_df.empty else pd.DataFrame()

    if isinstance(active_model, dict) and active_model.get("_ensemble"):
        rec_movie_ids, ens_note = _recommend_ensemble_with_behavior(
            user_id, limit, active_model["lightgcn"], active_model["lightgcl"], db, mid_idx
        )
        if not rec_movie_ids:
            cold_rows, note = _cold_start_with_behavior(user_id, limit, db, movies_df)
            cold_rows = enhance_movies_with_posters(cold_rows)
            return {
                "recommendations": cold_rows,
                "note": ens_note or note,
                "active_model": model_name,
                "behavior_personalization": True,
            }
        rec_movies = movies_df[movies_df["movieId"].isin(rec_movie_ids)].to_dict(orient="records")
        rank = {mid: i for i, mid in enumerate(rec_movie_ids)}
        rec_movies.sort(key=lambda m: rank.get(int(m["movieId"]), 10**9))
        rec_movies = enhance_movies_with_posters(rec_movies)
        ens_out = {
            "recommendations": rec_movies,
            "active_model": model_name,
            "behavior_personalization": True,
        }
        if item_graph_attack_penalty:
            ens_out["graph_attack_defense_active"] = True
        return ens_out

    if model_name.startswith("lightgcn") or model_name.startswith("lightgcl"):
        user_to_idx = active_model.get("user_id_to_index", {})
        movie_ids = active_model.get("item_ids", [])
        user_embeddings = active_model.get("user_embeddings")
        item_embeddings = active_model.get("item_embeddings")
        if int(user_id) not in user_to_idx:
            cold_rows, note = _cold_start_with_behavior(user_id, limit, db, movies_df)
            cold_rows = enhance_movies_with_posters(cold_rows)
            return {
                "recommendations": cold_rows,
                "note": note,
                "active_model": model_name,
                "behavior_personalization": True,
            }

        user_idx = int(user_to_idx[int(user_id)])
        ids_list = [int(x) for x in movie_ids]
        base_scores = item_embeddings @ user_embeddings[user_idx]
        aff = _build_genre_affinity(db, user_id, mid_idx)
        seen = _seen_movie_ids(db, user_id)
        scores = _behavior_adjust_graph_scores(base_scores, ids_list, user_id, db, mid_idx, aff, seen)
        scores = _apply_graph_attack_penalties_to_scores(scores, ids_list)
        pool_n = min(len(ids_list), max(limit * HOME_REFRESH_POOL_FACTOR, limit + 8))
        top_indices = np.argsort(scores)[::-1][:pool_n]
        pool_ids = [movie_ids[i] for i in top_indices]
        rec_movie_ids = _shuffle_take(pool_ids, limit, pool_size=pool_n)
    else:
        user_ids = active_model["user_ids"]
        movie_ids = active_model["movie_ids"]
        user_factors = active_model["user_factors"]
        item_factors = active_model["item_factors"]

        # Model-only flow: unknown users receive cold-start recommendations.
        if user_id not in user_ids:
            sample = movies_df.sample(limit).to_dict(orient="records")
            sample = sorted(sample, key = lambda x: x['averageRating'], reverse=True)
            return {"recommendations": sample, "note": f"Cold start recommendations (User not found in {model_name} model).", "active_model": model_name}

        user_idx = user_ids.index(user_id)
        u_vector = user_factors[user_idx]
        scores = np.dot(u_vector, item_factors)
        pool_n = min(len(movie_ids), max(limit * HOME_REFRESH_POOL_FACTOR, limit + 8))
        top_indices = np.argsort(scores)[::-1][:pool_n]
        pool_ids = [movie_ids[i] for i in top_indices]
        rec_movie_ids = _shuffle_take(pool_ids, limit, pool_size=pool_n)

    if not rec_movie_ids:
        sample = movies_df.sample(limit).to_dict(orient="records")
        sample = sorted(sample, key=lambda x: x["averageRating"], reverse=True)
        return {"recommendations": sample, "note": f"Cold start recommendations (User not found in {model_name} model).", "active_model": model_name}
    
    rec_movies = movies_df[movies_df["movieId"].isin(rec_movie_ids)].to_dict(orient="records")
    rank = {mid: i for i, mid in enumerate(rec_movie_ids)}
    rec_movies.sort(key=lambda m: rank.get(int(m["movieId"]), 10**9))
    rec_movies = enhance_movies_with_posters(rec_movies)
    out: dict = {"recommendations": rec_movies, "active_model": model_name, "user_id": int(user_id)}
    if model_name.startswith("lightgcn") or model_name.startswith("lightgcl") or model_name == "ensemble":
        out["behavior_personalization"] = True
    if item_graph_attack_penalty:
        out["graph_attack_defense_active"] = True
    return out


load_resources()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
