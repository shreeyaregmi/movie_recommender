from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import pickle
from fastapi.middleware.cors import CORSMiddleware
import os

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
robust_model = None

@app.on_event("startup")
def load_resources():
    global movies_df, robust_model
    
    movies_path = "processed_data/movies.csv"
    if os.path.exists(movies_path):
        movies_df = pd.read_csv(movies_path)
    
    model_path = "models/robust_model.pkl"
    if os.path.exists(model_path):
        with open(model_path, "rb") as f:
            robust_model = pickle.load(f)

@app.get("/")
def read_root():
    return {"message": "Welcome to the Robust Movie Recommender API"}

@app.get("/api/movies")
def get_movies(limit: int = 20):
    if movies_df.empty:
        return {"movies": []}
    sample = movies_df.head(limit).to_dict(orient="records")
    return {"movies": sample}

@app.get("/api/movies/trending")
def get_trending_movies(limit: int = 10):
    if movies_df.empty:
        return {"movies": []}
    sample = movies_df.sample(limit, random_state=42).to_dict(orient="records")
    return {"movies": sample}

@app.get("/api/movie/{movie_id}")
def get_movie(movie_id: int):
    if movies_df.empty:
        raise HTTPException(status_code=404, detail="No movies data.")
    
    movie = movies_df[movies_df['movieId'] == movie_id]
    if movie.empty:
        raise HTTPException(status_code=404, detail="Movie not found.")
    
    return movie.iloc[0].to_dict()

class RatingRequest(BaseModel):
    user_id: int
    movie_id: int
    rating: float

@app.post("/api/rate")
def rate_movie(req: RatingRequest):
    return {"status": "success", "message": f"Rated movie {req.movie_id} with {req.rating}"}

@app.get("/api/recommendations/{user_id}")
def get_recommendations(user_id: int, limit: int = 10):
    if not robust_model:
        return {"recommendations": []}
        
    svd = robust_model['svd']
    user_ids = robust_model['user_ids']
    movie_ids = robust_model['movie_ids']
    user_factors = robust_model['user_factors']
    item_factors = robust_model['item_factors']
    
    if user_id not in user_ids:
        sample = movies_df.sample(limit).to_dict(orient="records")
        return {"recommendations": sample, "note": "Cold start recommendations"}
        
    user_idx = user_ids.index(user_id)
    u_vector = user_factors[user_idx]
    
    scores = np.dot(u_vector, item_factors)
    top_indices = np.argsort(scores)[::-1][:limit]
    rec_movie_ids = [movie_ids[i] for i in top_indices]
    
    rec_movies = movies_df[movies_df['movieId'].isin(rec_movie_ids)].to_dict(orient="records")
    return {"recommendations": rec_movies}
