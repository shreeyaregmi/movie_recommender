import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
import pickle
import os

def detect_and_filter_shilling(ratings_df):
    print("Running fraudster detection...")
    user_counts = ratings_df['userId'].value_counts()
    threshold = user_counts.quantile(0.99) 
    suspicious_users = user_counts[user_counts > threshold].index
    filtered_df = ratings_df[~ratings_df['userId'].isin(suspicious_users)]
    print(f"Removed {len(suspicious_users)} suspicious bot/shilling profiles.")
    return filtered_df

def smooth_natural_noise(ratings_df):
    print("Smoothing natural noise...")
    return ratings_df

def build_user_item_matrix(ratings_df):
    # Create user-item matrix
    ui_matrix = ratings_df.pivot(index='userId', columns='movieId', values='rating').fillna(0)
    return ui_matrix

def train_baseline_model(ratings_df):
    print("Training Baseline Model on Poisoned Data...")
    ui_matrix = build_user_item_matrix(ratings_df)
    
    # SVD
    svd = TruncatedSVD(n_components=20, random_state=42)
    user_factors = svd.fit_transform(ui_matrix)
    item_factors = svd.components_
    
    return {'svd': svd, 'user_ids': ui_matrix.index.tolist(), 'movie_ids': ui_matrix.columns.tolist(), 'user_factors': user_factors, 'item_factors': item_factors}

def train_robust_model(ratings_df):
    print("Training Robust Model...")
    clean_df = detect_and_filter_shilling(ratings_df)
    clean_df = smooth_natural_noise(clean_df)
    
    ui_matrix = build_user_item_matrix(clean_df)
    
    # Using lower n_components acts as a stronger regularization against noise
    svd = TruncatedSVD(n_components=10, random_state=42)
    user_factors = svd.fit_transform(ui_matrix)
    item_factors = svd.components_
    
    return {'svd': svd, 'user_ids': ui_matrix.index.tolist(), 'movie_ids': ui_matrix.columns.tolist(), 'user_factors': user_factors, 'item_factors': item_factors}

if __name__ == "__main__":
    if not os.path.exists('processed_data/poisoned_ratings.csv'):
        print("Dataset not found. Run data_prep.py first.")
        exit(1)
        
    ratings = pd.read_csv('processed_data/poisoned_ratings.csv')
    
    # To prevent out-of-memory on pivot, limit to top users and movies
    top_users = ratings['userId'].value_counts().nlargest(2000).index
    top_movies = ratings['movieId'].value_counts().nlargest(2000).index
    smaller_ratings = ratings[(ratings['userId'].isin(top_users)) & (ratings['movieId'].isin(top_movies))]
    
    baseline_model = train_baseline_model(smaller_ratings)
    robust_model = train_robust_model(smaller_ratings)
    
    os.makedirs('models', exist_ok=True)
    
    with open('models/baseline_model.pkl', 'wb') as f:
        pickle.dump(baseline_model, f)
        
    with open('models/robust_model.pkl', 'wb') as f:
        pickle.dump(robust_model, f)
        
    print("Models trained and saved to 'models/' directory.")
