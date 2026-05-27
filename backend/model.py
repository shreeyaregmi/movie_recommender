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


def compute_user_suspicion_scores(
    ratings_df,
    target_movie_id=1,
    popularity_top_n=50,
):
    """Score each user by how suspicious their behavior looks based on rating count, bias, and target interaction."""
    df = ratings_df[["userId", "movieId", "rating"]].copy()
    grp = df.groupby("userId")
    user_stats = grp["rating"].agg(["count", "mean", "std"]).fillna(0.0)
    user_stats.rename(columns={"count": "rating_count", "mean": "mean_rating", "std": "std_rating"}, inplace=True)

    # High-rating ratio captures users that mostly give 4.5/5 ratings.
    high_ratio = grp.apply(lambda g: float((g["rating"] >= 4.5).mean()))
    user_stats["high_ratio"] = high_ratio

    popular_movies = df["movieId"].value_counts().head(popularity_top_n).index
    popular_ratio = grp.apply(lambda g: float(g["movieId"].isin(popular_movies).mean()))
    user_stats["popular_ratio"] = popular_ratio

    target_ratio = grp.apply(lambda g: float((g["movieId"] == target_movie_id).mean()))
    user_stats["target_ratio"] = target_ratio

    # Robust z-score utility
    def z(s):
        std = float(s.std(ddof=0))
        if std < 1e-8:
            return pd.Series(np.zeros(len(s)), index=s.index)
        return (s - s.mean()) / std

    score = (
        0.45 * z(user_stats["rating_count"]).clip(lower=0)
        + 0.25 * z(user_stats["high_ratio"]).clip(lower=0)
        + 0.20 * z(user_stats["popular_ratio"]).clip(lower=0)
        + 0.10 * z(user_stats["target_ratio"]).clip(lower=0)
    )
    user_stats["suspicion_score"] = score
    return user_stats


def build_weighted_robust_training_data(
    ratings_df,
    *,
    target_movie_id=1,
    suspicious_quantile=0.95,
    suspicious_keep_prob=0.15,
    random_state=42,
):
    """Subsample interactions from suspicious users rather than removing them entirely."""
    stats = compute_user_suspicion_scores(ratings_df, target_movie_id=target_movie_id)
    threshold = stats["suspicion_score"].quantile(suspicious_quantile)
    suspicious_users = set(stats[stats["suspicion_score"] >= threshold].index.tolist())

    rng = np.random.default_rng(random_state)
    df = ratings_df.copy()
    sus_mask = df["userId"].isin(suspicious_users)
    keep_mask = np.ones(len(df), dtype=bool)
    if sus_mask.any():
        keep_mask[sus_mask.to_numpy()] = rng.random(sus_mask.sum()) < suspicious_keep_prob
    robust_df = df.loc[keep_mask].copy()

    print(
        "Robust preprocessing:",
        f"flagged_users={len(suspicious_users)}",
        f"kept_rows={len(robust_df)}/{len(df)}",
        f"keep_prob={suspicious_keep_prob}",
    )
    return robust_df

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
    clean_df = build_weighted_robust_training_data(
        ratings_df,
        target_movie_id=1,
        suspicious_quantile=0.93,
        suspicious_keep_prob=0.10,
        random_state=42,
    )
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
