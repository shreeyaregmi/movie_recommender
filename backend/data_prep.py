import pandas as pd
import numpy as np
import os

def load_data(data_dir):
    movies = pd.read_csv(os.path.join(data_dir, 'movies.csv'))
    ratings = pd.read_csv(os.path.join(data_dir, 'ratings.csv'))
    return movies, ratings

def inject_shilling_attack(ratings, num_fake_users=50, target_movie_id=1, attack_type='bandwagon'):
    """
    Simulates a shilling attack (e.g., bandwagon attack).
    Attackers give the target movie a 5-star rating, and also rate popular movies highly to blend in.
    """
    print(f"Injecting {attack_type} attack with {num_fake_users} fake users on movie {target_movie_id}...")
    
    # Find popular movies to blend in
    movie_counts = ratings['movieId'].value_counts()
    popular_movies = movie_counts.head(50).index.tolist()
    
    fake_ratings = []
    max_user_id = ratings['userId'].max()
    
    for i in range(num_fake_users):
        fake_user_id = max_user_id + i + 1
        
        # Rate target movie 5 stars
        fake_ratings.append({'userId': fake_user_id, 'movieId': target_movie_id, 'rating': 5.0, 'timestamp': 999999999})
        
        # Rate some popular movies to blend in
        num_filler = np.random.randint(10, 30)
        filler_movies = np.random.choice(popular_movies, num_filler, replace=False)
        for m_id in filler_movies:
            if m_id != target_movie_id:
                fake_ratings.append({'userId': fake_user_id, 'movieId': m_id, 'rating': np.random.choice([4.0, 4.5, 5.0]), 'timestamp': 999999999})
                
    fake_df = pd.DataFrame(fake_ratings)
    poisoned_ratings = pd.concat([ratings, fake_df], ignore_index=True)
    return poisoned_ratings

def inject_natural_noise(ratings, noise_ratio=0.05):
    """
    Randomly flips a percentage of ratings to simulate natural noise (accidental clicks).
    """
    print(f"Injecting natural noise to {noise_ratio*100}% of ratings...")
    noisy_ratings = ratings.copy()
    num_noisy = int(len(noisy_ratings) * noise_ratio)
    
    noisy_indices = np.random.choice(noisy_ratings.index, num_noisy, replace=False)
    
    # Randomly change ratings for these indices
    possible_ratings = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
    random_ratings = np.random.choice(possible_ratings, num_noisy)
    
    noisy_ratings.loc[noisy_indices, 'rating'] = random_ratings
    return noisy_ratings

if __name__ == "__main__":
    np.random.seed(42)
    data_dir = 'ml-latest-small'
    
    if not os.path.exists(data_dir):
        print(f"Directory {data_dir} not found. Please download the dataset.")
        exit(1)
        
    movies, ratings = load_data(data_dir)
    print(f"Original ratings shape: {ratings.shape}")
    
    # 1. Inject Shilling Attack (Targeting Toy Story, movieId=1)
    poisoned_ratings = inject_shilling_attack(ratings, num_fake_users=100, target_movie_id=1)
    
    # 2. Inject Natural Noise
    final_ratings = inject_natural_noise(poisoned_ratings, noise_ratio=0.05)
    
    print(f"Final ratings shape: {final_ratings.shape}")
    
    # Save processed data
    os.makedirs('processed_data', exist_ok=True)
    final_ratings.to_csv('processed_data/poisoned_ratings.csv', index=False)
    movies.to_csv('processed_data/movies.csv', index=False)
    print("Saved processed data to 'processed_data/'")
