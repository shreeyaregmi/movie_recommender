import pandas as pd
import numpy as np
import os

def load_data(data_dir):
    movies = pd.read_csv(os.path.join(data_dir, 'movies.csv'))
    ratings = pd.read_csv(os.path.join(data_dir, 'ratings.csv'))
    links = pd.read_csv(os.path.join(data_dir, 'links.csv'))
    movies = pd.merge(movies, links[['movieId', 'tmdbId']], on='movieId', how='left')
    return movies, ratings

def inject_shilling_attack(ratings, num_fake_users=50, target_movie_id=1, attack_type='bandwagon'):
    """
    Simulates a shilling attack (e.g., bandwagon attack).
    Attackers give the target movie a 5-star rating, and also rate popular movies highly to blend in.
    """
    print(f"Injecting {attack_type} attack with {num_fake_users} fake users on movie {target_movie_id}...")
    
    movie_counts = ratings['movieId'].value_counts()
    popular_movies = movie_counts.head(50).index.tolist()

    fake_ratings = []
    max_user_id = ratings['userId'].max()

    for i in range(num_fake_users):
        fake_user_id = max_user_id + i + 1

        fake_ratings.append({'userId': fake_user_id, 'movieId': target_movie_id, 'rating': 5.0, 'timestamp': 999999999})

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
    
    possible_ratings = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
    random_ratings = np.random.choice(possible_ratings, num_noisy)
    
    noisy_ratings.loc[noisy_indices, 'rating'] = random_ratings
    return noisy_ratings


def print_attack_summary(original_ratings, final_ratings, movies, target_movie_id=1, top_n=10):
    """
    Print a readable summary of injected poisoning/noise effects.
    """
    max_original_user_id = original_ratings['userId'].max()
    fake_rows = final_ratings[final_ratings['userId'] > max_original_user_id]

    original_users = original_ratings['userId'].nunique()
    poisoned_users = final_ratings['userId'].nunique()
    fake_users_created = fake_rows['userId'].nunique()

    original_target_count = (original_ratings['movieId'] == target_movie_id).sum()
    poisoned_target_count = (final_ratings['movieId'] == target_movie_id).sum()
    target_added = poisoned_target_count - original_target_count

    movie_titles = movies[['movieId', 'title']].drop_duplicates()
    top_exploited = (
        fake_rows['movieId']
        .value_counts()
        .head(top_n)
        .rename_axis('movieId')
        .reset_index(name='fake_rating_count')
        .merge(movie_titles, on='movieId', how='left')
    )

    print("\n=== Poisoning Summary ===")
    print(f"Original users: {original_users}")
    print(f"Poisoned users: {poisoned_users}")
    print(f"Fake users created: {fake_users_created}")
    print(f"Original ratings: {len(original_ratings)}")
    print(f"Poisoned ratings: {len(final_ratings)}")
    print(f"Added rows total: {len(final_ratings) - len(original_ratings)}")
    print(f"Movies exploited by fake users: {fake_rows['movieId'].nunique()}")
    print(f"Target movieId: {target_movie_id}")
    print(f"Target ratings (original): {original_target_count}")
    print(f"Target ratings (poisoned): {poisoned_target_count}")
    print(f"Target ratings added by attack: {target_added}")

    print(f"\nTop {top_n} exploited movies by fake users:")
    if top_exploited.empty:
        print("No fake-user rows found.")
    else:
        for row in top_exploited.itertuples(index=False):
            title = row.title if isinstance(row.title, str) else "Unknown title"
            print(f"- movieId={int(row.movieId):<6} fake_ratings={int(row.fake_rating_count):<4} title={title}")

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
    
    # Saving the poisoned processed data
    os.makedirs('processed_data', exist_ok=True)
    final_ratings.to_csv('processed_data/poisoned_ratings.csv', index=False)
    movies.to_csv('processed_data/movies.csv', index=False)
    print("Saved processed data to 'processed_data/'")
    print_attack_summary(ratings, final_ratings, movies, target_movie_id=1, top_n=10)
