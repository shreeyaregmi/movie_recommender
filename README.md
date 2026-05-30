# Robust Movie Recommendation System Using Collaborative Filtering and Graph-Based Learning Under Adversarial Conditions

## Project Overview

This project presents a robust movie recommendation system developed as part of a thesis research project. The system evaluates recommendation performance under both normal and adversarial environments. Traditional collaborative filtering techniques are combined with graph-based recommendation methods to investigate recommendation quality and robustness against profile injection attacks.

The project implements and compares multiple recommendation approaches including Singular Value Decomposition (SVD), Light Graph Convolution Networks (LightGCN), and Light Graph Contrastive Learning (LightGCL).

---

## Research Objectives

- Develop a movie recommendation system using collaborative filtering techniques.
- Implement matrix factorization and graph-based recommendation models.
- Simulate profile injection (shilling) attacks.
- Evaluate recommendation accuracy and robustness under adversarial conditions.
- Compare recommendation performance before and after attack scenarios.

---

## Dataset Information

### Dataset Used
MovieLens Dataset

Source:
https://grouplens.org/datasets/movielens/

### Files Used
- ratings.csv
- movies.csv

### Dataset Description
The MovieLens dataset contains movie ratings provided by users along with movie metadata such as titles and genres. The dataset is widely used in recommendation system research and benchmarking studies.

---

## Development Environment

### Programming Language
- Python 3.x

### Frontend Technologies
- React
- Vite

### Development Tools
- Visual Studio Code
- Jupyter Notebook

### Libraries and Frameworks
- Pandas
- NumPy
- Scikit-learn
- Scikit-Surprise (SVD)
- PyTorch
- Matplotlib

---

## Recommendation Models Implemented

### Singular Value Decomposition (SVD)
SVD is used as the baseline recommendation model. It applies matrix factorization to learn latent relationships between users and movies and predict unknown ratings.

### LightGCN
LightGCN is a graph-based collaborative filtering model that learns user-item relationships through neighborhood aggregation in a graph structure.

### LightGCL
LightGCL extends graph-based recommendation through contrastive learning techniques that improve representation quality and robustness.

---

## Data Preprocessing

The preprocessing pipeline includes:

- Dataset loading
- Duplicate record removal
- Missing value handling
- Noise reduction and data cleaning
- User-item interaction matrix construction
- Train-test dataset splitting
- Graph construction for graph-based models

---

## Attack Simulation

To evaluate recommendation robustness, profile injection attacks were simulated.

Attack process:

1. Create synthetic malicious user profiles.
2. Generate manipulated rating patterns.
3. Inject fake profiles into the original dataset.
4. Retrain recommendation models using poisoned datasets.
5. Compare performance before and after attacks.

---

## Experimental Setup

### Training Configuration

| Parameter | Value |
|------------|---------|
| Dataset | MovieLens |
| Training Split | 80% |
| Testing Split | 20% |
| Recommendation Size | Top 10 |
| Attack Type | Profile Injection Attack |

### Evaluation Metrics

- Root Mean Square Error (RMSE)
- Mean Absolute Error (MAE)
- Precision@K
- Recall@K
- Normalized Discounted Cumulative Gain (NDCG)

---

## Installation

### Clone Repository

```bash
git clone https://github.com/shreeyaregmi/movie_recommender.git
```

### Install Python Dependencies

```bash
pip install -r requirements.txt
```

### Install Frontend Dependencies

```bash
npm install
```

---

## Execution Steps

### Step 1: Prepare Dataset
Download the MovieLens dataset and place the files inside the dataset folder.

### Step 2: Run Data Preprocessing

```bash
python preprocessing.py
```

### Step 3: Train Recommendation Models

```bash
python train_model.py
```

### Step 4: Execute Attack Simulation

```bash
python attack_simulation.py
```

### Step 5: Run Evaluation

```bash
python evaluation.py
```

### Step 6: Start Frontend Application

```bash
npm run dev
```

---

## Experimental Results

| Experimental Scenario | RMSE | Precision@K | NDCG@K |
|----------------------|--------|-------------|---------|
| Baseline Original | 2.8429 | 0.2165 | 0.2915 |
| Baseline Poisoned | 2.8455 | 0.1928 | 0.2678 |
| Robust Original | 2.9394 | 0.1764 | 0.2474 |
| Robust Poisoned | 2.9177 | 0.1702 | 0.2448 |

### Key Findings

- Recommendation quality decreases after profile injection attacks.
- Ranking metrics are more sensitive to adversarial manipulation than prediction metrics.
- Robust recommendation approaches maintain more stable performance under attack conditions.
- A trade-off exists between recommendation accuracy and robustness.

---

## Project Structure

```text
movie_recommender/
│
├── frontend/
├── backend/
├── dataset/
├── models/
├── evaluation/
├── attack_simulation/
├── screenshots/
├── README.md
└── requirements.txt
```

---

## Thesis Information

**Project Title**

Robust Movie Recommendation System 

**Research Area**

Recommendation Systems, Machine Learning, Graph-Based Learning, and Adversarial Robustness.

---

## License

This project is developed for academic and research purposes.
