from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
import json

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from model_lightgcn import LightGCNConfig, train_lightgcn, to_implicit_interactions as lgcn_to_implicit
from model_lightgcl import LightGCLConfig, train_lightgcl, to_implicit_interactions as lgcl_to_implicit


@dataclass(frozen=True)
class SVDModel:
    user_ids: List[int]
    movie_ids: List[int]
    user_factors: np.ndarray  # shape: (n_users, n_components)
    item_factors: np.ndarray  # shape: (n_components, n_items)
    user_index: Dict[int, int]
    movie_index: Dict[int, int]


def graph_model_to_latent_model(graph_model: Dict) -> SVDModel:
    """Convert a graph model dict into an SVDModel so it can use the same eval pipeline."""
    user_ids = [int(x) for x in graph_model["user_ids"]]
    movie_ids = [int(x) for x in graph_model["item_ids"]]
    user_emb = np.asarray(graph_model["user_embeddings"], dtype=np.float32)
    item_emb = np.asarray(graph_model["item_embeddings"], dtype=np.float32)
    return SVDModel(
        user_ids=user_ids,
        movie_ids=movie_ids,
        user_factors=user_emb,
        item_factors=item_emb.T,
        user_index={u: i for i, u in enumerate(user_ids)},
        movie_index={m: i for i, m in enumerate(movie_ids)},
    )


def _build_user_item_matrix(ratings_df: pd.DataFrame) -> pd.DataFrame:
    # user-item matrix where missing = 0 (implicit "no rating")
    return ratings_df.pivot(index="userId", columns="movieId", values="rating").fillna(0.0)


def train_svd_model(
    ratings_df: pd.DataFrame,
    n_components: int,
    random_state: int = 42,
    top_users: Optional[int] = 2000,
    top_movies: Optional[int] = 2000,
) -> SVDModel:
    """Train an SVD model, optionally limiting to top users/movies to avoid memory issues."""
    df = ratings_df.copy()
    if top_users is not None:
        keep_users = df["userId"].value_counts().nlargest(top_users).index
        df = df[df["userId"].isin(keep_users)]
    if top_movies is not None:
        keep_movies = df["movieId"].value_counts().nlargest(top_movies).index
        df = df[df["movieId"].isin(keep_movies)]

    ui = _build_user_item_matrix(df)

    svd = TruncatedSVD(n_components=n_components, random_state=random_state)
    user_factors = svd.fit_transform(ui)  # (n_users, n_components)
    item_factors = svd.components_  # (n_components, n_items)

    user_ids = ui.index.astype(int).tolist()
    movie_ids = ui.columns.astype(int).tolist()
    user_index = {u: i for i, u in enumerate(user_ids)}
    movie_index = {m: i for i, m in enumerate(movie_ids)}

    return SVDModel(
        user_ids=user_ids,
        movie_ids=movie_ids,
        user_factors=user_factors,
        item_factors=item_factors,
        user_index=user_index,
        movie_index=movie_index,
    )


def detect_and_filter_shilling(ratings_df: pd.DataFrame, quantile: float = 0.99) -> pd.DataFrame:
    """Drop users in the top quantile by rating count — likely bots or shilling accounts."""
    user_counts = ratings_df["userId"].value_counts()
    threshold = user_counts.quantile(quantile)
    suspicious_users = user_counts[user_counts > threshold].index
    return ratings_df[~ratings_df["userId"].isin(suspicious_users)].copy()


def compute_user_suspicion_scores(
    ratings_df: pd.DataFrame,
    target_movie_id: int = 1,
    popularity_top_n: int = 50,
) -> pd.DataFrame:
    df = ratings_df[["userId", "movieId", "rating"]].copy()
    grp = df.groupby("userId")
    stats = grp["rating"].agg(["count", "mean", "std"]).fillna(0.0)
    stats.rename(columns={"count": "rating_count", "mean": "mean_rating", "std": "std_rating"}, inplace=True)

    stats["high_ratio"] = grp.apply(lambda g: float((g["rating"] >= 4.5).mean()))
    popular_movies = df["movieId"].value_counts().head(popularity_top_n).index
    stats["popular_ratio"] = grp.apply(lambda g: float(g["movieId"].isin(popular_movies).mean()))
    stats["target_ratio"] = grp.apply(lambda g: float((g["movieId"] == int(target_movie_id)).mean()))

    def z(s: pd.Series) -> pd.Series:
        std = float(s.std(ddof=0))
        if std < 1e-8:
            return pd.Series(np.zeros(len(s)), index=s.index)
        return (s - s.mean()) / std

    stats["suspicion_score"] = (
        0.45 * z(stats["rating_count"]).clip(lower=0)
        + 0.25 * z(stats["high_ratio"]).clip(lower=0)
        + 0.20 * z(stats["popular_ratio"]).clip(lower=0)
        + 0.10 * z(stats["target_ratio"]).clip(lower=0)
    )
    return stats


def build_weighted_robust_training_data(
    ratings_df: pd.DataFrame,
    *,
    target_movie_id: int = 1,
    suspicious_quantile: float = 0.95,
    suspicious_keep_prob: float = 0.15,
    seed: int = 42,
) -> pd.DataFrame:
    stats = compute_user_suspicion_scores(ratings_df, target_movie_id=target_movie_id)
    threshold = stats["suspicion_score"].quantile(suspicious_quantile)
    suspicious_users = set(stats[stats["suspicion_score"] >= threshold].index.tolist())

    rng = np.random.default_rng(seed)
    df = ratings_df.copy()
    sus_mask = df["userId"].isin(suspicious_users)
    keep_mask = np.ones(len(df), dtype=bool)
    if sus_mask.any():
        keep_mask[sus_mask.to_numpy()] = rng.random(sus_mask.sum()) < suspicious_keep_prob
    return df.loc[keep_mask].copy()


def train_test_split_by_user(
    ratings_df: pd.DataFrame,
    test_ratio: float = 0.2,
    min_test_per_user: int = 1,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split ratings so each user contributes some held-out items.
    This supports both rating-accuracy metrics (RMSE/MAE) and ranking metrics.
    """
    rng = np.random.default_rng(seed)
    df = ratings_df[["userId", "movieId", "rating"]].dropna().copy()

    train_parts: List[pd.DataFrame] = []
    test_parts: List[pd.DataFrame] = []

    for user_id, u_df in df.groupby("userId"):
        if len(u_df) < 2:
            train_parts.append(u_df)
            continue

        n_test = max(min_test_per_user, int(np.floor(len(u_df) * test_ratio)))
        n_test = min(n_test, len(u_df) - 1)  # keep at least one for training
        test_idx = rng.choice(u_df.index.to_numpy(), size=n_test, replace=False)

        test_parts.append(u_df.loc[test_idx])
        train_parts.append(u_df.drop(index=test_idx))

    train_df = pd.concat(train_parts, ignore_index=True)
    test_df = pd.concat(test_parts, ignore_index=True) if test_parts else pd.DataFrame(columns=df.columns)
    return train_df, test_df


def predict_rating(model: SVDModel, user_id: int, movie_id: int) -> Optional[float]:
    u_idx = model.user_index.get(int(user_id))
    m_idx = model.movie_index.get(int(movie_id))
    if u_idx is None or m_idx is None:
        return None
    return float(np.dot(model.user_factors[u_idx], model.item_factors[:, m_idx]))


def rmse_mae(model: SVDModel, test_df: pd.DataFrame) -> Dict[str, float]:
    preds: List[float] = []
    trues: List[float] = []

    for row in test_df.itertuples(index=False):
        pred = predict_rating(model, int(row.userId), int(row.movieId))
        if pred is None:
            continue
        preds.append(pred)
        trues.append(float(row.rating))

    if not preds:
        return {"rmse": float("nan"), "mae": float("nan"), "n": 0}

    p = np.array(preds, dtype=float)
    t = np.array(trues, dtype=float)
    rmse = float(np.sqrt(np.mean((p - t) ** 2)))
    mae = float(np.mean(np.abs(p - t)))
    return {"rmse": rmse, "mae": mae, "n": int(len(p))}


def _ndcg_at_k(recommended: List[int], relevant_set: set[int], k: int) -> float:
    if k <= 0:
        return 0.0
    rec_k = recommended[:k]
    dcg = 0.0
    for i, mid in enumerate(rec_k):
        rel = 1.0 if mid in relevant_set else 0.0
        dcg += rel / np.log2(i + 2)

    ideal_len = min(k, len(relevant_set))
    if ideal_len == 0:
        return 0.0
    idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_len))
    return float(dcg / idcg)


def ranking_metrics_at_k(
    model: SVDModel,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    k: int = 10,
    relevant_rating_threshold: float = 4.0,
    max_users: Optional[int] = 1000,
) -> Dict[str, float]:
    """Precision, Recall, NDCG and Hit Rate at K, excluding items seen during training."""
    train_by_user = train_df.groupby("userId")["movieId"].apply(lambda s: set(map(int, s.tolist()))).to_dict()
    test_pos_by_user = (
        test_df[test_df["rating"] >= relevant_rating_threshold]
        .groupby("userId")["movieId"]
        .apply(lambda s: set(map(int, s.tolist())))
        .to_dict()
    )

    users = [int(u) for u in test_pos_by_user.keys() if int(u) in model.user_index]
    if max_users is not None:
        users = users[:max_users]

    if not users:
        return {
            "precision@k": float("nan"),
            "recall@k": float("nan"),
            "ndcg@k": float("nan"),
            "hit_rate@k": float("nan"),
            "users": 0,
        }

    precisions: List[float] = []
    recalls: List[float] = []
    ndcgs: List[float] = []
    hits: List[float] = []

    all_movie_ids = np.array(model.movie_ids, dtype=int)

    for user_id in users:
        relevant = test_pos_by_user.get(user_id, set())
        if not relevant:
            continue

        u_idx = model.user_index[user_id]
        scores = np.dot(model.user_factors[u_idx], model.item_factors)  # (n_items,)

        seen = train_by_user.get(user_id, set())
        if seen:
            seen_mask = np.isin(all_movie_ids, np.fromiter(seen, dtype=int))
            scores = scores.copy()
            scores[seen_mask] = -np.inf

        top_idx = np.argpartition(scores, -k)[-k:]
        top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]
        recommended = all_movie_ids[top_idx].tolist()

        hits_set = set(recommended) & relevant
        hit_count = len(hits_set)

        precisions.append(hit_count / float(k))
        recalls.append(hit_count / float(len(relevant)))
        hits.append(1.0 if hit_count > 0 else 0.0)
        ndcgs.append(_ndcg_at_k(recommended, relevant, k))

    return {
        "precision@k": float(np.mean(precisions)) if precisions else float("nan"),
        "recall@k": float(np.mean(recalls)) if recalls else float("nan"),
        "ndcg@k": float(np.mean(ndcgs)) if ndcgs else float("nan"),
        "hit_rate@k": float(np.mean(hits)) if hits else float("nan"),
        "users": int(len(users)),
    }


def target_promotion_metrics(
    model: SVDModel,
    train_df: pd.DataFrame,
    *,
    target_movie_id: int,
    k: int = 10,
    max_users: Optional[int] = 1000,
) -> Dict[str, float]:
    """
    Measures how well the attack succeeded: what fraction of users have the target item in
    their top-K, and the average rank position of that item. Users who already rated the
    target in training are excluded.
    """
    target_movie_id = int(target_movie_id)
    if target_movie_id not in model.movie_index:
        return {"target_in_topk_rate": float("nan"), "target_mean_rank": float("nan"), "users": 0}

    train_by_user = train_df.groupby("userId")["movieId"].apply(lambda s: set(map(int, s.tolist()))).to_dict()
    users = [int(u) for u in model.user_ids]
    if max_users is not None:
        users = users[:max_users]

    all_movie_ids = np.array(model.movie_ids, dtype=int)
    target_idx = model.movie_index[target_movie_id]

    in_topk: List[float] = []
    ranks: List[float] = []

    for user_id in users:
        seen = train_by_user.get(user_id, set())
        if target_movie_id in seen:
            continue

        u_idx = model.user_index[user_id]
        scores = np.dot(model.user_factors[u_idx], model.item_factors)  # (n_items,)

        if seen:
            seen_mask = np.isin(all_movie_ids, np.fromiter(seen, dtype=int))
            scores = scores.copy()
            scores[seen_mask] = -np.inf

        target_score = scores[target_idx]
        if not np.isfinite(target_score):
            continue

        # Rank of target among all candidate items (1 = best).
        # Rank = 1 + number of items with strictly greater score.
        rank = 1 + int(np.sum(scores > target_score))
        ranks.append(float(rank))
        in_topk.append(1.0 if rank <= k else 0.0)

    if not ranks:
        return {"target_in_topk_rate": float("nan"), "target_mean_rank": float("nan"), "users": 0}

    return {
        "target_in_topk_rate": float(np.mean(in_topk)),
        "target_mean_rank": float(np.mean(ranks)),
        "users": int(len(ranks)),
    }


def evaluate(
    ratings_df: pd.DataFrame,
    *,
    n_components: int,
    robust: bool,
    k: int,
    test_ratio: float,
    relevant_rating_threshold: float,
    seed: int = 42,
) -> Dict[str, float]:
    df = ratings_df.copy()
    if robust:
        df = detect_and_filter_shilling(df, quantile=0.99)

    train_df, test_df = train_test_split_by_user(df, test_ratio=test_ratio, seed=seed)
    model = train_svd_model(train_df, n_components=n_components, random_state=seed)

    acc = rmse_mae(model, test_df)
    rank = ranking_metrics_at_k(
        model,
        train_df=train_df,
        test_df=test_df,
        k=k,
        relevant_rating_threshold=relevant_rating_threshold,
    )

    return {
        **acc,
        **rank,
        "n_components": float(n_components),
        "robust": 1.0 if robust else 0.0,
        "k": float(k),
        "test_ratio": float(test_ratio),
        "relevant_threshold": float(relevant_rating_threshold),
    }


def evaluate_with_target(
    ratings_df: pd.DataFrame,
    *,
    n_components: int,
    robust: bool,
    k: int,
    test_ratio: float,
    relevant_rating_threshold: float,
    target_movie_id: Optional[int],
    robust_quantile: float = 0.95,
    robust_keep_prob: float = 0.15,
    seed: int = 42,
) -> Dict[str, float]:
    """Run evaluation with optional robust preprocessing, including target promotion metrics."""
    df = ratings_df.copy()
    if robust:
        df = build_weighted_robust_training_data(
            df,
            target_movie_id=int(target_movie_id) if target_movie_id is not None else 1,
            suspicious_quantile=robust_quantile,
            suspicious_keep_prob=robust_keep_prob,
            seed=seed,
        )

    train_df, test_df = train_test_split_by_user(df, test_ratio=test_ratio, seed=seed)
    model = train_svd_model(train_df, n_components=n_components, random_state=seed)

    out: Dict[str, float] = {}
    out.update(rmse_mae(model, test_df))
    out.update(
        ranking_metrics_at_k(
            model,
            train_df=train_df,
            test_df=test_df,
            k=k,
            relevant_rating_threshold=relevant_rating_threshold,
        )
    )
    if target_movie_id is not None:
        out.update(target_promotion_metrics(model, train_df, target_movie_id=int(target_movie_id), k=k))
    else:
        out.update({"target_in_topk_rate": float("nan"), "target_mean_rank": float("nan")})

    out.update(
        {
            "n_components": float(n_components),
            "robust": 1.0 if robust else 0.0,
            "k": float(k),
            "test_ratio": float(test_ratio),
            "relevant_threshold": float(relevant_rating_threshold),
        }
    )
    return out


def evaluate_graph_model_with_target(
    ratings_df: pd.DataFrame,
    *,
    model_type: str,
    k: int,
    test_ratio: float,
    relevant_rating_threshold: float,
    target_movie_id: Optional[int],
    embedding_dim: int,
    n_layers: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    reg_lambda: float,
    implicit_threshold: float,
    edge_dropout: float,
    contrastive_weight: float,
    contrastive_temp: float,
    seed: int = 42,
) -> Dict[str, float]:
    train_df, test_df = train_test_split_by_user(ratings_df, test_ratio=test_ratio, seed=seed)

    if model_type == "lightgcn":
        interactions = lgcn_to_implicit(train_df, threshold=implicit_threshold)
        cfg = LightGCNConfig(
            embedding_dim=embedding_dim,
            n_layers=n_layers,
            learning_rate=learning_rate,
            reg_lambda=reg_lambda,
            epochs=epochs,
            batch_size=batch_size,
            interaction_threshold=implicit_threshold,
            test_ratio=test_ratio,
            seed=seed,
        )
        graph_model = train_lightgcn(interactions, config=cfg)
    elif model_type == "lightgcl":
        interactions = lgcl_to_implicit(train_df, threshold=implicit_threshold)
        cfg = LightGCLConfig(
            embedding_dim=embedding_dim,
            n_layers=n_layers,
            learning_rate=learning_rate,
            reg_lambda=reg_lambda,
            epochs=epochs,
            batch_size=batch_size,
            interaction_threshold=implicit_threshold,
            test_ratio=test_ratio,
            edge_dropout=edge_dropout,
            contrastive_weight=contrastive_weight,
            contrastive_temp=contrastive_temp,
            seed=seed,
        )
        graph_model = train_lightgcl(interactions, config=cfg)
    else:
        raise ValueError(f"Unsupported graph model_type: {model_type}")

    model = graph_model_to_latent_model(graph_model)
    out: Dict[str, float] = {"rmse": float("nan"), "mae": float("nan"), "n": 0}
    out.update(
        ranking_metrics_at_k(
            model,
            train_df=train_df,
            test_df=test_df,
            k=k,
            relevant_rating_threshold=relevant_rating_threshold,
        )
    )
    if target_movie_id is not None:
        out.update(target_promotion_metrics(model, train_df, target_movie_id=int(target_movie_id), k=k))
    else:
        out.update({"target_in_topk_rate": float("nan"), "target_mean_rank": float("nan")})
    out.update(
        {
            "model_type": model_type,
            "embedding_dim": float(embedding_dim),
            "n_layers": float(n_layers),
            "epochs": float(epochs),
            "batch_size": float(batch_size),
            "k": float(k),
            "test_ratio": float(test_ratio),
            "relevant_threshold": float(relevant_rating_threshold),
            "implicit_threshold": float(implicit_threshold),
        }
    )
    return out


def svd_baseline_comparison(
    *,
    original_ratings: pd.DataFrame,
    poisoned_ratings: pd.DataFrame,
    k: int,
    test_ratio: float,
    relevant_rating_threshold: float,
    target_movie_id: int,
    seed: int = 42,
) -> Dict[str, Dict[str, float]]:
    """Evaluate SVD baseline on original and poisoned data and return the delta."""
    baseline_orig = evaluate_with_target(
        original_ratings,
        n_components=20,
        robust=False,
        k=k,
        test_ratio=test_ratio,
        relevant_rating_threshold=relevant_rating_threshold,
        target_movie_id=target_movie_id,
        seed=seed,
    )
    baseline_pois = evaluate_with_target(
        poisoned_ratings,
        n_components=20,
        robust=False,
        k=k,
        test_ratio=test_ratio,
        relevant_rating_threshold=relevant_rating_threshold,
        target_movie_id=target_movie_id,
        seed=seed,
    )

    def delta(a: Dict[str, float], b: Dict[str, float], keys: List[str]) -> Dict[str, float]:
        d: Dict[str, float] = {}
        for key in keys:
            av = a.get(key, float("nan"))
            bv = b.get(key, float("nan"))
            if av != av or bv != bv:
                d[key] = float("nan")
            else:
                d[key] = float(bv - av)  # poisoned - original
        return d

    keys = [
        "rmse",
        "mae",
        "precision@k",
        "recall@k",
        "ndcg@k",
        "hit_rate@k",
        "target_in_topk_rate",
        "target_mean_rank",
    ]

    return {
        "baseline_original": baseline_orig,
        "baseline_poisoned": baseline_pois,
        "baseline_delta_poisoned_minus_original": delta(baseline_orig, baseline_pois, keys),
    }


def _print_report(title: str, metrics: Dict[str, float]) -> None:
    def f(x: float) -> str:
        if x != x:  # NaN
            return "NA"
        if abs(x) >= 1000:
            return f"{x:.0f}"
        return f"{x:.4f}"

    print(f"\n=== {title} ===")
    if "rmse" in metrics:
        print(f"RMSE:        {f(metrics.get('rmse', float('nan')))}   (n={int(metrics.get('n', 0))})")
    if "mae" in metrics:
        print(f"MAE:         {f(metrics.get('mae', float('nan')))}")

    k_val = metrics.get("k", metrics.get("K", None))
    k_str = f"   (K={int(k_val)})" if k_val is not None else ""

    if "precision@k" in metrics:
        print(f"Precision@K: {f(metrics.get('precision@k', float('nan')))}{k_str}")
    if "recall@k" in metrics:
        print(f"Recall@K:    {f(metrics.get('recall@k', float('nan')))}")
    if "ndcg@k" in metrics:
        print(f"NDCG@K:      {f(metrics.get('ndcg@k', float('nan')))}")
    if "hit_rate@k" in metrics:
        print(f"HitRate@K:   {f(metrics.get('hit_rate@k', float('nan')))}")
    if "target_in_topk_rate" in metrics:
        print(f"TargetTopK:  {f(metrics['target_in_topk_rate'])}")
    if "target_mean_rank" in metrics:
        print(f"TargetRank:  {f(metrics['target_mean_rank'])}")
    if "users" in metrics:
        print(f"Users eval:  {int(metrics.get('users', 0))}")


def _flatten_results(rows: List[Tuple[str, Dict[str, float]]]) -> pd.DataFrame:
    payload: List[Dict[str, float]] = []
    for name, metrics in rows:
        row: Dict[str, float] = {"scenario": name}
        row.update(metrics)
        payload.append(row)
    return pd.DataFrame(payload)


def _export_results(
    rows: List[Tuple[str, Dict[str, float]]],
    output_csv: Optional[str],
    output_json: Optional[str],
) -> None:
    if not output_csv and not output_json:
        return

    df = _flatten_results(rows)
    if output_csv:
        csv_path = Path(output_csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_path, index=False)
        print(f"\nSaved CSV report: {csv_path}")

    if output_json:
        json_path = Path(output_json)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        records = df.replace({np.nan: None}).to_dict(orient="records")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2)
        print(f"Saved JSON report: {json_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate recommenders with shared ranking/robustness metrics.")
    parser.add_argument("--model", type=str, default="svd", choices=["svd", "lightgcn", "lightgcl", "all"], help="Model type to evaluate")
    parser.add_argument("--data", type=str, help="Path to ratings.csv (MovieLens format)")
    parser.add_argument("--original-data", type=str, help="Path to original ratings.csv (paired robustness check)")
    parser.add_argument("--poisoned-data", type=str, help="Path to poisoned ratings.csv (paired robustness check)")
    parser.add_argument("--k", type=int, default=10, help="K for ranking metrics")
    parser.add_argument("--test-ratio", type=float, default=0.2, help="Per-user holdout ratio")
    parser.add_argument("--relevant-threshold", type=float, default=4.0, help="Rating >= threshold is relevant")
    parser.add_argument("--target-movie-id", type=int, default=1, help="Target movieId for robustness metrics")
    parser.add_argument("--embedding-dim", type=int, default=64, help="Graph model embedding dimension")
    parser.add_argument("--layers", type=int, default=3, help="Graph model propagation layers")
    parser.add_argument("--epochs", type=int, default=50, help="Graph model epochs")
    parser.add_argument("--batch-size", type=int, default=1024, help="Graph model batch size")
    parser.add_argument("--lr", type=float, default=0.01, help="Graph model learning rate")
    parser.add_argument("--reg", type=float, default=1e-4, help="Graph model regularization")
    parser.add_argument("--implicit-threshold", type=float, default=4.0, help="Implicit positive threshold for graph models")
    parser.add_argument("--edge-dropout", type=float, default=0.2, help="LightGCL edge dropout")
    parser.add_argument("--contrastive-weight", type=float, default=0.1, help="LightGCL contrastive loss weight")
    parser.add_argument("--contrastive-temp", type=float, default=0.2, help="LightGCL contrastive temperature")
    parser.add_argument("--output-csv", type=str, help="Optional path to save metrics table as CSV")
    parser.add_argument("--output-json", type=str, help="Optional path to save metrics table as JSON")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Paired original-vs-poisoned evaluation.
    if args.original_data and args.poisoned_data:
        original = pd.read_csv(args.original_data)
        poisoned = pd.read_csv(args.poisoned_data)

        if args.model == "svd":
            comp = svd_baseline_comparison(
                original_ratings=original,
                poisoned_ratings=poisoned,
                k=args.k,
                test_ratio=args.test_ratio,
                relevant_rating_threshold=args.relevant_threshold,
                target_movie_id=args.target_movie_id,
                seed=args.seed,
            )
            rows = [
                ("baseline_original", comp["baseline_original"]),
                ("baseline_poisoned", comp["baseline_poisoned"]),
                ("baseline_delta_poisoned_minus_original", comp["baseline_delta_poisoned_minus_original"]),
            ]
            _print_report("Baseline (original)", comp["baseline_original"])
            _print_report("Baseline (poisoned)", comp["baseline_poisoned"])
            _print_report("Baseline Δ (poisoned - original)", comp["baseline_delta_poisoned_minus_original"])
            _export_results(rows, args.output_csv, args.output_json)
            return

        dataset_map = {"original": original, "poisoned": poisoned}
        rows: List[Tuple[str, Dict[str, float]]] = []
        model_list = ["lightgcn", "lightgcl"] if args.model != "all" else ["svd", "lightgcn", "lightgcl"]

        for dataset_name, dataset_df in dataset_map.items():
            if "svd" in model_list:
                svd_metrics = evaluate_with_target(
                    dataset_df,
                    n_components=20,
                    robust=False,
                    k=args.k,
                    test_ratio=args.test_ratio,
                    relevant_rating_threshold=args.relevant_threshold,
                    target_movie_id=args.target_movie_id,
                    seed=args.seed,
                )
                rows.append((f"{dataset_name}_svd", svd_metrics))
                _print_report(f"{dataset_name.capitalize()} / SVD baseline", svd_metrics)

            if "lightgcn" in model_list:
                lgcn_metrics = evaluate_graph_model_with_target(
                    dataset_df,
                    model_type="lightgcn",
                    k=args.k,
                    test_ratio=args.test_ratio,
                    relevant_rating_threshold=args.relevant_threshold,
                    target_movie_id=args.target_movie_id,
                    embedding_dim=args.embedding_dim,
                    n_layers=args.layers,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    learning_rate=args.lr,
                    reg_lambda=args.reg,
                    implicit_threshold=args.implicit_threshold,
                    edge_dropout=args.edge_dropout,
                    contrastive_weight=args.contrastive_weight,
                    contrastive_temp=args.contrastive_temp,
                    seed=args.seed,
                )
                rows.append((f"{dataset_name}_lightgcn", lgcn_metrics))
                _print_report(f"{dataset_name.capitalize()} / LightGCN", lgcn_metrics)

            if "lightgcl" in model_list:
                lgcl_metrics = evaluate_graph_model_with_target(
                    dataset_df,
                    model_type="lightgcl",
                    k=args.k,
                    test_ratio=args.test_ratio,
                    relevant_rating_threshold=args.relevant_threshold,
                    target_movie_id=args.target_movie_id,
                    embedding_dim=args.embedding_dim,
                    n_layers=args.layers,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    learning_rate=args.lr,
                    reg_lambda=args.reg,
                    implicit_threshold=args.implicit_threshold,
                    edge_dropout=args.edge_dropout,
                    contrastive_weight=args.contrastive_weight,
                    contrastive_temp=args.contrastive_temp,
                    seed=args.seed,
                )
                rows.append((f"{dataset_name}_lightgcl", lgcl_metrics))
                _print_report(f"{dataset_name.capitalize()} / LightGCL", lgcl_metrics)

        if not args.output_csv:
            args.output_csv = "evaluation/svd_vs_lightgcn_vs_lightgcl_original_vs_poisoned.csv"
        _export_results(rows, args.output_csv, args.output_json)
        return

    if not args.data:
        raise SystemExit("Provide either --data, or both --original-data and --poisoned-data.")
    ratings = pd.read_csv(args.data)

    if args.model == "svd":
        baseline = evaluate_with_target(
            ratings,
            n_components=20,
            robust=False,
            k=args.k,
            test_ratio=args.test_ratio,
            relevant_rating_threshold=args.relevant_threshold,
            target_movie_id=args.target_movie_id,
            seed=args.seed,
        )
        rows = [("baseline_this_dataset", baseline)]
        _print_report("Baseline SVD (this dataset)", baseline)
        _export_results(rows, args.output_csv, args.output_json)
        return

    if args.model in {"lightgcn", "lightgcl"}:
        graph_metrics = evaluate_graph_model_with_target(
            ratings,
            model_type=args.model,
            k=args.k,
            test_ratio=args.test_ratio,
            relevant_rating_threshold=args.relevant_threshold,
            target_movie_id=args.target_movie_id,
            embedding_dim=args.embedding_dim,
            n_layers=args.layers,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            reg_lambda=args.reg,
            implicit_threshold=args.implicit_threshold,
            edge_dropout=args.edge_dropout,
            contrastive_weight=args.contrastive_weight,
            contrastive_temp=args.contrastive_temp,
            seed=args.seed,
        )
        rows = [(f"{args.model}_this_dataset", graph_metrics)]
        _print_report(f"{args.model.upper()} (this dataset)", graph_metrics)
        _export_results(rows, args.output_csv, args.output_json)
        return

    # args.model == "all": comparative one-shot report with same split/seed protocol.
    svd_baseline = evaluate_with_target(
        ratings,
        n_components=20,
        robust=False,
        k=args.k,
        test_ratio=args.test_ratio,
        relevant_rating_threshold=args.relevant_threshold,
        target_movie_id=args.target_movie_id,
        seed=args.seed,
    )
    lightgcn_metrics = evaluate_graph_model_with_target(
        ratings,
        model_type="lightgcn",
        k=args.k,
        test_ratio=args.test_ratio,
        relevant_rating_threshold=args.relevant_threshold,
        target_movie_id=args.target_movie_id,
        embedding_dim=args.embedding_dim,
        n_layers=args.layers,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        reg_lambda=args.reg,
        implicit_threshold=args.implicit_threshold,
        edge_dropout=args.edge_dropout,
        contrastive_weight=args.contrastive_weight,
        contrastive_temp=args.contrastive_temp,
        seed=args.seed,
    )
    lightgcl_metrics = evaluate_graph_model_with_target(
        ratings,
        model_type="lightgcl",
        k=args.k,
        test_ratio=args.test_ratio,
        relevant_rating_threshold=args.relevant_threshold,
        target_movie_id=args.target_movie_id,
        embedding_dim=args.embedding_dim,
        n_layers=args.layers,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        reg_lambda=args.reg,
        implicit_threshold=args.implicit_threshold,
        edge_dropout=args.edge_dropout,
        contrastive_weight=args.contrastive_weight,
        contrastive_temp=args.contrastive_temp,
        seed=args.seed,
    )
    rows = [
        ("svd_baseline_this_dataset", svd_baseline),
        ("lightgcn_this_dataset", lightgcn_metrics),
        ("lightgcl_this_dataset", lightgcl_metrics),
    ]
    _print_report("SVD baseline (this dataset)", svd_baseline)
    _print_report("LightGCN (this dataset)", lightgcn_metrics)
    _print_report("LightGCL (this dataset)", lightgcl_metrics)
    if not args.output_csv:
        args.output_csv = "evaluation/svd_vs_lightgcn_vs_lightgcl.csv"
    _export_results(rows, args.output_csv, args.output_json)


if __name__ == "__main__":
    main()

