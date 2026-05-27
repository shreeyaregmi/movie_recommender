from __future__ import annotations

import argparse
import os
import pickle
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from scipy import sparse


@dataclass
class LightGCNConfig:
    embedding_dim: int = 64
    n_layers: int = 3
    learning_rate: float = 0.01
    reg_lambda: float = 1e-4
    epochs: int = 50
    batch_size: int = 1024
    interaction_threshold: float = 4.0
    test_ratio: float = 0.1
    seed: int = 42


def train_test_split_by_user(
    ratings_df: pd.DataFrame,
    *,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Hold out some interactions per user.
    Useful so the graph is built from train interactions only.
    """
    rng = np.random.default_rng(seed)
    df = ratings_df[["userId", "movieId", "rating"]].dropna().copy()

    train_parts: List[pd.DataFrame] = []
    test_parts: List[pd.DataFrame] = []
    for _, u_df in df.groupby("userId"):
        if len(u_df) < 2:
            train_parts.append(u_df)
            continue
        n_test = max(1, int(np.floor(len(u_df) * test_ratio)))
        n_test = min(n_test, len(u_df) - 1)
        test_idx = rng.choice(u_df.index.to_numpy(), size=n_test, replace=False)
        test_parts.append(u_df.loc[test_idx])
        train_parts.append(u_df.drop(index=test_idx))

    train_df = pd.concat(train_parts, ignore_index=True)
    test_df = pd.concat(test_parts, ignore_index=True) if test_parts else pd.DataFrame(columns=df.columns)
    return train_df, test_df


def to_implicit_interactions(ratings_df: pd.DataFrame, threshold: float = 4.0) -> pd.DataFrame:
    """
    Convert explicit ratings to implicit positives:
    rating >= threshold => positive interaction.
    """
    implicit = ratings_df[ratings_df["rating"] >= threshold][["userId", "movieId"]].drop_duplicates().copy()
    return implicit


def build_id_mappings(interactions: pd.DataFrame) -> Tuple[List[int], List[int], Dict[int, int], Dict[int, int]]:
    user_ids = sorted(interactions["userId"].astype(int).unique().tolist())
    item_ids = sorted(interactions["movieId"].astype(int).unique().tolist())
    user_to_idx = {u: i for i, u in enumerate(user_ids)}
    item_to_idx = {m: i for i, m in enumerate(item_ids)}
    return user_ids, item_ids, user_to_idx, item_to_idx


def build_bipartite_norm_adj(
    interactions: pd.DataFrame,
    user_to_idx: Dict[int, int],
    item_to_idx: Dict[int, int],
) -> sparse.csr_matrix:
    """
    Build LightGCN normalized adjacency:
    A_hat = D^{-1/2} A D^{-1/2}
    on bipartite user-item graph.
    """
    n_users = len(user_to_idx)
    n_items = len(item_to_idx)
    n_nodes = n_users + n_items

    u_idx = interactions["userId"].map(user_to_idx).astype(int).to_numpy()
    i_idx = interactions["movieId"].map(item_to_idx).astype(int).to_numpy()
    i_nodes = i_idx + n_users

    rows = np.concatenate([u_idx, i_nodes])
    cols = np.concatenate([i_nodes, u_idx])
    data = np.ones(len(rows), dtype=np.float32)

    adj = sparse.coo_matrix((data, (rows, cols)), shape=(n_nodes, n_nodes), dtype=np.float32).tocsr()
    degree = np.asarray(adj.sum(axis=1)).flatten()
    degree[degree == 0.0] = 1.0
    d_inv_sqrt = np.power(degree, -0.5)
    d_mat = sparse.diags(d_inv_sqrt)
    norm_adj = (d_mat @ adj @ d_mat).tocsr()
    return norm_adj


def build_user_pos_items(
    interactions: pd.DataFrame,
    user_to_idx: Dict[int, int],
    item_to_idx: Dict[int, int],
) -> Dict[int, Set[int]]:
    user_pos: Dict[int, Set[int]] = {uidx: set() for uidx in range(len(user_to_idx))}
    for row in interactions.itertuples(index=False):
        uidx = user_to_idx[int(row.userId)]
        iidx = item_to_idx[int(row.movieId)]
        user_pos[uidx].add(iidx)
    return user_pos


def propagate_embeddings(
    e0: np.ndarray,
    norm_adj: sparse.csr_matrix,
    n_layers: int,
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    LightGCN propagation:
    E_final = mean(E^(0), E^(1), ..., E^(L))
    where E^(k+1) = A_hat E^(k)
    """
    embs = [e0]
    e = e0
    for _ in range(n_layers):
        e = norm_adj @ e
        embs.append(e)
    e_final = np.mean(np.stack(embs, axis=0), axis=0)
    return e_final, embs


def sample_bpr_batch(
    user_pos_items: Dict[int, Set[int]],
    n_items: int,
    batch_size: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    users = np.array([u for u, items in user_pos_items.items() if len(items) > 0], dtype=np.int64)
    sampled_users = rng.choice(users, size=batch_size, replace=True)

    pos_items = np.zeros(batch_size, dtype=np.int64)
    neg_items = np.zeros(batch_size, dtype=np.int64)

    for i, u in enumerate(sampled_users):
        pos_set = user_pos_items[int(u)]
        pos_items[i] = rng.choice(np.fromiter(pos_set, dtype=np.int64))

        # Rejection sample a negative item not interacted by this user.
        while True:
            cand = int(rng.integers(0, n_items))
            if cand not in pos_set:
                neg_items[i] = cand
                break

    return sampled_users, pos_items, neg_items


def train_lightgcn(
    interactions: pd.DataFrame,
    *,
    config: LightGCNConfig,
) -> Dict:
    user_ids, item_ids, user_to_idx, item_to_idx = build_id_mappings(interactions)
    norm_adj = build_bipartite_norm_adj(interactions, user_to_idx, item_to_idx)
    user_pos_items = build_user_pos_items(interactions, user_to_idx, item_to_idx)

    n_users = len(user_ids)
    n_items = len(item_ids)
    n_nodes = n_users + n_items

    rng = np.random.default_rng(config.seed)
    e0 = rng.normal(0.0, 0.1, size=(n_nodes, config.embedding_dim)).astype(np.float32)

    for epoch in range(1, config.epochs + 1):
        e_final, embs = propagate_embeddings(e0, norm_adj, config.n_layers)
        grad_final = np.zeros_like(e_final, dtype=np.float32)

        u, p, n = sample_bpr_batch(user_pos_items, n_items, config.batch_size, rng)
        p_nodes = p + n_users
        n_nodes_idx = n + n_users

        u_e = e_final[u]
        p_e = e_final[p_nodes]
        n_e = e_final[n_nodes_idx]

        x = np.sum(u_e * (p_e - n_e), axis=1)  # preference margin
        # d(-log(sigmoid(x)))/dx = -sigmoid(-x)
        coeff = -(1.0 / (1.0 + np.exp(x))).astype(np.float32)

        grad_u = coeff[:, None] * (p_e - n_e)
        grad_p = coeff[:, None] * u_e
        grad_n = -coeff[:, None] * u_e

        np.add.at(grad_final, u, grad_u)
        np.add.at(grad_final, p_nodes, grad_p)
        np.add.at(grad_final, n_nodes_idx, grad_n)

        grad_final /= float(config.batch_size)
        grad_final += config.reg_lambda * e_final

        layer_grads = [grad_final / float(config.n_layers + 1) for _ in range(config.n_layers + 1)]
        for k in range(config.n_layers, 0, -1):
            layer_grads[k - 1] = layer_grads[k - 1] + (norm_adj @ layer_grads[k])
        grad_e0 = layer_grads[0]

        e0 -= config.learning_rate * grad_e0

        if epoch % max(1, config.epochs // 10) == 0 or epoch == 1:
            loss = float(np.mean(np.log1p(np.exp(-x))))
            print(f"[LightGCN] epoch={epoch:03d}/{config.epochs} batch_bpr_loss={loss:.4f}")

    final_emb, _ = propagate_embeddings(e0, norm_adj, config.n_layers)
    user_emb = final_emb[:n_users]
    item_emb = final_emb[n_users:]

    return {
        "model_type": "lightgcn",
        "embedding_dim": config.embedding_dim,
        "n_layers": config.n_layers,
        "interaction_threshold": config.interaction_threshold,
        "user_ids": user_ids,
        "item_ids": item_ids,
        "user_id_to_index": user_to_idx,
        "item_id_to_index": item_to_idx,
        "user_embeddings": user_emb,
        "item_embeddings": item_emb,
        "train_user_pos_item_indices": {u: sorted(list(s)) for u, s in user_pos_items.items()},
    }


def recommend_top_k(
    model: Dict,
    user_id: int,
    k: int = 10,
    exclude_seen: bool = True,
) -> List[int]:
    """
    Return top-k recommended movieIds using dot-product scores.
    """
    user_to_idx = model["user_id_to_index"]
    item_ids = np.array(model["item_ids"], dtype=np.int64)
    user_emb = model["user_embeddings"]
    item_emb = model["item_embeddings"]

    if int(user_id) not in user_to_idx:
        return []

    uidx = user_to_idx[int(user_id)]
    scores = item_emb @ user_emb[uidx]

    if exclude_seen:
        seen = model.get("train_user_pos_item_indices", {}).get(uidx, [])
        if seen:
            scores = scores.copy()
            scores[np.array(seen, dtype=np.int64)] = -np.inf

    top_idx = np.argpartition(scores, -k)[-k:]
    top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]
    return item_ids[top_idx].tolist()


def save_model(model: Dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train LightGCN on implicit interactions with BPR loss.")
    parser.add_argument("--ratings-path", type=str, default="processed_data/poisoned_ratings.csv")
    parser.add_argument("--output-model", type=str, default="models/lightgcn_model.pkl")
    parser.add_argument("--threshold", type=float, default=4.0)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--reg", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ratings = pd.read_csv(args.ratings_path)
    train_ratings, test_ratings = train_test_split_by_user(
        ratings, test_ratio=args.test_ratio, seed=args.seed
    )
    train_interactions = to_implicit_interactions(train_ratings, threshold=args.threshold)
    test_interactions = to_implicit_interactions(test_ratings, threshold=args.threshold)

    print(f"Train ratings rows: {len(train_ratings)} | Test ratings rows: {len(test_ratings)}")
    print(f"Train implicit interactions: {len(train_interactions)} | Test implicit interactions: {len(test_interactions)}")

    cfg = LightGCNConfig(
        embedding_dim=args.embedding_dim,
        n_layers=args.layers,
        learning_rate=args.lr,
        reg_lambda=args.reg,
        epochs=args.epochs,
        batch_size=args.batch_size,
        interaction_threshold=args.threshold,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
    model = train_lightgcn(train_interactions, config=cfg)
    save_model(model, args.output_model)
    print(f"Saved LightGCN model to: {args.output_model}")


if __name__ == "__main__":
    main()
