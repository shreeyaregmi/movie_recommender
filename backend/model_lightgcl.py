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
class LightGCLConfig:
    embedding_dim: int = 64
    n_layers: int = 3
    learning_rate: float = 0.01
    reg_lambda: float = 1e-4
    epochs: int = 80
    batch_size: int = 1024
    interaction_threshold: float = 4.0
    test_ratio: float = 0.1
    edge_dropout: float = 0.2
    contrastive_weight: float = 0.1
    contrastive_temp: float = 0.2
    seed: int = 42


def train_test_split_by_user(
    ratings_df: pd.DataFrame,
    *,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
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
    return ratings_df[ratings_df["rating"] >= threshold][["userId", "movieId"]].drop_duplicates().copy()


def build_id_mappings(interactions: pd.DataFrame) -> Tuple[List[int], List[int], Dict[int, int], Dict[int, int]]:
    user_ids = sorted(interactions["userId"].astype(int).unique().tolist())
    item_ids = sorted(interactions["movieId"].astype(int).unique().tolist())
    user_to_idx = {u: i for i, u in enumerate(user_ids)}
    item_to_idx = {m: i for i, m in enumerate(item_ids)}
    return user_ids, item_ids, user_to_idx, item_to_idx


def build_bipartite_adj(
    interactions: pd.DataFrame,
    user_to_idx: Dict[int, int],
    item_to_idx: Dict[int, int],
) -> sparse.csr_matrix:
    n_users = len(user_to_idx)
    n_items = len(item_to_idx)
    n_nodes = n_users + n_items

    u_idx = interactions["userId"].map(user_to_idx).astype(int).to_numpy()
    i_idx = interactions["movieId"].map(item_to_idx).astype(int).to_numpy()
    i_nodes = i_idx + n_users

    rows = np.concatenate([u_idx, i_nodes])
    cols = np.concatenate([i_nodes, u_idx])
    data = np.ones(len(rows), dtype=np.float32)
    return sparse.coo_matrix((data, (rows, cols)), shape=(n_nodes, n_nodes), dtype=np.float32).tocsr()


def normalize_adj(adj: sparse.csr_matrix) -> sparse.csr_matrix:
    degree = np.asarray(adj.sum(axis=1)).flatten()
    degree[degree == 0.0] = 1.0
    d_inv_sqrt = np.power(degree, -0.5)
    d_mat = sparse.diags(d_inv_sqrt)
    return (d_mat @ adj @ d_mat).tocsr()


def edge_dropout(adj: sparse.csr_matrix, drop_rate: float, rng: np.random.Generator) -> sparse.csr_matrix:
    """
    Randomly drop undirected edges for graph augmentation.
    """
    if drop_rate <= 0.0:
        return adj
    coo = adj.tocoo()
    keep = rng.random(coo.data.shape[0]) > drop_rate
    if keep.sum() == 0:
        keep[rng.integers(0, len(keep))] = True
    dropped = sparse.coo_matrix((coo.data[keep], (coo.row[keep], coo.col[keep])), shape=adj.shape, dtype=np.float32)
    return dropped.tocsr()


def build_user_pos_items(
    interactions: pd.DataFrame,
    user_to_idx: Dict[int, int],
    item_to_idx: Dict[int, int],
) -> Dict[int, Set[int]]:
    user_pos: Dict[int, Set[int]] = {uidx: set() for uidx in range(len(user_to_idx))}
    for row in interactions.itertuples(index=False):
        user_pos[user_to_idx[int(row.userId)]].add(item_to_idx[int(row.movieId)])
    return user_pos


def propagate_embeddings(e0: np.ndarray, norm_adj: sparse.csr_matrix, n_layers: int) -> np.ndarray:
    embs = [e0]
    e = e0
    for _ in range(n_layers):
        e = norm_adj @ e
        embs.append(e)
    return np.mean(np.stack(embs, axis=0), axis=0)


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
        while True:
            cand = int(rng.integers(0, n_items))
            if cand not in pos_set:
                neg_items[i] = cand
                break
    return sampled_users, pos_items, neg_items


def _l2_normalize(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    norm = np.maximum(norm, eps)
    return x / norm


def contrastive_grad_symmetric(
    z1: np.ndarray,
    z2: np.ndarray,
    *,
    temp: float,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Simplified symmetric InfoNCE:
    L = CE(sim(z1,z2), diag) + CE(sim(z2,z1), diag)

    Returns scalar loss and gradients wrt z1/z2 logits-backprop approximation.
    """
    # cosine style by L2-normalization
    x1 = _l2_normalize(z1)
    x2 = _l2_normalize(z2)

    logits12 = (x1 @ x2.T) / temp
    logits21 = (x2 @ x1.T) / temp

    # softmax probs
    def softmax(a: np.ndarray) -> np.ndarray:
        a = a - np.max(a, axis=1, keepdims=True)
        e = np.exp(a)
        return e / np.sum(e, axis=1, keepdims=True)

    p12 = softmax(logits12)
    p21 = softmax(logits21)
    n = z1.shape[0]
    target = np.arange(n)

    loss12 = -np.mean(np.log(np.clip(p12[np.arange(n), target], 1e-9, 1.0)))
    loss21 = -np.mean(np.log(np.clip(p21[np.arange(n), target], 1e-9, 1.0)))
    loss = float(loss12 + loss21)

    # dCE/dlogits = p - y
    g12 = p12
    g12[np.arange(n), target] -= 1.0
    g12 /= n
    g21 = p21
    g21[np.arange(n), target] -= 1.0
    g21 /= n

    grad_x1 = (g12 @ x2 + g21.T @ x2) / temp
    grad_x2 = (g12.T @ x1 + g21 @ x1) / temp

    return loss, grad_x1.astype(np.float32), grad_x2.astype(np.float32)


def train_lightgcl(interactions: pd.DataFrame, *, config: LightGCLConfig) -> Dict:
    user_ids, item_ids, user_to_idx, item_to_idx = build_id_mappings(interactions)
    adj = build_bipartite_adj(interactions, user_to_idx, item_to_idx)
    user_pos_items = build_user_pos_items(interactions, user_to_idx, item_to_idx)

    n_users = len(user_ids)
    n_items = len(item_ids)
    n_nodes = n_users + n_items

    rng = np.random.default_rng(config.seed)
    e0 = rng.normal(0.0, 0.1, size=(n_nodes, config.embedding_dim)).astype(np.float32)

    for epoch in range(1, config.epochs + 1):
        view1 = normalize_adj(edge_dropout(adj, config.edge_dropout, rng))
        view2 = normalize_adj(edge_dropout(adj, config.edge_dropout, rng))

        e1 = propagate_embeddings(e0, view1, config.n_layers)
        e2 = propagate_embeddings(e0, view2, config.n_layers)
        e_final = 0.5 * (e1 + e2)

        grad_final = np.zeros_like(e_final, dtype=np.float32)

        u, p, n = sample_bpr_batch(user_pos_items, n_items, config.batch_size, rng)
        p_nodes = p + n_users
        n_nodes_idx = n + n_users
        u_e = e_final[u]
        p_e = e_final[p_nodes]
        n_e = e_final[n_nodes_idx]
        x = np.sum(u_e * (p_e - n_e), axis=1)
        coeff = -(1.0 / (1.0 + np.exp(x))).astype(np.float32)

        grad_u = coeff[:, None] * (p_e - n_e)
        grad_p = coeff[:, None] * u_e
        grad_n = -coeff[:, None] * u_e
        np.add.at(grad_final, u, grad_u)
        np.add.at(grad_final, p_nodes, grad_p)
        np.add.at(grad_final, n_nodes_idx, grad_n)
        grad_final /= float(config.batch_size)

        sample_nodes = min(2048, n_nodes)
        node_idx = rng.choice(np.arange(n_nodes), size=sample_nodes, replace=False)
        cl_loss, grad_e1_sub, grad_e2_sub = contrastive_grad_symmetric(
            e1[node_idx], e2[node_idx], temp=config.contrastive_temp
        )
        grad_cl = np.zeros_like(grad_final, dtype=np.float32)
        np.add.at(grad_cl, node_idx, 0.5 * (grad_e1_sub + grad_e2_sub))
        grad_final += config.contrastive_weight * grad_cl
        grad_final += config.reg_lambda * e_final

        g1 = grad_final.copy()
        g2 = grad_final.copy()
        for _ in range(config.n_layers):
            g1 = view1 @ g1
            g2 = view2 @ g2
        grad_e0 = 0.5 * (g1 + g2)

        e0 -= config.learning_rate * grad_e0

        if epoch % max(1, config.epochs // 10) == 0 or epoch == 1:
            bpr_loss = float(np.mean(np.log1p(np.exp(-x))))
            total_loss = bpr_loss + config.contrastive_weight * cl_loss
            print(
                f"[LightGCL] epoch={epoch:03d}/{config.epochs} "
                f"bpr={bpr_loss:.4f} cl={cl_loss:.4f} total={total_loss:.4f}"
            )

    # final embedding from non-dropped full normalized graph
    full_norm = normalize_adj(adj)
    final_emb = propagate_embeddings(e0, full_norm, config.n_layers)
    user_emb = final_emb[:n_users]
    item_emb = final_emb[n_users:]

    return {
        "model_type": "lightgcl",
        "embedding_dim": config.embedding_dim,
        "n_layers": config.n_layers,
        "interaction_threshold": config.interaction_threshold,
        "edge_dropout": config.edge_dropout,
        "contrastive_weight": config.contrastive_weight,
        "contrastive_temp": config.contrastive_temp,
        "user_ids": user_ids,
        "item_ids": item_ids,
        "user_id_to_index": user_to_idx,
        "item_id_to_index": item_to_idx,
        "user_embeddings": user_emb,
        "item_embeddings": item_emb,
        "train_user_pos_item_indices": {u: sorted(list(s)) for u, s in user_pos_items.items()},
    }


def recommend_top_k(model: Dict, user_id: int, k: int = 10, exclude_seen: bool = True) -> List[int]:
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
    parser = argparse.ArgumentParser(description="Train LightGCL (LightGCN + contrastive views) with BPR + CL.")
    parser.add_argument("--ratings-path", type=str, default="processed_data/poisoned_ratings.csv")
    parser.add_argument("--output-model", type=str, default="models/lightgcl_model.pkl")
    parser.add_argument("--threshold", type=float, default=4.0)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--reg", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--edge-dropout", type=float, default=0.2)
    parser.add_argument("--contrastive-weight", type=float, default=0.1)
    parser.add_argument("--contrastive-temp", type=float, default=0.2)
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

    cfg = LightGCLConfig(
        embedding_dim=args.embedding_dim,
        n_layers=args.layers,
        learning_rate=args.lr,
        reg_lambda=args.reg,
        epochs=args.epochs,
        batch_size=args.batch_size,
        interaction_threshold=args.threshold,
        test_ratio=args.test_ratio,
        edge_dropout=args.edge_dropout,
        contrastive_weight=args.contrastive_weight,
        contrastive_temp=args.contrastive_temp,
        seed=args.seed,
    )
    model = train_lightgcl(train_interactions, config=cfg)
    save_model(model, args.output_model)
    print(f"Saved LightGCL model to: {args.output_model}")


if __name__ == "__main__":
    main()
