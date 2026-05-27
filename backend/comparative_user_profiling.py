"""
Builds comparison tables for the three demo users (Bibek, Shreya, Shrawan).
Covers input difference (CSV + SQLite behaviors) and processing difference
(score decomposition and vector displacement).

Run from backend/:
  python -m comparison.comparative_user_profiling

Output: comparison/output/all_report_tables.csv
Filter by the Section column in Excel to read each table separately.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Allow running as script or module from backend/
_BACKEND = Path(__file__).resolve().parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from database import Rating, SessionLocal, WatchHistory  # noqa: E402

DEMO_USERS: Dict[int, Dict[str, str]] = {
    1: {"name": "Bibek", "email": "bibek@cdu.edu.au"},
    2: {"name": "Shreya", "email": "shreya@cdu.edu.au"},
    3: {"name": "Shrawan", "email": "shrawan@cdu.edu.au"},
}

RATINGS_CSV = _BACKEND / "processed_data" / "poisoned_ratings.csv"
MOVIES_CSV = _BACKEND / "processed_data" / "movies.csv"
LIGHTGCN_MODEL = _BACKEND / "models" / "lightgcn_model.pkl"
SVD_MODEL = _BACKEND / "models" / "baseline_model.pkl"
DEFAULT_OUTPUT = _BACKEND / "comparison" / "output"
REPORT_FILENAME = "all_report_tables.csv"

SECTION_LABELS = {
    "01_input_difference": "1 - Input difference",
    "02_processing_inception_action": "2 - Processing (Inception before/after)",
    "03_processing_all_steps": "3 - Processing (all steps)",
    "04_weight_dominance": "4 - Weight dominance",
    "05_vector_displacement": "5 - Vector displacement",
    "08_user_swap_sensitivity": "8 - System reactivity (user swap)",
}

# Demo action movie for "system reacting" trace (Inception)
INCEPTION_MOVIE_ID = 79132
INCEPTION_TITLE = "Inception (2010)"

# Per-user illustrative actions (Processing Difference section)
DEMO_PROCESSING_ACTIONS: Dict[int, Dict[str, Any]] = {
    1: {"action": "rate", "movie_id": INCEPTION_MOVIE_ID, "rating": 5.0, "label": "Rates Inception 5★"},
    2: {"action": "watch", "movie_id": INCEPTION_MOVIE_ID, "watch_pct": 85.0, "label": "Watches Inception 85%"},
    3: {"action": "rate", "movie_id": INCEPTION_MOVIE_ID, "rating": 2.0, "label": "Rates Inception 2★"},
}

GENRE_CLUSTERS = {
    "Sci-Fi / Mind-bending": ["Sci-Fi", "Mystery"],
    "Crime / Thriller": ["Crime", "Thriller"],
    "Adventure / Action": ["Adventure", "Action"],
}

# Maps to main.py behavior_adjust_graph_scores weights
W_GENRE = 0.08
W_RATING = 0.35
W_WATCH_FULL = 0.2
W_WATCH_PARTIAL = 0.06
W_SEEN = 5.0


def _parse_genres(genres_val) -> List[str]:
    if genres_val is None or (isinstance(genres_val, float) and np.isnan(genres_val)):
        return []
    s = str(genres_val).strip()
    if not s or s == "(no genres listed)":
        return []
    return [p.strip() for p in s.split("|") if p.strip()]


def _genre_weights_from_ratings(sub: pd.DataFrame, movies: pd.DataFrame, top_n: int = 5) -> List[Tuple[str, float]]:
    if sub.empty:
        return []
    mid_idx = movies.set_index("movieId", drop=False)
    counts: Dict[str, float] = {}
    for row in sub.itertuples(index=False):
        mid = int(row.movieId)
        if mid not in mid_idx.index:
            continue
        w = max(0.1, float(row.rating) - 2.5)
        for g in _parse_genres(mid_idx.loc[mid].get("genres")):
            counts[g] = counts.get(g, 0.0) + w
    return sorted(counts.items(), key=lambda t: t[1], reverse=True)[:top_n]


def _load_lightgcn() -> Optional[dict]:
    if not LIGHTGCN_MODEL.exists():
        return None
    try:
        with open(LIGHTGCN_MODEL, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None


def _load_svd() -> Optional[dict]:
    if not SVD_MODEL.exists():
        return None
    try:
        with open(SVD_MODEL, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None


def _load_lightgcn_user_set() -> set[int]:
    model = _load_lightgcn()
    if not model:
        return set()
    return set(int(u) for u in model.get("user_id_to_index", {}).keys())


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _build_cluster_centroids(
    gcn: dict, movies_df: pd.DataFrame, max_items_per_cluster: int = 250
) -> Dict[str, np.ndarray]:
    """Mean LightGCN item embedding per genre cluster (vector-space regions)."""
    item_ids = [int(x) for x in gcn["item_ids"]]
    item_emb = gcn["item_embeddings"]
    mid_to_i = {mid: i for i, mid in enumerate(item_ids)}
    centroids: Dict[str, np.ndarray] = {}

    if movies_df.empty:
        return centroids

    for cluster_name, genre_keys in GENRE_CLUSTERS.items():
        mask = np.zeros(len(item_ids), dtype=bool)
        for _, row in movies_df.iterrows():
            mid = int(row["movieId"])
            if mid not in mid_to_i:
                continue
            genres = _parse_genres(row.get("genres"))
            if any(g in genres for g in genre_keys):
                mask[mid_to_i[mid]] = True
        if not mask.any():
            continue
        idx = np.where(mask)[0]
        if len(idx) > max_items_per_cluster:
            idx = np.random.default_rng(42).choice(idx, size=max_items_per_cluster, replace=False)
        centroids[cluster_name] = np.mean(item_emb[idx], axis=0)
    return centroids


def _genre_affinity_from_csv(
    user_id: int, ratings_df: pd.DataFrame, movies_df: pd.DataFrame
) -> Dict[str, float]:
    sub = ratings_df[ratings_df["userId"] == int(user_id)] if not ratings_df.empty else pd.DataFrame()
    ranked = _genre_weights_from_ratings(sub, movies_df, top_n=50)
    return {g: float(w) for g, w in ranked}


def _genre_boost_for_movie(genre_aff: Dict[str, float], genres: List[str]) -> float:
    if not genre_aff or not genres:
        return 0.0
    gb = sum(genre_aff.get(p, 0.0) for p in genres)
    return W_GENRE * min(gb / max(len(genres), 1), 4.0)


def compute_score_decomposition(
    user_id: int,
    movie_id: int,
    gcn: dict,
    movies_df: pd.DataFrame,
    ratings_df: pd.DataFrame,
    db_session,
    *,
    extra_rating: Optional[Tuple[int, float]] = None,
    extra_watch: Optional[Tuple[int, float]] = None,
    seen_ids: Optional[set[int]] = None,
) -> Optional[Dict[str, Any]]:
    """
    Decompose final LightGCN score into report-friendly components:
      collaborative_filter  ≈ user_emb · item_emb
      genre_match           ≈ genre affinity boost
      live_behavior         ≈ rating + watch boosts − seen penalty
    """
    uid = int(user_id)
    mid = int(movie_id)
    user_to_idx = gcn.get("user_id_to_index", {})
    if uid not in user_to_idx:
        return None

    item_ids = [int(x) for x in gcn["item_ids"]]
    if mid not in item_ids:
        return None

    u_idx = int(user_to_idx[uid])
    i_idx = item_ids.index(mid)
    user_emb = gcn["user_embeddings"][u_idx]
    item_emb = gcn["item_embeddings"][i_idx]

    collaborative = float(np.dot(user_emb, item_emb))
    genre_aff = _genre_affinity_from_csv(uid, ratings_df, movies_df)

    for r in db_session.query(Rating).filter(Rating.user_id == uid).all():
        w = max(-1.0, min(1.0, (float(r.rating) - 3.5) / 1.5))
        row = movies_df[movies_df["movieId"] == int(r.movie_id)]
        if row.empty:
            continue
        for g in _parse_genres(row.iloc[0].get("genres")):
            genre_aff[g] = genre_aff.get(g, 0.0) + w * 0.6

    row = movies_df[movies_df["movieId"] == mid]
    genres = _parse_genres(row.iloc[0].get("genres")) if not row.empty else []
    genre_match = _genre_boost_for_movie(genre_aff, genres)

    rating_boost = 0.0
    watch_boost = 0.0
    for r in db_session.query(Rating).filter(Rating.user_id == uid).all():
        if int(r.movie_id) == mid:
            rating_boost += W_RATING * (float(r.rating) - 3.5)
    if extra_rating and int(extra_rating[0]) == mid:
        rating_boost += W_RATING * (float(extra_rating[1]) - 3.5)

    for w in db_session.query(WatchHistory).filter(WatchHistory.user_id == uid).all():
        if int(w.movie_id) != mid:
            continue
        pct = float(w.watch_percentage or 0.0)
        if pct >= 50.0:
            watch_boost += W_WATCH_FULL * (pct / 100.0)
        elif pct >= 25.0:
            watch_boost += W_WATCH_PARTIAL * (pct / 100.0)
    if extra_watch and int(extra_watch[0]) == mid:
        pct = float(extra_watch[1])
        if pct >= 50.0:
            watch_boost += W_WATCH_FULL * (pct / 100.0)
        elif pct >= 25.0:
            watch_boost += W_WATCH_PARTIAL * (pct / 100.0)

    seen = seen_ids if seen_ids is not None else set()
    for r in db_session.query(Rating).filter(Rating.user_id == uid).all():
        seen.add(int(r.movie_id))
    for w in db_session.query(WatchHistory).filter(WatchHistory.user_id == uid).all():
        if float(w.watch_percentage or 0.0) >= 50.0:
            seen.add(int(w.movie_id))
    seen_penalty = W_SEEN if mid in seen else 0.0

    live_behavior = rating_boost + watch_boost - seen_penalty
    final_score = collaborative + genre_match + live_behavior

    parts = {
        "collaborative_filter": collaborative,
        "genre_match": genre_match,
        "live_behavior": live_behavior,
    }
    abs_sum = sum(abs(v) for v in parts.values()) or 1.0
    weights_pct = {k: round(100.0 * abs(v) / abs_sum, 1) for k, v in parts.items()}
    dominant = max(weights_pct.items(), key=lambda t: t[1])[0]

    return {
        "movie_id": mid,
        "title": str(row.iloc[0]["title"]) if not row.empty else f"Movie {mid}",
        "collaborative_filter": round(collaborative, 6),
        "genre_match": round(genre_match, 6),
        "live_behavior": round(live_behavior, 6),
        "rating_boost": round(rating_boost, 6),
        "watch_boost": round(watch_boost, 6),
        "seen_penalty": round(seen_penalty, 6),
        "final_score": round(final_score, 6),
        "weight_pct_collaborative": weights_pct["collaborative_filter"],
        "weight_pct_genre": weights_pct["genre_match"],
        "weight_pct_live": weights_pct["live_behavior"],
        "dominant_component": dominant,
        "formula": (
            f"Score = (w_cf·Collaborative) + (w_g·Genre) + (w_live·Behavior) "
            f"= {collaborative:.4f} + {genre_match:.4f} + ({live_behavior:.4f}) = {final_score:.4f}"
        ),
    }


def _svd_vector_update(
    user_vector: np.ndarray, item_vector: np.ndarray, rating: float, lr: float = 0.2
) -> np.ndarray:
    centered = float(rating) - 2.5
    return (1.0 - lr) * user_vector + lr * (item_vector * centered)


def analyze_vector_displacement(
    user_id: int,
    action: Dict[str, Any],
    gcn: dict,
    svd: Optional[dict],
    movies_df: pd.DataFrame,
    centroids: Dict[str, np.ndarray],
) -> Dict[str, Any]:
    """Before/after vector proximity when a user rates or watches the action movie."""
    uid = int(user_id)
    mid = int(action["movie_id"])
    out: Dict[str, Any] = {
        "user_id": uid,
        "action_label": action.get("label", ""),
        "movie_id": mid,
        "movie_title": INCEPTION_TITLE,
    }

    cluster_sims_before: Dict[str, float] = {}
    cluster_sims_after: Dict[str, float] = {}

    user_to_idx = gcn.get("user_id_to_index", {})
    item_ids = [int(x) for x in gcn["item_ids"]]
    lightgcn_note = ""

    if uid in user_to_idx and mid in item_ids:
        u_idx = int(user_to_idx[uid])
        i_idx = item_ids.index(mid)
        u_before = np.asarray(gcn["user_embeddings"][u_idx], dtype=np.float64)
        item_vec = np.asarray(gcn["item_embeddings"][i_idx], dtype=np.float64)

        for cname, centroid in centroids.items():
            cluster_sims_before[cname] = round(_cosine_sim(u_before, centroid), 4)

        # LightGCN embeddings are fixed at serve time; illustrate pull toward item/cluster
        u_after_conceptual = u_before + 0.15 * item_vec
        for cname, centroid in centroids.items():
            cluster_sims_after[cname] = round(_cosine_sim(u_after_conceptual, centroid), 4)

        displacement_norm = float(np.linalg.norm(u_after_conceptual - u_before))
        out["lightgcn_embedding_norm_before"] = round(float(np.linalg.norm(u_before)), 4)
        out["lightgcn_conceptual_shift_norm"] = round(displacement_norm, 6)
        out["lightgcn_cosine_to_inception_item_before"] = round(_cosine_sim(u_before, item_vec), 4)
        out["lightgcn_cosine_to_inception_item_after_conceptual"] = round(
            _cosine_sim(u_after_conceptual, item_vec), 4
        )
        lightgcn_note = (
            "LightGCN offline user embeddings are fixed until retrain. "
            "Live reactions use score boosts (see weighted scoring). "
            "Conceptual shift (+15% item direction) shows proximity move toward Inception / Sci-Fi region."
        )
    else:
        lightgcn_note = "User or movie not in LightGCN index."

    out["cluster_cosine_before"] = cluster_sims_before
    out["cluster_cosine_after_conceptual"] = cluster_sims_after
    out["lightgcn_notes"] = lightgcn_note

    svd_block: Dict[str, Any] = {"available": False}
    if svd and uid in svd["user_ids"] and mid in svd["movie_ids"]:
        u_i = svd["user_ids"].index(uid)
        m_i = svd["movie_ids"].index(mid)
        u_vec = np.asarray(svd["user_factors"][u_i], dtype=np.float64)
        i_vec = np.asarray(svd["item_factors"][:, m_i], dtype=np.float64)
        rating = float(action.get("rating", 4.0))
        if action.get("action") == "watch":
            rating = 4.0 + 0.01 * float(action.get("watch_pct", 80.0))

        u_after = _svd_vector_update(u_vec, i_vec, rating)
        svd_block = {
            "available": True,
            "embedding_dim": int(u_vec.shape[0]),
            "vector_l2_before": round(float(np.linalg.norm(u_vec)), 4),
            "vector_l2_after": round(float(np.linalg.norm(u_after)), 4),
            "displacement_l2": round(float(np.linalg.norm(u_after - u_vec)), 6),
            "cosine_to_inception_item_before": round(_cosine_sim(u_vec, i_vec), 4),
            "cosine_to_inception_item_after": round(_cosine_sim(u_after, i_vec), 4),
            "update_rule": "u_new = (1−0.2)·u + 0.2·(item_vector × (rating−2.5))  [same as POST /api/rate]",
            "note": "SVD lives in a different vector space (dim≠LightGCN); cluster table uses LightGCN only.",
        }

    out["svd_vector_displacement"] = svd_block
    return out


def build_processing_analysis(
    user_ids: Optional[List[int]] = None,
    ratings_df: Optional[pd.DataFrame] = None,
    movies_df: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Weighted scoring table + vector displacement records."""
    user_ids = user_ids or sorted(DEMO_USERS.keys())
    if ratings_df is None:
        ratings_df = pd.read_csv(RATINGS_CSV)
    if movies_df is None:
        movies_df = pd.read_csv(MOVIES_CSV) if MOVIES_CSV.exists() else pd.DataFrame()

    gcn = _load_lightgcn()
    svd = _load_svd()
    if not gcn:
        raise FileNotFoundError(f"LightGCN model required: {LIGHTGCN_MODEL}")

    centroids = _build_cluster_centroids(gcn, movies_df)
    db = SessionLocal()
    scoring_rows: List[Dict[str, Any]] = []
    displacement_records: List[Dict[str, Any]] = []

    try:
        for uid in user_ids:
            action = DEMO_PROCESSING_ACTIONS.get(uid, DEMO_PROCESSING_ACTIONS[1])
            name = DEMO_USERS.get(uid, {}).get("name", f"User{uid}")

            before = compute_score_decomposition(uid, INCEPTION_MOVIE_ID, gcn, movies_df, ratings_df, db)
            extra_r = extra_w = None
            if action.get("action") == "rate":
                extra_r = (INCEPTION_MOVIE_ID, float(action["rating"]))
            elif action.get("action") == "watch":
                extra_w = (INCEPTION_MOVIE_ID, float(action["watch_pct"]))

            after = compute_score_decomposition(
                uid, INCEPTION_MOVIE_ID, gcn, movies_df, ratings_df, db,
                extra_rating=extra_r, extra_watch=extra_w,
            )

            disp = analyze_vector_displacement(uid, action, gcn, svd, movies_df, centroids)
            disp["score_on_inception_before"] = before
            disp["score_on_inception_after_action"] = after
            displacement_records.append(disp)

            if before and after:
                scoring_rows.append({
                    "user": name,
                    "user_id": uid,
                    "scenario": f"Score on {INCEPTION_TITLE}",
                    "phase": "Before action",
                    "collaborative_filter": before["collaborative_filter"],
                    "genre_match": before["genre_match"],
                    "live_behavior": before["live_behavior"],
                    "final_score": before["final_score"],
                    "pct_collaborative": before["weight_pct_collaborative"],
                    "pct_genre": before["weight_pct_genre"],
                    "pct_live": before["weight_pct_live"],
                    "dominant": before["dominant_component"],
                })
                scoring_rows.append({
                    "user": name,
                    "user_id": uid,
                    "scenario": action["label"],
                    "phase": "After action (live boost)",
                    "collaborative_filter": after["collaborative_filter"],
                    "genre_match": after["genre_match"],
                    "live_behavior": after["live_behavior"],
                    "final_score": after["final_score"],
                    "pct_collaborative": after["weight_pct_collaborative"],
                    "pct_genre": after["weight_pct_genre"],
                    "pct_live": after["weight_pct_live"],
                    "dominant": after["dominant_component"],
                })

            if uid in gcn.get("user_id_to_index", {}):
                uix = int(gcn["user_id_to_index"][uid])
                base = gcn["item_embeddings"] @ gcn["user_embeddings"][uix]
                top_i = int(np.argmax(base))
                top_mid = int(gcn["item_ids"][top_i])
                top_dec = compute_score_decomposition(uid, top_mid, gcn, movies_df, ratings_df, db)
                if top_dec:
                    scoring_rows.append({
                        "user": name,
                        "user_id": uid,
                        "scenario": f"Top collaborative pick: {top_dec['title'][:40]}",
                        "phase": "Current",
                        "collaborative_filter": top_dec["collaborative_filter"],
                        "genre_match": top_dec["genre_match"],
                        "live_behavior": top_dec["live_behavior"],
                        "final_score": top_dec["final_score"],
                        "pct_collaborative": top_dec["weight_pct_collaborative"],
                        "pct_genre": top_dec["weight_pct_genre"],
                        "pct_live": top_dec["weight_pct_live"],
                        "dominant": top_dec["dominant_component"],
                    })
    finally:
        db.close()

    scoring_df = pd.DataFrame(scoring_rows)

    weight_summary_rows = []
    for uid in user_ids:
        top_row = scoring_df[
            (scoring_df["user_id"] == uid) & (scoring_df["scenario"].str.startswith("Top collaborative"))
        ]
        inc_row = scoring_df[
            (scoring_df["user_id"] == uid) & (scoring_df["phase"] == "Before action")
        ]
        if top_row.empty:
            continue
        row = top_row.iloc[0]
        inc = inc_row.iloc[0] if not inc_row.empty else None
        weight_summary_rows.append({
            "user": DEMO_USERS[uid]["name"],
            "reference_movie": row["scenario"].replace("Top collaborative pick: ", ""),
            "dominant_signal": row["dominant"],
            "pct_collaborative_filter": row["pct_collaborative"],
            "pct_genre_match": row["pct_genre"],
            "pct_live_behavior": row["pct_live"],
            "dominant_on_inception": inc["dominant"] if inc is not None else "—",
            "interpretation": _dominance_interpretation(uid, row),
        })
    weight_summary = pd.DataFrame(weight_summary_rows)

    return scoring_df, weight_summary, displacement_records, list(centroids.keys())


def _dominance_interpretation(user_id: int, row: pd.Series) -> str:
    name = DEMO_USERS.get(user_id, {}).get("name", "User")
    dom = row["dominant"]
    if user_id == 1 and dom == "genre_match":
        return f"{name} (heavy CSV profile): genre match term dominates — taste from 200+ ratings shapes the blend."
    if user_id == 2 and dom == "collaborative_filter":
        return f"{name} (sparse profile): collaborative filter dominates — graph similarity leads over weak genre signal."
    if user_id == 3:
        return f"{name} (critical rater): mixed low ratings; {dom.replace('_', ' ')} leads on this title."
    if dom == "genre_match":
        return f"{name}: genre affinity term is largest for this movie."
    if dom == "collaborative_filter":
        return f"{name}: LightGCN dot-product (collaborative filter) is largest."
    return f"{name}: live behavior boosts/penalties dominate."


def _profile_label(train_count: int, mean_rating: float, top_genre: str) -> str:
    if train_count >= 100:
        return "Heavy rater — rich collaborative signal"
    if train_count >= 25:
        return "Moderate rater — usable but sparser profile"
    if train_count > 0:
        return "Light rater — limited historical signal"
    return "No CSV history — app-only signals"


def compute_user_profile(
    user_id: int,
    ratings_df: pd.DataFrame,
    movies_df: pd.DataFrame,
    in_lightgcn: set[int],
    db_session,
) -> Dict[str, Any]:
    uid = int(user_id)
    meta = DEMO_USERS.get(uid, {"name": f"User{uid}", "email": "unknown"})
    sub = ratings_df[ratings_df["userId"] == uid].copy() if not ratings_df.empty else pd.DataFrame()

    train_count = len(sub)
    unique_movies = int(sub["movieId"].nunique()) if train_count else 0
    mean_rating = round(float(sub["rating"].mean()), 2) if train_count else None
    std_rating = round(float(sub["rating"].std(ddof=0)), 2) if train_count > 1 else None
    high_ratio = (
        round(float((sub["rating"] >= 4.5).mean()), 2) if train_count else None
    )
    low_ratio = round(float((sub["rating"] <= 2.5).mean()), 2) if train_count else None

    genre_ranked = _genre_weights_from_ratings(sub, movies_df)
    top_genres = [g for g, _ in genre_ranked[:3]]
    top_genre_str = ", ".join(top_genres) if top_genres else "—"

    top_movies: List[str] = []
    if train_count and not movies_df.empty:
        merged = sub.merge(movies_df[["movieId", "title"]], on="movieId", how="left")
        for row in merged.nlargest(3, "rating").itertuples(index=False):
            title = str(getattr(row, "title", f"Movie {row.movieId}"))
            top_movies.append(f"{title} ({row.rating}★)")

    app_ratings = db_session.query(Rating).filter(Rating.user_id == uid).all()
    app_watches = db_session.query(WatchHistory).filter(WatchHistory.user_id == uid).all()

    return {
        "user_id": uid,
        "name": meta["name"],
        "email": meta["email"],
        "training_ratings_count": train_count,
        "unique_movies_rated": unique_movies,
        "mean_rating": mean_rating,
        "rating_std": std_rating,
        "high_rating_ratio_ge_4_5": high_ratio,
        "low_rating_ratio_le_2_5": low_ratio,
        "top_genres": top_genre_str,
        "top_rated_examples": "; ".join(top_movies) if top_movies else "—",
        "in_lightgcn_model": uid in in_lightgcn,
        "app_sqlite_ratings": len(app_ratings),
        "app_sqlite_watches": len(app_watches),
        "profile_type": _profile_label(train_count, mean_rating or 0.0, top_genres[0] if top_genres else ""),
    }


def build_comparison_table(user_ids: Optional[List[int]] = None) -> Tuple[pd.DataFrame, pd.DataFrame, List[Dict[str, Any]]]:
    """
    Returns:
      - wide_table: metrics as rows, users as columns (best for report paste)
      - per_user_table: one row per user
      - profiles: raw dict list
    """
    user_ids = user_ids or sorted(DEMO_USERS.keys())

    if not RATINGS_CSV.exists():
        raise FileNotFoundError(f"Ratings file not found: {RATINGS_CSV}")
    ratings_df = pd.read_csv(RATINGS_CSV)
    movies_df = pd.read_csv(MOVIES_CSV) if MOVIES_CSV.exists() else pd.DataFrame()
    in_gcn = _load_lightgcn_user_set()

    db = SessionLocal()
    try:
        profiles = [
            compute_user_profile(uid, ratings_df, movies_df, in_gcn, db) for uid in user_ids
        ]
    finally:
        db.close()

    per_user = pd.DataFrame(profiles)

    report_rows = [
        ("User ID", "user_id"),
        ("Name", "name"),
        ("Email", "email"),
        ("Training ratings (CSV input)", "training_ratings_count"),
        ("Unique movies rated", "unique_movies_rated"),
        ("Mean rating", "mean_rating"),
        ("Rating std dev", "rating_std"),
        ("Share of ratings ≥ 4.5", "high_rating_ratio_ge_4_5"),
        ("Share of ratings ≤ 2.5", "low_rating_ratio_le_2_5"),
        ("Top genre preferences", "top_genres"),
        ("Example highly rated titles", "top_rated_examples"),
        ("In LightGCN training set", "in_lightgcn_model"),
        ("Live app ratings (SQLite)", "app_sqlite_ratings"),
        ("Live watch events (SQLite)", "app_sqlite_watches"),
        ("Input profile summary", "profile_type"),
    ]

    wide_data = {"Metric": [label for label, _ in report_rows]}
    for p in profiles:
        col = f"{p['name']} (User {p['user_id']})"
        wide_data[col] = []
        for _, key in report_rows:
            val = p.get(key)
            if isinstance(val, bool):
                val = "Yes" if val else "No"
            elif val is None:
                val = "—"
            wide_data[col].append(val)

    wide_table = pd.DataFrame(wide_data)
    return wide_table, per_user, profiles


def _round_df(df: pd.DataFrame, decimals: int = 4) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_numeric_dtype(out[col]):
            out[col] = out[col].round(decimals)
    return out


def _format_input_table(per_user: pd.DataFrame) -> pd.DataFrame:
    """One row per user — easy to read in Excel/report."""
    rename = {
        "user_id": "User ID",
        "name": "Name",
        "email": "Email",
        "training_ratings_count": "Training ratings (CSV)",
        "unique_movies_rated": "Unique movies rated",
        "mean_rating": "Mean rating",
        "rating_std": "Rating std dev",
        "high_rating_ratio_ge_4_5": "Ratings >= 4.5 (share)",
        "low_rating_ratio_le_2_5": "Ratings <= 2.5 (share)",
        "top_genres": "Top genres",
        "top_rated_examples": "Top rated examples",
        "in_lightgcn_model": "In LightGCN",
        "app_sqlite_ratings": "App ratings (SQLite)",
        "app_sqlite_watches": "App watches (SQLite)",
        "profile_type": "Profile summary",
    }
    cols = [src for src in rename if src in per_user.columns]
    t = per_user[cols].rename(columns={src: rename[src] for src in cols})
    if "In LightGCN" in t.columns:
        t["In LightGCN"] = t["In LightGCN"].map({True: "Yes", False: "No"})
    return t


def _format_processing_action_table(displacement_records: List[Dict[str, Any]]) -> pd.DataFrame:
    """One row per user: before/after scores on Inception (core processing trace)."""
    rows = []
    for rec in displacement_records:
        uid = rec["user_id"]
        name = DEMO_USERS.get(uid, {}).get("name", f"User {uid}")
        before = rec.get("score_on_inception_before") or {}
        after = rec.get("score_on_inception_after_action") or {}
        rows.append(
            {
                "User": name,
                "User ID": uid,
                "Action": rec.get("action_label", ""),
                "Collaborative (before)": before.get("collaborative_filter"),
                "Genre match (before)": before.get("genre_match"),
                "Live behavior (before)": before.get("live_behavior"),
                "Final score (before)": before.get("final_score"),
                "Collaborative (after)": after.get("collaborative_filter"),
                "Genre match (after)": after.get("genre_match"),
                "Live behavior (after)": after.get("live_behavior"),
                "Final score (after)": after.get("final_score"),
                "Final score change": round(
                    float(after.get("final_score", 0) or 0) - float(before.get("final_score", 0) or 0),
                    4,
                ),
                "Dominant (before)": before.get("dominant_component"),
                "Dominant (after)": after.get("dominant_component"),
            }
        )
    return _round_df(pd.DataFrame(rows))


def _format_processing_components_table(scoring_df: pd.DataFrame) -> pd.DataFrame:
    """All score rows with short column names."""
    if scoring_df.empty:
        return scoring_df
    return _round_df(
        scoring_df.rename(
            columns={
                "user": "User",
                "user_id": "User ID",
                "scenario": "Scenario",
                "phase": "Phase",
                "collaborative_filter": "Collaborative",
                "genre_match": "Genre match",
                "live_behavior": "Live behavior",
                "final_score": "Final score",
                "pct_collaborative": "% Collaborative",
                "pct_genre": "% Genre",
                "pct_live": "% Live",
                "dominant": "Dominant term",
            }
        )
    )


def _format_vector_displacement_table(displacement_records: List[Dict[str, Any]]) -> pd.DataFrame:
    """One row per user — cluster cosines and SVD displacement."""
    rows = []
    for rec in displacement_records:
        uid = rec["user_id"]
        name = DEMO_USERS.get(uid, {}).get("name", f"User {uid}")
        row: Dict[str, Any] = {
            "User": name,
            "User ID": uid,
            "Action": rec.get("action_label", ""),
            "LCN shift norm": rec.get("lightgcn_conceptual_shift_norm"),
            "LCN cos to Inception (before)": rec.get("lightgcn_cosine_to_inception_item_before"),
            "LCN cos to Inception (after)": rec.get("lightgcn_cosine_to_inception_item_after_conceptual"),
        }
        for cname, val in (rec.get("cluster_cosine_before") or {}).items():
            short = cname.replace(" / ", "_").replace(" ", "_")
            row[f"Cos {short} (before)"] = val
            row[f"Cos {short} (after)"] = (rec.get("cluster_cosine_after_conceptual") or {}).get(cname)
        svd = rec.get("svd_vector_displacement") or {}
        if svd.get("available"):
            row["SVD displacement L2"] = svd.get("displacement_l2")
            row["SVD cos Inception (before)"] = svd.get("cosine_to_inception_item_before")
            row["SVD cos Inception (after)"] = svd.get("cosine_to_inception_item_after")
        rows.append(row)
    return _round_df(pd.DataFrame(rows))


def build_report_tables(
    user_ids: Optional[List[int]] = None,
) -> Dict[str, pd.DataFrame]:
    """All report tables as aligned DataFrames (one row per user where possible)."""
    wide, per_user, profiles = build_comparison_table(user_ids)
    ratings_df = pd.read_csv(RATINGS_CSV)
    movies_df = pd.read_csv(MOVIES_CSV) if MOVIES_CSV.exists() else pd.DataFrame()
    scoring_df, weight_summary, displacement_records, _ = build_processing_analysis(
        user_ids, ratings_df, movies_df
    )

    weight_clean = _round_df(
        weight_summary.rename(
            columns={
                "user": "User",
                "reference_movie": "Reference movie",
                "dominant_signal": "Dominant signal",
                "pct_collaborative_filter": "% Collaborative",
                "pct_genre_match": "% Genre",
                "pct_live_behavior": "% Live",
                "dominant_on_inception": "Dominant on Inception",
                "interpretation": "Notes",
            }
        )
    )

    return {
        "01_input_difference": _format_input_table(per_user),
        "02_processing_inception_action": _format_processing_action_table(displacement_records),
        "03_processing_all_steps": _format_processing_components_table(scoring_df),
        "04_weight_dominance": weight_clean,
        "05_vector_displacement": _format_vector_displacement_table(displacement_records),
    }


def _print_table(title: str, df: pd.DataFrame) -> None:
    print()
    print(title)
    print("-" * len(title))
    with pd.option_context(
        "display.max_columns",
        None,
        "display.width",
        200,
        "display.max_colwidth",
        40,
    ):
        print(df.to_string(index=False))
    print()


def export_report(
    output_dir: Path,
    user_ids: Optional[List[int]] = None,
) -> Path:
    """Write one combined CSV with every report table (filter by Section column)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Remove legacy extra CSV/JSON only; keep PNG/Mermaid and diagrams/
    for old in output_dir.iterdir():
        if old.is_dir():
            continue
        if not old.is_file() or old.name == REPORT_FILENAME:
            continue
        if old.suffix.lower() in {".png", ".mmd"}:
            continue
        if old.suffix.lower() in {".csv", ".json", ".md", ".txt"}:
            old.unlink()

    tables = build_report_tables(user_ids)
    report_path = output_dir / REPORT_FILENAME

    combined_parts = []
    for key, df in tables.items():
        part = df.copy()
        part.insert(0, "Section", SECTION_LABELS.get(key, key))
        combined_parts.append(part)
    pd.concat(combined_parts, ignore_index=True).to_csv(report_path, index=False)
    return report_path


def run_comparison(
    output_dir: Optional[Path] = None,
    user_ids: Optional[List[int]] = None,
    *,
    with_viz: bool = False,
    swap_test: bool = False,
) -> None:
    out = output_dir or DEFAULT_OUTPUT
    tables = build_report_tables(user_ids)
    report_path = export_report(out, user_ids)

    print("Comparative User Profiling — report tables")
    for key, df in tables.items():
        _print_table(SECTION_LABELS.get(key, key).upper(), df)

    print()
    print(f"Report file: {report_path.resolve()}")
    print("Use the Section column in Excel to filter each table.")

    if with_viz:
        from comparison.visualize_report import run_visualizations

        print()
        run_visualizations(out, user_ids)

    if swap_test:
        from comparison.user_swap_test import append_swap_section_to_report, run_user_swap_test

        print()
        print("Running user swap / system reactivity test...")
        timeline_df, stimulus_df, meta = run_user_swap_test(output_dir=out, restore_after=True)
        append_swap_section_to_report(out, timeline_df, stimulus_df)
        print(f"  Chart: {meta.get('chart_path', '—')}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build input + processing comparison tables for demo users (report)."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Directory for the combined report CSV",
    )
    parser.add_argument(
        "--user-ids",
        type=str,
        default="1,2,3",
        help="Comma-separated user IDs (default: 1,2,3)",
    )
    parser.add_argument(
        "--with-viz",
        action="store_true",
        help="Also generate heatmap, metrics charts, and Mermaid path diagrams",
    )
    parser.add_argument(
        "--swap-test",
        action="store_true",
        help="Run user swap sensitivity test (section 8) and append to report CSV",
    )
    args = parser.parse_args()
    ids = [int(x.strip()) for x in args.user_ids.split(",") if x.strip().isdigit()]
    run_comparison(
        args.output_dir,
        ids or None,
        with_viz=args.with_viz,
        swap_test=args.swap_test,
    )


if __name__ == "__main__":
    main()
