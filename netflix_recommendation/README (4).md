# Netflix Recommendation System using a Transformer Two-Tower Model

A content-aware recommendation system built in **PyTorch**, trained on the public **Netflix Titles** catalog. It uses a **two-tower architecture** in which both towers are Transformer encoders: one encodes a *user's watch history*, the other encodes an *item's text metadata*. Users and items are projected into a shared 64-dimensional space, and recommendations are produced by ranking items by dot-product similarity to the user.

> **Notebook:** `Recommendation_using_two_tower.ipynb` (written for Google Colab with a GPU)

---

## Table of Contents

1. [Overview](#1-overview)
2. [Key Features](#2-key-features)
3. [Dataset](#3-dataset)
4. [Pipeline at a Glance](#4-pipeline-at-a-glance)
5. [Step-by-Step Walkthrough](#5-step-by-step-walkthrough)
6. [Model Architecture](#6-model-architecture)
7. [Training Setup](#7-training-setup)
8. [Inference and Recommendation](#8-inference-and-recommendation)
9. [Evaluation](#9-evaluation)
10. [Results](#10-results)
11. [Limitations and Honest Caveats](#11-limitations-and-honest-caveats)
12. [Suggested Improvements](#12-suggested-improvements)
13. [Saved Artifacts](#13-saved-artifacts)
14. [Installation and Usage](#14-installation-and-usage)
15. [Hyperparameter Reference](#15-hyperparameter-reference)
16. [Project Structure](#16-project-structure)

---

## 1. Overview

Two-tower models are the standard first-stage ("candidate generation") architecture in large-scale recommenders such as those used for video, ads, and e-commerce. The idea:

- A **user tower** maps a user to a vector.
- An **item tower** maps an item to a vector.
- The **relevance score** is the dot product of the two vectors.

Because the towers are independent, all item vectors can be **precomputed once**, and serving a user is just one user-vector computation followed by a matrix multiplication (or an approximate nearest-neighbour search at scale).

This project implements that pattern end to end: data loading, cleaning, SQLite storage, synthetic interaction generation, tokenization, model definition, training with negative sampling, ranking, metric evaluation, and artifact export.

## 2. Key Features

- **Transformer-based user tower** that reads a user's last 50 watched titles as a sequence (with positional embeddings and padding masks).
- **Transformer-based item tower** that reads a title's text (title, type, genres, description) as a token sequence.
- **L2-normalised embeddings**, so the dot product equals cosine similarity (range −1 to 1).
- **Custom tokenizer and vocabulary** (no pretrained language model needed), capped at 20,000 words.
- **Binary cross-entropy training with random negative sampling.**
- **Top-K recommendation** with already-watched titles filtered out.
- **Ranking metrics:** Recall@K, HitRate@K, NDCG@K for K ∈ {5, 10, 20}.
- **SQLite persistence** of the catalog and interactions.
- **One-click export** of weights, vocabulary, mappings, config, metrics, and recommendations as a ZIP.

## 3. Dataset

| Property | Value |
|---|---|
| Source | Netflix Titles CSV (`netflix_titles.csv`), uploaded manually in Colab |
| Rows (titles) | 8,807 |
| Columns | 12 |
| Columns | `show_id, type, title, director, cast, country, date_added, release_year, rating, duration, listed_in, description` |

**Notes on data quality** (from the notebook's inspection step):

- `director`, `cast`, `country`, `date_added`, and `rating` contain missing values (e.g. `director` is non-null for only 6,173 rows, `cast` for 7,982).
- The dataset contains both **Movies** and **TV Shows**; a bar chart of this split is generated.
- A top-10 genre (`listed_in`) frequency table is printed.

## 4. Pipeline at a Glance

```
netflix_titles.csv
        │
        ▼
 Clean + fill NaNs ──► build "content" text field
        │                     │
        ▼                     ▼
  SQLite (netflix_titles)   Tokenize ► Vocabulary (20k) ► pad to 64 tokens
        │                     │
        ▼                     ▼
 Synthetic user interactions   item_sequences  [8807 × 64]
   (10 users × 100 watches)
        │
        ▼
 Per-user 80/20 split ──► user histories (last 50 items, padded)
        │
        ▼
 Build (history, positive, negative) training triples
        │
        ▼
 Train TwoTowerTransformer (BCE loss, AdamW, 10 epochs)
        │
        ▼
 Precompute item vectors + user vectors ► score matrix [10 × 8807]
        │
        ▼
 Recommend top-K  ►  Evaluate Recall / HitRate / NDCG  ►  Export artifacts
```

## 5. Step-by-Step Walkthrough

### 5.1 Environment setup
Installs `pandas, numpy, scikit-learn, torch, matplotlib`, checks for a CUDA GPU, and selects the device (`cuda` if available, otherwise `cpu`). The run recorded in the notebook used PyTorch 2.11.0 on a GPU.

### 5.2 Data loading and exploration
Loads the CSV, prints shape, columns, `info()`, missing-value counts, Movie/TV Show distribution, and the top 10 genre strings, followed by a bar chart.

### 5.3 Cleaning
- Drops duplicate `show_id`s.
- Fills missing values in all text columns with an empty string and casts them to `str`.
- Coerces `release_year` to integer (invalid → 0).
- Resets the index so that **row position = item index**.

### 5.4 Content text construction
A single lowercase `content` column is built by concatenating:

```
title + type + listed_in (genres) + description
```

Director, cast, country, and rating are cleaned but **not** included in `content`, so the item tower never sees them.

### 5.5 SQLite storage
The cleaned catalog is written to a local SQLite database (`netflix_recommendations.db`) as the table `netflix_titles`, and verified with `SELECT` queries (8,807 titles). Later, the interaction log is stored in a second table, `user_interactions`.

### 5.6 Synthetic user interactions
The dataset has **no real user behaviour**, so interactions are simulated:

- **10 users**, each with a hand-defined genre taste, for example:
  - User 1: Action, Crime, Thrillers
  - User 2: Documentaries, Docuseries
  - User 3: Romantic, Dramas
  - User 4: Comedies, Stand-Up Comedy
  - User 5: Horror, Thrillers
  - User 6: Children, Family Movies
  - User 7: International, TV Dramas
  - User 8: Science, Nature
  - User 9: Sports, Action
  - User 10: Mysteries, Crime
- Each user "watches" **100 titles**. With probability **0.8** the title is sampled from titles whose `listed_in` matches the user's genres; with probability **0.2** it is sampled uniformly from the whole catalog (exploration/noise).
- Seeds are fixed (`random.seed(42)`, `np.random.seed(42)`) for reproducibility.
- Result: **1,000 interactions** (all labelled `watched = 1`).

### 5.7 Tokenization and vocabulary
- Tokenizer: lowercase, regex `[a-z0-9]+`.
- Vocabulary: the **20,000 most frequent words** plus `<PAD>` (id 0) and `<UNK>` (id 1) → **size 20,002**.
- Each title's `content` is truncated or padded to **`MAX_LEN = 64`** tokens, giving a tensor `item_sequences` of shape **[8807, 64]**.

### 5.8 ID mapping and train/validation split
- `show_id → item_idx` (0…8806) and `user_id → user_idx` (0…9).
- A **stratified-by-user random 80/20 split** is applied to each user's interactions: **800 training / 200 validation** interactions (80 / 20 per user).

### 5.9 User history tensors
For each user, the training item indices are collected and converted into a fixed-length sequence of **`MAX_HISTORY = 50`**: the last 50 items are kept, and shorter histories are padded with the special ID `NUM_ITEMS` (8807), which is the padding index of the user tower's embedding table. This produces `user_history_tensor` of shape **[10, 50]**, used at inference time.

## 6. Model Architecture

Both towers share the same recipe: **embedding → + positional embedding → Transformer encoder → masked mean pooling → linear projection → L2 normalisation.**

### 6.1 Item Tower (text → vector)
| Component | Detail |
|---|---|
| Input | 64 token IDs from the item's `content` |
| Token embedding | `Embedding(20002, 64, padding_idx=0)` |
| Positional embedding | `Embedding(64, 64)` (learned) |
| Encoder | 2 × `TransformerEncoderLayer` (d_model 64, 4 heads, FFN 128, dropout 0.1, batch-first) |
| Padding | `src_key_padding_mask = (tokens == 0)` |
| Pooling | Mean over non-padding positions |
| Output | `Linear(64, 64)` then L2-normalise → 64-d unit vector |

### 6.2 User Tower (history → vector)
| Component | Detail |
|---|---|
| Input | Up to 50 previously watched item indices |
| Item-ID embedding | `Embedding(8808, 64, padding_idx=8807)` |
| Positional embedding | `Embedding(50, 64)` (learned) |
| Encoder | 2 × `TransformerEncoderLayer` (same configuration as above) |
| Padding | `src_key_padding_mask = (history == NUM_ITEMS)` |
| Pooling | Mean over non-padding positions |
| Output | `Linear(64, 64)` then L2-normalise → 64-d unit vector |

### 6.3 Combined model
```python
score = sum(user_vector * item_vector, dim=1)   # cosine similarity in [-1, 1]
```
The trained network has **1,993,344 trainable parameters**. Most of them sit in the two embedding tables (20,002 × 64 for words and 8,808 × 64 for item IDs).

## 7. Training Setup

**Training examples** are built per training interaction as a triple `(history, positive_item, negative_item)`:

- **history:** the user's training items *excluding the current positive item* (capped to the last 50 and padded).
- **positive_item:** the item the user actually watched.
- **negative_item:** one item drawn uniformly at random from items the user did **not** watch in training.

This gives **800 triples**, loaded in batches of **32** (25 batches per epoch).

**Objective:** each batch scores the positive and negative items, concatenates the scores, and applies `BCEWithLogitsLoss` with labels 1 (positive) and 0 (negative).

| Setting | Value |
|---|---|
| Loss | `BCEWithLogitsLoss` |
| Optimizer | AdamW |
| Learning rate | 1e-3 |
| Weight decay | 1e-5 |
| Batch size | 32 |
| Epochs | 10 |

**Training loss (recorded run):**

| Epoch | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| Loss | 0.6924 | 0.6684 | 0.6287 | 0.5876 | 0.5411 | 0.5015 | 0.4783 | 0.4429 | 0.4236 | 0.4186 |

The loss falls steadily from about ln 2 (0.693, i.e. random guessing) to 0.42.

## 8. Inference and Recommendation

1. **Item embeddings:** all 8,807 items are encoded in batches of 256 → `item_vectors` of shape [8807, 64].
2. **User embeddings:** all 10 user histories are encoded → `user_vectors` of shape [10, 64].
3. **Score matrix:** `user_vectors @ item_vectors.T` → [10, 8807].
4. **`recommend_for_user(user_id, top_k=10)`:**
   - takes the user's score row,
   - sets the scores of already-watched training items to `-inf`,
   - returns the top-K titles with `title, type, listed_in, release_year, score`.

Example usage:

```python
recommend_for_user(user_id=0, top_k=10)
```

The notebook demonstrates this for users 0 and 1. For user 0, the top results included titles such as *Fauda*, *Tremors 5: Bloodline*, *Maron*, *Smoke & Mirrors*, *Gridlocked*, and *Pusher*, with cosine scores around 0.99.

## 9. Evaluation

For each user, the **relevant set** is the validation items not present in training. All items are ranked (training items excluded) and three metrics are computed at **K = 5, 10, 20**, then averaged across users:

| Metric | Meaning |
|---|---|
| **Recall@K** | Fraction of the user's relevant items that appear in the top K |
| **HitRate@K** | 1 if at least one relevant item is in the top K, otherwise 0 |
| **NDCG@K** | Rank-aware gain: relevant items near the top score higher (normalised by the ideal ranking) |

## 10. Results

Metrics from the recorded run (averaged over 10 users):

| K | Recall | HitRate | NDCG |
|---|---|---|---|
| 5 | 0.0000 | 0.0 | 0.0000 |
| 10 | 0.0059 | 0.1 | 0.0066 |
| 20 | 0.0118 | 0.1 | 0.0086 |

The numbers are very low, which is explained in the next section.

## 11. Limitations and Honest Caveats

This notebook is best read as a **working proof-of-concept of the architecture and pipeline**, not a validated recommender. Please keep the following in mind.

1. **Synthetic data.** Interactions are generated from simple genre rules plus 20% random noise. The model is therefore learning a simulated behaviour, and results will not transfer to real user behaviour.
2. **Tiny scale.** 10 users, 800 training interactions, and 8,807 items. A Transformer with about 2M parameters is far larger than this data can support.
3. **Weak evaluation results.** HitRate@10 of 0.1 means only 1 of 10 users received a hit. Recall@10 of 0.006 is only a few times higher than random chance (roughly 10 / 8,807 ≈ 0.001). The model has not learned to rank held-out items well, even though the training loss decreased.
4. **Likely overfitting to user identity.** For a given user, every training example uses nearly the same history (all their other training items), so the user tower can effectively memorise the user and the classifier mostly learns "items this user watched vs. random items" rather than generalisable taste.
5. **Item-ID embeddings in the user tower.** The user tower embeds watched items by ID, so it cannot represent **new/unseen items** in a history (cold-start) and shares no information with the item tower's text understanding.
6. **Limited item features.** Only title, type, genres, and description are used; cast, director, country, rating, and release year are ignored.
7. **Evaluation scoring.** The score matrix uses the user vectors computed from the *training* history only, which is correct, but the metrics rest on just 20 validation items per user, so they are very noisy.
8. **No hyperparameter tuning, no early stopping, no validation loss tracking** during training, and the random negatives are "easy" (random items are rarely hard confusers).
9. **Colab-specific code.** `google.colab.files` is used for upload/download and will fail elsewhere unless adapted.
10. **Minor code note.** The training loop runs the full model twice per batch (once for positives, once for negatives), recomputing the user tower each time. This is correct but wasteful.

## 12. Suggested Improvements

- **Use real interaction data** (e.g. MovieLens, or logged Netflix-style watch events) and a **time-based split** (train on earlier events, validate on later ones).
- **Build user history representations more carefully:** for each target, use only items watched *before* it (a sequential next-item setup) rather than all other training items.
- **Share or reuse the item tower** to embed items inside the user's history (e.g. average or Transformer over item-tower vectors) to remove the dependence on item IDs and support cold-start.
- **In-batch negatives with a softmax/InfoNCE loss** and **hard-negative mining** instead of one random negative per positive.
- **Richer item features:** cast, director, country, rating, release year, duration, plus a pretrained text encoder (e.g. a sentence-transformer) for descriptions.
- **Regularisation and tuning:** larger dropout, smaller embedding dimension or fewer layers for small data, early stopping on a validation metric.
- **More users and a proper baseline** (popularity, TF-IDF content-based, matrix factorisation) so the Transformer's value can be measured.
- **ANN index** (FAISS, ScaNN) for serving at scale.
- **Add diversity and re-ranking** steps after candidate generation.

## 13. Saved Artifacts

The final cells export everything to `netflix_model/` and zip it as `Netflix_Transformer_Two_Tower.zip`:

| File | Description |
|---|---|
| `two_tower_transformer.pth` | Trained model `state_dict` |
| `vocabulary.pkl` | Word → ID vocabulary |
| `item_to_idx.pkl` / `idx_to_item.pkl` | `show_id` ↔ item index mappings |
| `config.json` | `vocab_size, num_items, embed_dim, num_heads, num_layers, max_len, max_history` |
| `evaluation_metrics.csv` | Recall / HitRate / NDCG at K = 5, 10, 20 |
| `recommendations.csv` | Top-10 recommendations for every user |
| `netflix_recommendations.db` | SQLite DB with `netflix_titles` and `user_interactions` tables |

**Reloading the model later:**

```python
import json, pickle, torch

with open("netflix_model/config.json") as f:
    cfg = json.load(f)
with open("netflix_model/vocabulary.pkl", "rb") as f:
    vocab = pickle.load(f)

# Re-define ItemTower, UserTower, TwoTowerTransformer and the globals
# NUM_HEADS, NUM_LAYERS, MAX_LEN, MAX_HISTORY, NUM_ITEMS exactly as in the notebook.
model = TwoTowerTransformer(vocab_size=cfg["vocab_size"], num_items=cfg["num_items"])
model.load_state_dict(torch.load("netflix_model/two_tower_transformer.pth", map_location="cpu"))
model.eval()
```

Note that the model classes read `NUM_HEADS`, `NUM_LAYERS`, `MAX_LEN`, `MAX_HISTORY`, and `NUM_ITEMS` from global variables, so these must be defined (values are in `config.json`) before instantiating the model.

## 14. Installation and Usage

**Requirements**

- Python 3.9+
- `torch`, `pandas`, `numpy`, `scikit-learn`, `matplotlib`
- Optional: a CUDA GPU (the model is small and also runs on CPU)

```bash
pip install pandas numpy scikit-learn torch matplotlib
```

**Run on Google Colab (as written)**

1. Open the notebook in Colab (a GPU runtime is optional but recommended).
2. Run the cells in order; when prompted, upload `netflix_titles.csv`.
3. After training, the final cell downloads `Netflix_Transformer_Two_Tower.zip`.

**Run locally**

1. Place `netflix_titles.csv` in the working directory.
2. Replace the Colab upload cell (`from google.colab import files; files.upload()`) with nothing, and remove the final `files.download(...)` cell.
3. Run the remaining cells in order.

## 15. Hyperparameter Reference

| Name | Value | Purpose |
|---|---|---|
| `EMBED_DIM` | 64 | Embedding / model dimension |
| `NUM_HEADS` | 4 | Attention heads per layer |
| `NUM_LAYERS` | 2 | Transformer layers per tower |
| FFN dim | 128 | Feed-forward hidden size |
| Dropout | 0.1 | Transformer dropout |
| `MAX_LEN` | 64 | Max tokens per item text |
| `MAX_HISTORY` | 50 | Max watched items per user |
| Vocabulary size | 20,002 | 20,000 words + `<PAD>` + `<UNK>` |
| `BATCH_SIZE` | 32 | Training batch size |
| `EPOCHS` | 10 | Training epochs |
| Learning rate | 1e-3 | AdamW |
| Weight decay | 1e-5 | AdamW |
| Train/val split | 80/20 | Per user, random, seed 42 |

## 16. Project Structure

```
.
├── Recommendation_using_two_tower.ipynb   # Main notebook
├── netflix_titles.csv                     # Input dataset (user-provided)
├── netflix_recommendations.db             # Generated SQLite database
└── netflix_model/                         # Generated artifacts
    ├── two_tower_transformer.pth
    ├── vocabulary.pkl
    ├── item_to_idx.pkl
    ├── idx_to_item.pkl
    ├── config.json
    ├── evaluation_metrics.csv
    ├── recommendations.csv
    └── netflix_recommendations.db
```

---

**Dataset credit:** Netflix Movies and TV Shows dataset (publicly available on Kaggle).
