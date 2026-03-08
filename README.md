


# Trademarkia Semantic Search

A lightweight semantic search system built over the 20 Newsgroups corpus (~18,846 posts).
Implements fuzzy GMM clustering, a cluster-partitioned semantic cache built from first
principles, and a FastAPI service with full Swagger UI support.

---

## Architecture Overview


```
[Place your architecture diagram image here]
```

### Two-Stage Design

**Offline Pipeline (runs once on first startup — ~15-40 minutes on CPU)**
```
20 Newsgroups Dataset
        |
        v
   Preprocessor         remove headers/footers/quotes, strip emails/URLs,
                        filter posts < 20 words
        |
        v
   MiniLM Embedder      sentence-transformers/all-MiniLM-L6-v2
                        384-dim L2-normalized vectors
        |
        v
   FAISS IndexFlatIP    exact cosine search via inner product on unit vectors
        |
        v
   PCA (384 -> 64 dims) stability reduction before GMM
        |
        v
   GMM Fuzzy Clustering BIC sweep K=5..40, optimal K selected automatically
                        output: membership matrix (N, K), each row sums to 1.0
        |
        v
   Adaptive Thresholds  per-cluster threshold computed from intra-cluster
                        pairwise cosine similarity distributions
        |
        v
   Persist Artifacts    faiss_index.bin, doc_embeddings.npy,
                        cluster_model.pkl, pca_model.pkl,
                        cluster_membership.npy, metadata_map.json,
                        adaptive_thresholds.json
```

**Online Pipeline (FastAPI — loads artifacts in ~30 seconds on subsequent runs)**
```
POST /query
        |
        v
   Embed query (MiniLM)
        |
        v
   GMM predict_proba -> cluster membership distribution
        |
        v
   SemanticCache lookup  multi-bucket search (all clusters > 0.20 membership)
   [HIT]  return cached result instantly
   [MISS] FAISS top-k search -> store in cache -> return results
```

---

## Problem Statement

Given the 20 Newsgroups dataset (~20,000 news posts across 20 categories), build:

1. **Fuzzy clustering** of the corpus using vector embeddings and a vector database
2. **A semantic cache** that avoids redundant computation on similar queries, built from
   first principles without Redis or any caching middleware
3. **A FastAPI service** that exposes the cache as a live API with proper state management

---

## Implementation Details

### Part 1 — Preprocessing and Embedding

**Dataset loading:**
`fetch_20newsgroups(remove=('headers', 'footers', 'quotes'))` strips metadata that would
cause the model to cluster by sender domain rather than by topic. Quoted replies are
removed because they repeat prior posts verbatim and inflate spurious similarity.

**Cleaning decisions:**
- Emails and URLs are stripped entirely — MiniLM treats them as noise tokens and they
  introduce institutional bias (e.g., `mit.edu` addresses creating fake similarity)
- Posts under 20 words are filtered — these are one-line reactions with no semantic signal
- No stopword removal or stemming — MiniLM is trained on natural sentences and degrades
  when grammatical structure is broken

**Result:** 16,771 documents retained from 18,846 raw posts (2,075 filtered)

**Embedding model: `sentence-transformers/all-MiniLM-L6-v2`**

Chosen for three reasons:
- 384-dimensional output, runs fully on CPU, no API keys required
- Consistently top-ranked on the MTEB semantic similarity benchmark among small models
- Fully offline and reproducible — no network calls at inference time

All embeddings are L2-normalized at encoding time so that FAISS inner product equals
cosine similarity exactly.

**Vector store: FAISS `IndexFlatIP`**

Exact exhaustive search chosen over approximate methods (HNSW, IVFPQ) because:
- 16,771 vectors at 384 dimensions completes an exhaustive search in ~2-5ms
- Approximate indexing adds engineering complexity with no practical latency benefit at
  this scale
- No server process required — the index lives in process memory

A Python dict `{faiss_id -> {text, category, cluster_distribution}}` bridges FAISS
integer indices back to document content and cluster membership.

---

### Part 2 — Fuzzy Clustering

**Why not hard clustering:**
A post about gun legislation does not belong exclusively to `talk.politics.guns` or
`talk.politics.misc`. It belongs to both with varying degrees of membership. Hard
cluster assignment (k-means, DBSCAN) destroys this information. GMM preserves it.

**Algorithm: Gaussian Mixture Model (GMM)**

GMM was chosen over Fuzzy C-Means because it provides a principled probabilistic output
— the posterior `P(cluster | document)` derived via Bayes' theorem through Expectation-
Maximization. FCM is distance-based and assumes spherical clusters, which is a poor
model for high-dimensional embedding space.

**PCA reduction (384 → 64 dimensions) before GMM:**
Fitting full covariance GMM in 384-D requires estimating 384×384 covariance matrices per
component. With ~16K samples these become near-singular. PCA to 64 dimensions:
- Retains ~55% of variance (acceptable for broad topic clustering)
- Reduces covariance matrix size from 384×384 to 64×64 per component
- Improves EM convergence and reduces wall-clock time significantly

**BIC sweep for cluster count selection:**

```
BIC = -2 * log_likelihood + n_parameters * log(n_samples)
```

BIC penalizes model complexity proportionally to log(N). The K that minimizes BIC
represents the best balance between data fit and parsimony. We swept K=5 to K=40.

**Result: Optimal K=7**

| K | BIC Score |
|---|-----------|
| 5 | -2,501,217 |
| 6 | -2,509,724 |
| **7** | **-2,515,728 (minimum)** |
| 8 | -2,510,960 |
| 10 | -2,502,492 |
| 20 | -2,399,614 |
| 40 | -2,097,320 |

K=7 is not K=20 (the original category count). The data shows that the semantic
embedding space has 7 broad groupings — topic categories that were taxonomically
separate in the original labelling collapse into single semantic clusters
(e.g., `comp.sys.mac.hardware` and `comp.sys.ibm.pc.hardware` merge into one hardware
cluster because MiniLM embeds them near-identically).

**BIC Curve:**


analysis/results/bic_curve.png


The curve shows a clean global minimum at K=7 with monotonic rise afterward — strong
evidence that K=7 is not a convenience choice but a data-driven one.

**Output: Membership matrix shape (16771, 7)**
Each row is a probability distribution summing to 1.0. Example:

```
Document: "Gun legislation debate in Congress"
Cluster 0 (hardware/tech):  0.02
Cluster 1 (sports):         0.01
Cluster 2 (general):        0.08
Cluster 3 (science/med):    0.04
Cluster 4 (politics/guns):  0.78   <- dominant
Cluster 5 (religion):       0.05
Cluster 6 (space/science):  0.02
```

**UMAP Visualization:**


analysis/results/umap_clusters.png


Cluster 0 (blue) is tightly isolated in the bottom-left — a highly distinct topic domain
with almost no overlap. Clusters 2, 3, and 4 show natural overlap in the centre —
these are the boundary documents the assignment calls the most interesting. 30 boundary
documents (where top-2 cluster memberships differ by less than 0.10) are saved to
`analysis/results/boundary_documents.csv`.

---

### Part 3 — Semantic Cache

**Design: Custom Python Singleton, no external dependencies.**

Redis, Memcached, and all caching libraries are explicitly excluded. The cache is
implemented entirely in Python in-process memory.

**Data structure:**
```python
_store: Dict[cluster_id (int), List[CacheEntry]]
```

Entries are partitioned into per-cluster buckets. This is not an aesthetic choice —
it is a scaling decision. As the cache grows to N entries, a flat list requires O(N)
comparisons per lookup. Cluster-partitioned lookup reduces this to O(N/K) on average.
For K=7, this is a 7x reduction in comparisons with no approximation.

**Multi-bucket lookup (not just argmax):**
A query with cluster membership `[0.02, 0.55, 0.08, 0.30, 0.05, 0.00, 0.00]` searches
both Cluster 1 (55%) and Cluster 3 (30%) because a cached semantically-equivalent query
may have been stored under either bucket. Any cluster exceeding 0.20 membership is
searched.

**Similarity metric:**
Cosine similarity via dot product on L2-normalized embeddings. Since all embeddings are
unit-normalized at encode time: `dot(query, cached) = cosine_sim(query, cached)`.

**Threshold sweep analysis:**

The similarity threshold `theta` is the most consequential design variable in the cache:

| theta | Hit Rate | Precision | False Positives | Behaviour |
|-------|----------|-----------|-----------------|-----------|
| 0.70  | 41.7%    | 1.00      | 0.000           | Catches clear paraphrases |
| 0.75  | 33.3%    | 1.00      | 0.000           | Slightly stricter |
| 0.80  | 8.3%     | 1.00      | 0.000           | Misses most paraphrases |
| 0.85+ | 0.0%     | 1.00      | 0.000           | Cache effectively disabled |

MiniLM paraphrase pairs on this corpus max out at ~0.82 cosine similarity. Setting
theta above 0.82 prevents any paraphrase from ever hitting the cache. The default
is `CACHE_SIM_THRESHOLD=0.70` — empirically verified as the correct operating point
for this model and corpus.

**Adaptive Per-Cluster Threshold (novel extension):**

A single global threshold assumes all clusters have the same semantic density — the UMAP
disproves this. Cluster 0 (isolated tight blob) has a higher mean intra-cluster
similarity than Cluster 2 (wide spread). Each cluster should use a threshold calibrated
to its own local geometry.

Formula:
```
theta_k = clip(mean_pairwise_sim_k + alpha * std_pairwise_sim_k, lower, upper)
```

Computed from pairwise cosine similarities among the cluster's member documents (sampled
up to 300 per cluster for efficiency). Dense clusters receive stricter thresholds. Sparse
clusters receive looser thresholds. The global threshold is retained as fallback.

This means the clustering from Part 2 is not only used for cache partitioning (O(N/K)
lookup) but also for threshold calibration — the cluster structure does double duty in
the cache system.

---

### Part 4 — FastAPI Service

**Startup pattern: `lifespan` context manager (not deprecated `@app.on_event`)**

All heavy artifacts (FAISS index, GMM, PCA, metadata map, adaptive thresholds,
SentenceTransformer) are loaded once into `app.state` at startup. Request handlers
read from `app.state` — no artifact is loaded or re-initialized per request.

**Endpoints:**

| Method | Path | Description |
|--------|------|-------------|
| POST | `/query` | Semantic search with cache lookup |
| GET | `/cache/stats` | Cache hit rate and entry count |
| DELETE | `/cache` | Flush cache and reset stats |
| GET | `/health` | Service health and artifact status |

**POST /query response:**
```json
{
  "query": "What is the impact of firearm regulations?",
  "cache_hit": true,
  "matched_query": "How do gun laws affect crime?",
  "similarity_score": 0.7369,
  "result": [
    {
      "rank": 1,
      "text": "...",
      "category": "talk.politics.guns",
      "similarity_score": 0.6292
    }
  ],
  "dominant_cluster": 4
}
```

**Logging:**
Rotating file logger (`logs/app.log`, 10MB per file, 5 backups) logs every significant
event: query received, cache hit/miss with score and adaptive threshold used, artifact
load times, pipeline stage completion, and all errors with full context.

---

## Analysis Artifacts

All outputs saved to `analysis/results/`:

| File | Description |
|------|-------------|
| `bic_curve.png` | BIC score vs K, dashed line at optimal K=7 |
| `bic_scores.csv` | Raw BIC scores for all K values |
| `umap_clusters.png` | 2D UMAP projection coloured by dominant cluster |
| `boundary_documents.csv` | 30 documents with top-2 cluster membership gap < 0.10 |
| `cluster_top_docs.csv` | Representative documents per cluster |
| `threshold_analysis.png` | Hit rate / precision / false positive rate vs theta |
| `threshold_analysis.csv` | Raw threshold sweep data |
| `adaptive_thresholds.png` | Per-cluster similarity histograms with threshold lines |
| `adaptive_thresholds.csv` | cluster_id, mean_sim, std_sim, adaptive_threshold, n_docs |

---

## Project Structure

```
trademarkia-semantic-search/
├── app/
│   ├── main.py                  FastAPI app, lifespan, build pipeline orchestration
│   ├── config.py                Settings (pydantic-settings), all artifact paths
│   ├── models.py                Pydantic request/response schemas
│   ├── pipeline/
│   │   ├── preprocessor.py      Dataset loading and cleaning
│   │   ├── embedder.py          MiniLM encoding (corpus + query)
│   │   ├── indexer.py           FAISS build/load/search, metadata map
│   │   ├── clusterer.py         PCA, BIC sweep, GMM, membership matrix
│   │   └── adaptive_threshold.py Per-cluster threshold computation
│   ├── cache/
│   │   └── semantic_cache.py    Singleton cache, cluster-partitioned store/lookup
│   └── api/
│       └── routes.py            FastAPI endpoint handlers
├── analysis/
│   ├── cluster_analysis.py      BIC plot, UMAP, boundary docs, adaptive threshold plot
│   ├── threshold_analysis.py    Threshold sweep experiment
│   └── results/                 All generated plots and CSVs
├── data/
│   ├── artifacts/               FAISS index, GMM, PCA, embeddings, thresholds
│   └── metadata/                metadata_map.json
├── logs/                        Rotating application logs
├── logger/
│   └── logging_config.py        Logger configuration (console + rotating file)
├── Dockerfile
├── .env.example
└── requirements.txt
```

---

## Setup and Running

### Prerequisites
- Python 3.11+
- 4GB RAM minimum (8GB recommended for build pipeline)
- ~2GB disk space for artifacts

### Without Docker

```bash
git clone https://github.com/YOUR_USERNAME/trademarkia-semantic-search.git
cd trademarkia-semantic-search

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt

cp .env.example .env            # Default values work out of the box

uvicorn app.main:app --host 0.0.0.0 --port 8000
```

**First run:** The system automatically runs the full build pipeline —
preprocessing, embedding (MiniLM), FAISS indexing, BIC sweep (K=5..40), GMM fitting,
membership matrix computation, and adaptive threshold calibration. This takes
**15-40 minutes on CPU**. All artifacts are persisted to `data/artifacts/`.

**Subsequent runs:** Artifacts are detected and loaded from disk. Server is ready in
approximately **30 seconds**.

Open Swagger UI: `http://localhost:8000/docs`

### With Docker

```bash
cp .env.example .env

docker build -t trademarkia-semantic-search .
docker run -p 8000:8000 --env-file .env trademarkia-semantic-search
```

> If `data/artifacts/` already contains built artifacts, include them before building
> the Docker image. The container will skip the 40-minute pipeline and start in ~30s.

### Pull from Docker Hub

```bash
docker pull manojdj/trademarkia-semantic-search:latest
docker run -p 8000:8000 manojdj/trademarkia-semantic-search:latest
```

---

## Environment Variables

Copy `.env.example` to `.env`. All defaults are production-ready and do not need to be
changed for a standard run.

```env
CACHE_SIM_THRESHOLD=0.70        # Global similarity threshold fallback (0.0 - 1.0)
CACHE_MULTI_BUCKET_THRESHOLD=0.2 # Min cluster membership to include in cache search
TOP_K_RESULTS=5                 # Number of FAISS results returned per query
PCA_COMPONENTS=64               # PCA output dimensions before GMM
GMM_K_MIN=5                     # BIC sweep lower bound
GMM_K_MAX=40                    # BIC sweep upper bound
LOG_LEVEL=INFO                  # DEBUG for verbose pipeline logs
```

---

## Running Analysis Scripts

After the server has completed its first build, run analysis independently:

```bash
# Generate all cluster analysis artifacts (BIC plot, UMAP, boundary docs, adaptive threshold plot)
python -m analysis.cluster_analysis

# Run threshold sweep experiment
python -m analysis.threshold_analysis
```

Outputs are saved to `analysis/results/`.

---

## Key Design Decisions Summary

| Decision | Choice | Reason |
|----------|--------|--------|
| Embedding model | `all-MiniLM-L6-v2` | CPU-efficient, offline, top MTEB score for size |
| Vector store | FAISS `IndexFlatIP` | Exact search sufficient at 16K vectors, no server needed |
| Clustering | GMM with full covariance | Probabilistic soft assignments, handles ellipsoidal clusters |
| Cluster count | BIC-selected K=7 | Data-driven, not assumed from original 20 categories |
| Cache structure | `Dict[cluster_id, List[CacheEntry]]` | O(N/K) lookup, not O(N) |
| Cache threshold | Adaptive per-cluster | Calibrated to local semantic density of each cluster |
| API pattern | FastAPI lifespan + `app.state` | Artifacts loaded once, never per-request |
| Logging | Rotating file handler | Production-grade, persistent across restarts |
```



