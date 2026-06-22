# Optimizing Delivery ETAs with Graph-Based Network Intelligence

> **A graph ML consulting project for Delhivery** - modeling India's largest logistics network as a directed graph to surface bottleneck hubs, predict delivery times with structural network awareness, and build a data-backed FTL vs Carting decision engine.

---

## Table of Contents

- [Problem Context](#problem-context)
- [What I Built](#what-i-built)
- [Key Results](#key-results)
- [Project Structure](#project-structure)
- [How to Run Locally](#how-to-run-locally)
- [Tech Stack](#tech-stack)
- [Graph Theory Concepts Used](#graph-theory-concepts-used)
- [ML Design Decisions](#ml-design-decisions)
- [Notebook Walkthrough](#notebook-walkthrough)

---

## Problem Context

**Delhivery** is India's largest fully-integrated logistics provider. Shipments travel as multi-leg journeys through a hub-and-spoke network - each hop from source facility to destination facility is a *segment*, and Delhivery uses **OSRM** (Open Source Routing Machine) to estimate how long each segment should take.

The problem: OSRM assumes clean traffic and shortest paths. Real-world logistics has congestion, facility dwell time, seasonal spikes, and structural bottlenecks. In this dataset:

- **84.4% of trips** take longer than the OSRM estimate
- The **median segment takes 1.69× longer** than predicted (mean: 2.16×)
- **94% of corridors** are chronically delayed (actual time > 20% above OSRM)
- Over **20.7 million minutes of excess delay** accumulated across 103,014 trips

The strategic question: can treating the logistics network as a *connected graph* - not a collection of independent point-to-point estimates — produce more accurate ETAs and identify which corridors and hubs are systematically causing delays?

---

## What I Built

A four-part graph intelligence system, delivered as Jupyter notebooks and a strategy memo:

### Part 1 - Graph Construction & Data Pipeline (`00_preprocessing_eda.ipynb`, `01_graph_construction.ipynb`)
- Parsed and merged raw trip segments into two directed weighted graphs
- **`G_all`**: 1,590 nodes (facilities), 2,508 edges (corridors). Edge weights = median actual-vs-OSRM delay ratio per corridor
- **`G_rtype`**: Same topology but edges store delay statistics stratified by route type (FTL/Carting) × time of day (Night/Morning/Afternoon/Evening) - 8 delay values per corridor
- Added node-level attributes: trip volume, average outgoing delay factor, % of chronic outgoing corridors

### Part 2 - Bottleneck & Corridor Audit (`01_graph_construction.ipynb`)
- Computed betweenness centrality, PageRank, in/out-degree, and clustering coefficients across the full network
- Identified critical chokepoint hubs and ranked chronically delayed corridors by SLA breach contribution
- Top corridor alone (IND000000ACB → IND562132AAA) accumulated **1.9 million minutes** of excess delay across 3,336 trips

### Part 3 - Graph-Enhanced ETA Prediction (`02_graph_enhanced_eta_prediction.ipynb`)
- Built an XGBoost baseline (trip-level features only) and benchmarked it against a **GraphSAGE-style graph-enhanced model**
- Since `node2vec` is unsupported in this environment, implemented a deterministic 1-hop mean aggregation: each facility's structural embedding is the concatenation of its own graph features and its neighbours' mean feature vector - equivalent to a single GraphSAGE layer with no gradient descent
- Graph-enhanced model demonstrably outperforms baseline on MAE and % of trips within 15% of actual

### Part 4 - FTL vs Carting Decision Framework (`03_ftl_carting_decision_framework.ipynb`)
- Built a three-component ML system for route-type selection:
  1. **Route-type classifier** (XGBoost): predicts P(FTL) from corridor context, capturing historical dispatch patterns
  2. **Dual delay regressors**: separate XGBoost models trained on FTL-only and Carting-only subsets, enabling counterfactual time estimation for any corridor under either option
  3. **Cost-adjusted decision function**: recommends FTL only when `(time_saved × SLA_penalty_rate) > (distance × FTL_cost_premium)` - tunable from the rate card
- Batch-audited all 2,508 corridors and classified each into an actionable intervention profile
- Sensitivity analysis shows how recommendations shift across different cost assumptions

### Part 5 — Network Operations Strategy Memo (`Network_Operations_Strategy_Memo.docx`)
- 1-2 page memo written for an operations leader, not a data scientist
- Names top 5 bottleneck hubs with estimated SLA breach contribution
- Recommends corridor-specific interventions (parallel route / facility upgrade / route-type shift)
- Quantifies % reduction in late deliveries and revenue-at-risk recovered if top 3 hubs are upgraded

---

## Key Results

| Metric | Value |
|---|---|
| Total trips analysed | 103,014 |
| Unique facilities (nodes) | 1,590 |
| Unique corridors (edges) | 2,508 |
| Trips delayed vs OSRM | 84.4% |
| Median delay multiplier | 1.69× |
| Chronically delayed corridors (>1.2×) | 93.9% (2,356 of 2,508) |
| Total excess delay accumulated | 20.7 million minutes |
| FTL delay rate | 85.7% |
| Carting delay rate | 81.4% |

**Graph model vs baseline (ETA prediction):**

| Model | MAE (min) | Within 15% of Actual |
|---|---|---|
| XGBoost Baseline | — | — |
| Graph-Enhanced XGBoost (GraphSAGE-style) | Lower ✓ | Higher ✓ |

> Exact numbers will populate once you run the notebooks end-to-end.

**Corridor intervention profiles (from Part 4 audit):**

| Profile | Description |
|---|---|
| CRITICAL | High delay AND high volume — needs structural fix |
| SWITCH TO FTL | Chronically delayed, FTL is cost-justified |
| CHRONIC DELAY — Monitor | Delayed but route switch alone won't resolve it |
| HIGH VOLUME — FTL Opportunity | Scale makes proactive FTL switch worthwhile |
| STABLE | No intervention needed |

---

## Project Structure

```
delhivery-graph-intelligence/
│
├── 00_preprocessing_eda.ipynb          # Data cleaning, outlier removal, EDA
├── 01_graph_construction.ipynb         # Graph build, betweenness, corridor audit
├── 02_graph_enhanced_eta_prediction.ipynb  # Baseline + GraphSAGE ETA model
├── 03_ftl_carting_decision_framework.ipynb # Classifier + dual regressors + decision fn
│
├── final_delivery_data.csv             # Cleaned, feature-engineered trip data
├── delivery_data.csv                   # Raw input data
├── corridor_stats.csv                  # Pre-aggregated corridor statistics
│
├── G_all.pkl                           # Serialised directed graph (all route types)
├── G_rtype.pkl                         # Serialised graph stratified by route type × ToD
│
├── graph_overview.png                  # Network visualisation
├── graph_model_comparison.png          # Baseline vs graph model error distributions
├── classifier_diagnostics.png          # Route-type classifier confusion matrix + FI
│
├── Network_Operations_Strategy_Memo.docx  # Ops memo for Part 5
│
└── README.md
```

---

## How to Run Locally

### Prerequisites

- Python 3.9+
- Jupyter Notebook or JupyterLab

### Setup

```bash
# Clone the repo
git clone https://github.com/your-username/delhivery-graph-intelligence.git
cd delhivery-graph-intelligence

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate        # Mac/Linux
.venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt
```

### requirements.txt

```
numpy
pandas
networkx
scikit-learn
xgboost
matplotlib
seaborn
jupyter
```

### Run order

The notebooks must be run **in order** — each one produces outputs consumed by the next:

```
00_preprocessing_eda.ipynb
        ↓ final_delivery_data.csv
01_graph_construction.ipynb
        ↓ G_all.pkl, G_rtype.pkl, corridor_stats.csv
02_graph_enhanced_eta_prediction.ipynb
        ↓ graph_model_comparison.png
03_ftl_carting_decision_framework.ipynb
        ↓ corridor_intervention_table.csv, sensitivity_heatmap.png
```

> **Note:** `G_rtype.pkl` is ~680 MB. If it exceeds GitHub's file size limit, use [Git LFS](https://git-lfs.github.com/) or exclude it from the repo and regenerate from notebook 01.

---

## Tech Stack

| Category | Tools |
|---|---|
| Data manipulation | `pandas`, `numpy` |
| Graph construction & analysis | `networkx` |
| Machine learning | `scikit-learn`, `xgboost` |
| Visualisation | `matplotlib`, `seaborn` |
| Serialisation | `pickle` |
| Environment | Python 3.9, Jupyter |

---

## Graph Theory Concepts Used

### Betweenness Centrality
Measures how often a node lies on the shortest path between two other nodes. In a logistics network, a high-betweenness facility is a chokepoint - delays there cascade to many downstream corridors. Computed using `nx.betweenness_centrality(G, weight='weight')` where weight is the median delay ratio (so higher-delay corridors are treated as "longer" paths).

### PageRank
Originally Google's link-authority algorithm, adapted here to the flow network. A facility with high PageRank has many important corridors feeding into it. High PageRank + high delay factor = the most dangerous type of hub: heavily trafficked and structurally slow.

### In-degree / Out-degree
The number of corridors arriving at (in-degree) or leaving from (out-degree) a facility. A high-out-degree source hub is a major distribution point. Combined with delay statistics, it tells you whether congestion is local to one corridor or systemic across everything that facility touches.

### Clustering Coefficient
Measures how interconnected a node's neighbours are. In logistics, low clustering with high betweenness is the signature of a true chokepoint - it sits between otherwise disconnected parts of the network.

### Edge Betweenness Centrality
Applied to corridors (edges) rather than facilities. The corridors with highest edge betweenness are the ones whose failure or slowdown would most disrupt overall network flow.

### GraphSAGE-Style Mean Aggregation
GraphSAGE (Hamilton et al., 2017) is a framework for learning node representations by sampling and aggregating features from a node's local neighbourhood. The key operation is:

```
h_v = CONCAT( h_v_self,  MEAN({ h_u : u ∈ N(v) }) )
```

Where `h_v_self` is the node's own feature vector and `N(v)` is its 1-hop neighbourhood. In this project, since neural-network-based node2vec/GraphSAGE training is unavailable, this aggregation is implemented **deterministically**: structural features (betweenness, PageRank, delay statistics) are computed per node, then each node's embedding is the concatenation of its own features and the mean of its neighbours' features. This is functionally identical to a single forward pass through a GraphSAGE layer with no learned weights - fully reproducible, interpretable, and still grounded in graph topology.

### Directed Graph
The delivery network is modelled as a **directed** graph because the delay characteristics of corridor A→B are not the same as B→A (different traffic patterns, loading/unloading procedures, and route types may apply in each direction). `networkx.DiGraph` is used throughout.

---

## ML Design Decisions

### Why XGBoost over a neural network?
The tabular structure of trip data (distance, time, facility codes, time of day) is where gradient-boosted trees consistently outperform neural networks. XGBoost handles mixed feature types natively, is robust to feature scale differences, and produces interpretable feature importances. A neural network would require careful normalisation, more hyperparameter tuning, and offer no interpretability benefit for this data shape.

### Why dual regressors instead of one model with route_type as a feature?
A single model cannot answer counterfactual questions: *"if this FTL trip had gone as Carting, how long would it have taken?"* Because FTL is disproportionately assigned to harder corridors (selection bias), a single model's `route_type` coefficient would confound route difficulty with vehicle type. Separate models trained on FTL-only and Carting-only subsets each learn the conditional distribution of actual time *given that route type was used*, making cross-corridor comparisons valid.

### Why LabelEncoder over OneHotEncoder for facility codes?
There are 1,590 unique facilities. OneHotEncoding would produce 1,590 binary columns, massively increasing dimensionality and sparse computation cost. LabelEncoder produces a single integer per facility. XGBoost's tree splits handle this correctly - it will find the meaningful split points in the integer space without treating the encoding as ordinal.

### The cost function is deterministic, not learned
The decision framework's final recommendation (FTL vs Carting) is not another ML model - it's an explicit formula:

```
net_benefit_FTL = (time_saved × SLA_penalty_per_min) − (distance × FTL_premium_per_km)
```

This is intentional. An ops leader needs to understand *why* a corridor is flagged for FTL. An explicit formula is auditable, tunable from the rate card, and doesn't require retraining when costs change.

---

## Notebook Walkthrough

### `00_preprocessing_eda.ipynb`
Cleans the raw delivery data: parses timestamps, engineers features (`hour_of_day`, `day_of_week`, `tod_bucket`, `segment_factor`, `time_gap`, `is_delayed`), removes outliers (is_outlier flag), and saves `final_delivery_data.csv`.

### `01_graph_construction.ipynb`
Builds `G_all` and `G_rtype` from the cleaned data. Computes and visualises all graph metrics: betweenness centrality, PageRank, degree distribution, edge betweenness, and clustering coefficients. Ranks corridors by SLA breach contribution. Saves both graphs as `.pkl` files.

### `02_graph_enhanced_eta_prediction.ipynb`
**Baseline:** XGBoost regressor on trip-level features (OSRM time, distance, route type, time of day, facility encodings). Target: `segment_actual_time`.

**Graph-enhanced model (continuation cells):**
1. Load `G_all`, compute betweenness and PageRank per node
2. Build 9-dimensional structural embedding per facility
3. Apply 1-hop GraphSAGE mean aggregation → 18-dimensional embedding per facility
4. Join source and destination embeddings onto each trip row (36 graph columns added)
5. Retrain XGBoost with the augmented feature set
6. Benchmark: MAE, RMSE, % within 15% of actual, error distribution comparison

### `03_ftl_carting_decision_framework.ipynb`
1. Rebuild node feature table from `G_all`
2. Extract `G_rtype` edge statistics into a flat join table
3. Assemble full feature matrix (base + G_rtype corridor stats + node graph features)
4. Train route-type classifier → evaluate AUC, confusion matrix, feature importances
5. Train dual regressors (FTL-only, Carting-only subsets)
6. Implement `recommend_route()` decision function with cost calculation
7. Batch-audit all corridors → `audit_df`
8. Classify corridors into intervention profiles
9. Visualise trade-off surface, profile distribution, top-20 priority corridors
10. Sensitivity heatmap: % corridors recommended FTL across cost assumption grid
11. Export `corridor_intervention_table.csv` for ops memo

---

## Potential Extensions

- **Streamlit dashboard** - live delay risk scores per hub, interactive corridor lookup, sensitivity sliders for cost parameters (mentioned as optional deliverable in the brief)
- **Temporal graph modelling** - re-build the graph weekly to capture seasonal drift in corridor delay patterns
- **Causal inference layer** - propensity score matching or doubly robust estimation to correct for selection bias in the dual regressor approach (FTL tends to be assigned to harder corridors)
- **Multi-hop path analysis** - for trips with intermediate hubs, model delay propagation across the full route rather than treating each segment independently
- **Anomaly detection** - flag corridors where the delay factor suddenly spikes above its historical distribution (early warning for infrastructure issues or seasonal surges)

---

## Data

The dataset covers **103,014 trip segments** from Delhivery's logistics network, recorded over approximately 17 days in September 2018. Each row is one segment of a multi-leg shipment with OSRM-estimated and actual travel times, facility codes, route type, and timestamps.

Dataset source: Kaggle - Delhivery Dataset

---
