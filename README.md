# JoPACC Payment Intelligence

A Flask app for exploring Jordan's national payment systems month by month. It uses figures from the monthly reports of the Jordan Payments and Clearing Company (JoPACC), covering November 2022 to June 2026 across five systems: ACH, CliQ, ECCU, JoMoPay and eFAWATEERcom.

The app has three pages:

| Page | Route | What it does |
|---|---|---|
| Monthly patterns | `/cluster` | Clusters 214 system-month observations, labels each pattern and flags unusual months |
| Report questions | `/text` | Retrieval-augmented Q&A over the original PDF reports; answers come with the passages they used |
| Activity graph | `/graph` | Turns one system's monthly indicator into activity states and analyses the transitions as a directed graph |

It started as a data mining coursework project. The notebooks in `notebooks/` hold the original analysis, and the Python modules are the reusable versions the app runs on.

## How it works

### Clustering (`activity2_analysis.py`)

Each row is one payment system in one month. Systems differ in scale by orders of magnitude, so every feature is measured against that system's own historical mean:

- `relative_transaction_level`: transactions this month / the system's average
- `relative_value_level`: value in JOD this month / the system's average
- `relative_average_transaction_value`: average ticket size / the system's average

The features are standardised and projected onto two principal components. Five algorithms are then compared on the PCA coordinates:

| Algorithm | Clusters | Noise | Silhouette | Calinski-Harabasz | Davies-Bouldin | Stability (ARI) |
|---|---|---|---|---|---|---|
| **K-means** | 3 | 0% | 0.590 | **371.9** | 0.571 | **0.998** |
| Hierarchical (Ward) | 3 | 0% | 0.578 | 290.1 | **0.509** | 0.861 |
| DBSCAN | 5 | 8.9% | 0.579 | 162.0 | 0.531 | 0.949 |
| BIRCH | 3 | 0% | 0.499 | 298.1 | 0.705 | 0.924 |
| HDBSCAN | 5 | 10.7% | **0.596** | 179.9 | 0.551 | 0.936 |

Stability is the mean adjusted Rand index after adding small random noise to the coordinates 20 times. K-means is the final model because it keeps every month, separates the clusters most clearly and barely changes under that noise. HDBSCAN's silhouette edge comes from discarding about 11% of months as noise.

The three clusters come out as *Normal activity*, *High activity with smaller transactions* and *Low activity with larger transactions*. A month is marked **unusual** when it sits further from its cluster centre than that cluster's 95th percentile. Pick a single month on the page and it also lists the five most similar months in PCA space.

### Report Q&A (`rag_analysis.py`)

1. Extract text page by page with `pypdf`. If a report yields under 1,000 characters (a scanned PDF), fall back to OCR with RapidOCR when it's installed.
2. Split pages into 200-word chunks with a 40-word overlap, keeping the source file, month and page number.
3. Embed the chunks with `all-MiniLM-L6-v2` and retrieve the top five by cosine similarity.
4. Send the passages to an LLM through OpenRouter with a prompt that restricts it to the supplied context. A top score below 0.45 flags the evidence as insufficient, but the model still reads the passages and only replies that the answer isn't in the reports when none of them help. It asks a clarifying question when the request is genuinely ambiguous.

### Activity graph (`graph_analysis.py`)

For the chosen system, date range and indicator, the months are ranked and split into four equal groups: Low, Medium, High and Very High. Each pair of consecutive months becomes a directed edge between their states, weighted by how often that transition happens. The page draws the graph and reports in/out degree, weighted frequencies, PageRank, closeness, betweenness and the longest run spent in a single state.

## Running it

Requires Python 3.10 or newer.

```bash
python -m venv .venv
# Windows:      .venv\Scripts\activate
# macOS/Linux:  source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

Then open <http://127.0.0.1:5000>. The clustering and graph pages work straight away.

The report Q&A page needs two extra things.

**The PDF reports.** These aren't in the repository because together they're close to 200 MB. Download the English monthly reports from JoPACC's website and put them in a `Reports/` folder next to `app.py`. The month and year are read from the filename, so keep names like `april_monthly_report_2025_en.pdf`.

**An OpenRouter API key**, set as an environment variable:

```bash
# macOS/Linux
export OPENROUTER_API_KEY="your-key"
# Windows PowerShell
$env:OPENROUTER_API_KEY = "your-key"
```

The default model is `minimax/minimax-m3:free`; set `OPENROUTER_MODEL` to use a different one. The first question builds the embedding index and caches it in `rag_index_cache.pkl`. Delete that file whenever you add or remove reports.

Scanned reports need the optional OCR packages:

```bash
pip install pypdfium2 rapidocr_onnxruntime
```

## Notebooks

| Notebook | Contents |
|---|---|
| `Activity_2.ipynb` | Data validation, feature engineering, PCA and the five-way clustering comparison |
| `Activity_3.ipynb` | The RAG pipeline step by step (expects `../Reports/` and `OPENROUTER_API_KEY`) |
| `Activity_4.ipynb` | Graph mining on ACH transaction volume for 2023: unweighted and weighted directed graphs with their measures |

Run them from inside `notebooks/`; they read the data from the repository root. `Activity_2` also uses `seaborn`.

## Data

- `jopacc_system_month_november_2022_to_june_2026.csv`: one row per payment system per month, built from JoPACC's public monthly reports. It holds aggregate figures only (user counts, transaction counts and values, demographic splits, cheque returns), with no individual or transaction-level records.
- `jopacc_clusters.csv`: the same observations with the clustering output added (cluster, PCA coordinates, distance to centre, normal/unusual state and pattern label).

## Project layout

```
app.py                  Flask routes and template filters
activity2_analysis.py   Clustering pipeline and charts
rag_analysis.py         PDF extraction, retrieval and OpenRouter call
graph_analysis.py       State-transition graph and graph measures
templates/, static/     Page templates and CSS
notebooks/              Original analysis notebooks
```
