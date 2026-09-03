# Football Transfer Intelligence Explorer

This repository contains the source code and supporting materials for the MSc Data Science Extended Research Project:

**Football Transfer Intelligence Explorer: Interactive Visual Analytics of Football Transfer Networks**

The project transforms player-level football transfer records into a directed, weighted club network and combines network analysis, Louvain community detection, Node2Vec embeddings, PCA and interactive visualisation.

## Project Overview

The system supports multi-level exploration of football transfer networks, from geographical and cross-border patterns to club relationships, communities, structural similarity and player-level transfer evidence.

The main analytical components are:

- Directed weighted club-level network construction
- Degree and weighted-degree analysis
- PageRank centrality
- Louvain community detection
- Node2Vec graph embeddings
- PCA projection for two-dimensional visual exploration
- Interactive visualisation using Streamlit and Plotly

The final analysed network contains:

- 3,606 clubs
- 31,272 directed club-to-club relationships
- 66,960 valid inter-club transfer records
- 7 Louvain communities

## Repository Structure

```text
football-transfer-intelligence-explorer/
│
├── app.py
├── requirements.txt
├── README.md
│
├── notebooks/
│   ├── 01_data_exploration_and_preprocessing.ipynb
│   ├── 02_network_construction.ipynb
│   ├── 03_louvain_clustering.ipynb
│   ├── 04_node2vec_pca.ipynb
│   ├── 05_interactive_visualisation.ipynb
│   └── README.md
│
├── data/
│   ├── README.md
│   └── processed/
│       ├── cleaned_transfers.csv
│       ├── club_edges.csv
│       ├── club_embeddings.csv
│       ├── club_nodes_with_clusters.csv
│       ├── louvain_cluster_summary.csv
│       ├── network_summary.csv
│       ├── top_countries_by_louvain_cluster.csv
│       └── README.md
│
├── figures/
└── docs/
```

## Data Source

The raw football transfer dataset used in this project is publicly available from Kaggle:

**Football Transfer Dataset**

https://www.kaggle.com/datasets/mexwell/football-transfer-dataset

The raw `transfers.csv` file is not redistributed in this repository.

After downloading the dataset, place the file in:

```text
data/raw/transfers.csv
```

The preprocessing notebook reconstructs source and target clubs, removes invalid or non-club endpoints, prepares numerical attributes and generates the processed data used in the analysis.

## Installation

Install the required Python packages using:

```bash
pip install -r requirements.txt
```

The main dependencies include pandas, NumPy, NetworkX, python-louvain, Node2Vec, gensim, scikit-learn, matplotlib, Plotly and Streamlit.

## Reproducing the Analysis

Run the notebooks in numerical order:

1. `01_data_exploration_and_preprocessing.ipynb`
   - data audit
   - missing-value analysis
   - transfer-direction reconstruction
   - data cleaning

2. `02_network_construction.ipynb`
   - directed weighted club-network construction
   - network measures
   - PageRank analysis

3. `03_louvain_clustering.ipynb`
   - Louvain community detection
   - community summaries

4. `04_node2vec_pca.ipynb`
   - 64-dimensional Node2Vec embeddings
   - PCA projection for visual exploration

5. `05_interactive_visualisation.ipynb`
   - preparation of outputs used by the interactive application

Processed outputs required by the application are included in `data/processed/`.

## Running the Interactive Application

Run the Streamlit application using:

```bash
streamlit run app.py
```

The application contains:

- Story Mode
- Country Overview
- Network Drill-down
- Temporal Evolution
- Club Intelligence
- Community Patterns
- Transfer Evidence

## Deployed Application

The deployed Football Transfer Intelligence Explorer is available at:

https://lacey106-football-transfer.hf.space

## Notes on Reproducibility

Computationally intensive analytical outputs, including network measures, Louvain assignments and Node2Vec embeddings, were generated offline and exported before application execution.

Fixed random seeds were used where supported to improve reproducibility. Node2Vec was executed with a single worker.

The two-dimensional PCA coordinates are used for visual exploration only. Club similarity in the application is calculated using Euclidean distance in the full 64-dimensional Node2Vec embedding space.

## Author

MSc Data Science  
The University of Manchester  
2026
