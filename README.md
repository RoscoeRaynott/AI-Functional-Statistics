# Functional Data Analysis with Clustering and LLMs

This Streamlit app allows you to upload functional data, perform clustering, and use large language models (LLMs) to interpret results. It is designed for easy deployment and exploration of time-series or curve data.

---

## 🚀 Features

* Upload CSVs with rows as functional observations (e.g., curves over time)
* Perform KMeans clustering
* Generate visualizations (line plots, heatmaps)
* Summarize clusters using statistics
* Use LLMs to:

  * Explain each cluster
  * Compare clusters
  * Answer user questions
  * Describe outliers
* Export Markdown report with LLM-generated insights

---

## 📁 Example CSV Format

Each row should represent one curve:

```csv
0.0,0.1,0.2,...,1.0
1.2,1.3,1.5,...,2.1
...
```

Ensure the columns represent ordered time points or measurement points.

---

## 🛠️ Installation

1. **Clone the repository**:

```bash
git clone https://github.com/yourusername/fda-cluster-llm.git
cd fda-cluster-llm
```

2. **Create Conda environment**:

```bash
conda create -n fda-llm python=3.10
conda activate fda-llm
```

3. **Install dependencies**:

```bash
pip install -r requirements.txt
```

> ✅ Alternatively, you can use `environment.yml` if you prefer conda-managed installs.

4. **Run the app**:

```bash
streamlit run app.py
```

