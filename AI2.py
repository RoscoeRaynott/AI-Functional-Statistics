import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import EllipticEnvelope
from llm_utils import explain_cluster, compare_clusters, answer_user_query, explain_outlier

st.set_page_config(page_title="Functional Data Clustering", layout="wide")
st.title("📈 Functional Data Analysis with Clustering and LLMs")

# Load API key securely
api_key = st.secrets['OPENROUTER_API_KEY']

# ----------------------------- #
# 📁 Upload Data
# ----------------------------- #
file = st.file_uploader("Upload a CSV file with functional data (rows = curves)")
if file:
    df = pd.read_csv(file)
    st.write("Data Preview:", df.head())

    # ----------------------------- #
    # ⚙️ Feature Engineering (Optional smoothing, derivs can be added here)
    # ----------------------------- #
    timepoints = df.columns.astype(float) if df.columns.str.replace('.', '', 1).str.isnumeric().all() else np.arange(df.shape[1])

    # ----------------------------- #
    # 📊 Clustering
    # ----------------------------- #
    n_clusters = st.slider("Number of clusters", 2, 10, 3)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)
    df["Cluster"] = clusters

    st.subheader("📉 Clustered Functional Curves")
    fig, ax = plt.subplots()
    for cluster_id in range(n_clusters):
        for row in df[df["Cluster"] == cluster_id].drop("Cluster", axis=1).values:
            ax.plot(timepoints, row, alpha=0.4, label=f"Cluster {cluster_id}" if cluster_id == 0 else "")
    st.pyplot(fig)

    # ----------------------------- #
    # 📦 Cluster Summary Stats
    # ----------------------------- #
    cluster_summary = {}
    for cluster_id in range(n_clusters):
        cluster_data = df[df["Cluster"] == cluster_id].drop("Cluster", axis=1)
        summary = {
            "mean": cluster_data.mean().mean(),
            "std": cluster_data.stack().std(),
            "min": cluster_data.min().min(),
            "max": cluster_data.max().max()
        }
        cluster_summary[cluster_id] = summary

    # ----------------------------- #
    # 🤖 LLM Comparison
    # ----------------------------- #
    if st.button("Compare Clusters via LLM"):
        st.subheader("🔍 LLM Comparison of Clusters")
        comparison = compare_clusters(cluster_summary, api_key)
        st.markdown(comparison)

    # ----------------------------- #
    # 📄 Generate Markdown Report
    # ----------------------------- #
    def generate_report(cluster_summary, explanations):
        report = "# Functional Data Cluster Report\n\n"
        for cid, stats in cluster_summary.items():
            report += f"## Cluster {cid}\n\n"
            report += f"**Stats:** {stats}\n\n"
            report += f"**LLM Insight:** {explanations.get(cid, 'N/A')}\n\n"
        return report

    if st.button("Download Cluster Report (Markdown)"):
        explanations = {cid: explain_cluster(stats, api_key) for cid, stats in cluster_summary.items()}
        report_md = generate_report(cluster_summary, explanations)
        st.download_button("Download Report", report_md, file_name="cluster_report.md")

    # ----------------------------- #
    # 💬 Ask a Question
    # ----------------------------- #
    st.subheader("💬 Ask a Question About Your Data")
    user_query = st.text_input("Ask something like: 'Which cluster has the steepest rise?'" )
    if st.button("Ask LLM"):
        if user_query.strip():
            answer = answer_user_query(user_query, api_key)
            st.markdown("**LLM Response:**")
            st.write(answer)

    # ----------------------------- #
    # 🔥 Heatmap
    # ----------------------------- #
    st.subheader("📊 Heatmap of Functional Curves")
    sorted_df = df.sort_values("Cluster")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(sorted_df.drop("Cluster", axis=1), cmap="viridis", ax=ax, cbar=True)
    st.pyplot(fig)

    # ----------------------------- #
    # 🚨 Outlier Detection
    # ----------------------------- #
    st.subheader("🚨 Outlier Detection in Functional Data")
    try:
        X = df.drop(columns=["Cluster"], errors="ignore")
        if len(X) > 10:
            envelope = EllipticEnvelope(contamination=0.1).fit(X)
            df["Outlier"] = envelope.predict(X) == -1
            st.write(f"Detected {df['Outlier'].sum()} outliers.")

            if st.checkbox("Show Outlier Curves"):
                fig, ax = plt.subplots()
                for row in df[df["Outlier"]].drop(["Cluster", "Outlier"], axis=1).values:
                    ax.plot(timepoints, row, color="orange", alpha=0.7)
                st.pyplot(fig)

            if st.button("Explain Outliers with LLM"):
                for i, row in df[df["Outlier"]].iterrows():
                    curve_data = row.drop(labels=["Cluster", "Outlier"], errors="ignore").to_dict()
                    explanation = explain_outlier(curve_data, api_key)
                    st.markdown(f"**Outlier {i}:** {explanation}")
        else:
            st.warning("Too few samples for reliable outlier detection (need >10 curves).")
    except Exception as e:
        st.error(f"Outlier detection failed: {e}")

