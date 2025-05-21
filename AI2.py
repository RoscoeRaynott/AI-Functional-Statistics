import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import EllipticEnvelope
# Make sure llm_utils is updated with the correct model ID
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
    # Ensure timepoints are numeric if possible, otherwise use index
    try:
        timepoints = df.columns.astype(float)
    except ValueError:
        st.warning("CSV columns are not numeric. Using row index as timepoints.")
        timepoints = np.arange(df.shape[1])


    # ----------------------------- #
    # 📊 Clustering
    # ----------------------------- #
    '''
    n_clusters = st.slider("Number of clusters", 2, 10, 3)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10) # Added n_init for KMeans
    clusters = kmeans.fit_predict(X_scaled)
    df["Cluster"] = clusters
    '''

    st.subheader("📊 Clustering Options")
    clustering_method = st.selectbox(
        "Select Clustering Algorithm",
        options=["k-means", "dbscan", "gmm", "hierarchical"]
    )
    
    if clustering_method == "k-means":
        n_clusters = st.slider("Number of clusters (for k-means)", 2, 10, 3)
    elif clustering_method == "gmm":
        n_clusters = st.slider("Number of components (for GMM)", 2, 10, 3)
    elif clustering_method == "hierarchical":
        n_clusters = st.slider("Number of clusters (for hierarchical)", 2, 10, 3)
    else:  # dbscan
        n_clusters = None
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df)
    
    # Initialize cluster labels
    clusters = np.zeros(len(X_scaled))
    
    # Choose and fit model
    if clustering_method == "k-means":
        model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    elif clustering_method == "dbscan":
        eps = st.slider("Epsilon (DBSCAN)", 0.1, 2.0, 0.5)
        min_samples = st.slider("Min Samples (DBSCAN)", 2, 10, 5)
        model = DBSCAN(eps=eps, min_samples=min_samples)
    elif clustering_method == "gmm":
        model = GaussianMixture(n_components=n_clusters, random_state=42)
    elif clustering_method == "hierarchical":
        linkage = st.selectbox("Linkage Criterion", ["ward", "complete", "average", "single"])
        model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
    
    # Fit the model
    model.fit(X_scaled)
    
    # Extract cluster labels
    if hasattr(model, 'labels_'):
        clusters = model.labels_
    elif hasattr(model, 'predict'):
        clusters = model.predict(X_scaled)
    
    df["Cluster"] = clusters
    
    # Optional: Exclude noise points for stats (e.g., DBSCAN)
    valid_clusters = [cid for cid in np.unique(clusters) if cid != -1]
    
    st.subheader("📉 Clustered Functional Curves")
    fig, ax = plt.subplots()
    for cluster_id in range(n_clusters):
        # Plotting logic needs adjustment to handle labels correctly
        cluster_data = df[df["Cluster"] == cluster_id].drop("Cluster", axis=1)
        for i, row in enumerate(cluster_data.values):
             ax.plot(timepoints, row, alpha=0.4, color=plt.cm.viridis(cluster_id / (n_clusters - 1)), label=f"Cluster {cluster_id}" if i == 0 else "") # Use color map and label only once per cluster

    ax.legend() # Add legend
    st.pyplot(fig)

    # ----------------------------- #
    # 📦 Cluster Summary Stats
    # ----------------------------- #
    '''
    cluster_summary = {}
    for cluster_id in range(n_clusters):
        cluster_data = df[df["Cluster"] == cluster_id].drop("Cluster", axis=1)
        # Calculate summary stats across all points in the cluster
        all_points = cluster_data.values.flatten()
        summary = {
            "cluster_id": cluster_id, # Include cluster ID in summary
            "num_curves": len(cluster_data),
            "mean_value": np.mean(all_points),
            "std_value": np.std(all_points),
            "min_value": np.min(all_points),
            "max_value": np.max(all_points),
            "mean_curve": cluster_data.mean().tolist() # Mean curve as a list
        }
        cluster_summary[cluster_id] = summary
    '''
    cluster_summary = {}
    for cluster_id in valid_clusters:
        cluster_data = df[df["Cluster"] == cluster_id].drop("Cluster", axis=1)
        all_points = cluster_data.values.flatten()
        summary = {
            "cluster_id": cluster_id,
            "num_curves": len(cluster_data),
            "mean_value": np.mean(all_points),
            "std_value": np.std(all_points),
            "min_value": np.min(all_points),
            "max_value": np.max(all_points),
            "mean_curve": cluster_data.mean().tolist(),
            "method": clustering_method  # Useful for LLM explanations
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
    # LLM explanation for each cluster (can be slow, maybe do on button click?)
    if st.button("Generate LLM Explanations for Each Cluster"):
         st.subheader("🧠 LLM Insights per Cluster")
         cluster_explanations = {}
         for cid, stats in cluster_summary.items():
             with st.spinner(f"Getting LLM explanation for Cluster {cid}..."):
                 explanation = explain_cluster(stats, api_key) # Pass the stats dict
                 cluster_explanations[cid] = explanation
                 st.markdown(f"**Cluster {cid}:** {explanation}")

    # Download report button (uses potentially generated explanations)
    if st.button("Download Cluster Report (Markdown)"):
        # Need to ensure explanations are available, maybe store in session state?
        # For simplicity, let's regenerate for the report download
        with st.spinner("Generating explanations for report..."):
             cluster_explanations_for_report = {cid: explain_cluster(stats, api_key) for cid, stats in cluster_summary.items()}

        def generate_report(cluster_summary, explanations):
            report = "# Functional Data Cluster Report\n\n"
            for cid, stats in cluster_summary.items():
                report += f"## Cluster {cid}\n\n"
                report += f"**Summary Stats:**\n```json\n{stats}\n```\n\n" # Format stats nicely
                report += f"**LLM Insight:** {explanations.get(cid, 'N/A')}\n\n"
                # Optional: Add mean curve plot to report (more complex)
            return report

        report_md = generate_report(cluster_summary, cluster_explanations_for_report)
        st.download_button("Download Report", report_md, file_name="functional_data_cluster_report.md")


    # ----------------------------- #
    # 💬 Ask a Question
    # ----------------------------- #
    st.subheader("💬 Ask a Question About Your Data")
    user_query = st.text_input("Ask something like: 'Which cluster has the steepest rise?'" )
    if st.button("Ask LLM"):
        if user_query.strip():
            # --- CHANGE THIS LINE ---
            answer = answer_user_query(cluster_summary, user_query, api_key) # Pass cluster_summary
            # -----------------------
            st.markdown("**LLM Response:**")
            st.write(answer)
        else:
            st.warning("Please enter a question.")


    # ----------------------------- #
    # 🔥 Heatmap
    # ----------------------------- #
    st.subheader("📊 Heatmap of Functional Curves")
    sorted_df = df.sort_values("Cluster")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(sorted_df.drop("Cluster", axis=1), cmap="viridis", ax=ax, cbar=True)
    ax.set_xlabel("Timepoints")
    ax.set_ylabel("Curve Index (Sorted by Cluster)")
    st.pyplot(fig)

    # ----------------------------- #
    # 🚨 Outlier Detection
    # ----------------------------- #
    st.subheader("🚨 Outlier Detection in Functional Data")
    try:
        X = df.drop(columns=["Cluster"], errors="ignore")
        if len(X) > 10:
            # Use a robust method like Isolation Forest or One-Class SVM for functional data
            # EllipticEnvelope assumes Gaussian distribution, which might not fit functional data well
            # For simplicity, keeping EllipticEnvelope but note its limitations
            envelope = EllipticEnvelope(contamination=0.1, random_state=42).fit(X) # Added random_state
            df["Outlier"] = envelope.predict(X) == -1
            num_outliers = df['Outlier'].sum()
            st.write(f"Detected {num_outliers} outliers.")

            if num_outliers > 0:
                if st.checkbox("Show Outlier Curves"):
                    fig, ax = plt.subplots()
                    outlier_data = df[df["Outlier"]].drop(["Cluster", "Outlier"], axis=1)
                    for i, row in outlier_data.iterrows():
                        ax.plot(timepoints, row.values, color="orange", alpha=0.7, label=f"Outlier {i}" if i == outlier_data.index[0] else "") # Label only first outlier
                    ax.legend()
                    st.pyplot(fig)

                if st.button("Explain Outliers with LLM"):
                    outlier_curves = df[df["Outlier"]].drop(["Cluster", "Outlier"], axis=1)
                    if not outlier_curves.empty:
                        for i, row in outlier_curves.iterrows():
                            curve_data = row.to_dict() # Pass the series as dict
                            with st.spinner(f"Getting LLM explanation for Outlier {i}..."):
                                explanation = explain_outlier(curve_data, api_key)
                                st.markdown(f"**Outlier {i}:** {explanation}")
                    else:
                        st.info("No outliers detected to explain.")
            else:
                 st.info("No outliers detected.")

        else:
            st.warning("Too few samples for reliable outlier detection (need >10 curves).")
    except Exception as e:
        st.error(f"Outlier detection failed: {e}")
