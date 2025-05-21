import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering # Added DBSCAN and AgglomerativeClustering
from sklearn.mixture import GaussianMixture # Added GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import EllipticEnvelope
# Make sure llm_utils is updated with the correct model ID
# Assuming llm_utils is a local file or part of your environment
try:
    from llm_utils import explain_cluster, compare_clusters, answer_user_query, explain_outlier
except ImportError:
    st.error("Could not import llm_utils. Please ensure it's available in your environment.")
    # Define dummy functions to prevent further errors if llm_utils is missing
    def explain_cluster(*args, **kwargs): return "LLM functionality not available."
    def compare_clusters(*args, **kwargs): return "LLM functionality not available."
    def answer_user_query(*args, **kwargs): return "LLM functionality not available."
    def explain_outlier(*args, **kwargs): return "LLM functionality not available."


st.set_page_config(page_title="Functional Data Clustering", layout="wide")
st.title("📈 Functional Data Analysis with Clustering and LLMs")

# Load API key securely
api_key = st.secrets.get('OPENROUTER_API_KEY', 'sk-your-default-api-key') # Use .get() for safer access

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
        # Attempt to convert columns to float, handling multi-index or string columns
        if isinstance(df.columns, pd.MultiIndex):
            # This case is complex and depends on your multi-index structure.
            # For simplicity, if multi-index, we'll try to get the lowest level.
            try:
                timepoints = df.columns.get_level_values(-1).astype(float)
            except ValueError:
                st.warning("MultiIndex columns are not fully numeric. Using column index as timepoints.")
                timepoints = np.arange(df.shape[1])
        else:
            timepoints = df.columns.astype(float)
    except ValueError:
        st.warning("CSV columns are not numeric. Using column index as timepoints.")
        timepoints = np.arange(df.shape[1])

    # Ensure df does not contain non-numeric columns before scaling/clustering
    # This might happen if the first column is an ID or label.
    # We'll assume all columns are timepoints for now, as per "rows = curves"
    # If the first column is an ID, you might do: df_numeric = df.iloc[:, 1:]
    # For now, let's just make sure df is all numeric for scaling
    df_numeric = df.select_dtypes(include=np.number)
    if df_numeric.empty:
        st.error("No numeric data found in the CSV for clustering. Please check your file format.")
        st.stop()
    if df_numeric.shape[1] != df.shape[1]:
        st.warning("Non-numeric columns detected and dropped for clustering.")
        df = df_numeric # Update df to only contain numeric columns

    # ----------------------------- #
    # 📊 Clustering
    # ----------------------------- #
    st.subheader("📊 Clustering Options")
    clustering_method = st.selectbox(
        "Select Clustering Algorithm",
        options=["k-means", "dbscan", "gmm", "hierarchical"]
    )
    
    n_clusters_input = None # Initialize to None for methods that don't use it directly
    if clustering_method == "k-means":
        n_clusters_input = st.slider("Number of clusters (for k-means)", 2, 10, 3)
    elif clustering_method == "gmm":
        n_clusters_input = st.slider("Number of components (for GMM)", 2, 10, 3)
    elif clustering_method == "hierarchical":
        n_clusters_input = st.slider("Number of clusters (for hierarchical)", 2, 10, 3)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df) # Scale the numeric data

    # Initialize cluster labels (default to a single cluster or noise if no valid clustering occurs)
    clusters = np.array([0] * len(X_scaled)) # Default to all in one cluster
    
    # Choose and fit model
    model = None
    if clustering_method == "k-means":
        if n_clusters_input is not None:
            model = KMeans(n_clusters=n_clusters_input, random_state=42, n_init=10)
    elif clustering_method == "dbscan":
        eps = st.slider("Epsilon (DBSCAN)", 0.1, 2.0, 0.5)
        min_samples = st.slider("Min Samples (DBSCAN)", 2, 10, 5)
        model = DBSCAN(eps=eps, min_samples=min_samples)
    elif clustering_method == "gmm":
        if n_clusters_input is not None:
            model = GaussianMixture(n_components=n_clusters_input, random_state=42)
    elif clustering_method == "hierarchical":
        if n_clusters_input is not None:
            linkage = st.selectbox("Linkage Criterion", ["ward", "complete", "average", "single"])
            model = AgglomerativeClustering(n_clusters=n_clusters_input, linkage=linkage)

    if model: # Only fit if a model was successfully initialized
        model.fit(X_scaled)
        
        # Extract cluster labels
        if hasattr(model, 'labels_'):
            clusters = model.labels_
        elif hasattr(model, 'predict'):
            clusters = model.predict(X_scaled)
    else:
        st.warning("Please select valid clustering parameters.")
        # If no model was fit or parameters were invalid, clusters remains default 0
        st.stop() # Stop execution if no valid clustering happened

    df["Cluster"] = clusters
    
    # Optional: Exclude noise points for stats (e.g., DBSCAN where -1 is noise)
    valid_clusters = [cid for cid in np.unique(clusters) if cid != -1]
    
    st.subheader("📉 Clustered Functional Curves")
    fig, ax = plt.subplots()
    num_actual_clusters = len(valid_clusters)
    
    if num_actual_clusters > 0:
        for i, cluster_id in enumerate(valid_clusters):
            cluster_data = df[df["Cluster"] == cluster_id].drop("Cluster", axis=1)
            
            # Dynamically determine color. Avoid division by zero if only one valid cluster.
            # Using i directly ensures a unique color for each valid cluster.
            if num_actual_clusters > 1:
                color_index = i / (num_actual_clusters - 1)
            else:
                color_index = 0.5 # A default color if only one cluster
            
            color = plt.cm.viridis(color_index)
            
            for j, row in enumerate(cluster_data.values):
                ax.plot(timepoints, row, alpha=0.4, color=color, label=f"Cluster {int(cluster_id)}" if j == 0 else "")
        ax.legend()
    else:
        st.info("No distinct clusters formed or all points are noise (e.g., DBSCAN).")
    
    st.pyplot(fig)

    # ----------------------------- #
    # 📦 Cluster Summary Stats
    # ----------------------------- #
    cluster_summary = {}
    for cluster_id in valid_clusters:
        cluster_data = df[df["Cluster"] == cluster_id].drop("Cluster", axis=1)
        # Ensure cluster_data is not empty before calculating statistics
        if not cluster_data.empty:
            all_points = cluster_data.values.flatten()
            summary = {
                "cluster_id": int(cluster_id), # Ensure cluster ID is integer
                "num_curves": len(cluster_data),
                "mean_value": float(np.mean(all_points)),
                "std_value": float(np.std(all_points)),
                "min_value": float(np.min(all_points)),
                "max_value": float(np.max(all_points)),
                "mean_curve": cluster_data.mean().tolist(), # Mean curve as a list
                "method": clustering_method  # Useful for LLM explanations
            }
            cluster_summary[cluster_id] = summary
        else:
            st.warning(f"Cluster {cluster_id} is empty and will be skipped in summary.")

    # ----------------------------- #
    # 🤖 LLM Comparison
    # ----------------------------- #
    if st.button("Compare Clusters via LLM"):
        if cluster_summary: # Ensure there are clusters to compare
            st.subheader("🔍 LLM Comparison of Clusters")
            comparison = compare_clusters(cluster_summary, api_key)
            st.markdown(comparison)
        else:
            st.info("No valid clusters found for comparison.")

    # ----------------------------- #
    # 📄 Generate Markdown Report
    # ----------------------------- #
    if st.button("Generate LLM Explanations for Each Cluster"):
         st.subheader("🧠 LLM Insights per Cluster")
         cluster_explanations = {}
         if cluster_summary:
             for cid, stats in cluster_summary.items():
                 with st.spinner(f"Getting LLM explanation for Cluster {cid}..."):
                     explanation = explain_cluster(stats, api_key) # Pass the stats dict
                     cluster_explanations[cid] = explanation
                     st.markdown(f"**Cluster {int(cid)}:** {explanation}")
             st.session_state['cluster_explanations'] = cluster_explanations # Store for download
         else:
             st.info("No valid clusters to explain.")

    # Download report button (uses potentially generated explanations)
    if st.button("Download Cluster Report (Markdown)"):
        # Check if explanations are already in session state, otherwise regenerate
        if 'cluster_explanations' in st.session_state and st.session_state['cluster_explanations']:
            cluster_explanations_for_report = st.session_state['cluster_explanations']
        else:
            with st.spinner("Generating explanations for report (may take a moment)..."):
                cluster_explanations_for_report = {cid: explain_cluster(stats, api_key) for cid, stats in cluster_summary.items() if stats}
            st.session_state['cluster_explanations'] = cluster_explanations_for_report # Store them

        def generate_report(summary, explanations):
            report = "# Functional Data Cluster Report\n\n"
            if not summary:
                report += "No cluster data to report."
                return report
            for cid_key in sorted(summary.keys()): # Sort by cluster ID
                stats = summary[cid_key]
                report += f"## Cluster {stats['cluster_id']}\n\n"
                report += f"**Summary Stats:**\n```json\n{json.dumps(stats, indent=2)}\n```\n\n" # Use json.dumps for clean formatting
                report += f"**LLM Insight:** {explanations.get(cid_key, 'N/A')}\n\n"
                # To include mean curve plots: This would require saving plots as images
                # and embedding them, which is more complex for a markdown download.
                # For a markdown-only report, describing the curve is usually enough.
            return report

        import json # Import json for dumps
        report_md = generate_report(cluster_summary, cluster_explanations_for_report)
        st.download_button("Download Report", report_md, file_name="functional_data_cluster_report.md")


    # ----------------------------- #
    # 💬 Ask a Question
    # ----------------------------- #
    st.subheader("💬 Ask a Question About Your Data")
    user_query = st.text_input("Ask something like: 'Which cluster has the steepest rise?'" )
    if st.button("Ask LLM"):
        if user_query.strip() and cluster_summary:
            answer = answer_user_query(cluster_summary, user_query, api_key)
            st.markdown("**LLM Response:**")
            st.write(answer)
        else:
            st.warning("Please enter a question and ensure clustering has been performed.")


    # ----------------------------- #
    # 🔥 Heatmap
    # ----------------------------- #
    st.subheader("📊 Heatmap of Functional Curves")
    # Ensure 'Cluster' column exists and df is not empty after potential filtering
    if "Cluster" in df.columns and not df.empty:
        sorted_df = df.sort_values("Cluster")
        fig, ax = plt.subplots(figsize=(10, 6))
        # Drop 'Cluster' column before plotting heatmap
        sns.heatmap(sorted_df.drop("Cluster", axis=1), cmap="viridis", ax=ax, cbar=True)
        ax.set_xlabel("Timepoints")
        ax.set_ylabel("Curve Index (Sorted by Cluster)")
        st.pyplot(fig)
    else:
        st.info("No data available to generate heatmap after clustering.")


    # ----------------------------- #
    # 🚨 Outlier Detection
    # ----------------------------- #
    st.subheader("🚨 Outlier Detection in Functional Data")
    try:
        # Create a copy to avoid SettingWithCopyWarning
        df_for_outliers = df.copy()
        X = df_for_outliers.drop(columns=["Cluster", "Outlier"], errors="ignore")
        
        if len(X) > 10:
            # EllipticEnvelope assumes Gaussian distribution, which might not fit functional data well
            # Consider other methods like IsolationForest or OneClassSVM for robustness in functional data.
            # For this example, keeping EllipticEnvelope as per original code.
            envelope = EllipticEnvelope(contamination=0.1, random_state=42)
            envelope.fit(X) # Fit on X
            df_for_outliers["Outlier"] = envelope.predict(X) == -1
            num_outliers = df_for_outliers['Outlier'].sum()
            st.write(f"Detected {num_outliers} outliers.")

            if num_outliers > 0:
                if st.checkbox("Show Outlier Curves"):
                    fig, ax = plt.subplots()
                    outlier_data = df_for_outliers[df_for_outliers["Outlier"]].drop(["Cluster", "Outlier"], axis=1)
                    if not outlier_data.empty: # Ensure there's data to plot
                        for i, row in outlier_data.iterrows():
                            # Label only first outlier for simplicity in legend
                            ax.plot(timepoints, row.values, color="orange", alpha=0.7, label=f"Outlier {i}" if i == outlier_data.index[0] else "")
                        ax.legend()
                    else:
                        st.info("No outlier curves to display.") # Should not happen if num_outliers > 0
                    st.pyplot(fig)

                if st.button("Explain Outliers with LLM"):
                    outlier_curves = df_for_outliers[df_for_outliers["Outlier"]].drop(["Cluster", "Outlier"], axis=1)
                    if not outlier_curves.empty:
                        for i, row in outlier_curves.iterrows():
                            # Convert series to a dictionary for LLM input
                            curve_data = row.to_dict()
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
