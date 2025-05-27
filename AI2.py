import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import EllipticEnvelope
from sklearn.decomposition import PCA # Import PCA from scikit-learn
import json # For formatting summary stats in report
import os # To check for example file existence

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


st.set_page_config(page_title="Functional Data Analysis", layout="wide")
st.title("📈 Functional Data Analysis with Clustering and LLMs")

# Load API key securely
api_key = st.secrets.get('OPENROUTER_API_KEY', 'sk-your-default-api-key') # Use .get() for safer access

# ----------------------------- #
# 📁 Data Input Options
# ----------------------------- #
st.sidebar.subheader("📁 Data Source")
data_source_options = ["Upload your own CSV", "Example Data 1", "Example Data 2"]
selected_data_source = st.sidebar.radio("Choose a data source:", data_source_options)

df_original = None
df = None
file_name_display = None # To display the name of the loaded file

# Define paths to your example files
# Adjust these paths if your files are in a subdirectory, e.g., "data/example_data_1.csv"
EXAMPLE_FILE_1 = "fourier_fixed_coeffs_100x50.csv"
EXAMPLE_FILE_2 = "fourier_varying_coeffs_100x50.csv"

if selected_data_source == "Upload your own CSV":
    file = st.sidebar.file_uploader("Upload a CSV file with functional data (rows = curves, columns = timepoints/features)", type=["csv"])
    if file:
        df_original = pd.read_csv(file)
        df = df_original.copy()
        file_name_display = file.name
elif selected_data_source == "Example Data 1":
    if os.path.exists(EXAMPLE_FILE_1):
        df_original = pd.read_csv(EXAMPLE_FILE_1)
        df = df_original.copy()
        file_name_display = EXAMPLE_FILE_1
        st.sidebar.success(f"Loaded {EXAMPLE_FILE_1}")
    else:
        st.sidebar.error(f"Error: {EXAMPLE_FILE_1} not found. Please make sure it's in the correct directory.")
elif selected_data_source == "Example Data 2":
    if os.path.exists(EXAMPLE_FILE_2):
        df_original = pd.read_csv(EXAMPLE_FILE_2)
        df = df_original.copy()
        file_name_display = EXAMPLE_FILE_2
        st.sidebar.success(f"Loaded {EXAMPLE_FILE_2}")
    else:
        st.sidebar.error(f"Error: {EXAMPLE_FILE_2} not found. Please make sure it's in the correct directory.")

# --- Main app logic starts here, only if df is loaded ---
if df is not None and not df.empty:
    st.subheader(f"📄 Data Preview: `{file_name_display}`")
    st.write(df.head())

    # ----------------------------- #
    # ⚙️ Feature Engineering
    # ----------------------------- #
    # Ensure timepoints are numeric if possible, otherwise use index
    try:
        if isinstance(df.columns, pd.MultiIndex):
            try:
                timepoints = np.sort(df.columns.get_level_values(-1).astype(float))
            except ValueError:
                st.warning("MultiIndex columns are not fully numeric. Using column index as timepoints.")
                timepoints = np.arange(df.shape[1])
        else:
            timepoints = np.sort(df.columns.astype(float))
    except ValueError:
        st.warning("CSV columns are not numeric. Using column index as timepoints.")
        timepoints = np.arange(df.shape[1])

    # Ensure df for processing is purely numeric
    df_numeric = df.select_dtypes(include=np.number)
    if df_numeric.shape[1] < df.shape[1]:
        st.warning(f"Non-numeric columns detected and dropped for numerical analysis. Original columns: {df.shape[1]}, Numeric columns: {df_numeric.shape[1]}")
    
    if df_numeric.empty:
        st.error("No numeric data found in the CSV for analysis. Please check your file format.")
        st.stop()
    
    # Handle NaNs in numeric data
    if df_numeric.isnull().values.any():
        st.warning("NaNs detected in numeric data. Filling with column means for analysis.")
        df_numeric = df_numeric.fillna(df_numeric.mean())


    # ----------------------------- #
    # 🔬 Principal Component Analysis (PCA on Curves) - using sklearn.PCA
    # ----------------------------- #
    st.subheader("🔬 Principal Component Analysis (PCA on Curves)")

    st.markdown("""
    Principal Component Analysis (PCA) applied to functional data (curves) helps identify the main modes of variation.
    - **Mean Function**: Represents the average curve in your dataset.
    - **Principal Components (Eigenvectors/Eigenfunctions)**: These are vectors (representing functions) that capture directions of greatest variance.
    - **Modes of Variation**: The plots showing "Mean +/- sqrt(λ) * Component" illustrate how the data varies around the mean. Each component captures a certain percentage of the total variance.
    """)

    if not df_numeric.empty:
        data_matrix_pca = df_numeric.values

        # Slider for number of components
        slider_max_val = min(data_matrix_pca.shape[0], data_matrix_pca.shape[1])
        slider_max_val = min(10, slider_max_val) # User-imposed practical limit
        slider_max_val = max(1, slider_max_val) # Ensure slider_max_val is at least 1

        slider_default_val = min(3, slider_max_val)
        slider_default_val = max(1, slider_default_val) # Ensure default is at least 1

        n_components_pca_val = st.slider(
            "Number of Principal Components (PCA)",
            1,
            slider_max_val,
            slider_default_val
        )

        try:
            pca_sklearn_model = PCA(n_components=n_components_pca_val)
            pca_sklearn_model.fit(data_matrix_pca)

            st.session_state['pca_sklearn_model'] = pca_sklearn_model
            st.session_state['timepoints_sklearn'] = timepoints # Use the derived timepoints
            st.success("PCA computation complete.")

        except Exception as e:
            st.error(f"PCA fitting failed: {e}")
            st.session_state.pop('pca_sklearn_model', None)

        # Visualization of PCA results
        if 'pca_sklearn_model' in st.session_state and 'timepoints_sklearn' in st.session_state:
            pca_sklearn = st.session_state['pca_sklearn_model']
            timepoints_sklearn_vis = st.session_state['timepoints_sklearn']

            # Ensure timepoints align with component feature length
            if len(timepoints_sklearn_vis) != pca_sklearn.components_.shape[1]:
                st.error(f"Mismatch in timepoints length ({len(timepoints_sklearn_vis)}) and PCA component features ({pca_sklearn.components_.shape[1]}). Please check data processing.")
            else:
                mean_function_sklearn = pca_sklearn.mean_
                components_sklearn = pca_sklearn.components_
                explained_variance_sklearn = pca_sklearn.explained_variance_
                explained_variance_ratio_sklearn = pca_sklearn.explained_variance_ratio_

                st.subheader("Mean Function and Principal Components (PCA)")
                st.write("Explained Variance Ratio per Component:",
                         [f"{ratio:.2%}" for ratio in explained_variance_ratio_sklearn])

                # Determine layout: how many plots per row
                plots_per_row = 3 
                num_components_to_plot = pca_sklearn.n_components_
                
                total_plots = 1 + num_components_to_plot 
                num_rows = (total_plots + plots_per_row - 1) // plots_per_row 

                fig_pca_grid, axes = plt.subplots(num_rows, plots_per_row, 
                                                  figsize=(plots_per_row * 5, num_rows * 4), 
                                                  squeeze=False) 
                axes_flat = axes.flatten() 

                # --- Plot 1: Mean Function ---
                ax_mean_sk = axes_flat[0]
                ax_mean_sk.plot(timepoints_sklearn_vis, mean_function_sklearn)
                ax_mean_sk.set_title("Mean Function")
                ax_mean_sk.set_xlabel("Timepoints")
                ax_mean_sk.set_ylabel("Value")
                ax_mean_sk.tick_params(axis='x', labelsize=8)
                ax_mean_sk.tick_params(axis='y', labelsize=8)


                # --- Plot 2 to N+1: Principal Components and Modes of Variation ---
                for i in range(num_components_to_plot):
                    ax_comp_sk = axes_flat[i + 1] 

                    component_values_sk = components_sklearn[i]
                    eigenvalue_sk = explained_variance_sklearn[i]
                    scaled_effect_sk = np.sqrt(eigenvalue_sk) * component_values_sk

                    ax_comp_sk.plot(timepoints_sklearn_vis, mean_function_sklearn, 
                                     label="Mean", color='black', linestyle='-', linewidth=0.8, alpha=0.6)
                    ax_comp_sk.plot(timepoints_sklearn_vis, mean_function_sklearn + scaled_effect_sk,
                                     label=f"Mean + Mode {i+1}", linestyle='--')
                    ax_comp_sk.plot(timepoints_sklearn_vis, mean_function_sklearn - scaled_effect_sk,
                                     label=f"Mean - Mode {i+1}", linestyle=':')

                    title_sk = f"PC {i+1} ({explained_variance_ratio_sklearn[i]:.1%} var)"
                    ax_comp_sk.set_title(title_sk, fontsize=10)
                    ax_comp_sk.set_xlabel("Timepoints", fontsize=8)
                    ax_comp_sk.set_ylabel("Value", fontsize=8)
                    ax_comp_sk.legend(fontsize=7)
                    ax_comp_sk.tick_params(axis='x', labelsize=8)
                    ax_comp_sk.tick_params(axis='y', labelsize=8)
                
                for j in range(total_plots, len(axes_flat)):
                    fig_pca_grid.delaxes(axes_flat[j])

                plt.tight_layout() 
                st.pyplot(fig_pca_grid)
        else:
            st.info("PCA results not available for visualization. Please ensure PCA ran successfully.")
    else:
        st.info("Please upload data with numeric columns to perform PCA.")


    # ----------------------------- #
    # 📊 Clustering
    # ----------------------------- #
    st.subheader("📊 Clustering Options")
    if not df_numeric.empty:
        clustering_method = st.selectbox(
            "Select Clustering Algorithm",
            options=["k-means", "dbscan", "gmm", "hierarchical"]
        )

        n_clusters_input = None
        if clustering_method == "k-means":
            n_clusters_input = st.slider("Number of clusters (for k-means)", 2, min(10, len(df_numeric)), 3)
        elif clustering_method == "gmm":
            n_clusters_input = st.slider("Number of components (for GMM)", 2, min(10, len(df_numeric)), 3)
        elif clustering_method == "hierarchical":
            n_clusters_input = st.slider("Number of clusters (for hierarchical)", 2, min(10, len(df_numeric)), 3)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df_numeric)

        clusters = np.array([0] * len(X_scaled))
        model = None
        if clustering_method == "k-means" and n_clusters_input:
            model = KMeans(n_clusters=n_clusters_input, random_state=42, n_init='auto')
        elif clustering_method == "dbscan":
            eps = st.slider("Epsilon (DBSCAN)", 0.1, 5.0, 0.5, step=0.1)
            min_samples = st.slider("Min Samples (DBSCAN)", 2, 20, 5)
            model = DBSCAN(eps=eps, min_samples=min_samples)
        elif clustering_method == "gmm" and n_clusters_input:
            model = GaussianMixture(n_components=n_clusters_input, random_state=42)
        elif clustering_method == "hierarchical" and n_clusters_input:
            linkage = st.selectbox("Linkage Criterion", ["ward", "complete", "average", "single"])
            model = AgglomerativeClustering(n_clusters=n_clusters_input, linkage=linkage)

        if model:
            try:
                model.fit(X_scaled)
                if hasattr(model, 'labels_'):
                    clusters = model.labels_
                elif hasattr(model, 'predict'): 
                    clusters = model.predict(X_scaled)
            except Exception as e:
                st.error(f"Clustering failed: {e}")
                st.stop()
        else:
            st.warning("Please select valid clustering parameters.")
            st.stop()

        df_clustered = df_numeric.copy()
        df_clustered["Cluster"] = clusters

        valid_clusters = sorted([cid for cid in np.unique(clusters) if cid != -1]) 

        st.subheader("📉 Clustered Functional Curves")
        fig_cluster_plot, ax_cluster_plot = plt.subplots()
        if num_actual_clusters := len(valid_clusters): 
            palette = sns.color_palette("viridis", num_actual_clusters)
            for i, cluster_id in enumerate(valid_clusters):
                cluster_data_plot = df_clustered[df_clustered["Cluster"] == cluster_id].drop("Cluster", axis=1)
                color = palette[i]
                for j, row_values in enumerate(cluster_data_plot.values):
                    ax_cluster_plot.plot(timepoints, row_values, alpha=0.4, color=color, label=f"Cluster {int(cluster_id)}" if j == 0 else "")
            if num_actual_clusters > 0 : ax_cluster_plot.legend()
        else:
            st.info("No distinct clusters formed (excluding noise).")
        st.pyplot(fig_cluster_plot)

        # ----------------------------- #
        # 📦 Cluster Summary Stats
        # ----------------------------- #
        cluster_summary = {}
        if num_actual_clusters > 0:
            for cluster_id in valid_clusters:
                cluster_data_stats = df_clustered[df_clustered["Cluster"] == cluster_id].drop("Cluster", axis=1)
                if not cluster_data_stats.empty:
                    all_points_stats = cluster_data_stats.values.flatten()
                    summary = {
                        "cluster_id": int(cluster_id),
                        "num_curves": len(cluster_data_stats),
                        "mean_value": float(np.mean(all_points_stats)),
                        "std_value": float(np.std(all_points_stats)),
                        "min_value": float(np.min(all_points_stats)),
                        "max_value": float(np.max(all_points_stats)),
                        "mean_curve": cluster_data_stats.mean().tolist(),
                        "method": clustering_method
                    }
                    cluster_summary[cluster_id] = summary
                else:
                    st.warning(f"Cluster {cluster_id} is empty and will be skipped in summary.")
            st.session_state['cluster_summary'] = cluster_summary 
        else:
            st.info("No valid clusters to summarize.")
            st.session_state.pop('cluster_summary', None)


        # ----------------------------- #
        # 🤖 LLM Comparison & Explanations
        # ----------------------------- #
        if 'cluster_summary' in st.session_state and st.session_state['cluster_summary']:
            current_cluster_summary = st.session_state['cluster_summary']
            if st.button("Compare Clusters via LLM"):
                st.subheader("🔍 LLM Comparison of Clusters")
                comparison = compare_clusters(current_cluster_summary, api_key, clustering_method)
                st.markdown(comparison)

            if st.button("Generate LLM Explanations for Each Cluster"):
                 st.subheader("🧠 LLM Insights per Cluster")
                 cluster_explanations = {}
                 for cid, stats in current_cluster_summary.items():
                     with st.spinner(f"Getting LLM explanation for Cluster {cid}..."):
                         explanation = explain_cluster(stats, api_key)
                         cluster_explanations[cid] = explanation
                         st.markdown(f"**Cluster {int(cid)}:** {explanation}")
                 st.session_state['cluster_explanations'] = cluster_explanations

        # ----------------------------- #
        # 📄 Generate Markdown Report
        # ----------------------------- #
        if st.button("Download Cluster Report (Markdown)"):
            report_summary = st.session_state.get('cluster_summary', {})
            report_explanations = st.session_state.get('cluster_explanations', {})
            
            if not report_explanations and report_summary : 
                 with st.spinner("Generating explanations for report..."):
                    report_explanations = {cid: explain_cluster(stats, api_key) for cid, stats in report_summary.items() if stats}
                    st.session_state['cluster_explanations'] = report_explanations


            def generate_report(summary_data, explanations_data):
                report = "# Functional Data Cluster Report\n\n"
                if not summary_data:
                    report += "No cluster data to report."
                    return report
                for cid_key in sorted(summary_data.keys()):
                    stats = summary_data[cid_key]
                    report += f"## Cluster {stats['cluster_id']}\n\n"
                    report += f"**Clustering Method:** {stats.get('method', 'N/A')}\n"
                    report += f"**Number of Curves:** {stats['num_curves']}\n\n"
                    report += f"**Aggregate Statistics:**\n"
                    report += f"- Mean Value (all points): {stats['mean_value']:.2f}\n"
                    report += f"- Std Value (all points): {stats['std_value']:.2f}\n"
                    report += f"- Min Value (all points): {stats['min_value']:.2f}\n"
                    report += f"- Max Value (all points): {stats['max_value']:.2f}\n\n"
                    report += f"**LLM Insight:** {explanations_data.get(cid_key, 'N/A')}\n\n"
                return report

            report_md_content = generate_report(report_summary, report_explanations)
            st.download_button("Download Report", report_md_content, file_name="functional_data_cluster_report.md", mime="text/markdown")


        # ----------------------------- #
        # 💬 Ask a Question
        # ----------------------------- #
        st.subheader("💬 Ask a Question About Your Data")
        user_query = st.text_input("Ask something like: 'Which cluster has the steepest rise?'")
        if st.button("Ask LLM"):
            query_summary = st.session_state.get('cluster_summary', {})
            if user_query.strip() and query_summary:
                answer = answer_user_query(query_summary, user_query, api_key)
                st.markdown("**LLM Response:**")
                st.write(answer)
            else:
                st.warning("Please enter a question and ensure clustering has been performed and summary is available.")

        # ----------------------------- #
        # 🔥 Heatmap
        # ----------------------------- #
        st.subheader("🔥 Heatmap of Functional Curves")
        if "Cluster" in df_clustered.columns and not df_clustered.empty:
            heatmap_data = df_clustered.sort_values("Cluster").drop("Cluster", axis=1)
            fig_heatmap, ax_heatmap = plt.subplots(figsize=(10, 6))
            sns.heatmap(heatmap_data, cmap="viridis", ax=ax_heatmap, cbar=True, yticklabels=False) 
            ax_heatmap.set_xlabel("Timepoints")
            ax_heatmap.set_ylabel("Curve Index (Sorted by Cluster)")
            st.pyplot(fig_heatmap)
        else:
            st.info("No data available to generate heatmap after clustering.")

        # ----------------------------- #
        # 🚨 Outlier Detection
        # ----------------------------- #
        st.subheader("🚨 Outlier Detection in Functional Data")
        estimated_contamination = 0.1 # Default
        if len(df_numeric) > 20: # Need a reasonable number of samples for score distribution
            try:
                # Fit a preliminary model to get scores
                # You might use a slightly higher initial contamination or a method that gives scores easily
                prelim_envelope = EllipticEnvelope(random_state=42) # Default contamination is 0.1
                prelim_envelope.fit(df_numeric)
                scores = prelim_envelope.decision_function(df_numeric)
                
                # Simple heuristic: look for a drop after sorting scores
                # More sophisticated methods exist (e.g., finding the "elbow")
                sorted_scores = np.sort(scores)
                # Example: consider points below the 5th percentile of scores as outliers
                # This is just one heuristic, you might need to adjust the percentile
                threshold_score = np.percentile(sorted_scores, 5) # Detect ~5% as outliers by this heuristic
                estimated_outliers = np.sum(scores < threshold_score)
                estimated_contamination = max(0.01, min(0.49, estimated_outliers / len(df_numeric))) # Ensure in valid range
                
                st.write(f"Heuristically estimated contamination: {estimated_contamination:.3f}")
            except Exception as est_e:
                st.warning(f"Could not estimate contamination automatically: {est_e}. Using default 0.1.")
                estimated_contamination = 0.1
        # --- End of estimation section ---
        
        # Use the estimated_contamination or allow user override
        contamination_value = st.slider(
            "Contamination (proportion of outliers)", 
            0.01, 0.49, 
            float(estimated_contamination), # Convert to float for slider
            0.01
        )
        
        if len(df_numeric) > 10:
            try:
                envelope = EllipticEnvelope(contamination=contamination_value, random_state=42)
                outlier_predictions = envelope.fit_predict(df_numeric) 
                
                df_clustered["Outlier"] = (outlier_predictions == -1)
                num_outliers = df_clustered['Outlier'].sum()
                st.write(f"Detected {num_outliers} outliers.")

                if num_outliers > 0:
                    if st.checkbox("Show Outlier Curves"):
                        fig_outlier, ax_outlier = plt.subplots()
                        non_outlier_data_plot = df_clustered[~df_clustered["Outlier"]].drop(["Cluster", "Outlier"], axis=1, errors='ignore')
                        for _, row_values in non_outlier_data_plot.iterrows():
                            ax_outlier.plot(timepoints, row_values.values, color="grey", alpha=0.3)
                        
                        outlier_data_plot = df_clustered[df_clustered["Outlier"]].drop(["Cluster", "Outlier"], axis=1, errors='ignore')
                        for idx, row_values in outlier_data_plot.iterrows():
                            ax_outlier.plot(timepoints, row_values.values, color="orange", alpha=0.9, label=f"Outlier {idx}")
                        if not outlier_data_plot.empty: 
                            ax_outlier.legend()
                        st.pyplot(fig_outlier)

                    if st.button("Explain Outliers with LLM"):
                        outlier_curves_explain = df_clustered[df_clustered["Outlier"]].drop(["Cluster", "Outlier"], axis=1, errors='ignore')
                        if not outlier_curves_explain.empty:
                            for i, row_series in outlier_curves_explain.iterrows():
                                curve_data_dict = row_series.to_dict()
                                with st.spinner(f"Getting LLM explanation for Outlier {i}..."):
                                    explanation = explain_outlier(curve_data_dict, api_key)
                                    st.markdown(f"**Outlier {i}:** {explanation}")
                        else:
                            st.info("No outliers detected to explain.") 
            except Exception as e:
                st.error(f"Outlier detection failed: {e}")
                # import traceback # Optional: for more detailed in-app traceback
                # st.code(traceback.format_exc())
        else:
            st.warning("Too few samples for reliable outlier detection (need >10 curves).")
else:
    st.sidebar.info("Please upload a CSV file or select an example dataset to begin analysis.")
    st.info("Welcome! Please select a data source from the sidebar to get started.")
