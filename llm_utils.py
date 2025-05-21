# -*- coding: utf-8 -*-
"""
Created on Mon May 19 03:03:51 2025

@author: peaco
"""

import requests

# OpenRouter API endpoint
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
# Choose an available free LLM on OpenRouter
# --- CHANGE THIS LINE ---
LLM_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"  # Replace with a valid model ID from OpenRouter
# e.g., "mistralai/Mistral-7B-Instruct-v0.2", "meta-llama/Llama-2-7b-chat-hf", etc.
# -----------------------

# Helper: Make a chat request to the LLM
# Note: api_key is passed from the Streamlit app via st.secrets or env vars

def call_llm(prompt: str, api_key: str) -> str:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    body = {
        "model": LLM_MODEL,
        "messages": [{"role": "user", "content": prompt}]
    }

    try:
        response = requests.post(OPENROUTER_API_URL, headers=headers, json=body)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except requests.HTTPError as e:
        # return the API’s own error message for debugging
        try:
            err = response.json()
        except Exception:
            err = response.text
        return f"[LLM Error {response.status_code}] {err}"

# 1. Explain a single cluster

def explain_cluster(summary_stats: dict, api_key: str) -> str: # Removed cluster_id from signature as it's in summary_stats
    prompt = f"""
    The following are summary statistics of a group of functional curves (cluster {summary_stats.get('cluster_id', 'N/A')}):

    {summary_stats}

    Please explain what kind of trend or behavior this cluster likely represents.
    """
    return call_llm(prompt, api_key)

# 2. Compare multiple clusters

def compare_clusters(cluster_summaries: dict, api_key: str, clustering_method) -> str:
    prompt = "Compare the following functional data clusters:\n\n"
    for cid, stats in cluster_summaries.items():
        prompt += f"Cluster {cid}:\n{stats}\n\n"
    prompt += "Summarize the main differences between these clusters."
    prompt += f"\n\nThe data was clustered using the '{clustering_method}' algorithm."
    return call_llm(prompt, api_key)

# 3. Natural language question about the data

def answer_user_query(cluster_summaries: dict, question: str, api_key: str) -> str: # Corrected signature
    formatted = "\n".join([f"Cluster {cid}: {summary}" for cid, summary in cluster_summaries.items()])
    prompt = f"""
    Here is a summary of clustered functional data:

    {formatted}

    Question: {question}

    Please provide a detailed but clear answer.
    """
    return call_llm(prompt, api_key)

# 4. Explain a specific outlier curve

def explain_outlier(curve_data: dict, api_key: str) -> str:
    prompt = f"""
    This curve has been identified as an outlier in a set of functional data:

    {curve_data}

    Explain why this curve might be considered unusual or different from typical patterns.
    """
    return call_llm(prompt, api_key)
