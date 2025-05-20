# -*- coding: utf-8 -*-
"""
Created on Mon May 19 03:03:51 2025

@author: peaco
"""

import requests
import os

# Set your OpenRouter API key here or via environment variable
#OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "your-api-key-here")
OPENROUTER_API_KEY = st.secrets["OPENROUTER_API_KEY"]

# Choose an available free LLM on OpenRouter
LLM_MODEL = "openrouter/mistral"  # e.g., mistral, llama2, etc.

# Helper: Make a chat request to the LLM
def call_llm(prompt: str) -> str:
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }

    body = {
        "model": LLM_MODEL,
        "messages": [{"role": "user", "content": prompt}]
    }

    try:
        response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=body)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"[LLM Error] {str(e)}"

# 1. Explain a single cluster
def explain_cluster(cluster_id: int, summary_stats: dict) -> str:
    prompt = f"""
    The following are summary statistics of a group of functional curves (cluster {cluster_id}):

    {summary_stats}

    Please explain what kind of trend or behavior this cluster likely represents.
    """
    return call_llm(prompt)

# 2. Compare multiple clusters
def compare_clusters(cluster_summaries: dict) -> str:
    prompt = "Compare the following functional data clusters:\n\n"
    for cid, stats in cluster_summaries.items():
        prompt += f"Cluster {cid}:\n{stats}\n\n"
    prompt += "Summarize the main differences between these clusters."
    return call_llm(prompt)

# 3. Natural language question about the data
def answer_user_query(cluster_summaries: dict, question: str) -> str:
    formatted = "\n".join([f"Cluster {cid}: {summary}" for cid, summary in cluster_summaries.items()])
    prompt = f"""
    Here is a summary of clustered functional data:

    {formatted}

    Question: {question}

    Please provide a detailed but clear answer.
    """
    return call_llm(prompt)

# 4. Explain a specific outlier curve
def explain_outlier(curve_data: dict) -> str:
    prompt = f"""
    This curve has been identified as an outlier in a set of functional data:

    {curve_data}

    Explain why this curve might be considered unusual or different from typical patterns.
    """
    return call_llm(prompt)
