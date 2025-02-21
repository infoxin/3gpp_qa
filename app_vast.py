import streamlit as st
import requests
import json
import uuid

st.set_page_config(layout="wide")

MODEL_PROVIDER_MAP = {
    "anthropic.claude-3-5-sonnet-20240620-v1:0": "bedrock",
    "us.meta.llama3-3-70b-instruct-v1:0": "bedrock",
    "meta.llama3-70b-instruct-v1:0": "bedrock",
    "mistral.mistral-large-2402-v1:0": "bedrock",
    "gemini-2.0-flash-exp": "vertexai",
    "gemini-1.5-pro-001": "vertexai",
    "openai.gpt-4o": "azure",
    "openai.o1-mini": "azure",
    "openai.o1-preview": "azure"
}

VAST_AI_INSTANCE_ID = 17811871
VAST_AI_API_URL = "https://console.vast.ai/api/v0"


def get_vast_instance_status(api_key):
    headers = {
   'Accept': 'application/json',
   'Content-Type': 'application/json',
   'Authorization': f'Bearer {api_key}'
}
    try:
        response = requests.get(f"{VAST_AI_API_URL}/instances/{VAST_AI_INSTANCE_ID}/", headers=headers)
        if response.status_code == 200:
            return response.json()["instances"]["actual_status"]
        else:
            return {"error": f"Error {response.status_code}: {response.text}"}
    except Exception as e:
        return {"error": str(e)}

def start_vast_instance(api_key):
    headers = {
   'Accept': 'application/json',
   'Content-Type': 'application/json',
   'Authorization': f'Bearer {api_key}'
}
    payload = json.dumps({
   "state": "running"
})
    try:
        # response = requests.put(f"{VAST_AI_API_URL}/instances/{VAST_AI_INSTANCE_ID}/", headers=headers, data=payload)
        response = requests.request("PUT", f"{VAST_AI_API_URL}/instances/{VAST_AI_INSTANCE_ID}/", headers=headers, data=payload)
        return response.json()
    except Exception as e:
        return {"error": str(e)}

def stop_vast_instance(api_key):
    headers = {
   'Accept': 'application/json',
   'Content-Type': 'application/json',
   'Authorization': f'Bearer {api_key}'
}
    payload = json.dumps({
   "state": "stopped"
})
    try:
        # response = requests.put(f"{VAST_AI_API_URL}/instances/{VAST_AI_INSTANCE_ID}/", headers=headers, data=payload)
        response = requests.request("PUT", f"{VAST_AI_API_URL}/instances/{VAST_AI_INSTANCE_ID}/", headers=headers, data=payload)
        return response.json()
    except Exception as e:
        return {"error": str(e)}

def query_3gpp_rag(question):
    api_url = "http://38.29.145.24:40179/generate"
    data = {"question": question}
    
    try:
        response = requests.post(api_url, json=data, timeout=30)
        if response.status_code == 200:
            result = response.json()
            return result
        else:
            return {"error": f"Error {response.status_code}: {response.text}"}
    except Exception as e:
        return {"error": str(e)}

def query_capgemini(question, api_key, model_name, provider, workspace_id):
    url = "https://api.generative.engine.capgemini.com/v2/llm/invoke"
    headers = {
        "accept": "application/json",
        "x-api-key": api_key,
        "Content-Type": "application/json"
    }
    data = {
        "action": "run",
        "modelInterface": "langchain",
        "data": {
            "mode": "chain",
            "text": question,
            "files": [],
            "modelName": model_name,
            "provider": provider,
            "systemPrompt": "You are a helpful and kind AI assistant.",
            "sessionId": str(uuid.uuid4()),
            "workspaceId": workspace_id if workspace_id else None,
            "modelKwargs": {
                "maxTokens": 512,
                "temperature": 0.6,
                "streaming": False,
                "topP": 0.9
            }
        }
    }
    
    try:
        response = requests.post(url, headers=headers, data=json.dumps(data), timeout=30)
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"Error {response.status_code}: {response.text}"}
    except Exception as e:
        return {"error": str(e)}

st.title("3GPP RAG Chatbot & Capgemini Generative Engine")

st.sidebar.header("Capgemini API Settings")
api_key = st.sidebar.text_input("Enter Capgemini API Key", type="password")
model_name = st.sidebar.selectbox("Select Model", list(MODEL_PROVIDER_MAP.keys()))
provider = MODEL_PROVIDER_MAP[model_name]
st.sidebar.write(f"Provider: {provider}")

use_rag_workspace = st.sidebar.checkbox("Use RAG Workspace")
workspace_id = "rag-workspace-id" if use_rag_workspace else ""

st.sidebar.header("Vast.ai Instance Control")
vast_ai_api_key = st.sidebar.text_input("Enter Vast.ai API Key", type="password")

if st.sidebar.button("Start Instance") and vast_ai_api_key:
    start_vast_instance(vast_ai_api_key)
if st.sidebar.button("Stop Instance") and vast_ai_api_key:
    stop_vast_instance(vast_ai_api_key)
if st.sidebar.button("Show Status") and vast_ai_api_key:
    instance_status = get_vast_instance_status(vast_ai_api_key)
    st.sidebar.write("Instance Status:", instance_status)

question = st.text_area("Enter your question")
st.text("Some example of question for testing Models:\nWhere are discussed LPP procedures?\nWhat are the important points of this 38501-i40 3GPP document?\nWhat is new in 3GPP Release 18?\nWhat is National Roaming, and how does it depend on the home PLMN and visited PLMN?\nWhat is the primary goal of the 5G system in terms of service continuity during inter- and/or intra-access technology changes?")

if "response_3gpp" not in st.session_state:
    st.session_state.response_3gpp = None
if "response_capgemini" not in st.session_state:
    st.session_state.response_capgemini = None

col1, col2 = st.columns(2)

with col1:
    if st.button("Ask 3GPP RAG API"):
        if not question:
            st.error("Please enter a question.")
        else:
            st.session_state.response_3gpp = query_3gpp_rag(question)
    if st.session_state.response_3gpp:
        st.subheader("3GPP RAG API Response")
        st.json(st.session_state.response_3gpp)

with col2:
    if st.button("Ask Capgemini API"):
        if not question:
            st.error("Please enter a question.")
        elif not api_key:
            st.error("Please enter your Capgemini API key.")
        else:
            st.session_state.response_capgemini = query_capgemini(question, api_key, model_name, provider, workspace_id)
    if st.session_state.response_capgemini:
        st.subheader("Capgemini API Response")
        st.json(st.session_state.response_capgemini)
