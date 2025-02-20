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

def query_3gpp_rag(question):
    api_url = "http://38.29.145.24:40179/generate"
    data = {"question": question}
    
    try:
        response = requests.post(api_url, json=data, timeout=30000)
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
        response = requests.post(url, headers=headers, data=json.dumps(data), timeout=30000)
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
workspace_id = "5a64501c-a4d1-45dd-bd8c-d69a87bac162" if use_rag_workspace else ""

question = st.text_area("Enter your question")
st.text("Some example of question for testing:\nWhere are discussed LPP procedures?\nWhat are the important points of this 38501-i40 3GPP document?\nWhat is new in 3GPP Release 18?\nWhat is National Roaming, and how does it depend on the home PLMN and visited PLMN?\nWhat is the primary goal of the 5G system in terms of service continuity during inter- and/or intra-access technology changes?")
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
