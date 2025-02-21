import streamlit as st
import requests
import json
import uuid
import yaml
from langchain.embeddings import HuggingFaceEmbeddings
from my_utils import *  # 确保这个模块包含 `format_good_prompt_RAG_api` 等函数
import sys
import os

current_working_directory = os.getcwd()
project_path = current_working_directory
sys.path.append(project_path)
# import importlib
# importlib.reload(my_utils)
from my_utils import *


# Streamlit 页面配置
st.set_page_config(layout="wide")

# 3GPP RAG 相关的加载
@st.cache_resource
def load_rag_components():
    embedding_net = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")

    folder_path_database = "Datasets/Rel-18/Database_Embeddings"
    folder_path_prompts_get_answer = "prompts/3GPP_RAG_prompt_template.yaml"
    folder_path_prompts_question_augment = "prompts/3GPP_RAG_prompt_question_augmentation.yaml"

    retriever = load_embedding_data_base(folder_path=folder_path_database, embedding_net=embedding_net, number_of_chunks=5)

    with open(folder_path_prompts_question_augment, "r", encoding="utf-8") as file:
        RAG_prompt_question_augmentation_template = yaml.safe_load(file)
    with open(folder_path_prompts_get_answer, "r", encoding="utf-8") as file:
        RAG_prompt_get_answer_template = yaml.safe_load(file)

    return retriever, RAG_prompt_get_answer_template, RAG_prompt_question_augmentation_template

retriever, RAG_prompt_get_answer_template, RAG_prompt_question_augmentation_template = load_rag_components()

# 可选模型及价格
MODEL_OPTIONS = {
    "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free": "Free",
    "meta-llama/Llama-3.3-70B-Instruct-Turbo": "$0.88",
    "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo": "$3.50",
    "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo-128K": "$0.18",
    "deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free": "Free",
    "deepseek-ai/DeepSeek-V3": "$1.25",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B": "$1.60",
    "Qwen/Qwen2.5-Coder-32B-Instruct": "$0.80",
    "Qwen/Qwen2.5-72B-Instruct-Turbo": "$1.20",
    "Qwen/Qwen2-VL-72B-Instruct": "$1.20",
    "Qwen/QwQ-32B-Preview": "$1.20",
    "mistralai/Mixtral-8x7B-Instruct-v0.1": "$0.60",
}

def call_together_ai(api_key_together, prompt, model):
    url = "https://api.together.xyz/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key_together}",  # 替换为你的together.ai API密钥
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,  # 可根据需求更换模型
        "messages": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_k": 40
    }
    response = requests.post(url, headers=headers, data=json.dumps(payload))
    response_data = response.json()
    print("Response from call_together_ai:", response_data)
    # print(response_data)
    # return response_data['choices'][0]['message']['content']
    return response_data

# Capgemini API 配置
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
                "maxTokens": max_tokens,
                "temperature": temperature,
                "streaming": False,
                "topP": top_p
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

st.title("Together.AI RAG Chatbot & Capgemini Generative Engine")

st.sidebar.header("Model Parameters")
max_tokens = st.sidebar.slider("Max Tokens", min_value=512, max_value=8192, value=4096, step=512)
temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.6, step=0.05)
top_p = st.sidebar.slider("Top P", min_value=0.0, max_value=1.0, value=0.9, step=0.05)


st.sidebar.header("Together.AI API Settings")
api_key_together = st.sidebar.text_input("Enter Together.AI API Key", type="password")
selected_model = st.sidebar.selectbox("Select Model", options=list(MODEL_OPTIONS.keys()), format_func=lambda x: f"{x} ({MODEL_OPTIONS[x]})")
st.sidebar.write(f"Price: {MODEL_OPTIONS[selected_model]}")

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

# 修改 3GPP RAG API 调用
with col1:
    if st.button("Ask 3GPP RAG API"):
        if not question:
            st.error("Please enter a question.")
        else:
            # 替换 query_3gpp_rag 逻辑
            concatenated_content = get_retrieved_docs_as_string(retriever=retriever, question=question)
            input_prompt = format_good_prompt_RAG_api(
                prompt_data=RAG_prompt_question_augmentation_template, context=concatenated_content, question=question, previous_input=""
            )
            response = call_together_ai(api_key_together, input_prompt, model=selected_model)
            AUGMENTED_QUESTION = extract_answer(response['choices'][0]['message']['content'], keyword="Augmented question")
            Questions = f"Original question: {question}\nAugmented question: {AUGMENTED_QUESTION}"
            
            concatenated_content = get_retrieved_docs_as_string(retriever=retriever, question=Questions)
            input_prompt = format_good_prompt_RAG_api(
                prompt_data=RAG_prompt_get_answer_template, context=concatenated_content, question=question, previous_input=""
            )

            response = call_together_ai(api_key_together, input_prompt, model=selected_model)
            # final_response = extract_answer(response['choices'][0]['message']['content'])

            st.session_state.response_3gpp = response

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
