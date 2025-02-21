import importlib
import requests
import json
import my_utils
importlib.reload(my_utils)
from my_utils import *
import sys
import os
current_working_directory = os.getcwd()
project_path = current_working_directory
sys.path.append(project_path)
# import importlib

def call_together_ai(prompt, max_tokens=1000, temperature=0.2, top_k=40):
    url = "https://api.together.xyz/v1/chat/completions"
    headers = {
        "Authorization": "Bearer 9d93ec297b48bc7b3f2b5272a3a37784e1a8176b5aabdad3a2582985cc39007e",  # 替换为你的together.ai API密钥
        "Content-Type": "application/json"
    }
    payload = {
        "model": "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo-128K",  # 可根据需求更换模型
        "messages": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_k": top_k
    }
    response = requests.post(url, headers=headers, data=json.dumps(payload))
    response_data = response.json()
    # print(response_data)
    return response_data['choices'][0]['message']['content']

def ask_for_input_and_process(RAG_prompt_template_get_answer, RAG_prompt_template_question_augment, retreiver):
    previous_input = ""
    while True:
        user_input = input("\n\nPlease enter your prompt (or type 'exit' to stop): ")
        if user_input.lower() == 'exit':
            print("Exiting the program...")
            break

        Question = user_input

        concatenated_content = get_retrieved_docs_as_string(retriever=retreiver, question=Question)
        input_prompt = format_good_prompt_RAG_api(
            prompt_data=RAG_prompt_template_question_augment, context=concatenated_content, question=Question, previous_input=previous_input)
        response = call_together_ai(input_prompt)
        # print(response)
        AUGMENTED_QUESTION = extract_answer(response, keyword="Augmented question")
        # print("AUGMENTED_QUESTION:",AUGMENTED_QUESTION
        Questions = "Original question : " + Question + "\nAugmented question " + AUGMENTED_QUESTION
        print("\n\nOriginal question and augmented question used for improved document search in the database:\n", Questions)

        concatenated_content = get_retrieved_docs_as_string(retriever=retreiver, question=Questions)
        input_prompt = format_good_prompt_RAG_api(
            prompt_data=RAG_prompt_template_get_answer, context=concatenated_content, question=Question, previous_input=previous_input)

        response = call_together_ai(input_prompt)
        final_response = extract_answer(response)

        print("\n\nModel response (only to original question): \n\n", final_response)
        previous_input = previous_input + "\n" + Question

if __name__ == "__main__":
    print("\n\n******LOADING THE MODELS AND THE DATABASE ... \n\n")

    embedding_net = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")

    folder_start_path = "/workspace"
    folder_path_database = "Datasets/Rel-18/Database_Embeddings"
    forder_path_prompts_get_answer = "prompts/3GPP_RAG_prompt_template.yaml"
    forder_path_prompts_question_augment = "prompts/3GPP_RAG_prompt_question_augmentation.yaml"
   

    retreiver = load_embedding_data_base(folder_path=folder_path_database, embedding_net=embedding_net, number_of_chunks=5)

    with open(forder_path_prompts_question_augment, "r", encoding="utf-8") as file:
        RAG_prompt_question_augmentation_template = yaml.safe_load(file)
    with open(forder_path_prompts_get_answer, "r", encoding="utf-8") as file:
        RAG_prompt_get_answer_template = yaml.safe_load(file)

    print("\n\n******LOADING PART COMPLETED! \n\n")

    ask_for_input_and_process(RAG_prompt_get_answer_template, RAG_prompt_question_augmentation_template, retreiver)
