# import subprocess
# import sys
#pip install faiss-gpu faiss-cpu

# # List of required packages
# required_packages = [
#     "torch",
#     "transformers",
#     "peft",
#     "langchain",
#     "langchain-community",
#     "langchain-huggingface",
#     "datasets",
#     "tqdm",
#     "scikit-learn",
#     "pyyaml"
# ]

# def install_package(package):
#     """Install a package using pip."""
#     subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# # Check and install missing packages
# for package in required_packages:
#     try:
#         __import__(package.replace("-", "_"))  # Handle different naming conventions
#     except ImportError:
#         print(f"Installing {package}...")
#         install_package(package)


# Now import the necessary libraries
import torch 
from transformers import AutoConfig, AutoModelForQuestionAnswering, AutoTokenizer, AutoModelForCausalLM, pipeline, TrainingArguments, Trainer
from transformers import DataCollatorForLanguageModeling
from transformers import BitsAndBytesConfig

from peft import LoraConfig, TaskType, get_peft_model, AutoPeftModelForCausalLM, PeftConfig, PeftModel

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document as LangchainDocument
#from langchain.vectorstores import FAISS
from langchain_community.vectorstores import FAISS
#from langchain.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

from datasets import load_dataset, Dataset, DatasetDict

import os
import re
from tqdm import tqdm
from sklearn.metrics import f1_score
import yaml

def check_memory():
    print("The memory state of the hardwares on this device are the following. \n\n")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        
        total_memory = torch.cuda.get_device_properties(i).total_memory
        allocated_memory = torch.cuda.memory_allocated(i)
        cached_memory = torch.cuda.memory_reserved(i)
    
        print(f"Device {i}: {torch.cuda.get_device_name(i)}")
        print(f"Total Memory: {total_memory / 1024**2:.2f} MB")
        print(f"Memory Allocated: {allocated_memory / 1024**2:.2f} MB")
        print(f"Memory Cached: {cached_memory / 1024**2:.2f} MB")
        print("")

def load_model(my_model ="microsoft/phi-4", my_device = "cuda:0"):
    device = torch.device(my_device)
    # Check the selected GPU
    print(f"Using device: {torch.cuda.get_device_name(device.index)}\n")

    quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    #Inital model
    model_checkpoint = my_model#"meta-llama/Llama-3.1-8B-Instruct"#"meta-llama/Llama-3.2-1B"#"meta-llama/Llama-3.1-8B-Instruct"#"microsoft/phi-4"##"deepseek-ai/DeepSeek-R1-Distill-Llama-8B"# ##"microsoft/phi-4"#"openai-community/gpt2"# #"meta-llama/Llama-3.2-1B"#"meta-llama/Llama-3.1-8B-Instruct"#"meta-llama/Llama-3.2-1B-Instruct"#"microsoft/phi-4"#"meta-llama/Meta-Llama-3-8B-Instruct"#"microsoft/phi-4" #meta-llama/Llama-3.3-70B-Instruct"  #"meta-llama/Llama-3.2-1B-Instruct" #"meta-llama/Llama-3.2-1B"#"distilbert/distilgpt2"  #"meta-llama/Llama-3.3-70B-Instruct" #
    model = AutoModelForCausalLM.from_pretrained(model_checkpoint,
                                                device_map=device,
                                                torch_dtype="auto",#,#torch.bfloat16,#auto,#torch.float16, "auto"
                                                quantization_config=quantization_config)#qauntization reduced model size

    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    #tokenizer_f1 = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B") #use same tokenizer to assess all models
    # Ensure the tokenizer can handle padding if required
    tokenizer.pad_token = tokenizer.eos_token
    check_memory()
    print("Model used is: "+my_model)
    return model,tokenizer,device

def extract_answer(response, keyword="assistant"):
    """Extracts the text starting from the last occurrence of keyord:' using regex."""
    matches = list(re.finditer(rf"{re.escape(keyword)}", response))  # Find all occurrences
    if matches:
        last_match = matches[-1]  # Get the last occurrence
        return response[last_match.end():].strip()  # Extract text after it
    return response  # Return full response if keyword not found

###RAG FUNCTIONS###

def load_and_chunks_documents(folder_path,chunk_size,chunk_overlap):
    #  Step 1: Load all .txt files
    print("Load all .txt files\n")
    documents = []
    # for filename in tqdm(os.listdir(folder_path)):
    #     if filename.endswith(".txt"):
    #         file_path = os.path.join(folder_path, filename)
    #         with open(file_path, "r", encoding="utf-8") as f:
    #             text = f.read()
    #             print(f"Loaded: {file_path}")
    #         # Convert into LangChain document
    #         documents.append(LangchainDocument(page_content=text, metadata={"source": filename}))

    # Walk through all subdirectories and files
    for root, _, files in os.walk(folder_path):
        for filename in tqdm(files):
            if filename.endswith(".txt"):
                file_path = os.path.join(root, filename)
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        text = f.read()
                    print(f"Loaded: {file_path}")
                    # Convert into LangChain document
                    documents.append(LangchainDocument(page_content=text, metadata={"source": file_path}))
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
                    
    #  Step 2: Initialize Text Splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,  # Max tokens per chunk
        chunk_overlap=chunk_overlap,  # Overlap between chunks
        add_start_index=True,  # Adds index metadata
        separators=["\n\n", "\n", ".", " ", ""]  # Split by paragraphs, sentences, etc.
    )

    #  Step 3: Split documents
    print("Split documents\n")
    docs_processed = []
    for doc in documents:
        scope_text = "Not Found"  # Default value if "1 Scope" is not found
        # Use regex to extract the "1 Scope" section
        ## 1 Scope, you can extract it more precisely by searching for this specific pattern and including everything until the next # (which marks a new section).
        match = re.search(r"#\s*1\s*Scope\s*\n(.*?)(?=\n#|\Z)", doc.page_content, re.DOTALL)
        if match:
            scope_text = match.group(1).strip()

        chunks = text_splitter.split_documents([doc])
        # Attach "1 Scope" as metadata
        for chunk in chunks: #add scope oth as meta data and in chunk
            chunk.metadata["scope"] = scope_text
            chunk.page_content = f"Scope: {scope_text}\n\n{chunk.page_content}"  # Prepend text
        docs_processed += chunks
        #print("One new doc splitted!\n")

    #  All `.txt` files are loaded and split into chunks!
    print(f"\nTotal chunks after splitting: {len(docs_processed)}")
    return docs_processed

def create_and_save_embeddings(docs_processed,embedding_net,folder_path):
    # Create FAISS vector store (create embeddings and store them)
    print("\nStarting to compute embeddings (may take some time)...\n")
    db = FAISS.from_documents(docs_processed, embedding_net)
    pkl = db.serialize_to_bytes()  # serializes the faiss
    # You can then save `pkl` to a database or a file
    print("\nSaving embeddings in database...\n")
    file_name = folder_path + "faiss_index.pkl"
    # Ensure the directory exists before writing the file
    os.makedirs(os.path.dirname(file_name), exist_ok=True)

    with open(file_name, 'wb') as f:
        f.write(pkl)
    #db.save_local(file_name) #also save model

def load_embedding_data_base(folder_path,embedding_net,number_of_chunks=4):
    #Load the serialized FAISS index from file
    file_name = folder_path + "/faiss_index.pkl"
   
    with open(file_name, 'rb') as f:
        pkl = f.read()
    #db = FAISS.load_local("Datasets/faiss_index", embedding_net)

    # Deserialize FAISS index and load it with the embeddings model
    db = FAISS.deserialize_from_bytes(embeddings=embedding_net, serialized=pkl, allow_dangerous_deserialization=True)
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": number_of_chunks})
    return retriever
    #Merge two FAISS databases
    # db1.merge_from(db2)

def format_good_prompt_RAG(prompt_data,context,question,tokenizer,previous_input):

    messages = []
    for entry in prompt_data:
        content = entry["content"]
        if "{context}" in content:  # Replace placeholder in 'user' role
            content = content.replace("{context}",context)
        if "{question}" in content:
            content = content.replace("{question}",question)
        if "{previous_input}" in entry["content"]:
            content = content.replace("{previous_input}",previous_input)

        messages.append({"role": entry["role"], "content": content})

    # input_text_system = ("You are a helpful assistant." 
    #                     "You are provided with technical information on the 3GPP standard (extracted via a RAG pipeline) to be used to answer a technical question.\n"
    #                     "The provided document chunk index have no meaning to the user asking the question (they are chunks of true documents).\n"
    #                     "However, the source information (document name) do have meaning to the user. \n"
    #                     "Hence, in the answer you should not refer to the document chunk index (and start index) but to its source.\n" 
    #                     "IMPORTANT: ALWAYS indicate the identifier of the sources (one or several) used at the BEGINING of your answer (among the one provided in the RAG retrieved document chunks) in the following form: The identifier of the sources used to answer the question are [add name of documents]. I insist indicate that before starting the answer of the question.\n")
    #                     #"The technical information to be used to answer the question is the following. \n")
    # input_text_system = input_text_system + "Also, as additional information the previous questions of the user were (if any)  "  + previous_input 
    # input_text_system = input_text_system + "\nThe RAG retrieved document chunks are the following:\n" +  context  
    
    # input_text_user = "\nQuestion:"  + question

    # messages = [
    # {"role": "system", "content": input_text_system},
    # {"role": "user", "content": input_text_user}
    # ]
    input_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    return input_text

def get_retrieved_docs_as_string(retriever,question):
    #return one string with docs

    # Retrieve relevant chunks
    retrieved_docs = retriever.invoke(question)
    #retrieved_docs = retriever.similarity_search(question, k=5)
    # Format context
    concatenated_content = ""
    # Iterate through the retrieved documents and concatenate their content
    for i, doc in enumerate(retrieved_docs):
        concatenated_content += f"RAG retrieved document chunk index: {i+1}" +"\nSource: " + doc.metadata['source'] +"\nScope: " + doc.metadata['scope'] + "\nStart Index (in document): " + f"{doc.metadata['start_index']}"
        concatenated_content += "\nContent:\n" + doc.page_content + "\n\n"  # Add content and separate with newlines
    
    return concatenated_content

