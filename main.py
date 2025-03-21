## initial database?
# -> Set to True if you run the notebook for the first time or if you changed the md files
initial_db = True

print("Starting Ollama RAG ChatApp...")

DATA_PATH = "data/"
OLLAMA_MODEL = "DeepSeek-R1:14b"
OLLAMA_URL = "http://localhost:11434"
CHROMA_PATH = "chroma/"

print(f"Configuration loaded: Using {OLLAMA_MODEL} model at {OLLAMA_URL}")
print(f"Data path: {DATA_PATH}, Chroma path: {CHROMA_PATH}")

## langchain split config
# md headers
headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
    ("####", "Header 4"),
]

# chunk sizes
chunk_size = 500
chunk_overlap = 100
print(f"Text splitting configuration: chunk size={chunk_size}, overlap={chunk_overlap}")

from langchain.document_loaders import TextLoader
import os

print("Loading documents from data directory...")
documents = []

for file in os.listdir(DATA_PATH):
    loader = TextLoader(DATA_PATH + file, encoding="utf-8")
    documents.append(loader.load()[0])

print(f"Loaded {len(documents)} documents")

documents[0].metadata
len(documents[0].page_content)

# for doc in documents:
#     print(doc.metadata)

from langchain.text_splitter import MarkdownHeaderTextSplitter

print("Starting document splitting by markdown headers...")
text_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
chunks_array= []

for doc in documents:
    chunks = text_splitter.split_text(doc.page_content)
    # append source metadata to each chunk
    for chunk in chunks:
        chunk.metadata.update(doc.metadata)  # Use update() to merge metadata
    chunks_array.append(chunks)

print(f"Created {len(chunks_array)} document chunk arrays after header splitting")

# Char-level splits
from langchain.text_splitter import RecursiveCharacterTextSplitter

print("Performing character-level splitting to create final chunks...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len, add_start_index=True
)

chunks_array_txt_base = []
counter = 0
for document in chunks_array:
    for chunk in document:
        splits = text_splitter.split_documents([chunk])
        chunks_array_txt_base.append(splits)
        counter += 1

print(f"Generated {counter} chunks after character-level splitting")
print(f"Total document chunk arrays: {len(chunks_array_txt_base)}")

all_document_chunks = [chunk for document in chunks_array_txt_base for chunk in document]
print(f"Total individual document chunks: {len(all_document_chunks)}")

# TEST OLLAMA CONNECTION ##
from langchain_community.llms import Ollama

print(f"Testing connection to Ollama at {OLLAMA_URL}...")
ollama = Ollama(base_url=OLLAMA_URL, model=OLLAMA_MODEL)

response = ollama("Who is Alice?")
print(f"Ollama connection test response: {response}")

all_document_chunks[0]

from langchain.embeddings import OllamaEmbeddings
from langchain.vectorstores import Chroma
from tqdm import tqdm
import time
import concurrent.futures

def create_vector_db(all_document_chunks, ollama_url, ollama_model, chroma_path, batch_size=30, max_workers=3):
    # Get total number of document chunks
    total_chunks = len(all_document_chunks)
    print(f"Processing {total_chunks} document chunks")
    
    # Initialize embedding function - without the unsupported parameter
    print(f"Initializing OllamaEmbeddings with model {ollama_model}")
    embeddings = OllamaEmbeddings(
        base_url=ollama_url, 
        model=ollama_model
    )
    
    # Create Chroma instance with optimized settings
    print(f"Creating Chroma vector database at {chroma_path}")
    chroma_db = Chroma(
        embedding_function=embeddings, 
        persist_directory=chroma_path
    )
    
    # Function to process a batch of documents
    def process_batch(batch):
        try:
            chroma_db.add_documents(documents=batch)
            return len(batch)
        except Exception as e:
            print(f"Error processing batch: {e}")
            # On failure, process documents one by one to identify problematic documents
            successful = 0
            for doc in batch:
                try:
                    chroma_db.add_documents(documents=[doc])
                    successful += 1
                except Exception as e:
                    print(f"Problem with document: {doc.page_content[:50]}... Error: {e}")
            return successful
    
    # Process in optimized batches with parallel workers
    print(f"Starting batch processing with batch_size={batch_size}, max_workers={max_workers}")
    start_time = time.time()
    processed_count = 0
    
    # Create batches
    batches = [all_document_chunks[i:i+batch_size] for i in range(0, total_chunks, batch_size)]
    print(f"Created {len(batches)} batches for processing")
    
    # Process batches with progress tracking
    with tqdm(total=total_chunks, desc="Creating Vector Database") as pbar:
        # Use ThreadPoolExecutor for parallel processing
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all batches to the executor
            future_to_batch = {executor.submit(process_batch, batch): batch for batch in batches}
            
            # Process results as they complete
            for future in concurrent.futures.as_completed(future_to_batch):
                batch_count = future.result()
                processed_count += batch_count
                pbar.update(batch_count)
                
                # Persist periodically (e.g., every ~10% of documents)
                if processed_count % max(int(total_chunks * 0.1), batch_size) < batch_size:
                                        chroma_db.persist()
                    
    # Final persistence
    print("Performing final database persistence...")
    chroma_db.persist()
    
    # Calculate and display metrics
    elapsed_time = time.time() - start_time
    docs_per_second = processed_count / elapsed_time
    
    print(f"Vector database created with {processed_count} document chunks!")
    print(f"Total time: {elapsed_time:.2f} seconds ({docs_per_second:.2f} docs/second)")
    
    return chroma_db

# Usage
if initial_db:
    print("\nInitializing new vector database...")
    chroma_db = create_vector_db(
        all_document_chunks=all_document_chunks,
        ollama_url=OLLAMA_URL,
        ollama_model=OLLAMA_MODEL,
        chroma_path=CHROMA_PATH,
        batch_size=512,  # Adjust based on your system and model
        max_workers=150  # Adjust based on CPU cores and memory
    )
else:
    print("\nSkipping vector database creation - using existing database")

## load chroma db from disk
from langchain.embeddings import OllamaEmbeddings
from langchain.vectorstores import Chroma

print("Loading Chroma vector database from disk...")
chroma_db = Chroma(persist_directory=CHROMA_PATH, embedding_function=OllamaEmbeddings(base_url=OLLAMA_URL, model=OLLAMA_MODEL))
print(f"Successfully loaded Chroma database from {CHROMA_PATH}")

# test similarity search
query = "Who is Alice?"
print(f"\nRunning test similarity search with query: '{query}'")

result_docs = chroma_db.similarity_search(query)
print(f"Retrieved {len(result_docs)} relevant documents")

for doc in result_docs:
    print(doc)

from langchain_community.llms import Ollama

def chat_ollama(message, history):
    print(f"\nProcessing chat query: '{message}'")
    
    # initiate ollama
    ollama = Ollama(base_url=OLLAMA_URL, model=OLLAMA_MODEL)

    # search for similar documents in chroma db
    print("Searching vector database for relevant documents...")
    result_chunks = chroma_db.similarity_search(message)
    print(f"Found {len(result_chunks)} relevant document chunks")
    
    chroma_knowledge = ""
    for id, chunk in enumerate(result_chunks):
        source_id = id + 1
        chroma_knowledge += "[" + str(source_id) +"] \n" + chunk.page_content + "\n"

    sources = ""
    for id, chunk in enumerate(result_chunks):
        source_id = id + 1
        sources += "[" + str(source_id) + "] \n" + chunk.metadata["source"] + "\n"

    print("Building prompt with retrieved knowledge and sending to Ollama...")
    prompt = "Answer the following question using the provided knowledge and the chat history:\n\n###KNOWLEDGE: " + chroma_knowledge + "\n###CHAT-HISTORY: " + str(history) + "\n\n###QUESTION: " + message
    result = ollama(prompt) + "\n\n\nReferences:\n" + sources 
    print("Response generated successfully")

    # print(prompt)
    
    return result

print("\nTesting chat function...")
test_response = chat_ollama("Who is Alice?", "")
print("Chat function test completed")

import gradio as gr
print("\nSetting up Gradio web interface...")
gradio_interface = gr.ChatInterface(
        chat_ollama,
        chatbot=gr.Chatbot(),
        textbox=gr.Textbox(placeholder="Example: Who is Alice?", container=False, scale=7),
        title="The Ollama test chatbot",
        description=f"Ask the {OLLAMA_MODEL} chatbot a question!",
        theme='gradio/base', # themes at https://huggingface.co/spaces/gradio/theme-gallery
        retry_btn=None,
        undo_btn="Delete Previous",
        clear_btn="Clear",

)

print("Starting Gradio web server...")
gradio_interface.launch()
print("Gradio server is running!")