import fitz  
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain_community.document_loaders import PyPDFLoader


# In[ ]:


def load_pdf(pdf_path):
    
    loader = PyPDFLoader(pdf_path)
    
    return loader.load()


def split_text_documents(pdf_documents, chunk_size=500, chunk_overlap=20):
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    
    return text_splitter.split_documents(pdf_documents)


# In[ ]:


# Loading pdf:
pdf_path = r"D:/the-intelligent-investor.pdf"
pdf_documents = load_pdf(pdf_path)
all_splits = split_text_documents(pdf_documents)


# In[ ]:


all_splits


# # Loading model for embeddings:

# In[ ]:


model = SentenceTransformerEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')


# # Assigning splitted book text and model to vector DB: 

# In[ ]:


from langchain_chroma import Chroma

#Initializing Chroma vector db with all book documents and model for embeddings
vector_DB = Chroma.from_documents(all_splits, model, persist_directory="vector_DB_2")


# In[ ]:


import pickle

embedding = vector_DB.embeddings

path_to_save = "vector_DB_embeddings.pkl"

# Saving the extracted embeddings
data_to_save = {
    "embedding": embedding,
}

with open(path_to_save, "wb") as f:
    pickle.dump(data_to_save, f) 

print("Vector database data embeddings saved successfully at:", path_to_save)


# # Loading the saved embeddings and initializing vector DB: 

# In[ ]:


from langchain_chroma import Chroma
import pickle

# Loading dumped vector db embeddings
path_to_load = "vector_DB_embeddings.pkl"

with open(path_to_load, "rb") as f:
    saved_data = pickle.load(f)


# In[ ]:


vector_DB = Chroma(persist_directory="vector_DB", embedding_function=saved_data["embedding"])


# In[ ]:


# Retrieving k number of chunks from vector db
retriever = vector_DB.as_retriever(k=4)


# In[ ]:


query = "What the book is about?"

docs = retriever.invoke(query)

docs


# In[ ]:


query2 = "What are the best investment strategies?"

docs2 = retriever.invoke(query2)

docs2


# In[ ]:


docslist = []
for i in range(len(docs)):
    docslist.append(docs[i].page_content)

context = "".join(docslist) 
context


# In[ ]:


docslist2 = []
for i in range(len(docs2)):
    docslist2.append(docs2[i].page_content)

context2 = "".join(docslist2) 
context2


# # Approach 1:

# In[ ]:


import transformers
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# In[ ]:


model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# Tokenizer initialization
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Pipeline setup for text generation
pipeline = transformers.pipeline(
    "text-generation",
    model=model_name,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)

# Preparing the input prompt
prompt = query if context is None else f"{context}\n\n{query}"

# Generating responses
response = pipeline(
    prompt,
    max_length=500,
    do_sample=True,
    top_k=1,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
    temperature=0.5,
)


# In[ ]:


generated_response = response


# In[ ]:


generated_response


# # Cosine Similarity testing:

# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# In[ ]:


# Vectorizing the text
vectorizer = CountVectorizer().fit([generated_response, context])
vector_model = vectorizer.transform([generated_response])
vector_book = vectorizer.transform([context])

# Calculating cosine similarity
cosine_sim = cosine_similarity(vector_model, vector_book)
print("Cosine Similarity:", cosine_sim[0][0])


# In[ ]:





# # Approach 2:

# In[ ]:


from transformers import BartForConditionalGeneration, BartTokenizer

# Loading BART model and tokenizer
model_name = "facebook/bart-large-cnn"
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)


# In[ ]:


# Defining a function to generate response
def generateResponse(document):
    input_ids = tokenizer.encode(document, return_tensors="pt", max_length=1024, truncation=True)
    response_ids = model.generate(input_ids, max_length=500, min_length=50, length_penalty=2.0, num_beams=4, early_stopping=True, do_sample=True, temperature=0.5)
    response = tokenizer.decode(response_ids[0], skip_special_tokens=True)
    return response


# In[ ]:


response = generateResponse(context)
print("Response:", response)


# In[ ]:


response2 = generateResponse(context2)
print("Response:", response2)


# # Cosine Similarity testing:

# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# In[ ]:


# Vectorizing the text
vectorizer = CountVectorizer().fit([response, context])
vector_model = vectorizer.transform([response])
vector_book = vectorizer.transform([context])

# Calculating cosine similarity
cosine_sim = cosine_similarity(vector_model, vector_book)
print("Cosine Similarity:", cosine_sim[0][0])


# In[ ]:


# Vectorizing the text
vectorizer = CountVectorizer().fit([response2, context2])
vector_model = vectorizer.transform([response2])
vector_book = vectorizer.transform([context2])

# Calculating cosine similarity
cosine_sim = cosine_similarity(vector_model, vector_book)
print("Cosine Similarity:", cosine_sim[0][0])


# In[ ]:





# In[ ]:


# # Saving the model
# model.save_pretrained("BART_Model")
# tokenizer.save_pretrained("BART_Tokenizer")


# # Approach 3:

# In[ ]:


from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever


# In[ ]:


# Initializing Ollama model
ollama_model_name = "phi3"
ollama_model = ChatOllama(model=ollama_model_name)


# In[ ]:


model_prompt = PromptTemplate(
    input_variables=["question"],
    template="""You are an AI language model assistant. Your task is to generate response
    on user question by retrieving relevant documents from a vector database. 
    By analyzing the douments according to user question, your goal is to help the user 
    overcome some of the limitations of the distance-based similarity search.
    Original question: {question}""",
)


# In[ ]:


dbretriever = MultiQueryRetriever.from_llm(
    vector_DB.as_retriever(), 
    ollama_model,
    prompt=model_prompt
)

template = """Answer the question based ONLY on the following context and just say that you don't know the answer on
              any other context or question asked by user out of context. Context:{context} 
              Question: {question}"""

prompt = ChatPromptTemplate.from_template(template)


# In[ ]:


chain = (
    {"context": dbretriever, "question": RunnablePassthrough()}
    | prompt
    | ollama_model
    | StrOutputParser()
)


# In[ ]:


response1 = chain.invoke(input(""))


# In[ ]:


response1


# In[ ]:


response2 =  chain.invoke("What are the best investment strategies?")


# In[ ]:


response2


# # Cosine Similarity testing:

# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# In[ ]:


# Vectorizing the text
vectorizer = CountVectorizer().fit([response1, context])
vector_model = vectorizer.transform([response1])
vector_book = vectorizer.transform([context])

# Calculating cosine similarity
cosine_sim = cosine_similarity(vector_model, vector_book)
print("Cosine Similarity:", cosine_sim[0][0])


# In[ ]:


# Vectorizing the text
vectorizer = CountVectorizer().fit([response2, context2])
vector_model = vectorizer.transform([response2])
vector_book = vectorizer.transform([context2])

# Calculating cosine similarity
cosine_sim = cosine_similarity(vector_model, vector_book)
print("Cosine Similarity:", cosine_sim[0][0])