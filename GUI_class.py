import pickle
import numpy
import torch
import transformers
from langchain_chroma import Chroma
from transformers import BartTokenizer, BartForConditionalGeneration
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever

class ChatBot:

    def __init__(self, query):

        self.query = query

        with open(r"vector_DB_embeddings.pkl", "rb") as f:
            saved_data = pickle.load(f)

        self.vectordb = Chroma(persist_directory=r"vector_DB", embedding_function=saved_data["embedding"])

        self.retriever = self.vectordb.as_retriever(k=4)

        self.docs = self.retriever.invoke(query)

        docslist = []
        for i in range(len(self.docs)):
            docslist.append(self.docs[i].page_content)

        self.context = "".join(docslist)

    def Approach1(self):

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
        prompt = self.query if self.context is None else f"{self.context}\n\n{self.query}"

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

        generated_response = response

        return generated_response


    def Approach2(self):

        # Loading the model
        BartModel = BartForConditionalGeneration.from_pretrained("BART_Model")

        # Loading the tokenizer
        Bart_Tokenizer = BartTokenizer.from_pretrained("BART_Tokenizer")

        input_ids = Bart_Tokenizer.encode(self.context, return_tensors="pt", max_length=1024, truncation=True)
        response_ids = BartModel.generate(input_ids, max_length=500, min_length=50, length_penalty=2.0, num_beams=4,
                                      early_stopping=True, do_sample=True, temperature=0.5)
        response = Bart_Tokenizer.decode(response_ids[0], skip_special_tokens=True)

        generated_response = response

        return generated_response


    def Approach3(self):

        # Ollama model
        ollama_model_name = "phi3"
        ollama_model = ChatOllama(model=ollama_model_name)

        model_prompt = PromptTemplate(
            input_variables=["question"],
            template="""You are an AI language model assistant. Your task is to generate response
            on user question by retrieving relevant documents from a vector database. 
            By analyzing the douments according to user question, your goal is to help the user 
            overcome some of the limitations of the distance-based similarity search.
            Original question: {question}""",
        )

        dbretriever = MultiQueryRetriever.from_llm(
            self.vectordb.as_retriever(),
            ollama_model,
            prompt = model_prompt
        )

        template = """Answer the question based ONLY on the following context and just say that you don't know the answer on
                      any other context or question asked by user out of context. Context:{context} 
                      Question: {question}"""

        prompt = ChatPromptTemplate.from_template(template)

        chain = (
                {"context": dbretriever, "question": RunnablePassthrough()}
                | prompt
                | ollama_model
                | StrOutputParser()
        )

        generated_response = chain.invoke(self.query)

        return generated_response