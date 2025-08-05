import os
import gc
import psutil
import traceback
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

# Memory optimization settings
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['TRANSFORMERS_CACHE'] = '/tmp'
os.environ['HF_HOME'] = '/tmp'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

load_dotenv()

# Global variables for lazy loading
embedding_model = None
llm = None
dbTherapy = None
dbResources = None
retrieverTherapy = None
retrieverResource = None

def log_memory_usage(context=""):
    """Log current memory usage"""
    try:
        process = psutil.Process()
        mem_in_mb = process.memory_info().rss / (1024 * 1024)
        print(f"[MEMORY] {context}: {mem_in_mb:.1f} MB")
        return mem_in_mb
    except:
        return 0

def get_embedding_model():
    """Lazy loading of embedding model with memory optimization"""
    global embedding_model
    if embedding_model is None:
        try:
            log_memory_usage("Before loading embeddings")
            
            # Fixed: Remove normalize_embeddings from model_kwargs, put in encode_kwargs
            embedding_model = HuggingFaceEmbeddings(
                model_name="all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={
                    'batch_size': 1,  # Process one at a time
                    'normalize_embeddings': True  # Move here
                }
            )
            
            log_memory_usage("After loading embeddings")
            gc.collect()
            
        except Exception as e:
            print(f"[ERROR] Failed to load embedding model: {e}")
            traceback.print_exc()
            raise
    return embedding_model

def get_llm():
    """Lazy loading of LLM"""
    global llm
    if llm is None:
        try:
            llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                temperature=0.7,
                api_key=os.getenv("GEMINI_API_KEY"),
            )
            print("[INFO] LLM loaded successfully")
        except Exception as e:
            print(f"[ERROR] Failed to load LLM: {e}")
            traceback.print_exc()
            raise
    return llm

def get_vectorstores():
    """Memory-optimized vector store loading"""
    global dbTherapy, dbResources, retrieverTherapy, retrieverResource
    
    if dbTherapy is None or dbResources is None:
        try:
            log_memory_usage("Before loading vectorstores")
            
            # Check if files exist
            if not os.path.exists("faiss_therapy_index"):
                raise FileNotFoundError("FAISS therapy index not found")
            if not os.path.exists("faiss_resource_index"):
                raise FileNotFoundError("FAISS resource index not found")
            
            embeddings = get_embedding_model()
            
            # Load therapy vectorstore
            dbTherapy = FAISS.load_local(
                "faiss_therapy_index",
                embeddings,
                allow_dangerous_deserialization=True
            )
            
            # Force garbage collection before loading second vectorstore
            gc.collect()
            
            # Load resource vectorstore
            dbResources = FAISS.load_local(
                "faiss_resource_index",
                embeddings,
                allow_dangerous_deserialization=True
            )
            
            # Create retrievers with limited results
            retrieverTherapy = dbTherapy.as_retriever(search_kwargs={"k": 2})
            retrieverResource = dbResources.as_retriever(search_kwargs={"k": 2})
            
            log_memory_usage("After loading vectorstores")
            gc.collect()
            
        except Exception as e:
            print(f"[ERROR] Failed to load vectorstores: {e}")
            traceback.print_exc()
            raise
            
    return retrieverTherapy, retrieverResource

def classify_agent(user_input: str) -> str:
    """Simple classification with memory management"""
    try:
        # Simple keyword-based classification to save memory
        keywords_resource = ['help', 'helpline', 'support', 'contact', 'number', 'resource', 'professional']
        keywords_therapy = ['feel', 'anxiety', 'depression', 'stress', 'sad', 'talk', 'chat', 'racing', 'ghosts', 'heart']
        
        user_lower = user_input.lower()
        
        # Count keyword matches
        resource_score = sum(1 for keyword in keywords_resource if keyword in user_lower)
        therapy_score = sum(1 for keyword in keywords_therapy if keyword in user_lower)
        
        # If no clear match, classify based on question words
        if resource_score == therapy_score:
            if any(word in user_lower for word in ['where', 'how to find', 'contact', 'call']):
                return "resource"
        
        return "resource" if resource_score > therapy_score else "therapist"
        
    except Exception as e:
        print(f"[ERROR] Classification error: {e}")
        return "therapist"

def get_therapy_response(user_input: str, user_profile: dict) -> str:
    """Get therapy response with memory optimization"""
    try:
        log_memory_usage("Before therapy response")
        
        therapy_retriever, _ = get_vectorstores()
        docs = therapy_retriever.invoke(user_input)
        
        if not docs:
            raise ValueError("No documents retrieved")
        
        # Limit context to save memory
        context = "\n".join([doc.page_content[:300] for doc in docs[:2]])
        
        profile_summary = f"Name: {user_profile.get('name', 'User')}, Country: {user_profile.get('country', 'unknown')}"
        
        # Simplified prompt to reduce token usage
        prompt = f"""You are a compassionate therapist.
Client: {profile_summary}
Question: {user_input}
Context: {context}

Provide a brief, supportive response (max 150 words):"""
        
        llm = get_llm()
        response = llm.invoke(prompt)
        
        log_memory_usage("After therapy response")
        gc.collect()
        
        return response.content if hasattr(response, 'content') else str(response)
        
    except Exception as e:
        print(f"[ERROR] Therapy response error: {e}")
        traceback.print_exc()
        return "I'm here to support you. Could you tell me more about what's on your mind?"

def get_resource_response(user_input: str, user_profile: dict) -> str:
    """Get resource response with memory optimization"""
    try:
        log_memory_usage("Before resource response")
        
        _, resource_retriever = get_vectorstores()
        docs = resource_retriever.invoke(user_input)
        
        if not docs:
            raise ValueError("No resource documents retrieved")
        
        # Limit context
        context = "\n".join([doc.page_content[:300] for doc in docs[:2]])
        country = user_profile.get('country', 'unknown')
        
        # Simplified prompt
        prompt = f"""Provide mental health resources for {country}.
Question: {user_input}
Available resources: {context}

List only contact numbers and websites (max 100 words):"""
        
        llm = get_llm()
        response = llm.invoke(prompt)
        
        log_memory_usage("After resource response")
        gc.collect()
        
        return response.content if hasattr(response, 'content') else str(response)
        
    except Exception as e:
        print(f"[ERROR] Resource response error: {e}")
        traceback.print_exc()
        return f"I'm unable to retrieve resources right now. Error: {str(e)}"

def multiagent_chain(user_input: str, user_profile: dict) -> dict:
    """Memory-optimized main function"""
    try:
        log_memory_usage("Start of multiagent_chain")
        
        # Classify intent
        agent = classify_agent(user_input)
        print(f"[INFO] Classified as: {agent}")
        
        if agent == "resource":
            response = get_resource_response(user_input, user_profile)
            agent_name = "Resource Assistant"
        else:
            response = get_therapy_response(user_input, user_profile)
            agent_name = "Therapist"
        
        log_memory_usage("End of multiagent_chain")
        
        # Force cleanup
        gc.collect()
        
        return {
            "agent": f"Gemini ({agent_name})",
            "response": response
        }
        
    except Exception as e:
        print(f"[ERROR] Multiagent chain error: {e}")
        traceback.print_exc()
        gc.collect()
        return {
            "agent": "System",
            "response": "Technical difficulties occurred. Please try again."
        }
