from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import os
import traceback

os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['TRANSFORMERS_CACHE'] = '/tmp'

from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import gc

load_dotenv()

# Initialize models as None - will load lazily
embedding_model = None
llm = None
dbTherapy = None
dbResources = None
retrieverTherapy = None
retrieverResource = None
memory = None
therapy_base_chain = None

def get_embedding_model():
    """Lazy loading of embedding model"""
    global embedding_model
    if embedding_model is None:
        try:
            embedding_model = HuggingFaceEmbeddings(
                model_name="all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            print("[INFO] Embedding model loaded successfully")
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
    """Lazy loading of vector stores with proper error handling"""
    global dbTherapy, dbResources, retrieverTherapy, retrieverResource
    
    if dbTherapy is None or dbResources is None:
        try:
            # Check if vectorstore files exist
            if not os.path.exists("faiss_therapy_index"):
                error_msg = "FAISS therapy index not found at faiss_therapy_index/"
                print(f"[ERROR] {error_msg}")
                raise FileNotFoundError(error_msg)
                
            if not os.path.exists("faiss_resource_index"):
                error_msg = "FAISS resource index not found at faiss_resource_index/"
                print(f"[ERROR] {error_msg}")
                raise FileNotFoundError(error_msg)
            
            print("[INFO] Loading FAISS vectorstores...")
            embeddings = get_embedding_model()
            
            # Load vectorstores
            dbTherapy = FAISS.load_local("faiss_therapy_index", embeddings, allow_dangerous_deserialization=True)
            dbResources = FAISS.load_local("faiss_resource_index", embeddings, allow_dangerous_deserialization=True)
            
            # Create retrievers
            retrieverTherapy = dbTherapy.as_retriever(search_kwargs={"k": 3})
            retrieverResource = dbResources.as_retriever(search_kwargs={"k": 3})
            
            print("[INFO] FAISS vectorstores loaded successfully")
            
        except Exception as e:
            print(f"[ERROR] Failed to load vectorstores: {e}")
            traceback.print_exc()
            raise
            
    return retrieverTherapy, retrieverResource

def get_therapy_chain():
    """Lazy loading of therapy chain with error handling"""
    global therapy_base_chain, memory
    
    if therapy_base_chain is None:
        try:
            print("[INFO] Initializing therapy chain...")
            
            memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True,
                output_key="answer",
                max_token_limit=1000
            )

            therapy_retriever, _ = get_vectorstores()
            
            therapy_base_chain = ConversationalRetrievalChain.from_llm(
                llm=get_llm(),
                retriever=therapy_retriever,
                memory=memory,
                return_source_documents=True,
                output_key="answer",
                verbose=False
            )
            
            print("[INFO] Therapy chain initialized successfully")
            
        except Exception as e:
            print(f"[ERROR] Failed to initialize therapy chain: {e}")
            traceback.print_exc()
            raise
            
    return therapy_base_chain

# Prompt templates
therapist_prompt = PromptTemplate(input_variables=["question", "raw_answer", "user_profile"], template="""
You are a compassionate clinical psychologist speaking directly with a client.

Client Profile: {user_profile}
Client's Question: {question}
Knowledge Base Insights: "{raw_answer}"

Remember this client's information from their profile. You can refer to their name, age, country, and other details naturally in conversation.

If the user provides symptoms, you must:
1. List possible diagnoses in bullet points
2. Include a disclaimer that you are an AI agent, not a professional
3. Ask if they want professional help or want to talk about it
4. If they want professional help, direct them to resources
5. If they want to talk, provide gentle and supportive guidance

Respond naturally and directly to the client. Avoid saying stuff like of course here is a gentle and supportive reply because then the user will feel that you are not talking to them directly.

Your response:
""")

resource_prompt = PromptTemplate(input_variables=["question", "raw_answer", "user_profile"], template="""
User asked: "{question}"
User Profile: {user_profile}
Resources retrieved: "{raw_answer}"

You are a mental health assistant. Suggest **only country-specific and free (if user is financially struggling or on a limited budget otherwise suggest paid resources too)** support links or helpline numbers.

Keep it short, practical, and clear.
Output only contact options, links, or phone numbers:
""")

classification_prompt = PromptTemplate(input_variables=["question"], template="""
You are an intent classifier for a multi-agent mental health system.

Classify the user input:
- therapist: if the user is asking for a definition, explanation, symptoms, needs emotional support, asks personal questions about themselves, or wants to chat
- resource: if the user is asking for support links, professional help, helpline numbers, or country-specific services

Input: "{question}"
Respond with one word only: therapist or resource
""")

def classify_agent(user_input: str) -> str:
    """Classify user intent"""
    try:
        prompt = classification_prompt.format(question=user_input)
        response = get_llm().invoke(prompt)
        result = response.content.strip().lower() if hasattr(response, "content") else str(response).strip().lower()
        return result if result in ["therapist", "resource"] else "therapist"
    except Exception as e:
        print(f"[ERROR] Classification error: {e}")
        traceback.print_exc()
        return "therapist"  # Default fallback

def therapist_wrapper(user_input, raw_answer, user_profile):
    """Therapist response wrapper"""
    try:
        prompt = therapist_prompt.format(question=user_input, raw_answer=raw_answer, user_profile=user_profile)
        styled = get_llm().invoke(prompt)
        return styled.content if hasattr(styled, "content") else str(styled)
    except Exception as e:
        print(f"[ERROR] Therapist wrapper error: {e}")
        traceback.print_exc()
        return "I'm here to help. Could you tell me more about what's on your mind?"

def resource_wrapper(user_input, user_profile):
    """Resource response wrapper"""
    try:
        _, resource_retriever = get_vectorstores()
        docs = resource_retriever.invoke(user_input)
        
        if not docs:
            raise ValueError("No documents retrieved from resource vectorstore")
        
        # Limit text length to save memory
        raw_text = "\n\n".join([doc.page_content[:500] for doc in docs[:3]])
        prompt = resource_prompt.format(question=user_input, raw_answer=raw_text, user_profile=user_profile)
        styled = get_llm().invoke(prompt)
        return styled.content if hasattr(styled, "content") else str(styled)
    except Exception as e:
        print(f"[ERROR] Resource wrapper error: {e}")
        traceback.print_exc()
        return f"I'm unable to retrieve resource documents right now. Error: {str(e)}. Please try again or contact support if this persists."

def multiagent_chain(user_input: str, user_profile: dict) -> dict:
    """Main multiagent entry point with comprehensive error handling"""
    try:
        print(f"[INFO] Processing user input: {user_input}")
        
        # Classify the agent
        agent = classify_agent(user_input)
        profile_summary = f"Country: {user_profile.get('country', 'unknown')}, Financial: {user_profile.get('financial', 'unknown')}, Name: {user_profile.get('name', 'unknown')}, Age: {user_profile.get('age', 'unknown')}"
        
        print(f"[INFO] Agent classified as: {agent}")

        if agent == "resource":
            try:
                answer = resource_wrapper(user_input, profile_summary)
                gc.collect()
                return {"agent": "Gemini (Resource Assistant)", "response": answer}
            except Exception as e:
                print(f"[ERROR] Resource agent failed: {e}")
                traceback.print_exc()
                return {"agent": "System", "response": f"I'm unable to retrieve resource documents right now. Error: {str(e)}. Please try again or contact support if this persists."}
        else:
            try:
                # Include user profile context in the question
                contextualized_question = f"User Profile: {profile_summary}\nUser Question: {user_input}"
                
                chain = get_therapy_chain()
                result = chain.invoke({"question": contextualized_question})
                raw_answer = result.get("answer", "")

                if not raw_answer:
                    raise ValueError("No answer retrieved from therapy chain")

                styled = therapist_wrapper(user_input, raw_answer, profile_summary)
                gc.collect()
                return {"agent": "Gemini (Therapist)", "response": styled}
                
            except Exception as e:
                print(f"[ERROR] Therapy agent failed: {e}")
                traceback.print_exc()
                return {"agent": "System", "response": f"I'm unable to retrieve therapy documents right now. Error: {str(e)}. Please try again or contact support if this persists."}

    except Exception as e:
        print(f"[ERROR] Multiagent chain critical error: {e}")
        traceback.print_exc()
        gc.collect()
        return {"agent": "System", "response": f"I'm experiencing technical difficulties. Error: {str(e)}. Please try again."}
