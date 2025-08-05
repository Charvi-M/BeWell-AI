from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from FAISS_rag_pipeline import multiagent_chain, log_memory_usage
import os
import gc

app = Flask(__name__)
CORS(app)

# Global user session state
user_session_data = {}

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/userdata", methods=["POST"])
def receive_user_data():
    try:
        data = request.get_json()
        name = data.get("userName", "unknown")
        
        # Store minimal user profile
        user_session_data["user"] = {
            "name": name,
            "age": data.get("userAge", ""),
            "country": data.get("userCountry", ""),
            "financial": data.get("financialStatus", ""),
            "diagnosis": data.get("hasDiagnosis", False),
            "is_minor": data.get("isMinor", False)
        }
        
        print(f"[BeeWell] New session: {name}")
        return jsonify({"status": "success"})
        
    except Exception as e:
        print(f"[ERROR] User data error: {e}")
        return jsonify({"status": "error"}), 500

@app.route("/api/chat", methods=["POST"])
def chat_handler():
    try:
        log_memory_usage("Start chat_handler")
        
        data = request.get_json()
        user_input = data.get("message", "")
        user_profile = user_session_data.get("user", {})
        
        print(f"[BeeWell] Processing: {user_input[:50]}...")
        
        # Process with multiagent chain
        result = multiagent_chain(user_input, user_profile)
        
        log_memory_usage("End chat_handler")
        
        # Force garbage collection after each request
        gc.collect()
        
        return jsonify({
            "agent": result.get("agent", "Therapist"),
            "response": result.get("response", "I'm here to help.")
        })
        
    except Exception as e:
        print(f"[BeeWell] Error: {str(e)}")
        gc.collect()  # Clean up on error too
        return jsonify({
            "agent": "System",
            "response": "Technical difficulties. Please try again."
        }), 500

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
