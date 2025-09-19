from flask import Flask, request, jsonify
from twilio.rest import Client
import os
import json
from dotenv import load_dotenv
from datetime import datetime, timedelta
import redis
from urllib.parse import urlparse
import logging
import time

# === New: LangChain / LLM & search imports ===
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_tavily import TavilySearch
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage, SystemMessage

# ------------------------------------------------------------------------------
# Setup
# ------------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv("FLASK_SECRET_KEY", "change-me")

# Redis
redis_url = os.environ.get("REDIS_URL", "")
if not redis_url:
    logger.warning("REDIS_URL not set; defaulting to local Redis.")
url = urlparse(redis_url or "redis://localhost:6379/0")
redis_client = redis.Redis(
    host=url.hostname,
    port=url.port,
    password=url.password,
    ssl=(url.scheme == "rediss"),
    ssl_cert_reqs=None
)

# Twilio
TWILIO_WHATSAPP_NUMBER = os.getenv("TWILIO_WHATSAPP_NUMBER")
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_TEMPLATE_SID = os.getenv("TWILIO_TEMPLATE_SID")  # optional

if not (TWILIO_WHATSAPP_NUMBER and TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN):
    logger.warning("Twilio env vars missing; sending messages will fail.")

client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

# ------------------------------------------------------------------------------
# LLMs & Search tools (initialized once)
# ------------------------------------------------------------------------------
# You need: GROQ_API_KEY, OPENAI_API_KEY, SERPER_API_KEY, TAVILY_API_KEY in env.
llm_groq = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
llm_openai = ChatOpenAI(model="gpt-4o-mini", temperature=0)

serper_tool = GoogleSerperAPIWrapper()  # uses SERPER_API_KEY
tavily_tool = TavilySearch()            # uses TAVILY_API_KEY

# ------------------------------------------------------------------------------
# Sessions
# ------------------------------------------------------------------------------
class ChatSession:
    def __init__(self, sender_number):
        self.sender_number = sender_number
        self.last_activity = datetime.now()
        self.conversation_history = []
        self.last_message_id = None
        self.is_new_session = True

    def to_dict(self):
        return {
            "sender_number": self.sender_number,
            "last_activity": self.last_activity.isoformat(),
            "conversation_history": self.conversation_history,
            "last_message_id": self.last_message_id,
            "is_new_session": self.is_new_session
        }

    @staticmethod
    def from_dict(data):
        session = ChatSession(data["sender_number"])
        session.last_activity = datetime.fromisoformat(data["last_activity"])
        session.conversation_history = data.get("conversation_history", [])
        session.last_message_id = data.get("last_message_id")
        session.is_new_session = False
        return session

def get_chat_session(sender_number):
    key = f"chat_session:{sender_number}"
    try:
        raw = redis_client.get(key)
        if raw:
            session_dict = json.loads(raw.decode("utf-8"))
            last_activity = datetime.fromisoformat(session_dict.get("last_activity"))
            if datetime.now() - last_activity > timedelta(hours=24):
                return ChatSession(sender_number)
            return ChatSession.from_dict(session_dict)
        return ChatSession(sender_number)
    except Exception as e:
        logger.error(f"Error getting chat session: {e}")
        return ChatSession(sender_number)

def save_chat_session(session: ChatSession):
    try:
        key = f"chat_session:{session.sender_number}"
        redis_client.setex(key, timedelta(hours=24), json.dumps(session.to_dict()))
    except Exception as e:
        logger.error(f"Error saving chat session: {e}")

# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------
def needs_rating(response_text: str) -> bool:
    casual_patterns = [
        "thank you", "thanks", "you're welcome", "noted",
        "got it", "understood", "👍", "🙏", "nice", "bravo", "amazing", "impressive",
        "sorry", "please", "hi", "hello", "hey", "good morning", "good afternoon",
        "good evening", "bye", "goodbye", "cool", "yeah", "yah", "alright", "ok", "okay"
    ]
    text = (response_text or "").lower().strip()
    is_short = len(text.split()) < 10
    is_casual = any(p in text for p in casual_patterns)
    is_error = "error" in text or "an error occurred" in text
    has_emoji_ending = text.endswith(('!', '👋', '🙂', '😊'))
    return not (is_short and (is_casual or has_emoji_ending or is_error))

def get_greeting_message():
    hour = datetime.now().hour
    if 5 <= hour < 12:
        return "Good morning! 🌅"
    elif 12 <= hour < 17:
        return "Good afternoon! 🌞"
    return "Good evening! 🌙"

def create_welcome_message():
    greeting = get_greeting_message()
    return (
        f"{greeting} Welcome to AI Fact Checker! 🤖✨\n\n"
        "I'm here to help you verify information and check facts. "
        "Share a factual statement you'd like me to verify, and I’ll search reliable sources and reply clearly.\n\n"
        "To get started, simply type your question or statement! 📝"
    )

def store_feedback(message_id, feedback_type, sender_number):
    try:
        feedback_key = f"feedback:{message_id}"
        feedback_data = {
            "timestamp": datetime.now().isoformat(),
            "feedback_type": feedback_type,
            "sender_number": sender_number
        }
        redis_client.setex(feedback_key, timedelta(days=30), json.dumps(feedback_data))
    except Exception as e:
        logger.error(f"Error storing feedback: {e}")

def send_message_with_template(to_number, body_text, user_input, is_greeting=False):
    """
    Sends a normal message. If not greeting and 'needs_rating', optionally sends
    a template follow-up when TWILIO_TEMPLATE_SID is set.
    """
    try:
        main_message = client.messages.create(
            from_=TWILIO_WHATSAPP_NUMBER,
            to=to_number,
            body=body_text
        )
        time.sleep(1)

        if not is_greeting and needs_rating(user_input) and TWILIO_TEMPLATE_SID:
            template_message = client.messages.create(
                from_=TWILIO_WHATSAPP_NUMBER,
                to=to_number,
                body="Was this response helpful?",
                content_sid=TWILIO_TEMPLATE_SID
            )
            return template_message  # last sent message id (buttons)
        return main_message
    except Exception as e:
        logger.error(f"Error sending message: {str(e)}")
        raise

def handle_button_response(button_text, chat_session, sender_number):
    try:
        if button_text in ["Pleased", "Not Pleased"]:
            feedback_type = "positive" if button_text == "Pleased" else "negative"
            if chat_session.last_message_id:
                store_feedback(chat_session.last_message_id, feedback_type, sender_number)
                message = client.messages.create(
                    from_=TWILIO_WHATSAPP_NUMBER,
                    to=sender_number,
                    body="Thank you for your feedback! 🙏\nWould you like to verify another claim?"
                )
                return True, message.sid
        return False, None
    except Exception as e:
        logger.error(f"Error handling button response: {e}")
        return False, None

# ------------------------------------------------------------------------------
# Greeting detection (reused from your API)
# ------------------------------------------------------------------------------
greeting_keywords = [
    "hello", "hi", "hey", "howdy", "yo", "greetings", "good day",
    "good morning", "good afternoon", "good evening", "how are you",
    "how are you doing", "how's it going", "what's up", "sup",
    "how have you been", "nice to meet you", "pleased to meet you",
    "peace be upon you", "as-salamu alaykum", "salamu alaikum",
    "sannu", "barka da rana", "barka da safiya", "barka da yamma", "wagwan"
]

def is_greeting(text: str) -> bool:
    lower_text = (text or "").strip().lower()
    return any(greet in lower_text for greet in greeting_keywords)

# ------------------------------------------------------------------------------
# Core fact-checking function (Serper → Tavily → Groq/OpenAI)
# ------------------------------------------------------------------------------
def fact_check_reply(user_input: str) -> dict:
    """
    Runs web search + LLM and returns:
      { "message": "<final reply string>", "sources": [url, ...] }
    """
    if not user_input:
        return {"message": "Please provide a claim to verify.", "sources": []}

    # Handle greeting inline (fast path)
    if is_greeting(user_input):
        return {
            "message": "Hello! 😊 I'm your fact-checking assistant. Share a factual statement you’d like me to verify, and I’ll check credible sources for you.",
            "sources": []
        }

    sources = []
    search_context = ""
    serper_failed = False

    # Try Serper
    try:
        serper_data = serper_tool.results(user_input)
        if isinstance(serper_data, dict) and "organic" in serper_data:
            # collect urls
            sources += [item.get("link") for item in serper_data["organic"] if "link" in item]
            # context lines (title + snippet)
            search_context = "\n\n".join(
                f"- {item.get('title', '')}\n  {item.get('snippet', '')}"
                for item in serper_data["organic"]
            )
        else:
            raise ValueError("No useful results from Serper.")
    except Exception as e:
        logger.warning(f"Serper failed: {e}")
        serper_failed = True

    # Fallback: Tavily
    if serper_failed:
        try:
            tavily_data = tavily_tool.invoke({"query": user_input})
            if isinstance(tavily_data, dict) and "results" in tavily_data:
                sources += [item.get("url") for item in tavily_data["results"] if "url" in item]
                search_context = "\n\n".join(
                    f"- {item.get('title', '')}\n  {item.get('content', '')}"
                    for item in tavily_data["results"]
                )
        except Exception as e2:
            logger.error(f"Tavily failed: {e2}")
            search_context = "No search results available."

    combined_context = f"Search results:\n\n{search_context}"

    messages = [
        SystemMessage(content="""
You are a professional AI fact-checking assistant.
Your primary role is to verify the accuracy of claims using the search results provided.
Respond in a clear, formal, and direct tone in narrative form.
Rules:
1) Clearly state whether the claim is true, false, misleading, or unverifiable.
2) If it's a non-fact-check question, say your role is limited to verifying factual claims.
3) If useful, include one or two URLs to support your answer — only if essential for credibility.
"""),
        HumanMessage(content=f"{combined_context}\n\nUser Claim: {user_input}")
    ]

    # Try Groq → fallback OpenAI
    try:
        response = llm_groq.invoke(messages)
        text = response.content if hasattr(response, "content") else str(response)
    except Exception as e:
        logger.warning(f"Groq failed: {e}")
        try:
            response = llm_openai.invoke(messages)
            text = response.content if hasattr(response, "content") else str(response)
        except Exception as e2:
            logger.error(f"All LLMs failed: {e2}")
            return {
                "message": "I couldn't complete the fact-check due to an internal error. Please try again.",
                "sources": []
            }

    return {"message": text, "sources": [s for s in sources if s][:5]}  # keep it short

# ------------------------------------------------------------------------------
# Webhook
# ------------------------------------------------------------------------------
@app.route("/whatsapp", methods=["POST"])
def whatsapp_reply():
    try:
        incoming_message = (request.form.get("Body") or "").strip()
        sender_number = request.form.get("From")

        if not sender_number:
            return jsonify({"status": "error", "message": "Missing sender"}), 400

        chat_session = get_chat_session(sender_number)

        # Button feedback
        button_text = request.form.get("ButtonText")
        if button_text:
            is_feedback, message_sid = handle_button_response(button_text, chat_session, sender_number)
            if is_feedback:
                return jsonify({"status": "success", "message_sid": message_sid})

        # Welcome for new sessions (24h)
        if chat_session.is_new_session:
            welcome_text = create_welcome_message()
            welcome_msg = send_message_with_template(
                sender_number, welcome_text, incoming_message, is_greeting=True
            )
            chat_session.conversation_history.append({
                "timestamp": datetime.now().isoformat(),
                "message": welcome_text,
                "type": "outgoing",
                "message_id": welcome_msg.sid
            })

        # Save incoming
        chat_session.conversation_history.append({
            "timestamp": datetime.now().isoformat(),
            "message": incoming_message,
            "type": "incoming"
        })

        # === Core: run local fact-check ===
        fc = fact_check_reply(incoming_message)
        response_text = fc.get("message") or "I am unable to provide a response now. Please try again."

        # Send reply
        message = send_message_with_template(sender_number, response_text, incoming_message)
        chat_session.last_message_id = message.sid

        # Save outgoing
        chat_session.conversation_history.append({
            "timestamp": datetime.now().isoformat(),
            "message": response_text,
            "type": "outgoing",
            "message_id": message.sid
        })

        chat_session.last_activity = datetime.now()
        save_chat_session(chat_session)

        return jsonify({"status": "success", "message_sid": message.sid})
    except Exception as e:
        logger.error(f"Error in whatsapp_reply: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500

# ------------------------------------------------------------------------------
# Entry
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    # Flask dev server; in production use gunicorn/uvicorn, etc.
    app.run(host='0.0.0.0', port=int(os.getenv("PORT", "5000")), debug=True)
