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
from googletrans import Translator
from pydub import AudioSegment
import openai
import re

# --- NEW: LangChain / search / LLM imports ---
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
redis_url = os.environ.get("REDIS_URL")
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

# OpenAI
openai.api_key = os.getenv("OPENAI_API_KEY")

# Translator
translator = Translator()

# ------------------------------------------------------------------------------
# LLMs & Search tools (initialized once)
# ------------------------------------------------------------------------------
# Need: GROQ_API_KEY, OPENAI_API_KEY, SERPER_API_KEY, TAVILY_API_KEY
llm_groq = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
llm_openai = ChatOpenAI(model="gpt-4o-mini", temperature=0)

serper_tool = GoogleSerperAPIWrapper()  # uses SERPER_API_KEY
tavily_tool = TavilySearch()            # uses TAVILY_API_KEY

# ------------------------------------------------------------------------------
# Session model
# ------------------------------------------------------------------------------
class ChatSession:
    def __init__(self, sender_number):
        self.sender_number = sender_number
        self.last_activity = datetime.now()
        self.conversation_history = []
        self.last_message_id = None
        self.is_new_session = True
        self.language = "en"

    def to_dict(self):
        return {
            "sender_number": self.sender_number,
            "last_activity": self.last_activity.isoformat(),
            "conversation_history": self.conversation_history,
            "last_message_id": self.last_message_id,
            "is_new_session": self.is_new_session,
            "language": self.language
        }

    @staticmethod
    def from_dict(data):
        session = ChatSession(data["sender_number"])
        session.last_activity = datetime.fromisoformat(data["last_activity"])
        session.conversation_history = data.get("conversation_history", [])
        session.last_message_id = data.get("last_message_id")
        session.is_new_session = False
        session.language = data.get("language", "en")
        return session

def get_chat_session(sender_number):
    session_key = f"chat_session:{sender_number}"
    try:
        session_data = redis_client.get(session_key)
        if session_data:
            session_dict = json.loads(session_data.decode('utf-8'))
            last_activity = datetime.fromisoformat(session_dict.get("last_activity"))
            # Reset after 24h idle (you had 2h before—24h is more forgiving)
            if datetime.now() - last_activity > timedelta(hours=24):
                return ChatSession(sender_number)
            return ChatSession.from_dict(session_dict)
        return ChatSession(sender_number)
    except Exception as e:
        logger.error(f"Error getting chat session: {e}")
        return ChatSession(sender_number)

def save_chat_session(session):
    try:
        session_key = f"chat_session:{session.sender_number}"
        session_data = json.dumps(session.to_dict())
        redis_client.setex(session_key, timedelta(hours=24), session_data)
    except Exception as e:
        logger.error(f"Error saving chat session: {e}")

# ------------------------------------------------------------------------------
# Utilities: translation, greetings, rating, feedback
# ------------------------------------------------------------------------------
def translate_text(text, dest_language):
    if not text or dest_language == "en":
        return text
    try:
        # protect URLs from translation
        url_pattern = re.compile(r'(https?://\S+)')
        urls = re.findall(url_pattern, text)
        placeholder_fmt = '__URL_PLACEHOLDER_{}__'
        for i, u in enumerate(urls):
            text = text.replace(u, placeholder_fmt.format(i))
        translated = translator.translate(text, dest=dest_language).text
        for i, u in enumerate(urls):
            translated = translated.replace(placeholder_fmt.format(i), u)
        return translated
    except Exception as e:
        logger.error(f"Error translating text: {e}")
        return text

def needs_rating(response_text):
    casual_patterns = [
        "thank you", "thanks", "you're welcome", "noted",
        "got it", "understood", "👍", "🙏", "nice","bravo","amazing","impressive",
        "sorry", "please", "hi", "hello", "hey", "good morning", "good afternoon",
        "good evening", "bye", "goodbye", "cool","yeah","yah","alright","oh","oops","ok","okay","yes"
    ]
    text = (response_text or "").lower().strip()
    is_short = len(text.split()) < 10
    is_casual = any(p in text for p in casual_patterns)
    is_error = "error" in text or "an error occurred" in text
    has_emoji_ending = text.endswith(('!', '👋', '🙂', '😊'))
    return not (is_short and (is_casual or has_emoji_ending or is_error))

def get_greeting_message(language="en"):
    hour = datetime.now().hour
    if 5 <= hour < 12:
        greeting = translate_text("Good morning! 🌅", language)
    elif 12 <= hour < 17:
        greeting = translate_text("Good afternoon! 🌞", language)
    else:
        greeting = translate_text("Good evening! 🌙", language)
    return greeting

def create_welcome_message(profile_name, language="en"):
    greeting = get_greeting_message(language)
    name = f"{profile_name}!" if profile_name else "User!"
    welcome_text = translate_text(
        "Welcome to AI Fact Checker! 🤖✨\n\n"
        "I'm here to help you verify information and check facts. "
        "Feel free to ask me any questions or share statements you'd like to fact-check.\n\n"
        "To get started, simply type your question or statement! 📝",
        language
    )
    return f"{greeting} {name}\n{welcome_text}"

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

def send_message_with_template(to_number, body_text, user_input, is_greeting=False, language="en"):
    """
    Sends a normal message. If not greeting and 'needs_rating', optionally sends
    a follow-up (either twilio content template if TWILIO_TEMPLATE_SID is set,
    or a simple thumbs up/down prompt).
    """
    try:
        translated_body = translate_text(body_text, language)
        main_message = client.messages.create(
            from_=TWILIO_WHATSAPP_NUMBER,
            to=to_number,
            body=translated_body
        )
        time.sleep(1)

        if not is_greeting and needs_rating(user_input):
            if TWILIO_TEMPLATE_SID:
                client.messages.create(
                    from_=TWILIO_WHATSAPP_NUMBER,
                    to=to_number,
                    body="Was this response helpful?",
                    content_sid=TWILIO_TEMPLATE_SID
                )
            else:
                client.messages.create(
                    from_=TWILIO_WHATSAPP_NUMBER,
                    to=to_number,
                    body=translate_text("Was this response helpful? Reply with 👍 for Yes or 👎 for No.", language)
                )
        return main_message
    except Exception as e:
        logger.error(f"Error sending message: {str(e)}")
        raise

def handle_button_response(user_response, chat_session, previous_lang, sender_number):
    try:
        if user_response in ["👍", "👎"]:
            feedback_type = "positive" if user_response == "👍" else "negative"
            if chat_session.last_message_id:
                store_feedback(chat_session.last_message_id, feedback_type, sender_number)
                message = client.messages.create(
                    from_=TWILIO_WHATSAPP_NUMBER,
                    to=sender_number,
                    body=translate_text("Thank you for your feedback! 🙏\nWould you like to verify another claim?", previous_lang)
                )
                return True, message.sid
        return False, None
    except Exception as e:
        logger.error(f"Error handling button response: {e}")
        return False, None

# ------------------------------------------------------------------------------
# Voice note transcription (OpenAI Whisper)
# ------------------------------------------------------------------------------
def transcribe_voice_message(audio_url, chat_session):
    try:
        # Download the audio
        import requests
        resp = requests.get(audio_url, timeout=60)
        resp.raise_for_status()
        with open("temp_audio.ogg", "wb") as f:
            f.write(resp.content)

        # Convert OGG → WAV
        audio = AudioSegment.from_file("temp_audio.ogg", format="ogg")
        audio.export("temp_audio.wav", format="wav")

        # Transcribe using Whisper (legacy v1)
        with open("temp_audio.wav", "rb") as audio_file:
            # openai.Audio.transcribe is from the legacy SDK interface
            transcript = openai.Audio.transcribe(model="whisper-1", file=audio_file, response_format="text")

        detected_language = translator.detect(transcript).lang
        chat_session.language = detected_language

        # Cleanup
        os.remove("temp_audio.ogg")
        os.remove("temp_audio.wav")

        return transcript
    except Exception as e:
        logger.error(f"Error transcribing voice message: {e}")
        return None

# ------------------------------------------------------------------------------
# Greeting detection for text
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
# Integrated fact-checking (Serper → Tavily → Groq/OpenAI)
# ------------------------------------------------------------------------------
def fact_check_reply(user_input: str) -> dict:
    """
    Returns { "message": "<reply>", "sources": [urls...] }
    """
    if not user_input:
        return {"message": "Please provide a claim to verify.", "sources": []}

    # Special-case greeting
    if is_greeting(user_input):
        return {
            "message": "Hello! 😊 I'm your fact-checking assistant. Share a factual statement you’d like me to verify, and I’ll check credible sources.",
            "sources": []
        }

    sources = []
    search_context = ""
    serper_failed = False

    # Try Serper
    try:
        serper_data = serper_tool.results(user_input)
        if isinstance(serper_data, dict) and "organic" in serper_data:
            sources += [item.get("link") for item in serper_data["organic"] if "link" in item]
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
Respond in a clear, formal, direct narrative.
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

    return {"message": text, "sources": [s for s in sources if s][:5]}

# ------------------------------------------------------------------------------
# Webhook
# ------------------------------------------------------------------------------
@app.route("/whatsapp", methods=["POST"])
def whatsapp_reply():
    try:
        sender_number = request.form.get("From")
        if not sender_number:
            return jsonify({"status": "error", "message": "Missing sender"}), 400

        chat_session = get_chat_session(sender_number)
        profile_name = request.form.get("ProfileName", "User")

        # Check for media (voice notes)
        num_media = int(request.form.get("NumMedia", 0))
        if num_media > 0:
            media_url = request.form.get("MediaUrl0")
            media_type = request.form.get("MediaContentType0", "")
            if media_type == "audio/ogg":
                client.messages.create(
                    from_=TWILIO_WHATSAPP_NUMBER,
                    to=sender_number,
                    body=translate_text("Processing your voice note... ⏳", chat_session.language)
                )
                transcribed_text = transcribe_voice_message(media_url, chat_session)
                incoming_message = transcribed_text if transcribed_text else translate_text(
                    "Sorry, I couldn't process the voice note.", chat_session.language
                )
            else:
                incoming_message = translate_text(
                    "Unsupported media type. Please send a voice note.", chat_session.language
                )
        else:
            incoming_message = (request.form.get("Body") or "").strip()

        # Previous language snapshot
        previous_lang = chat_session.language

        # Detect language from the incoming message (best effort)
        try:
            detected_language = translator.detect(incoming_message).lang
            chat_session.language = detected_language or chat_session.language
        except Exception:
            pass

        # Handle thumbs-up/down feedback
        if incoming_message in ["👍", "👎"]:
            is_feedback, message_sid = handle_button_response(incoming_message, chat_session, previous_lang, sender_number)
            if is_feedback:
                return jsonify({"status": "success", "message_sid": message_sid})

        # Welcome message for new 24h session
        if chat_session.is_new_session:
            welcome_msg = send_message_with_template(
                sender_number,
                create_welcome_message(profile_name, chat_session.language),
                incoming_message,
                is_greeting=True,
                language=chat_session.language
            )
            chat_session.conversation_history.append({
                "timestamp": datetime.now().isoformat(),
                "message": create_welcome_message(profile_name, chat_session.language),
                "type": "outgoing",
                "message_id": welcome_msg.sid
            })

        # Save incoming
        chat_session.conversation_history.append({
            "timestamp": datetime.now().isoformat(),
            "message": incoming_message,
            "type": "incoming"
        })

        # Inform user we’re processing (for non-media, substantive msgs)
        if num_media == 0 and needs_rating(incoming_message):
            client.messages.create(
                from_=TWILIO_WHATSAPP_NUMBER,
                to=sender_number,
                body=translate_text("Processing your request. ⏳", chat_session.language)
            )

        # === Core: run local fact-check ===
        fc = fact_check_reply(incoming_message)
        response_text = fc.get("message") or "I am unable to provide a response now. Please try again."

        # Optionally append sources (simple / compact)
        sources = fc.get("sources") or []
        if sources:
            # Keep short: show up to two URLs
            shown = sources[:2]
            response_text = f"{response_text}\n\nSources:\n- {shown[0]}" + (f"\n- {shown[1]}" if len(shown) > 1 else "")

        # Send reply
        message = send_message_with_template(
            sender_number, response_text, incoming_message, language=chat_session.language
        )
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
        # attempt a graceful error message in user's last language
        try:
            err_text = translate_text("An error occurred. Please try again later.", chat_session.language)
            client.messages.create(from_=TWILIO_WHATSAPP_NUMBER, to=sender_number, body=err_text)
        except Exception:
            pass
        return jsonify({"status": "error", "message": str(e)}), 500

# ------------------------------------------------------------------------------
# Entry
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.getenv("PORT", "5000")), debug=True)
