from flask import Flask, request, jsonify, session
from twilio.twiml.messaging_response import MessagingResponse
from twilio.rest import Client
import requests
import os
import json
from dotenv import load_dotenv
from datetime import datetime, timedelta
import redis
from urllib.parse import urlparse
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv("FLASK_SECRET_KEY")

url = urlparse(os.environ.get("REDIS_URL"))
redis_client = redis.Redis(
    host=url.hostname,
    port=url.port,
    password=url.password,
    ssl=(url.scheme == "rediss"),
    ssl_cert_reqs=None
)

EXTERNAL_API_URL = os.getenv("EXTERNAL_API")
TWILIO_WHATSAPP_NUMBER = os.getenv("TWILIO_WHATSAPP_NUMBER")
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")

client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

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
        session.conversation_history = data["conversation_history"]
        session.last_message_id = data.get("last_message_id")
        session.is_new_session = False
        return session

def needs_rating(response_text):
    # Responses that don't need rating
    casual_patterns = [
        "thank you", "thanks", "you're welcome", "noted",
        "got it", "understood", "👍", "🙏", "nice","bravo","amazing","impressive",
        "sorry", "please", "hi", "hello", "hey", "good morning", "good afternoon", 
        "good evening", "thanks", "thank you", "bye", "goodbye", "cool","yeah","yah","alright",
        "oh","oops","ok"
    ]
    
    text = response_text.lower().strip()
    
    # Check conditions
    is_short = len(text.split()) < 10
    is_casual = any(pattern in text for pattern in casual_patterns)
    is_error = "error" in text or "an error occurred" in text
    has_emoji_ending = text.endswith(('!', '👋', '🙂', '😊'))
    
    return not (is_short and (is_casual or has_emoji_ending or is_error))

def get_chat_session(sender_number):
    session_key = f"chat_session:{sender_number}"
    try:
        session_data = redis_client.get(session_key)
        if session_data:
            session_dict = json.loads(session_data.decode('utf-8'))
            session = ChatSession.from_dict(session_dict)
            last_activity = datetime.fromisoformat(session_dict["last_activity"])
            if datetime.now() - last_activity > timedelta(hours=24):
                session = ChatSession(sender_number)
        else:
            session = ChatSession(sender_number)
        return session
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
        "Feel free to ask me any questions or share statements you'd like to fact-check.\n\n"
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
    try:
        main_message = client.messages.create(
            from_=TWILIO_WHATSAPP_NUMBER,
            to=to_number,
            body=body_text
        )
        time.sleep(1)
        if not is_greeting and needs_rating(user_input):
            template_message = client.messages.create(
                from_=TWILIO_WHATSAPP_NUMBER,
                to=to_number,
                body="Was this response helpful?",
                content_sid=os.getenv("TWILIO_TEMPLATE_SID")
            )
            return template_message
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
                    body="Thank you for your feedback! 🙏.\n Would you like to verify another claim?"
                )
                return True, message.sid
        return False, None
    except Exception as e:
        logger.error(f"Error handling button response: {e}")
        return False, None

def call_external_api(user_query, chat_session):
    try:
        payload = {"user_input": user_query}
        response = requests.post(EXTERNAL_API_URL, json=payload, timeout=600)
        response.raise_for_status()
        data = response.json()
        return {"message": data.get("result", "Unexpected API response format.")}
    except Exception as e:
        logger.error(f"Error calling external API: {e}")
        return {"message": f"An error occurred: {e}", "status": "error"}

@app.route("/whatsapp", methods=["POST"])
def whatsapp_reply():
    try:
        incoming_message = request.form.get("Body", "").strip()
        sender_number = request.form.get("From")
        
        chat_session = get_chat_session(sender_number)
        
        button_text = request.form.get("ButtonText")
        if button_text:
            is_feedback, message_sid = handle_button_response(button_text, chat_session, sender_number)
            if is_feedback:
                return jsonify({"status": "success", "message_sid": message_sid})
        
        if chat_session.is_new_session:
            welcome_message = send_message_with_template(
                sender_number,
                create_welcome_message(),
                incoming_message,
                is_greeting=True
            )
            chat_session.conversation_history.append({
                "timestamp": datetime.now().isoformat(),
                "message": create_welcome_message(),
                "type": "outgoing",
                "message_id": welcome_message.sid
            })
        
        chat_session.conversation_history.append({
            "timestamp": datetime.now().isoformat(),
            "message": incoming_message,
            "type": "incoming"
        })
        
        api_response = call_external_api(incoming_message, chat_session)
        response_text = api_response.get("message", "I am unable to provide response now, please try your query again.")

        print(response_text)
        
        message = send_message_with_template(sender_number, response_text,incoming_message)
        chat_session.last_message_id = message.sid
        
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

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
