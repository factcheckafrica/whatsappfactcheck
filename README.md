# Fact Check WhatsApp

## 📌 Overview

**Fact Check WhatsApp** is a Flask-based chatbot that integrates directly with **WhatsApp via Twilio**.  
It allows users to submit text (and optionally voice messages) and receive **fact-checked responses** powered by **LLMs (Groq & OpenAI)** and **real-time search tools** (Google Serper, Tavily).  

The bot manages **user sessions** with Redis, supports feedback collection, and can handle greetings, ratings, and conversational history.

---

## ✨ Features

- **Real-Time Fact-Checking**  
  Uses Google Serper (default) and Tavily (fallback) for web search, and processes results with **Groq (LLaMA 3.1 8B)** or **OpenAI GPT-4o-mini**.

- **Multi-Language & Greetings**  
  Detects greetings in English, Hausa, and other languages for more natural interactions.

- **Session Management**  
  Tracks conversations with Redis, resets automatically after 24 hours.

- **Feedback Mechanism**  
  Supports interactive buttons (“Pleased” / “Not Pleased”) to rate responses.  
  Feedback is stored in Redis for 30 days.

- **WhatsApp Integration via Twilio**  
  Handles both incoming text and interactive button responses.  
  Optionally sends a **rating template** if `TWILIO_TEMPLATE_SID` is set.

---

## 📦 Requirements

- Python 3.9+
- Flask
- Redis
- Twilio
- LangChain & integrations:
  - `langchain-community`
  - `langchain-openai`
  - `langchain-groq`
  - `langchain-tavily`

---

## ⚙️ Environment Variables

Create a `.env` file in the root directory and add the following:

```env
# Flask
FLASK_SECRET_KEY=your_flask_secret_key

# Redis
REDIS_URL=your_redis_url

# Twilio
TWILIO_WHATSAPP_NUMBER=whatsapp:+1415XXXXXXX
TWILIO_ACCOUNT_SID=your_twilio_account_sid
TWILIO_AUTH_TOKEN=your_twilio_auth_token
TWILIO_TEMPLATE_SID=your_twilio_template_sid   # optional (for rating template)

# APIs
OPENAI_API_KEY=your_openai_api_key
GROQ_API_KEY=your_groq_api_key
SERPER_API_KEY=your_serper_api_key
TAVILY_API_KEY=your_tavily_api_key
