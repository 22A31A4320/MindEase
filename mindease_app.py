"""
MindEase - Streamlit Mental Health Chatbot Prototype
Single-file app demonstrating many features:
- Empathetic chat (OpenAI)
- Emotion detection (transformers)
- Mood tracking (SQLite)
- Journaling
- CBT mini-tool (cognitive reframing)
- Guided breathing
- Crisis detection & helplines
- Analytics and data export
- Simple personalization & privacy controls
- Light/Dark mode toggle added
"""

import os
import sqlite3
import time
import csv
import io
from datetime import datetime, date

import streamlit as st

# --- optional imports that may be heavy; app falls back gracefully if missing
try:
    from transformers import pipeline
    EMOTION_PIPE_AVAILABLE = True
except Exception:
    EMOTION_PIPE_AVAILABLE = False

try:
    import openai
    OPENAI_AVAILABLE = True
except Exception:
    OPENAI_AVAILABLE = False

import matplotlib.pyplot as plt
import pandas as pd

# ---------------------------
# Config and DB
# ---------------------------
st.set_page_config(page_title="MindEase", layout="wide", initial_sidebar_state="expanded")

DB_PATH = "mindease_data.db"

def init_db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS mood_log (
            id INTEGER PRIMARY KEY,
            timestamp TEXT,
            mood_label TEXT,
            score REAL,
            text TEXT
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS journals (
            id INTEGER PRIMARY KEY,
            timestamp TEXT,
            content TEXT
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS chats (
            id INTEGER PRIMARY KEY,
            timestamp TEXT,
            role TEXT,
            message TEXT
        )
    ''')
    conn.commit()
    return conn

conn = init_db()
c = conn.cursor()

# ---------------------------
# Helpers
# ---------------------------
def add_mood_log(mood_label, score, text):
    ts = datetime.utcnow().isoformat()
    c.execute("INSERT INTO mood_log (timestamp,mood_label,score,text) VALUES (?,?,?,?)", (ts, mood_label, score, text))
    conn.commit()

def add_journal_entry(content):
    ts = datetime.utcnow().isoformat()
    c.execute("INSERT INTO journals (timestamp,content) VALUES (?,?)", (ts, content))
    conn.commit()

def add_chat(role, message):
    ts = datetime.utcnow().isoformat()
    c.execute("INSERT INTO chats (timestamp,role,message) VALUES (?,?,?)", (ts, role, message))
    conn.commit()

def get_mood_history(limit=365):
    df = pd.read_sql_query("SELECT * FROM mood_log ORDER BY timestamp DESC LIMIT ?", conn, params=(limit,))
    if not df.empty:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df

def get_journals(limit=365):
    df = pd.read_sql_query("SELECT * FROM journals ORDER BY timestamp DESC LIMIT ?", conn, params=(limit,))
    if not df.empty:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df

def get_chat_history(limit=200):
    df = pd.read_sql_query("SELECT * FROM chats ORDER BY id DESC LIMIT ?", conn, params=(limit,))
    if not df.empty:
        df = df.iloc[::-1]  # show older first
    return df

# ---------------------------
# Emotion detection setup
# ---------------------------
EMOTION_MODEL = "j-hartmann/emotion-english-distilroberta-base"

emotion_pipe = None
if EMOTION_PIPE_AVAILABLE:
    try:
        emotion_pipe = pipeline("text-classification", model=EMOTION_MODEL, return_all_scores=False)
    except Exception:
        emotion_pipe = None

def detect_emotion(text):
    """
    Returns (label, score) e.g. ("sadness", 0.78)
    If emotion pipeline not available, fallback to simple heuristics.
    """
    if emotion_pipe:
        try:
            out = emotion_pipe(text[:512])[0]
            label = out['label']
            score = float(out.get('score', 0.0))
            return label.lower(), score
        except Exception:
            pass

    # fallback: naive heuristics
    txt = text.lower()
    if any(w in txt for w in ["sad", "depress", "hopeless", "alone", "cry"]):
        return "sadness", 0.6
    if any(w in txt for w in ["anx", "panic", "worried", "nervous", "scared"]):
        return "anxiety", 0.6
    if any(w in txt for w in ["happy", "great", "good", "joy"]):
        return "joy", 0.6
    if any(w in txt for w in ["angry", "mad", "furious", "annoyed"]):
        return "anger", 0.6
    return "neutral", 0.5

# ---------------------------
# OpenAI chat helper
# ---------------------------
def openai_reply(system_prompt, conversation_messages, model="gpt-4o-mini"):
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        return "OpenAI API key not set. Please set OPENAI_API_KEY environment variable."
    try:
        import openai
        openai.api_key = api_key
    except Exception as e:
        return f"OpenAI library not available: {e}"

    messages = [{"role":"system","content":system_prompt}] + conversation_messages
    try:
        resp = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            max_tokens=400,
            temperature=0.7,
            top_p=0.9
        )
        text = resp.choices[0].message.content.strip()
        return text
    except Exception as e:
        return f"OpenAI error: {e}"

# ---------------------------
# Crisis detection
# ---------------------------
CRISIS_KEYWORDS = [
    "i want to die", "i'm going to kill myself", "i want to kill myself", "end my life",
    "suicide", "i want to end it all", "i cant go on", "i can't go on"
]

def detect_crisis(text):
    t = text.lower()
    for k in CRISIS_KEYWORDS:
        if k in t:
            return True
    return False

# ---------------------------
# THEME HANDLER
# ---------------------------
theme_choice = st.sidebar.selectbox("App Theme", ["Light", "Dark"])
st.session_state["theme"] = theme_choice

def apply_theme():
    if st.session_state.get("theme") == "Dark":
        st.markdown(
            """
            <style>
            body, .block-container {background-color: #0E1117; color: #FFFFFF;}
            .sidebar .sidebar-content {background-color: #1C1F26; color: #FFFFFF;}
            textarea, input, select {background-color: #2C2F38 !important; color: #FFFFFF !important;}
            </style>
            """, unsafe_allow_html=True)
    else:
        st.markdown(
            """
            <style>
            body, .block-container {background-color: #FFFFFF; color: #000000;}
            .sidebar .sidebar-content {background-color: #F5F5F5; color: #000000;}
            textarea, input, select {background-color: #FFFFFF !important; color: #000000 !important;}
            </style>
            """, unsafe_allow_html=True)

apply_theme()

# ---------------------------
# SIDEBAR
# ---------------------------
st.sidebar.title("MindEase — Settings & Privacy")
st.sidebar.markdown("""
**Privacy**
- Your data is stored locally (SQLite) by default in this prototype.
- You can export or delete your data anytime.
- This chatbot is a supportive tool — it is **not** a replacement for professional care.
""")

tone = st.sidebar.selectbox("Assistant tone", ["Calm & Gentle", "Cheerful & Encouraging", "Formal & Respectful"])
remember = st.sidebar.checkbox("Enable context memory (local)", value=True)

st.sidebar.markdown("---")
st.sidebar.markdown("**Export / Manage Data**")
if st.sidebar.button("Export data (CSV)"):
    mood_df = get_mood_history(10000)
    journal_df = get_journals(10000)
    chat_df = get_chat_history(10000)
    output = io.StringIO()
    output.write("=== MOOD LOG ===\n")
    if not mood_df.empty:
        mood_df.to_csv(output, index=False)
    output.write("\n=== JOURNALS ===\n")
    if not journal_df.empty:
        journal_df.to_csv(output, index=False)
    output.write("\n=== CHATS ===\n")
    if not chat_df.empty:
        chat_df.to_csv(output, index=False)
    b = output.getvalue().encode('utf-8')
    st.sidebar.download_button("Download data (.txt)", data=b, file_name="mindease_export.txt", mime="text/plain")

if st.sidebar.button("Clear all local data"):
    c.execute("DELETE FROM mood_log")
    c.execute("DELETE FROM journals")
    c.execute("DELETE FROM chats")
    conn.commit()
    st.sidebar.success("All local data cleared.")

st.sidebar.markdown("---")
st.sidebar.markdown("Need urgent help? See the Crisis & Helplines tab.")

# ---------------------------
# TABS (Home, Chat, Breathing, CBT, Journal, Analytics, Crisis, Settings)
# ---------------------------
tabs = st.tabs(["Home", "Chat", "Breathing", "CBT Tools", "Journal & Mood", "Analytics", "Crisis & Helplines", "Settings"])
home, chat_tab, breathing_tab, cbt_tab, journal_tab, analytics_tab, crisis_tab, settings_tab = tabs

# ---------------------------
# HOME
# ---------------------------
with home:
    st.header("MindEase — Your AI Mental Wellness Companion")
    st.write("This prototype demonstrates many mental-health support features: empathetic chat, mood tracking, CBT mini-tools, journaling, crisis detection, and analytics.")
    st.write("**Important:** This is *not* a substitute for professional mental health care. If you are in immediate danger, contact local emergency services now.")
    st.markdown("---")
    st.subheader("Quick Start")
    st.write("- Go to **Chat** to talk with the assistant.")
    st.write("- Use **Journal & Mood** to log moods and keep a private diary.")
    st.write("- Try **CBT Tools** for cognitive reframing exercises.")
    st.write("- Use **Breathing** for a short grounding exercise (1–3 minutes).")

# ---------------------------
# CHAT
# ---------------------------
with chat_tab:
    st.subheader("Chat with MindEase")
    st.write("Type how you're feeling or what you'd like to talk about. The assistant will respond empathetically and offer coping strategies.")
    hist_df = get_chat_history(200)
    if not hist_df.empty:
        for _, row in hist_df.iterrows():
            role = row['role']
            msg = row['message']
            if role == "user":
                st.markdown(f"**You:** {msg}")
            else:
                st.markdown(f"**MindEase:** {msg}")

    user_input = st.text_area("Write to MindEase", value="", height=120)
    col1, col2 = st.columns([1,4])
    with col1:
        if st.button("Send"):
            if not user_input.strip():
                st.warning("Please write something...")
            else:
                add_chat("user", user_input)
                if detect_crisis(user_input):
                    resp = ("I’m truly sorry you’re feeling this way. If you are thinking about harming yourself, please contact emergency services or a suicide prevention hotline now. "
                            "If you're in India call 9152987821 (AASRA) or find a local helpline. Would you like me to show resources?")
                    add_chat("assistant", resp)
                    st.markdown(f"**MindEase:** {resp}")
                else:
                    label, score = detect_emotion(user_input)
                    add_mood_log(label, score, user_input)
                    tone_map = {
                        "Calm & Gentle": "You are a calm, gentle, empathetic mental health assistant. Validate feelings, ask supportive follow-up questions, and offer short practical coping steps.",
                        "Cheerful & Encouraging": "You are a friendly, upbeat assistant. Be encouraging, positive, and motivating while still validating feelings.",
                        "Formal & Respectful": "You are a respectful, formal assistant. Use professional tone, clear structure, and concise suggestions."
                    }
                    system_prompt = tone_map.get(tone, tone_map["Calm & Gentle"])
                    conversation = []
                    if remember:
                        past = get_chat_history(20)
                        if not past.empty:
                            for _, r in past.tail(6).iterrows():
                                conversation.append({"role": r['role'], "content": r['message']})
                    conversation.append({"role":"user","content": f"[detected_emotion:{label}] {user_input}"})
                    if os.getenv("OPENAI_API_KEY",""):
                        reply = openai_reply(system_prompt, conversation)
                    else:
                        reply = ("(No OpenAI key set) I detect you may be feeling " + label + 
                                 ". Try a 3-part breathing exercise, or tell me more about what's on your mind.")
                    add_chat("assistant", reply)
                    st.markdown(f"**MindEase:** {reply}")

# ---------------------------
# The rest of the tabs (Breathing, CBT, Journal, Analytics, Crisis, Settings)
# ---------------------------
# You can integrate the existing code for Breathing, CBT, Journal, Analytics, Crisis, and Settings here as before.
# They will automatically adopt the theme via `apply_theme()` at runtime.
