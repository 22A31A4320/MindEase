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
- Light/Dark mode toggle added
- Streamlit-native charts (matplotlib removed)
"""

import os
import sqlite3
import time
import io
from datetime import datetime

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
    if emotion_pipe:
        try:
            out = emotion_pipe(text[:512])[0]
            label = out['label']
            score = float(out.get('score', 0.0))
            return label.lower(), score
        except Exception:
            pass
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
# TABS
# ---------------------------
tab_selection = st.sidebar.radio(
    "Select Feature",
    ["Home", "Chat", "Breathing", "CBT Tools", "Journal & Mood", "Analytics", "Crisis & Helplines"]
)

# ---------------------------
# HOME
# ---------------------------
if tab_selection == "Home":
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
elif tab_selection == "Chat":
    st.subheader("Chat with MindEase")
    st.write("Type how you're feeling or what you'd like to talk about. The assistant will respond empathetically and offer coping strategies.")
    hist_df = get_chat_history(200)
    if not hist_df.empty:
        for _, row in hist_df.iterrows():
            role = row['role']
            msg = row['message']
            st.markdown(f"**{'You' if role=='user' else 'MindEase'}:** {msg}")

    user_input = st.text_area("Write to MindEase", value="", height=120)
    if st.button("Send"):
        if user_input.strip():
            add_chat("user", user_input)
            if detect_crisis(user_input):
                resp = ("I’m truly sorry you’re feeling this way. If you are thinking about harming yourself, please contact emergency services or a suicide prevention hotline now. "
                        "If you're in India call 9152987821 (AASRA) or find a local helpline.")
                add_chat("assistant", resp)
                st.markdown(f"**MindEase:** {resp}")
            else:
                label, score = detect_emotion(user_input)
                add_mood_log(label, score, user_input)
                tone_map = {
                    "Calm & Gentle": "You are a calm, gentle, empathetic mental health assistant.",
                    "Cheerful & Encouraging": "You are a friendly, upbeat assistant.",
                    "Formal & Respectful": "You are a respectful, formal assistant."
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
                    reply = f"(No OpenAI key set) I detect you may be feeling {label}. Try a 3-part breathing exercise."
                add_chat("assistant", reply)
                st.markdown(f"**MindEase:** {reply}")

# ---------------------------
# BREATHING
# ---------------------------
elif tab_selection == "Breathing":
    st.subheader("Guided Breathing (Box breathing / 4-4-4)")
    st.write("Follow the prompts. Try to breathe slowly and focus on the rhythm.")
    length = st.selectbox("Duration", [30, 60, 90])
    if st.button("Start breathing exercise"):
        placeholder = st.empty()
        step_length = 4
        cycles = max(1, length // (step_length*3))
        for i in range(cycles):
            placeholder.markdown(f"**Inhale** for {step_length} seconds ...")
            time.sleep(step_length)
            placeholder.markdown(f"**Hold** for {step_length} seconds ...")
            time.sleep(step_length)
            placeholder.markdown(f"**Exhale** for {step_length} seconds ...")
            time.sleep(step_length)
        placeholder.markdown("**Done.** How do you feel now?")

# ---------------------------
# CBT TOOLS
# ---------------------------
elif tab_selection == "CBT Tools":
    st.subheader("CBT Mini-Tools — Cognitive Restructuring")
    situation = st.text_area("Describe a recent upsetting situation", height=80)
    auto_thought = st.text_input("Automatic thought (e.g., 'I will fail at my job')")
    emotion_name = st.text_input("Emotion (e.g., anxiety, shame)")
    intensity = st.slider("Intensity", 0, 100, 60)
    pro = st.text_area("Evidence supporting the thought", height=80)
    contra = st.text_area("Evidence against the thought", height=80)
    if st.button("Generate balanced alternative thought"):
        if auto_thought.strip():
            alt = f"Although I feel {emotion_name} and think '{auto_thought}', the evidence against it includes: {contra or 'none listed'}. A more balanced view might be: 'I may struggle sometimes, but I have handled challenges before and can take small steps to improve.'"
            st.success("Balanced thought:")
            st.write(alt)
        else:
            st.warning("Please enter the automatic thought.")

# ---------------------------
# JOURNAL & MOOD
# ---------------------------
elif tab_selection == "Journal & Mood":
    st.subheader("Private Journal & Mood Tracker")
    journal_text = st.text_area("Write a journal entry (private)", height=140)
    if st.button("Save journal entry"):
        if journal_text.strip():
            add_journal_entry(journal_text.strip())
            st.success("Journal saved.")
        else:
            st.warning("Write something to save.")

    st.markdown("---")
    mood_options = ["joy", "calm", "neutral", "anxiety", "sadness", "anger", "other"]
    mood_choice = st.selectbox("How are you feeling now?", mood_options, index=2)
    mood_note = st.text_input("Optional short note")
    if st.button("Log mood"):
        score = 0.7
        add_mood_log(mood_choice, score, mood_note or "")
        st.success(f"Mood '{mood_choice}' logged.")

    st.markdown("---")
    st.write("Recent journal entries")
    journals = get_journals(20)
    if not journals.empty:
        for _, r in journals.iterrows():
            st.markdown(f"**{r['timestamp'].strftime('%Y-%m-%d %H:%M')}** — {r['content']}")
    else:
        st.write("No entries yet.")

# ---------------------------
# ANALYTICS
# ---------------------------
elif tab_selection == "Analytics":
    st.subheader("Analytics — Your Mood Trends (Local)")
    mh = get_mood_history(1000)
    if not mh.empty:
        counts = mh['mood_label'].value_counts()
        st.bar_chart(counts)
    else:
        st.write("No mood logs yet. Use Journal & Mood or Chat to create mood entries.")

# ---------------------------
# CRISIS & HELPLINES
# ---------------------------
elif tab_selection == "Crisis & Helplines":
    st.subheader("Crisis & Helplines")
    st.warning("If you are in immediate danger, call emergency services now.")
    st.write("- International: 988 (US) for suicide & crisis lifeline.")
    st.write("- India: AASRA — 9152987821 (emotional support).")
    st.write("- International database: https://findahelpline.com (search by country).")
