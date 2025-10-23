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
- Light/Dark mode toggle
"""

import os
import sqlite3
import time
import io
from datetime import datetime

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Optional imports
try:
    from transformers import pipeline
    EMOTION_PIPE_AVAILABLE = True
except:
    EMOTION_PIPE_AVAILABLE = False

try:
    import openai
    OPENAI_AVAILABLE = True
except:
    OPENAI_AVAILABLE = False

# ---------------------------
# Config and DB
# ---------------------------
st.set_page_config(page_title="MindEase", layout="wide", initial_sidebar_state="expanded")
DB_PATH = "mindease_data.db"

def init_db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS mood_log (
        id INTEGER PRIMARY KEY, timestamp TEXT, mood_label TEXT, score REAL, text TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS journals (
        id INTEGER PRIMARY KEY, timestamp TEXT, content TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS chats (
        id INTEGER PRIMARY KEY, timestamp TEXT, role TEXT, message TEXT)''')
    conn.commit()
    return conn

conn = init_db()
c = conn.cursor()

# ---------------------------
# Helpers
# ---------------------------
def add_mood_log(mood_label, score, text):
    ts = datetime.utcnow().isoformat()
    c.execute("INSERT INTO mood_log (timestamp,mood_label,score,text) VALUES (?,?,?,?)",
              (ts, mood_label, score, text))
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
    if not df.empty: df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df

def get_journals(limit=365):
    df = pd.read_sql_query("SELECT * FROM journals ORDER BY timestamp DESC LIMIT ?", conn, params=(limit,))
    if not df.empty: df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df

def get_chat_history(limit=200):
    df = pd.read_sql_query("SELECT * FROM chats ORDER BY id DESC LIMIT ?", conn, params=(limit,))
    if not df.empty: df = df.iloc[::-1]
    return df

# ---------------------------
# Emotion detection
# ---------------------------
EMOTION_MODEL = "j-hartmann/emotion-english-distilroberta-base"
emotion_pipe = None
if EMOTION_PIPE_AVAILABLE:
    try: emotion_pipe = pipeline("text-classification", model=EMOTION_MODEL, return_all_scores=False)
    except: emotion_pipe = None

def detect_emotion(text):
    if emotion_pipe:
        try:
            out = emotion_pipe(text[:512])[0]
            return out['label'].lower(), float(out.get('score', 0.0))
        except: pass
    txt = text.lower()
    if any(w in txt for w in ["sad","depress","hopeless","alone","cry"]): return "sadness",0.6
    if any(w in txt for w in ["anx","panic","worried","nervous","scared"]): return "anxiety",0.6
    if any(w in txt for w in ["happy","great","good","joy"]): return "joy",0.6
    if any(w in txt for w in ["angry","mad","furious","annoyed"]): return "anger",0.6
    return "neutral",0.5

# ---------------------------
# OpenAI chat helper
# ---------------------------
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", "")
if OPENAI_API_KEY: openai.api_key = OPENAI_API_KEY

def openai_reply(system_prompt, conversation_messages, model="gpt-4o-mini"):
    if not OPENAI_API_KEY: return "(No OpenAI key set)"
    messages = [{"role":"system","content":system_prompt}] + conversation_messages
    try:
        resp = openai.ChatCompletion.create(
            model=model, messages=messages, max_tokens=400, temperature=0.7, top_p=0.9)
        return resp.choices[0].message.content.strip()
    except Exception as e: return f"OpenAI error: {e}"

# ---------------------------
# Crisis detection
# ---------------------------
CRISIS_KEYWORDS = ["i want to die","i'm going to kill myself","suicide","end my life"]
def detect_crisis(text):
    t = text.lower()
    return any(k in t for k in CRISIS_KEYWORDS)

# ---------------------------
# Theme
# ---------------------------
theme_choice = st.sidebar.selectbox("App Theme", ["Light","Dark"])
def apply_theme():
    if theme_choice=="Dark":
        st.markdown("""<style>
        body, .block-container {background-color:#0E1117;color:#FFFFFF;}
        .sidebar .sidebar-content {background-color:#1C1F26;color:#FFFFFF;}
        textarea, input, select {background-color:#2C2F38!important;color:#FFFFFF!important;}
        </style>""", unsafe_allow_html=True)
    else:
        st.markdown("""<style>
        body, .block-container {background-color:#FFFFFF;color:#000000;}
        .sidebar .sidebar-content {background-color:#F5F5F5;color:#000000;}
        textarea, input, select {background-color:#FFFFFF!important;color:#000000!important;}
        </style>""", unsafe_allow_html=True)
apply_theme()

# ---------------------------
# Sidebar & Settings
# ---------------------------
st.sidebar.title("MindEase — Settings & Privacy")
st.sidebar.markdown("""
- Your data is stored locally in SQLite.
- This chatbot is supportive — not a replacement for professional care.
""")
tone = st.sidebar.selectbox("Assistant tone", ["Calm & Gentle","Cheerful & Encouraging","Formal & Respectful"])
remember = st.sidebar.checkbox("Enable context memory", value=True)

st.sidebar.markdown("---")
st.sidebar.markdown("**Export / Manage Data**")
if st.sidebar.button("Export data (CSV)"):
    output = io.StringIO()
    for df, name in zip([get_mood_history(10000), get_journals(10000), get_chat_history(10000)],
                        ["MOOD LOG","JOURNALS","CHATS"]):
        output.write(f"=== {name} ===\n")
        if not df.empty: df.to_csv(output, index=False)
    st.sidebar.download_button("Download data (.txt)", data=output.getvalue().encode('utf-8'), file_name="mindease_export.txt")

if st.sidebar.button("Clear all local data"):
    c.execute("DELETE FROM mood_log"); c.execute("DELETE FROM journals"); c.execute("DELETE FROM chats")
    conn.commit(); st.sidebar.success("All local data cleared.")

# ---------------------------
# Main Tabs
# ---------------------------
tab = st.selectbox("Select Feature", ["Home","Chat","Breathing","CBT Tools","Journal & Mood","Analytics","Crisis & Helplines","Settings"])

# ---------------------------
# Home
# ---------------------------
if tab=="Home":
    st.header("MindEase — Your AI Mental Wellness Companion")
    st.write("Features: empathetic chat, mood tracking, CBT mini-tools, journaling, crisis detection, analytics.")
    st.write("**Not a substitute for professional care.**")
    st.markdown("---")
    st.subheader("Quick Start")
    st.write("- Chat with MindEase\n- Journal & Mood\n- CBT Tools\n- Breathing exercises")

# ---------------------------
# Chat
# ---------------------------
elif tab=="Chat":
    st.subheader("Chat with MindEase")
    hist_df = get_chat_history(200)
    if not hist_df.empty:
        for _, r in hist_df.iterrows():
            st.markdown(f"**{'You' if r['role']=='user' else 'MindEase'}:** {r['message']}")
    user_input = st.text_area("Write to MindEase", height=120)
    if st.button("Send"):
        if user_input.strip():
            add_chat("user", user_input)
            if detect_crisis(user_input):
                resp = "If you are thinking of harming yourself, please contact local emergency services now."
                add_chat("assistant", resp); st.markdown(f"**MindEase:** {resp}")
            else:
                label, score = detect_emotion(user_input)
                add_mood_log(label, score, user_input)
                tone_map = {
                    "Calm & Gentle":"You are calm, empathetic, and supportive.",
                    "Cheerful & Encouraging":"You are friendly and motivating.",
                    "Formal & Respectful":"You are professional and concise."
                }
                system_prompt = tone_map.get(tone)
                conversation = []
                if remember:
                    past = get_chat_history(20)
                    if not past.empty:
                        for _, r in past.tail(6).iterrows():
                            conversation.append({"role":r['role'], "content":r['message']})
                conversation.append({"role":"user","content":f"[emotion:{label}] {user_input}"})
                reply = openai_reply(system_prompt, conversation)
                add_chat("assistant", reply)
                st.markdown(f"**MindEase:** {reply}")

# ---------------------------
# Breathing
# ---------------------------
elif tab=="Breathing":
    st.subheader("Guided Breathing (Box breathing)")
    length = st.selectbox("Duration (seconds)", [30,60,90])
    if st.button("Start"):
        placeholder = st.empty()
        step=4; cycles=max(1,length//(step*3))
        for i in range(cycles):
            placeholder.markdown(f"**Inhale** {step}s"); time.sleep(step)
            placeholder.markdown(f"**Hold** {step}s"); time.sleep(step)
            placeholder.markdown(f"**Exhale** {step}s"); time.sleep(step)
        placeholder.markdown("**Done.**")

# ---------------------------
# CBT Tools
# ---------------------------
elif tab=="CBT Tools":
    st.subheader("CBT Mini-Tools — Cognitive Restructuring")
    situation = st.text_area("Recent upsetting situation", height=80)
    auto_thought = st.text_input("Automatic thought")
    emotion_name = st.text_input("Emotion felt")
    intensity = st.slider("Intensity", 0,100,60)
    pro = st.text_area("Evidence for thought", height=80)
    contra = st.text_area("Evidence against thought", height=80)
    if st.button("Generate balanced thought"):
        if auto_thought:
            alt=f"Although I feel {emotion_name} and think '{auto_thought}', evidence against it includes: {contra or 'none'}. A balanced view: 'I may struggle but can take small steps.'"
            st.success(alt)
        else: st.warning("Enter automatic thought.")

# ---------------------------
# Journal & Mood
# ---------------------------
elif tab=="Journal & Mood":
    st.subheader("Journal & Mood Tracker")
    journal_text = st.text_area("Write a journal entry", height=140)
    if st.button("Save journal entry") and journal_text.strip():
        add_journal_entry(journal_text.strip()); st.success("Journal saved.")
    mood_choice = st.selectbox("Mood now", ["joy","calm","neutral","anxiety","sadness","anger","other"], index=2)
    mood_note = st.text_input("Optional note")
    if st.button("Log mood"):
        add_mood_log(mood_choice, 0.7, mood_note or ""); st.success(f"Mood '{mood_choice}' logged.")
    st.markdown("---")
    journals=get_journals(20)
    if not journals.empty:
        for _, r in journals.iterrows():
            st.markdown(f"**{r['timestamp'].strftime('%Y-%m-%d %H:%M')}** — {r['content']}")
    else: st.write("No entries yet.")

# ---------------------------
# Analytics
# ---------------------------
elif tab=="Analytics":
    st.subheader("Analytics — Mood Trends")
    mh=get_mood_history(1000)
    if mh.empty: st.write("No mood logs yet.")
    else:
        counts = mh['mood_label'].value_counts()
        st.table(counts.reset_index().rename(columns={'index':'mood','mood_label':'count'}))
        mapping={'joy':1.0,'calm':0.8,'neutral':0.5,'anxiety':0.2,'sadness':0.1,'anger':0.15}
        mh['num_score']=mh['mood_label'].map(mapping).fillna(mh['score'])
        mh=mh.sort_values('timestamp')
        plt.figure(figsize=(8,3)); plt.plot(mh['timestamp'], mh['num_score'])
        plt.title("Mood trend"); plt.xlabel("Date"); plt.ylabel("Score")
        st.pyplot(plt.gcf())

# ---------------------------
# Crisis & Helplines
# ---------------------------
elif tab=="Crisis & Helplines":
    st.subheader("Crisis & Helplines")
    st.warning("If in immediate danger, call local emergency services.")
    st.write("- India: AASRA — 9152987821")
    st.write("- International: 988 (US)")
    st.write("- https://findahelpline.com")

# ---------------------------
# Settings
# ---------------------------
elif tab=="Settings":
    st.subheader("Settings & Info")
    st.write("- Data stored locally in `mindease_data.db`")
    st.write("- Secure OpenAI integration via Streamlit Secrets")
    st.write("- Light/Dark mode available")
