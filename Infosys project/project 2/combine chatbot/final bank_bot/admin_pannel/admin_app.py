import streamlit as st
import pandas as pd
import os, json, csv
from datetime import datetime

# ------------------ DATASET APPEND FUNCTIONS ------------------
def append_to_bank_dataset(text, intent, response="", entities=""):
    csv_path = os.path.abspath("bank_chatbot_dataset.csv")
    samples = [line.strip() for line in str(text).splitlines() if line and line.strip()]
    if not samples:
        return
    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["text", "intent", "response", "entities"])
        for sample in samples:
            writer.writerow([sample, intent, response or "", entities or ""])

def append_to_faq_dataset(question, answer, data_dir):
    csv_path = os.path.join(data_dir, "faq_dataset.csv")
    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["question", "answer"])
        writer.writerow([question, answer])

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="Admin Panel",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------ GLOBAL CSS ------------------
st.markdown("""
<style>
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(8px); }
    to { opacity: 1; transform: translateY(0); }
}
.main > div { animation: fadeIn 0.6s ease-in-out; }

.stApp {
    background: linear-gradient(135deg, #f4f8ff, #eef3fb);
}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0a2a66, #061a3a);
}
section[data-testid="stSidebar"] * {
    color: white !important;
}

section[data-testid="stSidebar"] .stRadio > label {
    background: rgba(255, 255, 255, 0.1);
    padding: 10px 15px;
    border-radius: 8px;
    margin: 5px 0;
    transition: all 0.3s;
}

section[data-testid="stSidebar"] .stRadio > label:hover {
    background: rgba(255, 255, 255, 0.2);
    transform: translateX(5px);
}

.metric-card {
    background: white;
    padding: 20px;
    border-radius: 14px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.08);
    margin-bottom: 12px;
    transition: all 0.3s;
}

.metric-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 15px 40px rgba(0,0,0,0.12);
}

.metric-title {
    font-size: 14px;
    color: #6c757d;
}

.metric-value {
    font-size: 28px;
    font-weight: bold;
    color: #0a2a66;
}

.stButton>button {
    background: linear-gradient(135deg, #0a2a66, #154c9c);
    color: white;
    border-radius: 10px;
    padding: 10px 20px;
    font-weight: 600;
    border: none;
    transition: all 0.3s;
}

.stButton>button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 20px rgba(10, 42, 102, 0.3);
}

h1, h2, h3 {
    color: #0a2a66;
}

.stForm {
    background: white;
    padding: 25px;
    border-radius: 15px;
    box-shadow: 0 8px 25px rgba(0,0,0,0.06);
}

.stTextInput>div>div>input, .stTextArea>div>div>textarea {
    border-radius: 8px;
    border: 2px solid #e2e8f0;
    transition: all 0.3s;
}

.stTextInput>div>div>input:focus, .stTextArea>div>div>textarea:focus {
    border-color: #0a2a66;
    box-shadow: 0 0 0 3px rgba(10, 42, 102, 0.1);
}

div[data-testid="stDataFrame"] {
    border-radius: 12px;
    overflow: hidden;
    box-shadow: 0 8px 25px rgba(0,0,0,0.06);
}
</style>
""", unsafe_allow_html=True)

# ------------------ PATHS ------------------
BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

training_path = os.path.join(DATA_DIR, "training.json")
faq_path = os.path.join(DATA_DIR, "faq.json")
queries_path = os.path.join(DATA_DIR, "user_queries.csv")
faq_csv_path = os.path.join(DATA_DIR, "faq_dataset.csv")

# ------------------ AUTO LOGIN ------------------
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = True
    st.session_state["user"] = "admin"

# ------------------ LOAD DATA ------------------
def load_training():
    if os.path.exists(training_path):
        with open(training_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if "intents" not in data:
            data["intents"] = []
    else:
        data = {"intents": []}
    return data

def load_faq():
    if os.path.exists(faq_path):
        with open(faq_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = []
    return data

training = load_training()
faq = load_faq()

df_queries = pd.read_csv(queries_path) if os.path.exists(queries_path) else pd.DataFrame(
    columns=["query", "intents", "confidence", "date"]
)
faq_df = pd.read_csv(faq_csv_path) if os.path.exists(faq_csv_path) else pd.DataFrame(
    columns=["question", "answer"]
)

# ------------------ SIDEBAR ------------------
st.sidebar.title("🏦 Admin Panel")
page = st.sidebar.radio(
    "Navigation",
    ["Dashboard", "Training Data", "User Queries", "FAQs", "Analytics", "Settings", "Logout"]
)

# ------------------ DASHBOARD ------------------
if page == 'Dashboard':
    st.title("🏦 Admin Dashboard")
    
    cols = st.columns([1,1,1,1])
    with cols[0]:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-title">Total Queries</div>
            <div class="metric-value">📊 {}</div>
        </div>
        """.format(len(df_queries)), unsafe_allow_html=True)
    
    with cols[1]:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-title">Success Rate</div>
            <div class="metric-value">✅ 94.2%</div>
        </div>
        """, unsafe_allow_html=True)
    
    with cols[2]:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-title">Intents</div>
            <div class="metric-value">🎯 {}</div>
        </div>
        """.format(len(training.get('intents',[]))), unsafe_allow_html=True)
    
    with cols[3]:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-title">Entity Types</div>
            <div class="metric-value">🏷️ 42</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('### 📈 Recent Queries')
    if df_queries.empty:
        st.info('📭 No queries logged yet.')
    else:
        st.dataframe(df_queries.tail(20), use_container_width=True)

# ------------------ TRAINING DATA ------------------
elif page == "Training Data":
    st.header("📝 Training Data Editor")

    with st.form("add_intent_form", clear_on_submit=True):
        new_intent = st.text_input("New intent name")
        examples_text = st.text_area("Example phrases (comma-separated)", placeholder="what is my balance, show my balance")
        response_text = st.text_area("Bot response", placeholder="Please provide your account number")
        submit = st.form_submit_button("Add Intent")

        if submit:
            if not new_intent or not examples_text or not response_text:
                st.error("Intent, examples, and response are required")
            else:
                examples = [e.strip() for e in examples_text.split(",") if e.strip()]
                duplicate = any(i["intent"] == new_intent for i in training["intents"])
                if duplicate:
                    st.warning("Intent already exists. Updating examples and response...")
                    for i in training["intents"]:
                        if i["intent"] == new_intent:
                            i["examples"].extend([e for e in examples if e not in i["examples"]])
                            i["response"] = response_text
                else:
                    training["intents"].append({
                        "intent": new_intent.strip(),
                        "examples": examples,
                        "response": response_text.strip()
                    })

                # Save JSON
                with open(training_path, "w", encoding="utf-8") as f:
                    json.dump(training, f, indent=2)

                # Append examples to CSV
                for ex in examples:
                    append_to_bank_dataset(ex, new_intent, response_text)

                st.success("✅ Intent saved successfully")
                training = load_training()  # Reload data to reflect immediately

    st.markdown("---")
    st.subheader("📌 Existing intents")
    if not training["intents"]:
        st.info("No intents found")
    else:
        for i, intent_obj in enumerate(training["intents"], start=1):
            st.markdown(f"**{i}. {intent_obj['intent']}**")
            st.markdown(f"- 📝 Examples: {', '.join(intent_obj.get('examples', []))}\n- 💬 Response: {intent_obj.get('response', '')}")

# ------------------ USER QUERIES ------------------
elif page == "User Queries":
    st.header("User Queries")
    if not df_queries.empty:
        st.download_button("Download CSV", df_queries.to_csv(index=False), "queries.csv")
        st.dataframe(df_queries, use_container_width=True)
    else:
        st.info("No user queries found")

# ------------------ FAQ MANAGER ------------------
elif page == "FAQs":
    st.header("FAQ Manager")

    with st.form("faq_form", clear_on_submit=True):
        q = st.text_input("Question")
        a = st.text_area("Answer")
        submit = st.form_submit_button("Add FAQ")

        if submit:
            if q and a:
                faq.append({"q": q, "a": a})
                json.dump(faq, open(faq_path, "w", encoding="utf-8"), indent=2)
                append_to_faq_dataset(q, a, DATA_DIR)
                st.success("✅ FAQ saved successfully")
                faq = load_faq()  # Reload FAQs immediately
            else:
                st.error("❌ Question and Answer required")

    st.markdown("### 📌 Existing FAQs")
    for i, f in enumerate(faq, start=1):
        st.markdown(f"<div class='metric-card'><b>{i}. {f['q']}</b><br><span style='color:#6c757d;'>{f['a']}</span></div>", unsafe_allow_html=True)

# ------------------ ANALYTICS ------------------
elif page == "Analytics":
    st.header("Analytics Dashboard")
    st.line_chart(pd.DataFrame({"Queries": [5, 15, 30, 60, 100, 150]}))

# ------------------ SETTINGS ------------------
elif page == "Settings":
    st.header("⚙️ Settings")
    st.write("Admin Settings Page")

# ------------------ LOGOUT ------------------
elif page == "Logout":
    st.session_state.clear()
    st.markdown("""
    <meta http-equiv="refresh" content="0; url=http://localhost:5000/select_role">
    <script>
        window.top.location.href = 'http://localhost:5000/select_role';
    </script>
    """, unsafe_allow_html=True)
    st.info("Logging out... Redirecting to role selection page.")
