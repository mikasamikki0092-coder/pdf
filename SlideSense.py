import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import asyncio
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import random, smtplib, time
from email.message import EmailMessage

# -------------------- CONFIG --------------------
EMAIL_SENDER = "your_email@gmail.com"
EMAIL_PASSWORD = "your_gmail_app_password"

OTP_EXPIRY_TIME = 120       # seconds
OTP_RESEND_COOLDOWN = 30    # seconds

# -------------------- Page Configuration --------------------
st.set_page_config(
    page_title="SlideSense",
    page_icon="📘",
    layout="wide",
    initial_sidebar_state="expanded"
)

load_dotenv()

# -------------------- Session State --------------------
defaults = {
    "chat_history": [],
    "vector_db": None,
    "authenticated": False,
    "users": {"admin": "admin123"},
    "otp": None,
    "otp_email": None,
    "otp_time": None,
    "otp_last_sent": 0
}

for k,v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# -------------------- OTP SYSTEM --------------------
def send_otp(email):
    now = time.time()
    if now - st.session_state.otp_last_sent < OTP_RESEND_COOLDOWN:
        raise Exception("Resend cooldown active")

    otp = random.randint(100000, 999999)
    st.session_state.otp = str(otp)
    st.session_state.otp_email = email
    st.session_state.otp_time = now
    st.session_state.otp_last_sent = now

    msg = EmailMessage()
    msg.set_content(f"Your SlideSense OTP is: {otp}\nValid for 2 minutes.")
    msg["Subject"] = "SlideSense Login OTP"
    msg["From"] = EMAIL_SENDER
    msg["To"] = email

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(EMAIL_SENDER, EMAIL_PASSWORD)
        server.send_message(msg)

def verify_otp(user_otp):
    if not st.session_state.otp:
        return False, "No OTP generated"

    if time.time() - st.session_state.otp_time > OTP_EXPIRY_TIME:
        st.session_state.otp = None
        return False, "OTP expired"

    if user_otp == st.session_state.otp:
        st.session_state.otp = None
        return True, "Success"
    else:
        return False, "Invalid OTP"

# -------------------- AUTH UI --------------------
def login_ui():
    st.markdown("<h1 style='text-align:center;'>🔐 SlideSense Login</h1>", unsafe_allow_html=True)
    tab1, tab2, tab3 = st.tabs(["Login", "Sign Up", "Email OTP Login"])

    # ----- Login -----
    with tab1:
        u = st.text_input("Username")
        p = st.text_input("Password", type="password")
        if st.button("Login"):
            if u in st.session_state.users and st.session_state.users[u] == p:
                st.session_state.authenticated = True
                st.success("✅ Login successful")
                st.rerun()
            else:
                st.error("❌ Invalid credentials")

    # ----- Signup -----
    with tab2:
        nu = st.text_input("Create Username")
        np = st.text_input("Create Password", type="password")
        if st.button("Sign Up"):
            if nu in st.session_state.users:
                st.warning("User exists")
            elif nu == "" or np == "":
                st.warning("Fields required")
            else:
                st.session_state.users[nu] = np
                st.success("Account created")

    # ----- OTP Login -----
    with tab3:
        email = st.text_input("Email Address")

        if st.button("Send OTP"):
            if email:
                try:
                    send_otp(email)
                    st.success("📧 OTP sent")
                except:
                    st.warning("⏳ Wait before resending OTP")
            else:
                st.warning("Enter email")

        otp_in = st.text_input("Enter OTP")

        if st.button("Verify OTP"):
            ok, msg = verify_otp(otp_in)
            if ok:
                st.session_state.authenticated = True
                st.success("✅ OTP Verified")
                st.rerun()
            else:
                st.error(msg)

# -------------------- BLIP Model --------------------
@st.cache_resource
def load_blip():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

processor, blip_model = load_blip()

def describe_image(image: Image.Image):
    inputs = processor(image, return_tensors="pt")
    out = blip_model.generate(**inputs)
    return processor.decode(out[0], skip_special_tokens=True)

# -------------------- AUTH CHECK --------------------
if not st.session_state.authenticated:
    login_ui()
    st.stop()

# -------------------- Sidebar --------------------
st.sidebar.success("Logged in")
if st.sidebar.button("🚪 Logout"):
    for k in defaults:
        st.session_state[k] = defaults[k]
    st.rerun()

page = st.sidebar.selectbox("Choose Mode", ["PDF Analyzer", "Image Recognition"])

st.sidebar.markdown("## 💬 Chat History")
for i,(q,a) in enumerate(st.session_state.chat_history[-10:],1):
    st.sidebar.markdown(f"Q{i}: {q[:40]}")

# -------------------- HERO --------------------
st.markdown("""
<h1 style='text-align:center;'>📘 SlideSense</h1>
<p style='text-align:center;'>AI Powered PDF Analyzer & Image Intelligence System</p>
<hr>
""", unsafe_allow_html=True)

# -------------------- PDF ANALYZER --------------------
if page == "PDF Analyzer":
    pdf = st.file_uploader("Upload PDF", type="pdf")

    if pdf:
        if st.session_state.vector_db is None:
            with st.spinner("Processing PDF..."):
                reader = PdfReader(pdf)
                text = ""
                for p in reader.pages:
                    if p.extract_text():
                        text += p.extract_text()+"\n"

                splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=80)
                chunks = splitter.split_text(text)

                try:
                    asyncio.get_running_loop()
                except:
                    asyncio.set_event_loop(asyncio.new_event_loop())

                embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
                st.session_state.vector_db = FAISS.from_texts(chunks, embeddings)

        st.success("Document processed")

        q = st.text_input("Ask a question")

        if q:
            docs = st.session_state.vector_db.similarity_search(q, k=5)
            llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

            history = ""
            for x,y in st.session_state.chat_history[-5:]:
                history += f"Q:{x}\nA:{y}\n"

            prompt = ChatPromptTemplate.from_template("""
History:
{history}

Context:
{context}

Question:
{question}

Rules:
- Answer only from document
- If not found say: Information not found in the document
""")

            chain = create_stuff_documents_chain(llm, prompt)
            res = chain.invoke({"context":docs,"question":q,"history":history})

            st.session_state.chat_history.append((q,res))

        st.markdown("## 💬 Conversation")
        for q,a in st.session_state.chat_history:
            st.markdown(f"**User:** {q}")
            st.markdown(f"**SlideSense:** {a}")
            st.markdown("---")

# -------------------- IMAGE RECOGNITION --------------------
if page == "Image Recognition":
    img_file = st.file_uploader("Upload Image", type=["png","jpg","jpeg"])
    if img_file:
        img = Image.open(img_file)
        st.image(img, use_column_width=True)
        with st.spinner("Analyzing image..."):
            desc = describe_image(img)
        st.success(desc)
