import os
import io
import time
from datetime import datetime, timezone
import streamlit as st
from dotenv import load_dotenv

# Load environment variables
load_dotenv(override=True)

from pdf_processor import process_uploaded_pdfs, extract_pages_from_pdf, split_documents_into_chunks
from vector_store import VectorStoreManager, get_embedding_function
from qa_chain import get_llm, generate_answer
from supabase_client import SupabaseManager
from email_validator import validate_email_authenticity

import base64

# Production Quota Defaults
DEFAULT_PROVIDER = "gemini"
DEFAULT_MODEL = "gemini-3.6-flash"
DEFAULT_EMBEDDING_PROVIDER = "local"
DEFAULT_TOP_K = 5
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 150
DEFAULT_TEMPERATURE = 0.2
GUEST_FREE_QUESTION_LIMIT = 3
DAILY_FREE_QUESTION_LIMIT = 10

# Preload API Key from environment
API_KEY = os.getenv("GEMINI_API_KEY", "") or os.getenv("GOOGLE_API_KEY", "")

# Helper to encode background image as Base64
def get_background_css():
    bg_path = os.path.join(os.path.dirname(__file__), "bg.png")
    if os.path.exists(bg_path):
        try:
            with open(bg_path, "rb") as img_file:
                b64_str = base64.b64encode(img_file.read()).decode()
                return f"background-image: url('data:image/png;base64,{b64_str}');"
        except Exception:
            pass
    return "background: linear-gradient(135deg, #00A8B5 0%, #0F4C75 50%, #051923 100%);"

BG_CSS = get_background_css()

# Streamlit Page Configuration
st.set_page_config(
    page_title="Talk to My Notes | AI Study Companion",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Modern Glassmorphism & Gradient SaaS CSS
st.markdown(f"""
<style>
    /* Hide Default Streamlit Header, Fork button, GitHub icon, 3-dots menu & Viewer Badges */
    #MainMenu {{
        visibility: hidden !important;
        display: none !important;
    }}
    header {{
        visibility: hidden !important;
        display: none !important;
        height: 0px !important;
    }}
    footer {{
        visibility: hidden !important;
        display: none !important;
        height: 0px !important;
    }}
    [data-testid="stHeader"] {{
        visibility: hidden !important;
        display: none !important;
        height: 0px !important;
    }}
    [data-testid="stToolbar"] {{
        visibility: hidden !important;
        display: none !important;
    }}
    [data-testid="stDecoration"] {{
        visibility: hidden !important;
        display: none !important;
        height: 0px !important;
    }}
    [data-testid="stStatusWidget"] {{
        visibility: hidden !important;
        display: none !important;
    }}
    [data-testid="manage-app-button"] {{
        display: none !important;
    }}
    div[class*="viewerBadge"], .viewerBadge_container__1QSob, .viewerBadge_link__1S137 {{
        display: none !important;
    }}
    div[class*="ProfileAvatar"] {{
        display: none !important;
    }}

    /* Full App Background */
    [data-testid="stAppViewContainer"], .stApp {{
        {BG_CSS}
        background-size: cover !important;
        background-position: center !important;
        background-repeat: no-repeat !important;
        background-attachment: fixed !important;
        color: #F8FAFC !important;
        padding-top: 0px !important;
    }}

    /* Sidebar Glassmorphism */
    [data-testid="stSidebar"] {{
        background: rgba(5, 25, 40, 0.88) !important;
        backdrop-filter: blur(20px) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.12) !important;
    }}

    /* Navbar styling */
    .nav-container {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 10px 0;
        margin-bottom: 25px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.15);
    }}
    .brand-logo {{
        font-size: 1.7rem;
        font-weight: 800;
        color: #FFFFFF;
        letter-spacing: -0.5px;
        text-shadow: 0 2px 10px rgba(0,0,0,0.3);
    }}
    .brand-badge {{
        background: rgba(56, 229, 255, 0.2);
        color: #38E5FF;
        font-size: 0.78rem;
        font-weight: 700;
        padding: 3px 10px;
        border-radius: 9999px;
        margin-left: 8px;
        border: 1px solid rgba(56, 229, 255, 0.4);
        vertical-align: middle;
    }}
    
    /* Hero Glassmorphism Box */
    .hero-box {{
        background: rgba(255, 255, 255, 0.08) !important;
        backdrop-filter: blur(20px) !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        border-radius: 20px !important;
        padding: 50px 40px !important;
        color: white !important;
        text-align: center !important;
        margin-bottom: 36px !important;
        box-shadow: 0 20px 40px -15px rgba(0, 0, 0, 0.5) !important;
    }}
    .hero-title {{
        font-size: 2.9rem;
        font-weight: 800;
        line-height: 1.15;
        margin-bottom: 16px;
        color: #FFFFFF;
        letter-spacing: -1px;
        text-shadow: 0 2px 12px rgba(0,0,0,0.4);
    }}
    .hero-subtitle {{
        font-size: 1.18rem;
        color: #E0F7FA;
        max-width: 800px;
        margin: 0 auto;
        line-height: 1.6;
        text-shadow: 0 1px 4px rgba(0,0,0,0.3);
    }}
    
    /* Feature Glassmorphism Cards */
    .feature-card {{
        background: rgba(255, 255, 255, 0.08) !important;
        backdrop-filter: blur(16px) !important;
        border: 1px solid rgba(255, 255, 255, 0.18) !important;
        border-radius: 16px !important;
        padding: 28px 24px !important;
        text-align: center !important;
        color: white !important;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3) !important;
        height: 100% !important;
        transition: transform 0.2s, box-shadow 0.2s, background 0.2s;
    }}
    .feature-card:hover {{
        transform: translateY(-5px);
        background: rgba(255, 255, 255, 0.14) !important;
        box-shadow: 0 15px 30px -5px rgba(0, 0, 0, 0.5) !important;
    }}
    .feature-icon {{
        font-size: 2.4rem;
        margin-bottom: 14px;
    }}
    .feature-title {{
        font-size: 1.25rem !important;
        font-weight: 700 !important;
        color: #38E5FF !important;
        margin-bottom: 10px !important;
        text-shadow: 0 1px 8px rgba(0,0,0,0.3);
    }}
    .feature-desc {{
        font-size: 0.95rem !important;
        color: #E2E8F0 !important;
        line-height: 1.6 !important;
    }}

    /* Step Glassmorphism Cards */
    .step-card {{
        background: rgba(255, 255, 255, 0.08) !important;
        backdrop-filter: blur(14px) !important;
        border: 1px solid rgba(56, 229, 255, 0.3) !important;
        border-radius: 14px !important;
        padding: 22px 18px !important;
        text-align: center !important;
        color: white !important;
        box-shadow: 0 8px 20px -4px rgba(0, 0, 0, 0.3) !important;
        height: 100% !important;
    }}
    .step-title {{
        font-size: 1.15rem !important;
        font-weight: 700 !important;
        color: #38E5FF !important;
        margin-bottom: 8px !important;
        text-shadow: 0 1px 6px rgba(0,0,0,0.3);
    }}
    .step-desc {{
        font-size: 0.92rem !important;
        color: #E2E8F0 !important;
        line-height: 1.5 !important;
    }}

    /* Auth Glassmorphism Card */
    .auth-card {{
        background: rgba(8, 25, 40, 0.88) !important;
        backdrop-filter: blur(25px) !important;
        border: 1px solid rgba(56, 229, 255, 0.3) !important;
        border-radius: 20px !important;
        padding: 36px !important;
        box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.7) !important;
    }}

    .main-title {{
        font-size: 2.3rem;
        font-weight: 800;
        color: #38E5FF !important;
        margin-bottom: 0.2rem;
        text-shadow: 0 2px 10px rgba(0,0,0,0.3);
    }}
    .sub-title {{
        font-size: 1.05rem;
        color: #E0F2FE !important;
        margin-bottom: 1.5rem;
    }}
    .stats-card {{
        background: rgba(255, 255, 255, 0.08) !important;
        border: 1px solid rgba(255, 255, 255, 0.16) !important;
        border-radius: 10px !important;
        padding: 14px !important;
        margin-top: 10px;
        margin-bottom: 14px;
        color: #F8FAFC !important;
    }}
    .info-card {{
        background: rgba(255, 255, 255, 0.08) !important;
        border-left: 4px solid #38E5FF !important;
        border-radius: 0 12px 12px 0;
        padding: 20px !important;
        margin-bottom: 20px;
        color: #F8FAFC !important;
    }}
    .user-badge {{
        background: rgba(56, 229, 255, 0.15) !important;
        border: 1px solid rgba(56, 229, 255, 0.4) !important;
        border-radius: 10px;
        padding: 10px 12px;
        margin-bottom: 12px;
        font-size: 0.95rem;
        color: #E0F7FA !important;
        font-weight: 500;
    }}
    .guest-badge {{
        background: rgba(255, 255, 255, 0.1) !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        border-radius: 10px;
        padding: 10px 12px;
        margin-bottom: 12px;
        font-size: 0.95rem;
        color: #E2E8F0 !important;
        font-weight: 500;
    }}
    .premium-badge {{
        background: linear-gradient(135deg, rgba(254, 243, 199, 0.25) 0%, rgba(253, 230, 138, 0.25) 100%) !important;
        border: 1px solid #F59E0B !important;
        border-radius: 10px;
        padding: 10px 12px;
        margin-bottom: 12px;
        font-size: 0.95rem;
        color: #FDE68A !important;
        font-weight: 600;
    }}
    .quota-card {{
        background: rgba(255, 255, 255, 0.1) !important;
        border: 1px solid rgba(56, 229, 255, 0.35) !important;
        border-radius: 10px;
        padding: 14px;
        margin-bottom: 14px;
        color: #FFFFFF !important;
    }}
    .locked-card {{
        background: rgba(239, 68, 68, 0.15) !important;
        border-left: 4px solid #EF4444 !important;
        border-radius: 0 12px 12px 0;
        padding: 18px;
        margin: 16px 0;
        color: #FEE2E2 !important;
    }}
    .google-btn {{
        display: block;
        width: 100%;
        background-color: #4285F4;
        color: white;
        text-align: center;
        padding: 12px 16px;
        border-radius: 10px;
        font-weight: 600;
        font-size: 1rem;
        text-decoration: none;
        margin-bottom: 16px;
        box-shadow: 0 4px 12px rgba(66, 133, 244, 0.4);
        transition: background-color 0.2s, transform 0.1s;
    }}
    .google-btn:hover {{
        background-color: #3367D6;
        color: white;
        text-decoration: none;
        transform: translateY(-1px);
    }}

    /* Universal High-Contrast Text for Sidebar & Labels */
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] div,
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {{
        color: #FFFFFF !important;
    }}

    /* Expander Styling */
    [data-testid="stExpander"] {{
        background: rgba(10, 35, 55, 0.85) !important;
        border: 1px solid rgba(56, 229, 255, 0.4) !important;
        border-radius: 12px !important;
        margin-bottom: 16px !important;
    }}
    [data-testid="stExpander"] details summary {{
        color: #38E5FF !important;
        font-weight: 700 !important;
        background: rgba(56, 229, 255, 0.12) !important;
        border-radius: 10px !important;
        padding: 10px 14px !important;
    }}
    [data-testid="stExpander"] details summary svg {{
        fill: #38E5FF !important;
        color: #38E5FF !important;
    }}

    /* File Uploader Dropzone Styling */
    [data-testid="stFileUploader"] section {{
        background-color: rgba(10, 35, 55, 0.85) !important;
        border: 1.5px dashed rgba(56, 229, 255, 0.5) !important;
        border-radius: 12px !important;
        padding: 18px 14px !important;
    }}
    [data-testid="stFileUploader"] section:hover {{
        border-color: #38E5FF !important;
        background-color: rgba(15, 45, 75, 0.95) !important;
    }}
    [data-testid="stFileUploader"] section button {{
        background: linear-gradient(135deg, #00C0FA 0%, #0077C0 100%) !important;
        border: 1px solid rgba(255, 255, 255, 0.4) !important;
        border-radius: 8px !important;
        padding: 8px 16px !important;
        box-shadow: 0 4px 12px rgba(0, 192, 250, 0.35) !important;
    }}
    [data-testid="stFileUploader"] section button * {{
        color: #FFFFFF !important;
        font-weight: 700 !important;
    }}
    [data-testid="stFileUploader"] section button:hover {{
        background: linear-gradient(135deg, #38E5FF 0%, #008DDA 100%) !important;
        border-color: #38E5FF !important;
    }}
    [data-testid="stFileUploader"] section button:hover * {{
        color: #031B2C !important;
    }}
    [data-testid="stFileUploaderDropzoneInstructions"] * {{
        color: #E0F2FE !important;
        font-weight: 500 !important;
    }}
    [data-testid="stFileUploaderDropzoneInstructions"] small {{
        color: #94A3B8 !important;
    }}
    [data-testid="stFileUploaderFile"] {{
        background-color: rgba(255, 255, 255, 0.1) !important;
        border: 1px solid rgba(56, 229, 255, 0.3) !important;
        border-radius: 8px !important;
    }}
    [data-testid="stFileUploaderFile"] * {{
        color: #FFFFFF !important;
    }}

    /* Inputs, Selectboxes, and Labels */
    .stTextInput label, .stSelectbox label, .stFileUploader label {{
        color: #FFFFFF !important;
        font-weight: 700 !important;
        font-size: 0.95rem !important;
        text-shadow: 0 1px 4px rgba(0,0,0,0.5) !important;
    }}
    .stTextInput input {{
        background-color: #FFFFFF !important;
        color: #031B2C !important;
        font-weight: 600 !important;
        border-radius: 8px !important;
    }}
    .stSelectbox div[data-baseweb="select"] > div {{
        background-color: #FFFFFF !important;
        color: #031B2C !important;
        font-weight: 600 !important;
        border-radius: 8px !important;
    }}
    .stSelectbox div[data-baseweb="select"] * {{
        color: #031B2C !important;
    }}

    /* Progress bar color */
    .stProgress > div > div > div > div {{
        background: linear-gradient(90deg, #38E5FF 0%, #00C0FA 100%) !important;
    }}
    .stProgress > div > div > div {{
        background-color: rgba(255, 255, 255, 0.2) !important;
    }}

    /* Crisp High-Contrast Streamlit Buttons */
    .stButton > button {{
        background: rgba(10, 35, 55, 0.85) !important;
        color: #FFFFFF !important;
        border: 1px solid rgba(56, 229, 255, 0.5) !important;
        border-radius: 10px !important;
        font-weight: 700 !important;
        font-size: 0.95rem !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3) !important;
        transition: all 0.2s ease !important;
    }}
    .stButton > button p, .stButton > button span {{
        color: #FFFFFF !important;
        font-weight: 700 !important;
    }}
    .stButton > button:hover {{
        background: #38E5FF !important;
        border-color: #38E5FF !important;
        box-shadow: 0 6px 18px rgba(56, 229, 255, 0.5) !important;
        transform: translateY(-1px) !important;
    }}
    .stButton > button:hover p, .stButton > button:hover span {{
        color: #031B2C !important;
    }}

    /* Primary Accent Buttons */
    .stButton > button[kind="primary"], .stButton > button[data-testid="baseButton-primary"] {{
        background: linear-gradient(135deg, #00C0FA 0%, #0077C0 100%) !important;
        color: #FFFFFF !important;
        border: 1px solid rgba(255, 255, 255, 0.35) !important;
        box-shadow: 0 4px 15px rgba(0, 192, 250, 0.4) !important;
    }}
    .stButton > button[kind="primary"] p, .stButton > button[data-testid="baseButton-primary"] p {{
        color: #FFFFFF !important;
        font-weight: 800 !important;
    }}
    .stButton > button[kind="primary"]:hover, .stButton > button[data-testid="baseButton-primary"]:hover {{
        background: linear-gradient(135deg, #38E5FF 0%, #00A8FF 100%) !important;
        box-shadow: 0 6px 20px rgba(56, 229, 255, 0.6) !important;
    }}
    .stButton > button[kind="primary"]:hover p, .stButton > button[data-testid="baseButton-primary"]:hover p {{
        color: #031B2C !important;
    }}
</style>
""", unsafe_allow_html=True)


# Initialize Supabase Manager
supabase_db = SupabaseManager()

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

if "vector_manager" not in st.session_state:
    st.session_state.vector_manager = VectorStoreManager()

if "indexed_stats" not in st.session_state:
    st.session_state.indexed_stats = None

if "is_indexed" not in st.session_state:
    st.session_state.is_indexed = False

if "active_file_signature" not in st.session_state:
    st.session_state.active_file_signature = None

if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

if "current_user" not in st.session_state:
    st.session_state.current_user = None

if "is_premium" not in st.session_state:
    st.session_state.is_premium = False

if "daily_queries_used" not in st.session_state:
    st.session_state.daily_queries_used = 0

if "auth_view" not in st.session_state:
    st.session_state.auth_view = "welcome"  # "welcome", "login", "signup"


# ==============================================================================
# HANDLE GOOGLE OAUTH CALLBACK (From Redirect URL)
# ==============================================================================
query_params = st.query_params
if "code" in query_params:
    auth_code = query_params["code"]
    with st.spinner("Authenticating with Google..."):
        oauth_res = supabase_db.handle_oauth_callback(auth_code)
        if oauth_res["success"]:
            st.session_state.current_user = oauth_res["user"]
            st.session_state.is_premium = oauth_res["user"].get("is_premium", False)
            st.session_state.daily_queries_used = supabase_db.get_user_daily_query_count(oauth_res["user"]["id"])
            st.session_state.auth_view = "welcome"
            st.query_params.clear()
            st.rerun()
        else:
            st.error(f"Google login failed: {oauth_res.get('error')}")


def reset_everything_for_new_chat():
    """
    Completely wipes chat history, purges vector store, and clears loaded files.
    """
    if "vector_manager" in st.session_state and st.session_state.vector_manager:
        st.session_state.vector_manager.clear()
    st.session_state.messages = []
    st.session_state.vector_manager = VectorStoreManager()
    st.session_state.indexed_stats = None
    st.session_state.is_indexed = False
    st.session_state.active_file_signature = None
    st.session_state.uploader_key += 1
    st.rerun()


# ==============================================================================
# WELCOME & AUTHENTICATION LANDING EXPERIENCE (Not Logged In)
# ==============================================================================
if st.session_state.current_user is None:

    # --------------------------------------------------------------------------
    # TOP NAVBAR WITH LOGO & CORNER AUTH BUTTONS
    # --------------------------------------------------------------------------
    nav_left, nav_space, nav_right1, nav_right2, nav_right3 = st.columns([4, 3, 1.4, 1.4, 1.4])
    with nav_left:
        st.markdown('<div class="brand-logo">📚 Talk to My Notes <span class="brand-badge">AI 2.0</span></div>', unsafe_allow_html=True)
    with nav_right1:
        if st.button("🚀 Try Demo", use_container_width=True):
            st.session_state.current_user = {
                "id": "guest_student",
                "email": "guest@student.local",
                "name": "Guest Student",
                "is_premium": False
            }
            st.session_state.is_premium = False
            st.session_state.daily_queries_used = 0
            st.rerun()
    with nav_right2:
        if st.button("🔑 Log In", use_container_width=True, type="secondary"):
            st.session_state.auth_view = "login"
            st.rerun()
    with nav_right3:
        if st.button("✨ Sign Up", use_container_width=True, type="primary"):
            st.session_state.auth_view = "signup"
            st.rerun()

    st.markdown("<hr style='margin-top: 5px; margin-bottom: 25px; border: none; border-top: 1px solid #E5E7EB;'>", unsafe_allow_html=True)

    # --------------------------------------------------------------------------
    # VIEW A: AUTHENTICATION MODAL (If user clicked Log In / Sign Up)
    # --------------------------------------------------------------------------
    if st.session_state.auth_view in ["login", "signup"]:
        auth_col1, auth_col2, auth_col3 = st.columns([1, 2, 1])
        with auth_col2:
            if st.button("← Back to Welcome Page", type="secondary"):
                st.session_state.auth_view = "welcome"
                st.rerun()

            st.markdown('<div class="auth-card">', unsafe_allow_html=True)
            
            # Google One-Click OAuth
            google_oauth_url, oauth_verifier = supabase_db.get_google_oauth_url("http://localhost:8501")
            if google_oauth_url:
                st.markdown(
                    f'<a href="{google_oauth_url}" target="_self" class="google-btn">🔵 Continue with Google (Verified Account)</a>',
                    unsafe_allow_html=True
                )
                st.markdown("<div style='text-align: center; color: #9CA3AF; margin: 12px 0;'>— OR WITH EMAIL —</div>", unsafe_allow_html=True)

            tab_index = 0 if st.session_state.auth_view == "login" else 1
            tab_login, tab_register = st.tabs(["🔑 Sign In", "✨ Create Account"])

            # 1. Sign In Tab
            with tab_login:
                with st.form("modal_login_form"):
                    login_email = st.text_input("Email Address", placeholder="student@gmail.com")
                    login_password = st.text_input("Password", type="password")
                    login_submit = st.form_submit_button("Log In", type="primary", use_container_width=True)
                    
                    if login_submit:
                        if not login_email or not login_password:
                            st.error("Please provide both email and password.")
                        else:
                            is_valid, val_msg = validate_email_authenticity(login_email)
                            if not is_valid:
                                st.error(val_msg)
                            else:
                                res = supabase_db.sign_in_with_password(login_email.strip(), login_password)
                                if res["success"]:
                                    st.session_state.current_user = res["user"]
                                    st.session_state.is_premium = res["user"].get("is_premium", False)
                                    st.session_state.daily_queries_used = supabase_db.get_user_daily_query_count(res["user"]["id"])
                                    st.session_state.auth_view = "welcome"
                                    st.success(f"Welcome back, {res['user']['name']}!")
                                    st.rerun()
                                else:
                                    st.error(f"Login failed: {res.get('error')}")

            # 2. Create Account Tab
            with tab_register:
                with st.form("modal_signup_form"):
                    reg_name = st.text_input("Full Name", placeholder="Khush Vaghani")
                    reg_email = st.text_input("Email Address", placeholder="student@gmail.com")
                    reg_pass = st.text_input("Choose Password", type="password")
                    reg_confirm = st.text_input("Confirm Password", type="password")
                    signup_submit = st.form_submit_button("✨ Create Account & Log In", type="primary", use_container_width=True)

                    if signup_submit:
                        if not reg_name or not reg_email or not reg_pass or not reg_confirm:
                            st.error("Please fill in all required fields.")
                        elif reg_pass != reg_confirm:
                            st.error("Passwords do not match.")
                        elif len(reg_pass) < 6:
                            st.error("Password must be at least 6 characters.")
                        else:
                            is_valid, val_msg = validate_email_authenticity(reg_email)
                            if not is_valid:
                                st.error(val_msg)
                            else:
                                res = supabase_db.sign_up_direct(
                                    email=reg_email.strip(),
                                    password=reg_pass,
                                    full_name=reg_name.strip()
                                )
                                if res["success"]:
                                    st.session_state.current_user = res["user"]
                                    st.session_state.is_premium = False
                                    st.session_state.daily_queries_used = 0
                                    st.session_state.auth_view = "welcome"
                                    st.success(f"🎉 Account created successfully! Welcome, {res['user']['name']}.")
                                    time.sleep(1.0)
                                    st.rerun()
                                else:
                                    st.error(res.get("error", "Sign up failed."))

            st.markdown('</div>', unsafe_allow_html=True)

        st.stop()

    # --------------------------------------------------------------------------
    # VIEW B: WELCOME / LANDING PAGE (SaaS Homepage)
    # --------------------------------------------------------------------------
    else:
        # Hero Banner
        st.markdown("""
        <div class="hero-box">
            <div class="hero-title">Transform Your Lecture Notes into an Intelligent AI Study Companion</div>
            <div class="hero-subtitle">
                Upload your slides, syllabus, or textbooks. Ask complex questions in plain English and receive comprehensive, step-by-step master notes grounded 100% in your actual syllabus.
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div style='margin: 30px 0;'></div>", unsafe_allow_html=True)

        # 3 Value Proposition / Feature Cards
        feat_col1, feat_col2, feat_col3 = st.columns(3)
        with feat_col1:
            st.markdown("""
            <div class="feature-card">
                <div class="feature-icon">🎯</div>
                <div class="feature-title">Exam-Ready Accuracy</div>
                <div class="feature-desc">
                    Zero hallucinations. Get crystal-clear, step-by-step answers and formulas derived directly from your course slides, textbooks, and past lecture materials.
                </div>
            </div>
            """, unsafe_allow_html=True)

        with feat_col2:
            st.markdown("""
            <div class="feature-card">
                <div class="feature-icon">📝</div>
                <div class="feature-title">Master Topics in Minutes</div>
                <div class="feature-desc">
                    Turn dense 100-page textbooks and complex slide decks into structured revision summaries, key formulas, and high-yield exam takeaways instantly.
                </div>
            </div>
            """, unsafe_allow_html=True)

        with feat_col3:
            st.markdown("""
            <div class="feature-card">
                <div class="feature-icon">💡</div>
                <div class="feature-title">Learn at Your Own Pace</div>
                <div class="feature-desc">
                    Ask any question in plain English 24/7. Get patient, tailored breakdowns that make difficult concepts simple and easy to master before your exams.
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div style='margin: 40px 0;'></div>", unsafe_allow_html=True)

        # How It Works Walkthrough
        st.markdown("<h3 style='text-align: center; color: #FFFFFF; font-weight: 800; margin-bottom: 24px; text-shadow: 0 2px 10px rgba(0,0,0,0.4);'>📖 How It Works in 3 Simple Steps</h3>", unsafe_allow_html=True)
        step_c1, step_c2, step_c3 = st.columns(3)
        with step_c1:
            st.markdown("""
            <div class="step-card">
                <div class="step-title">1. 📂 Upload Your Notes</div>
                <div class="step-desc">Drop in your lecture slides, class handouts, or textbook PDFs in one click.</div>
            </div>
            """, unsafe_allow_html=True)
        with step_c2:
            st.markdown("""
            <div class="step-card">
                <div class="step-title">2. 🤖 Ask Any Question</div>
                <div class="step-desc">Ask for problem solutions, complex concept breakdowns, or revision summaries.</div>
            </div>
            """, unsafe_allow_html=True)
        with step_c3:
            st.markdown("""
            <div class="step-card">
                <div class="step-title">3. 🌟 Get Master Notes</div>
                <div class="step-desc">Receive clear, beautifully structured study guides grounded directly in your syllabus.</div>
            </div>
            """, unsafe_allow_html=True)

    st.stop()


# ==============================================================================
# MAIN APPLICATION INTERFACE (Authenticated & Guest Users)
# ==============================================================================

user = st.session_state.current_user
is_guest = (user.get("id") == "guest_student")

# Sync daily query count from Supabase for registered users
if supabase_db.is_configured() and not is_guest:
    st.session_state.daily_queries_used = supabase_db.get_user_daily_query_count(user["id"])

# Active question limit based on user status
active_limit = GUEST_FREE_QUESTION_LIMIT if is_guest else DAILY_FREE_QUESTION_LIMIT

# ==========================================
# SIDEBAR
# ==========================================
with st.sidebar:
    # User Profile & Tier Badge
    if st.session_state.is_premium:
        st.markdown(f"""
        <div class="premium-badge">
            👑 <b>{user.get('name', 'Student')}</b> (⭐ Premium VIP)<br>
            <span style="font-size:0.85rem; color:#FEF08A; font-weight:600;">♾️ Unlimited Daily Questions</span>
        </div>
        """, unsafe_allow_html=True)
    elif is_guest:
        st.markdown(f"""
        <div class="guest-badge">
            👤 <b>Guest Student</b> (Preview Mode)<br>
            <span style="font-size:0.85rem; color:#E0F2FE; font-weight:500;">3 Free Demo Questions</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="user-badge">
            👤 <b>{user.get('name', 'Student')}</b> (Free Student Plan)<br>
            <span style="font-size:0.85rem; color:#E0F2FE; font-weight:500;">{user.get('email', '')}</span>
        </div>
        """, unsafe_allow_html=True)

    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("➕ New Chat", use_container_width=True, type="secondary"):
            reset_everything_for_new_chat()
    with col_btn2:
        btn_label = "Exit Demo" if is_guest else "🚪 Log Out"
        if st.button(btn_label, use_container_width=True):
            supabase_db.sign_out()
            st.session_state.current_user = None
            st.session_state.is_premium = False
            st.session_state.auth_view = "welcome"
            reset_everything_for_new_chat()

    st.divider()

    # Quota Tracker
    if not st.session_state.is_premium:
        queries_used = st.session_state.daily_queries_used
        queries_left = max(0, active_limit - queries_used)
        pct = min(1.0, queries_used / active_limit)
        
        title_txt = "⚡ Guest Demo Quota" if is_guest else "⚡ Daily Questions Quota"
        st.markdown(f"""
        <div class="quota-card">
            <div style="font-weight: 700; font-size: 1rem; color: #38E5FF; margin-bottom: 4px;">{title_txt}</div>
            <div style="font-size: 0.92rem; color: #FFFFFF; margin-bottom: 6px;"><b>{queries_used} / {active_limit}</b> questions used</div>
        </div>
        """, unsafe_allow_html=True)
        st.progress(pct)
        if queries_left == 0:
            if is_guest:
                st.warning("🔒 0 demo questions left. Create a free account for 10 daily questions!")
            else:
                st.warning("🔒 0 questions left today. Redeem a promo code below for unlimited access!")
        else:
            st.markdown(f"<div style='color: #38E5FF; font-size: 0.9rem; font-weight: 700; margin-top: 6px;'>✨ {queries_left} questions remaining.</div>", unsafe_allow_html=True)
    else:
        st.success("⭐ **Premium Active**: Unlimited questions per day!")

    # Promo Code Redemption Section
    with st.expander("🎟️ Redeem Promo Code / Premium", expanded=not st.session_state.is_premium):
        promo_input = st.text_input("Promo Code", key="promo_code_in")
        if st.button("✨ Apply Promo Code", type="primary", use_container_width=True):
            if not promo_input:
                st.error("Please enter a promo code.")
            else:
                res = supabase_db.redeem_promo_code(user.get("id"), promo_input)
                if res["success"]:
                    st.session_state.is_premium = True
                    st.success(res["message"])
                    time.sleep(1.0)
                    st.rerun()
                else:
                    st.error(res["error"])

    st.divider()

    # Document Upload Section
    st.markdown("## 📄 Document Upload")
    uploaded_files = st.file_uploader(
        "Upload Lecture Slides or Textbook PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        key=f"pdf_uploader_{st.session_state.uploader_key}",
        help="Upload single or multiple subject PDF documents."
    )

    # Automatic Stale File Invalidation
    current_signature = [(f.name, f.size) for f in uploaded_files] if uploaded_files else None
    if st.session_state.is_indexed and st.session_state.active_file_signature != current_signature:
        st.session_state.vector_manager.clear()
        st.session_state.vector_manager = VectorStoreManager()
        st.session_state.indexed_stats = None
        st.session_state.is_indexed = False
        st.session_state.messages = []
        st.session_state.active_file_signature = None

    # Explanation Style Selection
    explanation_style = st.selectbox(
        "🧠 Explanation Style",
        options=[
            "📘 Comprehensive & In-Depth Master Notes (Recommended)",
            "🌟 Simplified & Plain English Lecture Notes",
            "⚡ Quick & Concise (Key Bullet Points)",
            "🔬 Exhaustive Academic Analysis"
        ],
        index=0,
        help="Select how notes are organized and explained."
    )

    # Process Button
    if uploaded_files:
        if st.button("🚀 Process & Index Notes", type="primary", use_container_width=True):
            with st.spinner("Extracting text and building fresh vector index..."):
                try:
                    st.session_state.vector_manager.clear()
                    st.session_state.vector_manager = VectorStoreManager()
                    
                    processed_data = process_uploaded_pdfs(
                        uploaded_files,
                        chunk_size=DEFAULT_CHUNK_SIZE,
                        chunk_overlap=DEFAULT_CHUNK_OVERLAP
                    )
                    chunks = processed_data["chunks"]

                    if not chunks:
                        st.error("No readable text found in the uploaded PDFs.")
                    else:
                        emb_fn = get_embedding_function(
                            provider=DEFAULT_EMBEDDING_PROVIDER,
                            api_key=API_KEY
                        )

                        st.session_state.vector_manager.create_vector_store(
                            chunks=chunks,
                            embedding_function=emb_fn
                        )

                        st.session_state.indexed_stats = processed_data["stats"]
                        st.session_state.is_indexed = True
                        st.session_state.active_file_signature = current_signature
                        st.session_state.messages = []

                        if supabase_db.is_configured() and not is_guest:
                            for f_info in processed_data["stats"].get("files_summary", []):
                                supabase_db.save_document(
                                    user_id=user["id"],
                                    file_name=f_info["file_name"],
                                    page_count=f_info["page_count"],
                                    chunk_count=len(chunks)
                                )

                        st.success("✅ Notes indexed and ready for study!")
                        st.rerun()

                except Exception as e:
                    st.error(f"Failed to process PDFs: {str(e)}")

    # Index Summary Stats
    if st.session_state.is_indexed and st.session_state.indexed_stats:
        stats = st.session_state.indexed_stats
        st.divider()
        st.markdown("### 📊 Active Document")
        files_list = "<br>".join([f"• <b>{f['file_name']}</b> ({f['page_count']} pgs)" for f in stats.get('files_summary', [])])
        st.markdown(f"""
        <div class="stats-card">
            <div>{files_list}</div>
            <hr style="margin: 8px 0; border: none; border-top: 1px solid #E2E8F0;">
            <div>📖 <b>Total Pages:</b> {stats['total_pages']}</div>
            <div>🧩 <b>Total Chunks:</b> {stats['total_chunks']}</div>
        </div>
        """, unsafe_allow_html=True)

        if st.button("🗑️ Remove Document", use_container_width=True):
            reset_everything_for_new_chat()

    # Past Notes / History from Supabase (Only for registered users)
    if supabase_db.is_configured() and not is_guest:
        with st.expander("📜 My Saved Notes & History", expanded=False):
            saved_notes = supabase_db.get_user_chat_history(user["id"], limit=10)
            if saved_notes:
                for note in saved_notes:
                    st.markdown(f"**Q:** {note['question']}")
                    st.caption(f"📁 {note.get('file_name', 'Document')} • {note.get('created_at', '')[:10]}")
                    st.divider()
            else:
                st.caption("No past notes saved yet.")


# ==========================================
# MAIN CHAT AREA
# ==========================================
header_col1, header_col2 = st.columns([5, 1])
with header_col1:
    st.markdown('<div class="main-title">📚 Talk to My Notes</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Upload your subject notes or textbooks and ask questions. '
        'Get comprehensive, beautifully structured master study notes grounded directly in your syllabus.</div>',
        unsafe_allow_html=True
    )
with header_col2:
    if st.session_state.is_indexed:
        st.markdown("<div style='padding-top: 10px;'></div>", unsafe_allow_html=True)
        if st.button("➕ New Chat", key="main_new_chat", use_container_width=True):
            reset_everything_for_new_chat()

# If not indexed yet, show clean instructions
if not st.session_state.is_indexed:
    st.markdown("""
    <div class="info-card">
        <h3 style="margin-top:0; color:#1E3A8A;">🚀 How to Get Started</h3>
        <ol style="margin-bottom:0; font-size:1.02rem; line-height:1.7;">
            <li>Upload your lecture slides or subject textbook PDFs in the sidebar on the left.</li>
            <li>Click <b>"🚀 Process & Index Notes"</b> to extract and prepare your documents.</li>
            <li>Ask any question in natural language and receive comprehensive, step-by-step master notes!</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)

else:
    # Render Chat History
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Check Limit Status
    is_locked = (not st.session_state.is_premium) and (st.session_state.daily_queries_used >= active_limit)

    if is_locked:
        if is_guest:
            st.markdown(f"""
            <div class="locked-card">
                <h3 style="margin-top: 0; color: #DC2626;">🔒 Guest Demo Limit Reached ({GUEST_FREE_QUESTION_LIMIT}/{GUEST_FREE_QUESTION_LIMIT} Questions Used)</h3>
                <p style="font-size: 1.05rem; color: #374151; margin-bottom: 10px;">
                    You have completed all <b>{GUEST_FREE_QUESTION_LIMIT} free demo questions</b>!
                </p>
                <p style="font-size: 0.96rem; color: #4B5563; margin-bottom: 14px;">
                    🚀 <b>Sign in or Create a Free Account</b> to get <b>10 Free Questions Every Day</b> + cloud notes syncing, or upgrade to <b>⭐ Premium</b> for unlimited questions!
                </p>
            </div>
            """, unsafe_allow_html=True)

            lock_col1, lock_col2 = st.columns(2)
            with lock_col1:
                if st.button("✨ Create Free Account (10 Questions/Day)", type="primary", use_container_width=True):
                    st.session_state.current_user = None
                    st.session_state.auth_view = "signup"
                    st.rerun()
            with lock_col2:
                if st.button("🔑 Sign In to Existing Account", use_container_width=True):
                    st.session_state.current_user = None
                    st.session_state.auth_view = "login"
                    st.rerun()

        else:
            st.markdown(f"""
            <div class="locked-card">
                <h3 style="margin-top: 0; color: #DC2626;">🔒 Daily Limit Reached ({DAILY_FREE_QUESTION_LIMIT}/{DAILY_FREE_QUESTION_LIMIT} Questions Used Today)</h3>
                <p style="font-size: 1.02rem; color: #374151; margin-bottom: 8px;">
                    You've used all <b>{DAILY_FREE_QUESTION_LIMIT} free questions</b> for today. Your quota resets tomorrow.
                </p>
                <p style="font-size: 0.95rem; color: #4B5563; margin-bottom: 0;">
                    🎟️ <b>Have a special Promo Code or VIP Key?</b><br>
                    Enter your code in the sidebar under <b>"🎟️ Redeem Promo Code / Premium"</b> to unlock <b>Unlimited Questions</b> instantly!
                </p>
            </div>
            """, unsafe_allow_html=True)

    else:
        # Process Input from user
        user_prompt = st.chat_input("Ask a question about your subject notes...")

        if user_prompt:
            # 1. Add user message
            st.session_state.messages.append({"role": "user", "content": user_prompt})
            with st.chat_message("user"):
                st.markdown(user_prompt)

            # 2. Retrieve top chunks & generate master notes
            with st.chat_message("assistant"):
                with st.spinner("Analyzing your notes & crafting detailed master notes..."):
                    try:
                        retrieved_docs_with_scores = st.session_state.vector_manager.similarity_search_with_scores(
                            query=user_prompt,
                            k=DEFAULT_TOP_K
                        )

                        result = generate_answer(
                            query=user_prompt,
                            retrieved_docs_with_scores=retrieved_docs_with_scores,
                            provider=DEFAULT_PROVIDER,
                            api_key=API_KEY,
                            explanation_style=explanation_style
                        )

                        answer_text = result["answer"]

                        # Display Clean Master Notes
                        st.markdown(answer_text)

                        # Store in history
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": answer_text
                        })

                        # Increment queries used
                        st.session_state.daily_queries_used += 1

                        # Store in Supabase Cloud Database (if logged in)
                        if supabase_db.is_configured() and not is_guest:
                            active_file = "Document"
                            if st.session_state.indexed_stats and st.session_state.indexed_stats.get("files_summary"):
                                active_file = st.session_state.indexed_stats["files_summary"][0]["file_name"]
                            supabase_db.save_chat_message(
                                user_id=user["id"],
                                file_name=active_file,
                                question=user_prompt,
                                answer=answer_text,
                                explanation_style=explanation_style
                            )

                        st.rerun()

                    except Exception as e:
                        st.error(f"An error occurred: {str(e)}")
