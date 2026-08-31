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

# Production Defaults
DEFAULT_PROVIDER = "gemini"
DEFAULT_MODEL = "gemini-3.6-flash"
DEFAULT_EMBEDDING_PROVIDER = "local"
DEFAULT_TOP_K = 5
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 150
DEFAULT_TEMPERATURE = 0.2
DAILY_FREE_QUESTION_LIMIT = 10

# Preload API Key from environment
API_KEY = os.getenv("GEMINI_API_KEY", "") or os.getenv("GOOGLE_API_KEY", "")

# Streamlit Page Configuration
st.set_page_config(
    page_title="Talk to My Notes | AI Study Companion",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Clean Production CSS
st.markdown("""
<style>
    .main-title {
        font-size: 2.3rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .stats-card {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 14px;
        margin-top: 10px;
        margin-bottom: 14px;
    }
    .info-card {
        background: #EFF6FF;
        border-left: 4px solid #3B82F6;
        border-radius: 0 8px 8px 0;
        padding: 16px;
        margin-bottom: 20px;
    }
    .user-badge {
        background: #EEF2FF;
        border: 1px solid #C7D2FE;
        border-radius: 8px;
        padding: 10px 12px;
        margin-bottom: 12px;
        font-size: 0.95rem;
        color: #3730A3;
        font-weight: 500;
    }
    .premium-badge {
        background: linear-gradient(135deg, #FEF3C7 0%, #FDE68A 100%);
        border: 1px solid #F59E0B;
        border-radius: 8px;
        padding: 10px 12px;
        margin-bottom: 12px;
        font-size: 0.95rem;
        color: #92400E;
        font-weight: 600;
    }
    .quota-card {
        background: #F3F4F6;
        border: 1px solid #E5E7EB;
        border-radius: 8px;
        padding: 12px;
        margin-bottom: 14px;
    }
    .locked-card {
        background: #FEF2F2;
        border-left: 4px solid #EF4444;
        border-radius: 0 8px 8px 0;
        padding: 16px;
        margin: 16px 0;
    }
    .google-btn {
        display: block;
        width: 100%;
        background-color: #4285F4;
        color: white;
        text-align: center;
        padding: 12px 16px;
        border-radius: 8px;
        font-weight: 600;
        font-size: 1rem;
        text-decoration: none;
        margin-bottom: 16px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        transition: background-color 0.2s;
    }
    .google-btn:hover {
        background-color: #3367D6;
        color: white;
        text-decoration: none;
    }
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
# AUTHENTICATION SCREEN (Google Sign-In & Email Authentication)
# ==============================================================================
if st.session_state.current_user is None:
    st.markdown('<div class="main-title">📚 Talk to My Notes</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Sign in with your verified Google account or student email to automatically sync your notes.</div>',
        unsafe_allow_html=True
    )

    auth_col1, auth_col2, auth_col3 = st.columns([1, 2, 1])
    with auth_col2:
        # 1. Primary: Google Sign-In (Verified & Impossible to Fake)
        google_oauth_url, oauth_verifier = supabase_db.get_google_oauth_url("http://localhost:8501")
        if google_oauth_url:
            st.markdown(
                f'<a href="{google_oauth_url}" target="_self" class="google-btn">🔵 Continue with Google (Verified Account)</a>',
                unsafe_allow_html=True
            )
            st.markdown("<div style='text-align: center; color: #9CA3AF; margin: 12px 0;'>— OR SIGN IN WITH EMAIL —</div>", unsafe_allow_html=True)

        tab_login, tab_register = st.tabs([
            "🔑 Email Sign In",
            "✨ Create Account"
        ])

        # 2. Email & Password Sign In
        with tab_login:
            with st.form("login_form"):
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
                                st.success(f"Welcome back, {res['user']['name']}!")
                                st.rerun()
                            else:
                                st.error(f"Login failed: {res.get('error')}")

        # 3. Direct Account Creation
        with tab_register:
            with st.form("signup_form"):
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
                                st.success(f"🎉 Account created successfully! Welcome, {res['user']['name']}.")
                                time.sleep(1.0)
                                st.rerun()
                            else:
                                st.error(res.get("error", "Sign up failed."))

        st.divider()
        
        # Quick Demo / Guest Mode
        if st.button("🚀 Continue as Guest (Quick Demo)", use_container_width=True):
            st.session_state.current_user = {
                "id": "guest_student",
                "email": "guest@student.local",
                "name": "Guest Student",
                "is_premium": False
            }
            st.session_state.is_premium = False
            st.session_state.daily_queries_used = 0
            st.rerun()

    st.stop()


# ==============================================================================
# MAIN APPLICATION INTERFACE (Authenticated Users)
# ==============================================================================

user = st.session_state.current_user

# Sync daily query count from Supabase
if supabase_db.is_configured() and user.get("id") and user["id"] != "guest_student":
    st.session_state.daily_queries_used = supabase_db.get_user_daily_query_count(user["id"])

# ==========================================
# SIDEBAR
# ==========================================
with st.sidebar:
    # User Profile & Tier Badge
    if st.session_state.is_premium:
        st.markdown(f"""
        <div class="premium-badge">
            👑 <b>{user.get('name', 'Student')}</b> (⭐ Premium VIP)<br>
            <span style="font-size:0.82rem; color:#78350F;">♾️ Unlimited Daily Questions</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="user-badge">
            👤 <b>{user.get('name', 'Student')}</b> (Free Plan)<br>
            <span style="font-size:0.82rem; color:#6B7280;">{user.get('email', '')}</span>
        </div>
        """, unsafe_allow_html=True)

    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("➕ New Chat", use_container_width=True, type="secondary"):
            reset_everything_for_new_chat()
    with col_btn2:
        if st.button("🚪 Log Out", use_container_width=True):
            supabase_db.sign_out()
            st.session_state.current_user = None
            st.session_state.is_premium = False
            reset_everything_for_new_chat()

    st.divider()

    # Daily Quota Tracker
    if not st.session_state.is_premium:
        queries_used = st.session_state.daily_queries_used
        queries_left = max(0, DAILY_FREE_QUESTION_LIMIT - queries_used)
        pct = min(1.0, queries_used / DAILY_FREE_QUESTION_LIMIT)
        
        st.markdown(f"""
        <div class="quota-card">
            <div style="font-weight: 600; font-size: 0.95rem; margin-bottom: 4px;">⚡ Daily Questions Quota</div>
            <div style="font-size: 0.88rem; color: #4B5563; margin-bottom: 6px;"><b>{queries_used} / {DAILY_FREE_QUESTION_LIMIT}</b> questions used today</div>
        </div>
        """, unsafe_allow_html=True)
        st.progress(pct)
        if queries_left == 0:
            st.warning("🔒 0 questions left today. Redeem a promo code below for unlimited access!")
        else:
            st.caption(f"✨ **{queries_left}** free questions remaining today.")
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

                        if supabase_db.is_configured() and user.get("id"):
                            for f_info in processed_data["stats"].get("files_summary", []):
                                supabase_db.save_document(
                                    user_id=user["id"],
                                    file_name=f_info["file_name"],
                                    page_count=f_info["page_count"],
                                    chunk_count=len(chunks)
                                )

                        st.success("✅ Notes indexed and synced to Supabase!")
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

    # Past Notes / History from Supabase
    if supabase_db.is_configured() and user.get("id"):
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

    # Check Daily Limit Status
    is_locked = (not st.session_state.is_premium) and (st.session_state.daily_queries_used >= DAILY_FREE_QUESTION_LIMIT)

    if is_locked:
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

                        # Increment daily queries used
                        st.session_state.daily_queries_used += 1

                        # Store in Supabase Cloud Database
                        if supabase_db.is_configured() and user.get("id"):
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
