import os
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List, Tuple
from dotenv import load_dotenv

load_dotenv(override=True)

try:
    from supabase import create_client, Client
    SUPABASE_AVAILABLE = True
except ImportError:
    Client = Any
    SUPABASE_AVAILABLE = False


# Default Master Promo Codes (can be customized via .env PROMO_CODES=CODE1,CODE2)
DEFAULT_VALID_PROMO_CODES = [
    "PREMIUM2026",
    "UNLIMITED",
    "STUDENTVIP",
    "KHUSHVIP",
    "NOTESPRO",
    "VIPACCESS"
]


def get_valid_promo_codes() -> List[str]:
    env_codes = os.getenv("PROMO_CODES", "")
    if env_codes:
        custom_list = [c.strip().upper() for c in env_codes.split(",") if c.strip()]
        return list(set(DEFAULT_VALID_PROMO_CODES + custom_list))
    return DEFAULT_VALID_PROMO_CODES


def get_supabase_config() -> Tuple[str, str]:
    url = os.getenv("SUPABASE_URL", "").strip()
    key = os.getenv("SUPABASE_KEY", "").strip() or os.getenv("SUPABASE_ANON_KEY", "").strip()
    return url, key


class SupabaseManager:
    """
    Manages Supabase Authentication (Google OAuth + Email), Cloud Database Operations,
    and Daily Quota / Promo Code Subscriptions.
    """

    def __init__(self):
        self.url = os.getenv("SUPABASE_URL", "").strip()
        self.key = os.getenv("SUPABASE_KEY", "").strip() or os.getenv("SUPABASE_ANON_KEY", "").strip()
        self.client: Optional[Client] = None

        if SUPABASE_AVAILABLE and self.url and self.key:
            try:
                self.client = create_client(self.url, self.key)
            except Exception as e:
                print(f"[Supabase Init Warning]: {str(e)}")
                self.client = None

    def is_configured(self) -> bool:
        return self.client is not None

    # ==========================================
    # 1. GOOGLE OAUTH & DIRECT AUTHENTICATION
    # ==========================================

    def get_google_oauth_url(self, redirect_to: str = "http://localhost:8501") -> Tuple[Optional[str], Optional[str]]:
        """
        Generates official Google OAuth login URL via Supabase and captures PKCE code verifier.
        """
        if not self.is_configured():
            return None, None
        try:
            res = self.client.auth.sign_in_with_oauth({
                "provider": "google",
                "options": {
                    "redirect_to": redirect_to
                }
            })
            storage_dict = getattr(self.client.auth._storage, "storage", {})
            verifier = storage_dict.get("supabase.auth.token-code-verifier")
            if verifier:
                try:
                    with open(".oauth_pkce_verifier", "w") as f:
                        f.write(verifier)
                except Exception:
                    pass
            return res.url, verifier
        except Exception as e:
            print(f"[Supabase get_google_oauth_url error]: {str(e)}")
            return None, None

    def handle_oauth_callback(self, auth_code: str, code_verifier: Optional[str] = None) -> Dict[str, Any]:
        """
        Exchanges Google OAuth code for authenticated user session.
        """
        if not self.is_configured():
            return {"success": False, "error": "Supabase is not configured."}
        try:
            if not code_verifier:
                storage_dict = getattr(self.client.auth._storage, "storage", {})
                code_verifier = storage_dict.get("supabase.auth.token-code-verifier")
            if not code_verifier and os.path.exists(".oauth_pkce_verifier"):
                try:
                    with open(".oauth_pkce_verifier", "r") as f:
                        code_verifier = f.read().strip()
                except Exception:
                    pass

            if code_verifier:
                self.client.auth._storage.set_item("supabase.auth.token-code-verifier", code_verifier)
                res = self.client.auth.exchange_code_for_session({
                    "auth_code": auth_code,
                    "code_verifier": code_verifier
                })
            else:
                res = self.client.auth.exchange_code_for_session({
                    "auth_code": auth_code
                })

            if res.user:
                full_name = res.user.user_metadata.get("full_name") or res.user.user_metadata.get("name") or res.user.email.split("@")[0]
                is_premium = res.user.user_metadata.get("is_premium", False)
                return {
                    "success": True,
                    "user": {
                        "id": str(res.user.id),
                        "email": res.user.email,
                        "name": full_name,
                        "is_premium": is_premium
                    },
                    "session": res.session
                }
            return {"success": False, "error": "Could not authenticate with Google."}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def sign_in_with_password(self, email: str, password: str) -> Dict[str, Any]:
        """
        Direct user login with email and password.
        """
        if not self.is_configured():
            return {"success": False, "error": "Supabase is not configured."}
        try:
            res = self.client.auth.sign_in_with_password({"email": email.strip(), "password": password})
            if res.user:
                is_premium = res.user.user_metadata.get("is_premium", False)
                return {
                    "success": True,
                    "user": {
                        "id": str(res.user.id),
                        "email": res.user.email,
                        "name": res.user.user_metadata.get("full_name", res.user.email.split("@")[0]),
                        "is_premium": is_premium
                    },
                    "session": res.session
                }
            return {"success": False, "error": "Invalid email or password."}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def sign_up_direct(self, email: str, password: str, full_name: str = "") -> Dict[str, Any]:
        """
        Direct instant account creation.
        """
        if not self.is_configured():
            return {"success": False, "error": "Supabase is not configured."}
        try:
            res = self.client.auth.sign_up({
                "email": email.strip(),
                "password": password,
                "options": {
                    "data": {
                        "full_name": full_name or email.split("@")[0],
                        "is_premium": False
                    }
                }
            })
            if res.user:
                if res.session:
                    return {
                        "success": True,
                        "user": {
                            "id": str(res.user.id),
                            "email": res.user.email,
                            "name": full_name or email.split("@")[0],
                            "is_premium": False
                        },
                        "session": res.session
                    }
                else:
                    login_attempt = self.sign_in_with_password(email, password)
                    if login_attempt["success"]:
                        return login_attempt
                    return {
                        "success": True,
                        "user": {
                            "id": str(res.user.id),
                            "email": res.user.email,
                            "name": full_name or email.split("@")[0],
                            "is_premium": False
                        },
                        "session": None
                    }
            return {"success": False, "error": "Sign up request failed."}
        except Exception as e:
            err_msg = str(e)
            if "already registered" in err_msg.lower() or "already exists" in err_msg.lower():
                return {"success": False, "error": "An account with this email already exists. Please log in."}
            return {"success": False, "error": err_msg}

    def sign_out(self) -> Dict[str, Any]:
        if not self.is_configured():
            return {"success": True}
        try:
            self.client.auth.sign_out()
            return {"success": True}
        except Exception as e:
            return {"success": False, "error": str(e)}

    # ==========================================
    # 2. DAILY USAGE & PROMO CODE MANAGEMENT
    # ==========================================

    def get_user_daily_query_count(self, user_id: str) -> int:
        """
        Returns the number of questions asked by the user today (UTC).
        """
        if not self.is_configured() or not user_id or user_id == "guest_student":
            return 0
        try:
            today_start = datetime.now(timezone.utc).strftime("%Y-%m-%dT00:00:00Z")
            res = self.client.table("user_chat_history") \
                .select("*", count="exact") \
                .eq("user_id", user_id) \
                .gte("created_at", today_start) \
                .execute()
            return res.count or 0
        except Exception as e:
            print(f"[Supabase get_daily_query_count error]: {str(e)}")
            return 0

    def redeem_promo_code(self, user_id: str, promo_code: str) -> Dict[str, Any]:
        """
        Validates promo code and upgrades the user to Premium.
        """
        cleaned_code = promo_code.strip().upper()
        valid_codes = get_valid_promo_codes()

        if cleaned_code in valid_codes:
            if self.is_configured() and user_id and user_id != "guest_student":
                try:
                    self.client.auth.update_user({
                        "data": {"is_premium": True, "promo_code_used": cleaned_code}
                    })
                except Exception as e:
                    print(f"[Supabase update_user metadata warning]: {str(e)}")

            return {
                "success": True,
                "message": f"🎉 Promo Code '{cleaned_code}' redeemed successfully! You now have Unlimited Questions."
            }
        else:
            return {
                "success": False,
                "error": "❌ Invalid Promo Code. Please verify your code and try again."
            }

    # ==========================================
    # 3. DATABASE PERSISTENCE METHODS
    # ==========================================

    def save_document(
        self,
        user_id: str,
        file_name: str,
        page_count: int,
        chunk_count: int
    ) -> Dict[str, Any]:
        if not self.is_configured() or not user_id:
            return {"success": False, "error": "Database not configured or guest user"}
        try:
            data = {
                "user_id": user_id,
                "file_name": file_name,
                "page_count": page_count,
                "chunk_count": chunk_count
            }
            res = self.client.table("user_documents").insert(data).execute()
            return {"success": True, "data": res.data}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def get_user_documents(self, user_id: str) -> List[Dict[str, Any]]:
        if not self.is_configured() or not user_id:
            return []
        try:
            res = self.client.table("user_documents").select("*").eq("user_id", user_id).order("created_at", desc=True).execute()
            return res.data or []
        except Exception as e:
            print(f"[Supabase get_documents error]: {str(e)}")
            return []

    def save_chat_message(
        self,
        user_id: str,
        file_name: str,
        question: str,
        answer: str,
        explanation_style: str = ""
    ) -> Dict[str, Any]:
        if not self.is_configured() or not user_id:
            return {"success": False, "error": "Database not configured or guest user"}
        try:
            data = {
                "user_id": user_id,
                "file_name": file_name,
                "question": question,
                "answer": answer,
                "explanation_style": explanation_style
            }
            res = self.client.table("user_chat_history").insert(data).execute()
            return {"success": True, "data": res.data}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def get_user_chat_history(self, user_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        if not self.is_configured() or not user_id:
            return []
        try:
            res = self.client.table("user_chat_history").select("*").eq("user_id", user_id).order("created_at", desc=True).limit(limit).execute()
            return res.data or []
        except Exception as e:
            print(f"[Supabase get_chat_history error]: {str(e)}")
            return []

    def clear_user_chat_history(self, user_id: str) -> bool:
        if not self.is_configured() or not user_id:
            return True
        try:
            self.client.table("user_chat_history").delete().eq("user_id", user_id).execute()
            return True
        except Exception as e:
            print(f"[Supabase clear_chat_history error]: {str(e)}")
            return False
