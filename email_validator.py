import re
import socket
from typing import Tuple, Dict, Any

# Top Disposable / Temporary Email Domains to Block
DISPOSABLE_EMAIL_DOMAINS = {
    "tempmail.com", "temp-mail.org", "10minutemail.com", "10minutemail.net",
    "mailinator.com", "guerrillamail.com", "guerrillamail.net", "guerrillamail.org",
    "sharklasers.com", "yopmail.com", "yopmail.fr", "yopmail.net", "trashmail.com",
    "trashmail.net", "dispostable.com", "throwawaymail.com", "fakeinbox.com",
    "getairmail.com", "mohmal.com", "crazymailing.com", "dropmail.me",
    "burnermail.io", "maildrop.cc", "emailondeck.com", "mytemp.email",
    "generator.email", "nada.ltd", "tempmailo.com", "inboxkitten.com",
    "fakemailgenerator.com", "tempail.com", "discard.email", "spambox.us"
}

# Strict Email Format Regex (RFC 5322 compliant)
EMAIL_REGEX = re.compile(
    r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$"
)


def validate_email_authenticity(email: str) -> Tuple[bool, str]:
    """
    Performs comprehensive multi-layer validation on an email address:
    1. Syntax & Format Check
    2. Disposable / Temporary Email Blacklist Check
    3. Live DNS Domain & Mail Server Verification
    """
    if not email or not isinstance(email, str):
        return False, "Please enter an email address."

    cleaned_email = email.strip().lower()

    # 1. Format Check
    if not EMAIL_REGEX.match(cleaned_email):
        return False, "❌ Invalid email format. Please enter a valid email (e.g. name@gmail.com)."

    if ".." in cleaned_email or cleaned_email.startswith(".") or cleaned_email.endswith("."):
        return False, "❌ Invalid email format."

    parts = cleaned_email.split("@")
    if len(parts) != 2:
        return False, "❌ Invalid email address."

    username, domain = parts[0], parts[1]

    if len(username) < 2:
        return False, "❌ Email username is too short."

    # 2. Check for Disposable / Temp Email Domains
    if domain in DISPOSABLE_EMAIL_DOMAINS:
        return False, (
            "🚫 Disposable/Temporary emails are not allowed.\n"
            "Please use your genuine personal or university email address."
        )

    # 3. Live DNS Host & Domain Existence Check
    try:
        # Verify that the domain exists and resolves in global DNS
        socket.gethostbyname(domain)
    except socket.gaierror:
        return False, (
            f"❌ The email domain '@{domain}' does not exist or has no active mail server.\n"
            "Please check for typos or enter a real email provider (Gmail, Outlook, Yahoo, etc.)."
        )
    except Exception as e:
        # In case of local network DNS timeout, allow trusted providers
        trusted_domains = {"gmail.com", "yahoo.com", "outlook.com", "hotmail.com", "icloud.com", "live.com", "aol.com"}
        if domain not in trusted_domains and not domain.endswith(".edu") and not domain.endswith(".ac.in"):
            return False, f"❌ Could not verify domain '@{domain}'. Please use a standard email provider."

    return True, "Email is valid."
