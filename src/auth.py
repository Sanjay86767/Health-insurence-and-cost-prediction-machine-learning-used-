"""
Policyholder Authentication, Multi-User Registry, and Credential Management
Persists accounts to data/users.json with password reset support.
"""

import json
from pathlib import Path
import hashlib
from typing import Dict, Any, Optional, Tuple

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
USERS_FILE = DATA_DIR / "users.json"


def _hash_password(password: str) -> str:
    """Create SHA-256 hash of password."""
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


DEFAULT_SEED_USERS = {
    "sanjay86767": {
        "username": "sanjay86767",
        "full_name": "Sanjay Kumar",
        "email": "sanjay@pussgrc.ac.in",
        "password_hash": _hash_password("password123"),
        "policy_id": "PU-INS-2026-98604",
        "tier": "Comprehensive Platinum Care",
        "avatar": "👨‍🔬",
        "age": 42,
        "height": 168,
        "weight": 77,
        "surgeries": 0,
        "diabetes": 0,
        "bp": 1,
        "transplants": 0,
        "chronic": 0,
        "allergies": 0,
        "cancer": 0,
    },
    "ramesh104": {
        "username": "ramesh104",
        "full_name": "Ramesh Chandra",
        "email": "ramesh.chandra@sample.org",
        "password_hash": _hash_password("password123"),
        "policy_id": "PU-INS-2026-104",
        "tier": "High-Risk Chronic Care Tier",
        "avatar": "👴",
        "age": 56,
        "height": 168,
        "weight": 98,
        "surgeries": 1,
        "diabetes": 1,
        "bp": 1,
        "transplants": 0,
        "chronic": 0,
        "allergies": 0,
        "cancer": 0,
    },
    "priya_fit": {
        "username": "priya_fit",
        "full_name": "Priya Sharma",
        "email": "priya.sharma@sample.org",
        "password_hash": _hash_password("password123"),
        "policy_id": "PU-INS-2026-701",
        "tier": "Standard Essential Health Pass",
        "avatar": "👩‍💼",
        "age": 23,
        "height": 172,
        "weight": 64,
        "surgeries": 0,
        "diabetes": 0,
        "bp": 0,
        "transplants": 0,
        "chronic": 0,
        "allergies": 0,
        "cancer": 0,
    },
    "vikram289": {
        "username": "vikram289",
        "full_name": "Vikram Singh",
        "email": "vikram.singh@sample.org",
        "password_hash": _hash_password("password123"),
        "policy_id": "PU-INS-2026-289",
        "tier": "Active Care Gold Plan",
        "avatar": "👨‍💼",
        "age": 48,
        "height": 170,
        "weight": 90,
        "surgeries": 0,
        "diabetes": 0,
        "bp": 0,
        "transplants": 0,
        "chronic": 1,
        "allergies": 0,
        "cancer": 0,
    },
}


def load_all_users() -> Dict[str, Dict[str, Any]]:
    """Load users from data/users.json or initialize with defaults."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not USERS_FILE.exists():
        save_all_users(DEFAULT_SEED_USERS)
        return DEFAULT_SEED_USERS.copy()

    try:
        with open(USERS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            # Ensure seed users exist
            updated = False
            for k, v in DEFAULT_SEED_USERS.items():
                if k not in data:
                    data[k] = v
                    updated = True
            if updated:
                save_all_users(data)
            return data
    except Exception:
        save_all_users(DEFAULT_SEED_USERS)
        return DEFAULT_SEED_USERS.copy()


def save_all_users(users: Dict[str, Dict[str, Any]]) -> None:
    """Save users dictionary to data/users.json."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(USERS_FILE, "w", encoding="utf-8") as f:
        json.dump(users, f, indent=2)


def authenticate_user(username_or_email: str, password: str) -> Tuple[bool, Optional[Dict[str, Any]], str]:
    """Verify credentials and return user profile."""
    users = load_all_users()
    identifier = username_or_email.strip().lower()
    p_hash = _hash_password(password)

    for username, user_data in users.items():
        if username.lower() == identifier or user_data.get("email", "").lower() == identifier:
            if user_data.get("password_hash") == p_hash:
                return True, user_data, "Login successful!"
            else:
                return False, None, "Incorrect password. Please try again or use Forgot Password."

    return False, None, "Account not found. Please register a new policyholder profile."


def register_user(
    username: str,
    full_name: str,
    email: str,
    password: str,
    vitals: Dict[str, Any],
) -> Tuple[bool, Optional[Dict[str, Any]], str]:
    """Register a new policyholder and save to data/users.json."""
    users = load_all_users()
    clean_u = username.strip().lower()
    clean_email = email.strip().lower()

    if not clean_u:
        return False, None, "Username cannot be empty."
    if len(password) < 4:
        return False, None, "Password must be at least 4 characters."
    if clean_u in users:
        return False, None, f"Username '{clean_u}' is already taken. Please choose another."

    for _, udata in users.items():
        if udata.get("email", "").lower() == clean_email:
            return False, None, f"Email '{clean_email}' is already registered."

    # Generate sequential or hashed Policy ID
    count = len(users) + 1
    policy_id = f"PU-INS-2026-{count:04d}"

    bmi = round(vitals["weight"] / ((vitals["height"] / 100) ** 2), 1)
    tier = "Platinum Comprehensive" if bmi < 25 and vitals.get("bp", 0) == 0 else "Active Health Care"

    new_user = {
        "username": clean_u,
        "full_name": full_name.strip(),
        "email": clean_email,
        "password_hash": _hash_password(password),
        "policy_id": policy_id,
        "tier": tier,
        "avatar": "👤",
        "age": int(vitals.get("age", 35)),
        "height": int(vitals.get("height", 170)),
        "weight": int(vitals.get("weight", 70)),
        "surgeries": int(vitals.get("surgeries", 0)),
        "diabetes": int(vitals.get("diabetes", 0)),
        "bp": int(vitals.get("bp", 0)),
        "transplants": int(vitals.get("transplants", 0)),
        "chronic": int(vitals.get("chronic", 0)),
        "allergies": int(vitals.get("allergies", 0)),
        "cancer": int(vitals.get("cancer", 0)),
    }

    users[clean_u] = new_user
    save_all_users(users)
    return True, new_user, f"Welcome {full_name}! Registration successful. Policy ID: {policy_id}"


def reset_password(username_or_email: str, new_password: str) -> Tuple[bool, str]:
    """Reset password for existing policyholder."""
    users = load_all_users()
    identifier = username_or_email.strip().lower()

    if len(new_password) < 4:
        return False, "New password must be at least 4 characters."

    target_user = None
    for username, udata in users.items():
        if username.lower() == identifier or udata.get("email", "").lower() == identifier:
            target_user = username
            break

    if not target_user:
        return False, f"No policyholder found with username or email '{username_or_email}'."

    users[target_user]["password_hash"] = _hash_password(new_password)
    save_all_users(users)
    return True, f"Password reset successfully for {users[target_user]['full_name']}! You can now log in."


def update_user_profile(username: str, updated_fields: Dict[str, Any]) -> Tuple[bool, Optional[Dict[str, Any]], str]:
    """Update profile attributes for active policyholder."""
    users = load_all_users()
    clean_u = username.strip().lower()

    if clean_u not in users:
        return False, None, "User not found."

    users[clean_u].update(updated_fields)
    save_all_users(users)
    return True, users[clean_u], "Profile updated successfully!"
