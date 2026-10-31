"""
Unit tests for multi-user authentication, registration, and password reset.
"""

import pytest
from src.auth import (
    authenticate_user,
    load_all_users,
    register_user,
    reset_password,
    update_user_profile,
)


def test_seed_users_exist():
    users = load_all_users()
    assert "sanjay86767" in users
    assert users["sanjay86767"]["full_name"] == "Sanjay Kumar"
    assert "PU-INS-2026" in users["sanjay86767"]["policy_id"]


def test_authenticate_success():
    ok, user, msg = authenticate_user("sanjay86767", "password123")
    assert ok is True
    assert user is not None
    assert user["full_name"] == "Sanjay Kumar"


def test_authenticate_wrong_password():
    ok, user, msg = authenticate_user("sanjay86767", "wrongpass")
    assert ok is False
    assert user is None
    assert "Incorrect password" in msg


def test_authenticate_not_found():
    ok, user, msg = authenticate_user("non_existent_person_999", "password123")
    assert ok is False
    assert user is None
    assert "not found" in msg.lower()


def test_register_and_reset_workflow():
    import uuid
    rand_id = uuid.uuid4().hex[:6]
    test_u = f"test_applicant_{rand_id}"
    test_name = "Rajesh Sharma"
    test_email = f"rajesh_{rand_id}@healthai.in"
    vitals = {
        "age": 38,
        "height": 175,
        "weight": 76,
        "surgeries": 0,
        "diabetes": 0,
        "bp": 1,
        "transplants": 0,
        "chronic": 0,
        "allergies": 0,
        "cancer": 0,
    }

    # Register
    ok, udata, msg = register_user(
        username=test_u,
        full_name=test_name,
        email=test_email,
        password="oldpassword123",
        vitals=vitals,
    )
    assert ok is True
    assert udata["full_name"] == test_name
    assert "PU-INS-2026" in udata["policy_id"]

    # Authenticate with old password
    auth_ok, _, _ = authenticate_user(test_u, "oldpassword123")
    assert auth_ok is True

    # Reset Password
    reset_ok, reset_msg = reset_password(test_u, "newpassword456")
    assert reset_ok is True
    assert "successfully" in reset_msg.lower()

    # Old password should now fail
    old_fail, _, _ = authenticate_user(test_u, "oldpassword123")
    assert old_fail is False

    # New password should succeed
    new_ok, logged_in, _ = authenticate_user(test_u, "newpassword456")
    assert new_ok is True
    assert logged_in["full_name"] == test_name

    # Clean up test user
    from src.auth import save_all_users
    all_u = load_all_users()
    if test_u in all_u:
        del all_u[test_u]
        save_all_users(all_u)
