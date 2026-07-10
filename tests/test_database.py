"""Test database layer directly."""

from api import database as db
from api.auth import hash_password


def test_create_user():
    u = db.create_user("dbtest_user_1", hash_password("testpass"), "DB Tester")
    assert u["username"] == "dbtest_user_1"
    assert u["display_name"] == "DB Tester"
    assert "id" in u


def test_get_user_by_username():
    db.create_user("dbtest_lookup", hash_password("pass"), "")
    u = db.get_user_by_username("dbtest_lookup")
    assert u is not None
    assert u["username"] == "dbtest_lookup"


def test_get_user_by_username_nonexistent():
    u = db.get_user_by_username("nonexistent_user_xyz")
    assert u is None


def test_create_and_list_sessions():
    u = db.create_user("dbtest_sessions", hash_password("pass"), "")
    s = db.create_session(u["id"], "Test Session")
    assert s["title"] == "Test Session"

    sessions = db.get_user_sessions(u["id"])
    assert any(sess["id"] == s["id"] for sess in sessions)


def test_add_and_get_messages():
    u = db.create_user("dbtest_msgs", hash_password("pass"), "")
    s = db.create_session(u["id"])
    db.add_message(s["id"], "user", "Hello!")
    db.add_message(s["id"], "assistant", "Hi there!", [{"file": "test.py"}])

    msgs = db.get_session_messages(s["id"])
    assert len(msgs) == 2
    assert msgs[0]["role"] == "user"
    assert msgs[0]["content"] == "Hello!"
    assert msgs[1]["role"] == "assistant"


def test_delete_session_cascades():
    u = db.create_user("dbtest_cascade", hash_password("pass"), "")
    s = db.create_session(u["id"])
    db.add_message(s["id"], "user", "test msg")

    db.delete_session(s["id"])
    msgs = db.get_session_messages(s["id"])
    assert len(msgs) == 0


def test_update_session_title():
    u = db.create_user("dbtest_rename", hash_password("pass"), "")
    s = db.create_session(u["id"], "Original")
    db.update_session_title(s["id"], "Updated Title")

    sessions = db.get_user_sessions(u["id"])
    updated = next(sess for sess in sessions if sess["id"] == s["id"])
    assert updated["title"] == "Updated Title"


def test_get_latest_integrity_fingerprints():
    result = db.get_latest_integrity_fingerprints()
    assert isinstance(result, dict)
