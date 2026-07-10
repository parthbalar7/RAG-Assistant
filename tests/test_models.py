"""Test Pydantic model validation."""

import pytest
from pydantic import ValidationError

from api.models import AuthReq, QueryReq, RenameSessionReq, MemoryAddReq


def test_auth_req_valid():
    req = AuthReq(username="testuser", password="pass1234")
    assert req.username == "testuser"


def test_auth_req_short_username():
    with pytest.raises(ValidationError):
        AuthReq(username="ab", password="pass1234")


def test_auth_req_short_password():
    with pytest.raises(ValidationError):
        AuthReq(username="testuser", password="abc")


def test_query_req_valid():
    req = QueryReq(query="What is Python?")
    assert req.use_hybrid is True
    assert req.use_reranking is True
    assert req.use_memory is True


def test_query_req_empty_query():
    with pytest.raises(ValidationError):
        QueryReq(query="")


def test_query_req_long_query():
    with pytest.raises(ValidationError):
        QueryReq(query="x" * 4001)


def test_query_req_all_options():
    req = QueryReq(
        query="test",
        session_id="abc",
        top_k=5,
        use_reranking=False,
        use_hybrid=False,
        use_routing=False,
        use_agent=True,
        use_pageindex=True,
        use_memory=False,
    )
    assert req.use_agent is True
    assert req.use_memory is False


def test_rename_session_req_valid():
    req = RenameSessionReq(title="My Chat")
    assert req.title == "My Chat"


def test_rename_session_req_empty():
    with pytest.raises(ValidationError):
        RenameSessionReq(title="")


def test_memory_add_req_defaults():
    req = MemoryAddReq(content="Remember this fact")
    assert req.memory_type == "fact"
    assert req.importance == 0.7
    assert req.tags == []


def test_memory_add_req_empty_content():
    with pytest.raises(ValidationError):
        MemoryAddReq(content="")
