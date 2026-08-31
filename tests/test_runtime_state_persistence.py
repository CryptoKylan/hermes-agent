"""Focused tests for the provider-neutral AgentRuntime v1 persistence seam."""

from __future__ import annotations

import math

import pytest

from agent.runtime_api import RuntimeStateEnvelope, RuntimeUsageReceipt
from hermes_state import SessionDB


def _receipt(*, correlation_id: str | None = "turn-1") -> RuntimeUsageReceipt:
    return RuntimeUsageReceipt(
        runtime_id="example-runtime",
        provider="example-provider",
        model="example-model",
        billing_mode="subscription",
        cost_status="known",
        input_tokens=10,
        output_tokens=4,
        cache_read_tokens=2,
        cache_write_tokens=1,
        reasoning_tokens=3,
        replay_safe=True,
        correlation_id=correlation_id,
    )


def test_fresh_schema_contains_runtime_tables(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        tables = {
            row[0]
            for row in db._conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }
        assert {"runtime_session_state", "runtime_usage_receipts"} <= tables

        state_columns = {
            row[1]
            for row in db._conn.execute(
                "PRAGMA table_info(runtime_session_state)"
            ).fetchall()
        }
        receipt_columns = {
            row[1]
            for row in db._conn.execute(
                "PRAGMA table_info(runtime_usage_receipts)"
            ).fetchall()
        }
        assert state_columns == {
            "session_id",
            "runtime_id",
            "schema_version",
            "state_json",
            "updated_at",
        }
        assert {
            "id",
            "session_id",
            "runtime_id",
            "provider",
            "model",
            "billing_mode",
            "cost_status",
            "input_tokens",
            "output_tokens",
            "cache_read_tokens",
            "cache_write_tokens",
            "reasoning_tokens",
            "replay_safe",
            "correlation_id",
            "recorded_at",
        } <= receipt_columns
    finally:
        db.close()


def test_runtime_state_update_and_read_are_scoped_by_session_and_runtime(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        db.create_session("session-a", source="cli")
        first = RuntimeStateEnvelope(
            runtime_id="example-runtime",
            schema_version=1,
            state={"sdk_session_id": "synthetic-sdk-session", "attempt": 1},
        )
        db.update_runtime_state("session-a", first)
        assert db.get_runtime_state("session-a", "example-runtime") == first
        assert db.get_runtime_state("session-a", "other-runtime") is None

        second = RuntimeStateEnvelope(
            runtime_id="example-runtime",
            schema_version=2,
            state={"attempt": 2, "nested": {"ready": True}},
        )
        db.update_runtime_state("session-a", second)
        assert db.get_runtime_state("session-a", "example-runtime") == second
        assert (
            db._conn.execute(
                "SELECT COUNT(*) FROM runtime_session_state WHERE session_id = ?",
                ("session-a",),
            ).fetchone()[0]
            == 1
        )
    finally:
        db.close()


@pytest.mark.parametrize(
    "state",
    [
        ["state-must-be-an-object"],
        {"access_token": "synthetic-secret-placeholder"},
        {"value": math.nan},
    ],
)
def test_runtime_state_rejects_unsafe_payloads(tmp_path, state):
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        db.create_session("session-a", source="cli")
        with pytest.raises(ValueError):
            db.update_runtime_state(
                "session-a",
                RuntimeStateEnvelope(
                    runtime_id="example-runtime",
                    schema_version=1,
                    state=state,
                ),
            )
        assert (
            db._conn.execute(
                "SELECT COUNT(*) FROM runtime_session_state"
            ).fetchone()[0]
            == 0
        )
    finally:
        db.close()


def test_runtime_usage_receipts_are_append_only_and_correlated_retries_are_idempotent(
    tmp_path,
):
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        db.create_session("session-a", source="cli")
        original = _receipt()
        assert db.record_runtime_usage_receipt("session-a", original) is True
        session_before = db.get_session("session-a")
        assert session_before["input_tokens"] == 0
        assert session_before["output_tokens"] == 0
        assert (
            db._conn.execute(
                "SELECT COUNT(*) FROM session_model_usage WHERE session_id = ?",
                ("session-a",),
            ).fetchone()[0]
            == 0
        )

        changed_retry = RuntimeUsageReceipt(
            **{**original.__dict__, "output_tokens": 999}
        )
        assert db.record_runtime_usage_receipt("session-a", changed_retry) is False
        assert db.list_runtime_usage_receipts("session-a") == [original]

        without_correlation = _receipt(correlation_id=None)
        assert db.record_runtime_usage_receipt("session-a", without_correlation) is True
        assert db.record_runtime_usage_receipt("session-a", without_correlation) is True
        assert len(db.list_runtime_usage_receipts("session-a")) == 3
        assert len(db.list_runtime_usage_receipts("session-a", "example-runtime")) == 3
    finally:
        db.close()


def test_runtime_state_and_receipts_are_inert_across_reopen(tmp_path):
    db_path = tmp_path / "state.db"
    db = SessionDB(db_path=db_path)
    db.create_session("session-a", source="cli")
    state = RuntimeStateEnvelope(
        runtime_id="example-runtime",
        schema_version=1,
        state={"retained": True},
    )
    receipt = _receipt()
    db.update_runtime_state("session-a", state)
    db.record_runtime_usage_receipt("session-a", receipt)
    db.close()

    reopened = SessionDB(db_path=db_path)
    try:
        assert reopened.get_runtime_state("session-a", "example-runtime") == state
        assert reopened.list_runtime_usage_receipts("session-a") == [receipt]
        assert (
            reopened._conn.execute(
                "SELECT COUNT(*) FROM runtime_usage_receipts"
            ).fetchone()[0]
            == 1
        )
    finally:
        reopened.close()
