"""Interruption-atomic routing of pending reviews; never implementation rework.

All task, checkpoint-run and provenance writes share one native transaction.
No filesystem staging, worker termination or other pre-commit side effects.
"""

from __future__ import annotations

import sqlite3


def _canonical_profile(value):
    from hermes_cli.profiles import normalize_profile_name, validate_profile_name

    if value is None:
        return None
    try:
        canonical = normalize_profile_name(value)
        validate_profile_name(canonical)
        return canonical
    except ValueError:
        return None


def _review_origin(conn, task_id):
    from hermes_cli import kanban_db as kb

    event = conn.execute(
        "SELECT id, run_id, payload FROM task_events "
        "WHERE task_id = ? AND kind = 'review_requested' ORDER BY id DESC LIMIT 1",
        (task_id,),
    ).fetchone()
    if event is None:
        return None, {}, None
    payload = kb._json_dict(event["payload"])
    implementer = kb._nonblank_str(payload.get("implementer"))
    return event, payload, _canonical_profile(implementer)


def _checkpoint(conn, task_id, run_id):
    from hermes_cli import kanban_db as kb

    row = conn.execute(
        "SELECT summary, metadata FROM task_runs WHERE id = ? AND task_id = ? AND ended_at IS NOT NULL",
        (run_id, task_id),
    ).fetchone()
    return (row["summary"], kb._json_dict(row["metadata"])) if row else (None, {})


def _inherited_checkpoint(conn, task_id, origin_run, checkpoint_run):
    original_summary, original_metadata = _checkpoint(conn, task_id, origin_run)
    summary, metadata = _checkpoint(conn, task_id, checkpoint_run)
    return (original_summary if summary is None else summary,
            {**original_metadata, **metadata})


def _worker_review_parking(conn, task_id, origin, phase):
    """Recognize native worker scheduling by its review claim, never by reason text."""
    from hermes_cli import kanban_db as kb


    claim = conn.execute(
        "SELECT payload FROM task_events WHERE task_id = ? AND run_id = ? "
        "AND kind = 'claimed' AND id > ? AND id < ? ORDER BY id DESC LIMIT 1",
        (task_id, phase["run_id"], origin["id"], phase["id"]),
    ).fetchone()
    if claim is None or kb._json_dict(claim["payload"]).get("source_status") != "review":
        return None
    run = conn.execute(
        "SELECT profile FROM task_runs WHERE task_id = ? AND id = ? AND ended_at IS NOT NULL",
        (task_id, phase["run_id"]),
    ).fetchone()
    reviewer = _canonical_profile(run["profile"]) if run else None
    if not reviewer:
        return None
    return phase["run_id"], reviewer


def _routing_source(conn, task_id, status, origin, payload, implementer):
    from hermes_cli import kanban_db as kb

    phase = conn.execute(
        "SELECT id, kind, payload, run_id FROM task_events WHERE task_id = ? AND kind IN "
        "('review_requested', 'review_reopened', 'changes_requested', 'scheduled', 'completed', 'status') "
        "ORDER BY id DESC LIMIT 1", (task_id,),
    ).fetchone()
    if status == "review":
        if phase is None or phase["kind"] != "review_requested":
            return None
        return origin["run_id"], payload.get("reviewer")
    parking = kb._json_dict(phase["payload"]) if phase else {}
    if phase is None or phase["kind"] != "scheduled":
        return None
    if "resume_status" not in parking:
        return _worker_review_parking(conn, task_id, origin, phase)
    if (parking.get("resume_status") != "review"
            or parking.get("origin_review_event_id") != origin["id"]
            or parking.get("implementer") != implementer):
        return None
    return phase["run_id"], parking.get("prior_reviewer")


def _ownership_problem(conn, task):
    if any(task[key] is not None for key in (
        "claim_lock", "claim_expires", "worker_pid", "current_run_id",
    )):
        return "task has inconsistent ownership"
    if conn.execute(
        "SELECT 1 FROM task_runs WHERE task_id = ? AND ended_at IS NULL LIMIT 1", (task["id"],),
    ).fetchone():
        return "task has an open run"
    return None


def _retired_worker_birth(conn, task, run_id):
    """Prove legacy debris using the closed phase's exit receipt and process identities."""
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_dispatch as dispatch

    birth = task["worker_started_at"]
    if birth is None:
        return True
    if not birth or birth == dispatch.UNVERIFIED_WORKER_FINGERPRINT:
        return False
    if not conn.execute(
        "SELECT 1 FROM task_runs WHERE task_id = ? AND id = ? AND ended_at IS NOT NULL",
        (task["id"], run_id),
    ).fetchone():
        return False
    receipt = conn.execute(
        "SELECT id, payload FROM task_events WHERE task_id = ? AND run_id = ? "
        "AND kind = 'worker_result' ORDER BY id DESC LIMIT 1", (task["id"], run_id),
    ).fetchone()
    result = kb._json_dict(receipt["payload"]) if receipt else {}
    if type(result.get("exit_code")) is not int:
        return False
    events = conn.execute(
        "SELECT payload FROM task_events WHERE task_id = ? AND run_id = ? "
        "AND kind IN ('spawned', 'worker_bound') AND id < ? ORDER BY id",
        (task["id"], run_id, receipt["id"]),
    ).fetchall()
    workers = [kb._json_dict(event["payload"]) for event in events]
    if not any(worker.get("started_at") == birth for worker in workers):
        return False
    if not any(worker.get("pid") == result.get("pid") for worker in workers):
        return False
    for worker in workers:
        pid, started_at = worker.get("pid"), worker.get("started_at")
        if type(pid) is not int or pid < 1 or not started_at or started_at == dispatch.UNVERIFIED_WORKER_FINGERPRINT:
            return False
        if dispatch._worker_alive(pid, started_at):
            return False
    return True


def route_review(
    conn: sqlite3.Connection, task_id: str, *, target: str, expected_event_id: int,
    reviewer: str | None = None, summary: str | None = None, metadata: dict | None = None,
) -> tuple[bool, str | None]:
    """Route pending Review or explicitly parked review to Review/Scheduled.

    ``expected_event_id`` is the latest *task* event (including comments), not
    just the review event. Scheduled is an indefinite, unassigned parking lane;
    resumption requires an explicit reviewer distinct from the implementer.
    Metadata overlays prior checkpoint fields; omitted summary is inherited.
    Refusals write nothing. Nested transactions are intentionally forbidden.
    """
    from hermes_cli import kanban_db as kb

    if target not in {"review", "scheduled"}:
        return False, "target must be review or scheduled"
    if type(expected_event_id) is not int or expected_event_id < 1:
        return False, "expected_event_id must be a positive integer"
    if metadata is not None and not isinstance(metadata, dict):
        return False, "metadata must be an object"
    reviewer = _canonical_profile(reviewer)
    with kb.write_txn(conn):
        task = conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()
        if task is None:
            return False, "task not found"
        if task["status"] not in {"review", "scheduled"}:
            return False, "task is not a pending or parked review"
        ownership_problem = _ownership_problem(conn, task)
        if ownership_problem:
            return False, ownership_problem
        latest_id = conn.execute(
            "SELECT MAX(id) FROM task_events WHERE task_id = ?", (task_id,),
        ).fetchone()[0]
        if latest_id != expected_event_id:
            return False, "task event changed"
        origin, payload, implementer = _review_origin(conn, task_id)
        if origin is None or implementer is None:
            return False, "review handoff has no valid implementer provenance"
        source = _routing_source(conn, task_id, task["status"], origin, payload, implementer)
        if source is None:
            return False, "task has no current pending/parked-review provenance"
        checkpoint_run, prior_reviewer = source
        if not _retired_worker_birth(conn, task, checkpoint_run):
            return False, "task has unproven worker birth metadata"
        if target == "review":
            if not reviewer or reviewer == implementer:
                return False, "an explicit reviewer distinct from the implementer is required"
            if not kb._parents_satisfied(conn, task_id):
                return False, "parent dependencies are not satisfied"
        prior_summary, prior_metadata = _inherited_checkpoint(conn, task_id, origin["run_id"], checkpoint_run)
        summary = kb.redact_review_value(prior_summary if summary is None else summary)
        metadata = kb.redact_review_value({**prior_metadata, **(metadata or {})})
        conn.execute(
            "UPDATE tasks SET status = ?, assignee = ?, claim_lock = NULL, claim_expires = NULL, "
            "worker_pid = NULL, worker_started_at = NULL, current_run_id = NULL WHERE id = ?",
            (target, reviewer if target == "review" else None, task_id),
        )
        run_id = kb._synthesize_ended_run(
            conn, task_id, outcome="review_requested" if target == "review" else "scheduled",
            summary=summary, metadata=metadata, profile=implementer,
        )
        event_payload = {
            "summary": kb._first_line(summary, 400) or None,
            "implementer": implementer, "reviewer": reviewer if target == "review" else None,
            "prior_reviewer": prior_reviewer, "origin_review_event_id": origin["id"],
            "resume_status": "review",
        }
        if task["worker_started_at"] is not None:
            event_payload["retired_worker_started_at"] = task["worker_started_at"]
            event_payload["retired_worker_run_id"] = checkpoint_run
        kb._append_event(
            conn, task_id, "review_requested" if target == "review" else "scheduled",
            event_payload, run_id=run_id,
        )
    return True, None


def _cmd_route_review(args):
    import os
    from hermes_cli import kanban as cli

    if os.environ.get("HERMES_KANBAN_TASK"):
        return cli._err("kanban route-review is orchestrator-only")
    metadata, rc = cli._parse_metadata_flag(args.metadata)
    if rc:
        return rc
    with cli.kbc.connect_closing() as conn:
        ok, reason = route_review(
            conn, args.task_id, target=args.target, expected_event_id=args.expected_event_id,
            reviewer=args.reviewer, summary=args.summary, metadata=metadata,
        )
    return cli._ok_or_err(ok, f"cannot route review for {args.task_id}: {reason}",
                          f"Routed {args.task_id} → {args.target}")
