"""Native review routing preserves the implementation handoff across parking."""

import argparse
import contextlib
import json
import multiprocessing
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc


@pytest.fixture
def store(tmp_path, monkeypatch):
    home = tmp_path / "hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    with kbc.connect() as conn:
        yield conn


def _event_id(conn, task_id):
    return conn.execute(
        "SELECT MAX(id) FROM task_events WHERE task_id = ?", (task_id,),
    ).fetchone()[0]


def _task(conn, task_id):
    task = kb.get_task(conn, task_id)
    assert task is not None
    return task


@pytest.mark.parametrize("park", [False, True])
def test_park_and_reroute_preserves_checkpoint_and_implementer(store, park):
    from hermes_cli.kanban_review_routing import route_review

    conn = store
    task_id = kb.create_task(conn, title="Review implementation", assignee="implementer")
    claimed = kb.claim_task(conn, task_id)
    assert claimed is not None
    summary = "Implementation checkpoint\nFull verification and outstanding review context."
    metadata = {"head_sha": "a" * 40, "checkpoint": {"verified": ["unit tests"], "next": "review"}}
    assert kb.request_review(
        conn, task_id, reviewer="first-reviewer", summary=summary,
        metadata=metadata, expected_run_id=claimed.current_run_id,
    )
    with kb.write_txn(conn):
        conn.execute(
            "UPDATE tasks SET consecutive_failures = 1, block_recurrences = 2, "
            "block_kind = 'capability' WHERE id = ?", (task_id,),
        )
    if park:
        origin = _event_id(conn, task_id)
        assert route_review(conn, task_id, target="scheduled", expected_event_id=origin) == (True, None)
        parked = kb.get_task(conn, task_id)
        assert parked is not None
        assert parked.status == "scheduled"
        assert parked.assignee is None
        marker = kb.list_events(conn, task_id)[-1].payload
        assert marker is not None
        assert marker["resume_status"] == "review"
        assert marker["implementer"] == "implementer"
        assert marker["prior_reviewer"] == "first-reviewer"
        assert marker["origin_review_event_id"] == origin
    assert route_review(
        conn, task_id, target="review", reviewer="Second-Reviewer",
        expected_event_id=_event_id(conn, task_id), metadata={"routing_note": "independent reviewer"},
    ) == (True, None)
    routed = kb.get_task(conn, task_id)
    assert routed is not None
    assert (routed.status, routed.assignee) == ("review", "second-reviewer")
    routed_row = conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()
    for field in ("claim_lock", "claim_expires", "worker_pid", "worker_started_at", "current_run_id"):
        assert routed_row[field] is None
    counters = conn.execute(
        "SELECT consecutive_failures, block_recurrences, block_kind FROM tasks WHERE id = ?",
        (task_id,),
    ).fetchone()
    assert tuple(counters) == (1, 2, "capability")
    handoff = conn.execute(
        "SELECT summary, metadata, ended_at, profile FROM task_runs WHERE task_id = ? ORDER BY id DESC LIMIT 1",
        (task_id,),
    ).fetchone()
    assert handoff["summary"] == summary
    assert json.loads(handoff["metadata"]) == {**metadata, "routing_note": "independent reviewer"}
    assert handoff["ended_at"] is not None
    assert handoff["profile"] == "implementer"
    reviewer = kb.claim_review_task(conn, task_id)
    assert reviewer is not None
    assert kb.request_changes(
        conn, task_id, reason="Add missing edge case", expected_run_id=reviewer.current_run_id,
    ) == (True, "implementer")
    returned = kb.get_task(conn, task_id)
    assert returned is not None
    assert (returned.status, returned.assignee) == ("ready", "implementer")


def test_worker_scheduled_review_resumes_with_original_checkpoint(store):
    from hermes_cli.kanban_review_routing import route_review

    task_id = kb.create_task(store, title="Partial review", assignee="implementer")
    assert kb.request_review(store, task_id, reviewer="first-reviewer", metadata={"head_sha": "b" * 40})
    assert kb.claim_review_task(store, task_id)
    assert kb.schedule_task(store, task_id, reason="Partial review; resume remaining checks")
    assert route_review(
        store, task_id, target="review", reviewer="second-reviewer",
        expected_event_id=_event_id(store, task_id),
    ) == (True, None)
    handoff = kb.latest_run(store, task_id)
    assert handoff is not None
    assert handoff.metadata is not None
    assert handoff.metadata["head_sha"] == "b" * 40
    assert _task(store, task_id).assignee == "second-reviewer"


def _stamp_worker(conn, task_id):
    run_id = _task(conn, task_id).current_run_id
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET worker_pid = 123, worker_started_at = 'fixture-spawn' WHERE id = ?", (task_id,))
        conn.execute("UPDATE task_runs SET worker_pid = 456, worker_started_at = 'fixture-bound' WHERE id = ?", (run_id,))
        kb._append_event(conn, task_id, "spawned", {"pid": 123, "started_at": "fixture-spawn"}, run_id=run_id)
        kb._append_event(conn, task_id, "worker_bound", {"pid": 456, "started_at": "fixture-bound"}, run_id=run_id)
    return run_id


@pytest.mark.parametrize("transition", ["schedule", "block", "request-review"])
def test_worker_handoff_clears_task_birth_but_keeps_run_identity(store, transition):
    task_id = kb.create_task(store, title="Worker handoff", assignee="implementer")
    assert kb.claim_task(store, task_id)
    run_id = _stamp_worker(store, task_id)
    actions = {
        "schedule": lambda: kb.schedule_task(store, task_id, expected_run_id=run_id),
        "block": lambda: kb.block_task(store, task_id, kind="transient", expected_run_id=run_id),
        "request-review": lambda: kb.request_review(store, task_id, reviewer="first-reviewer", expected_run_id=run_id),
    }
    assert actions[transition]()
    assert store.execute("SELECT worker_started_at FROM tasks WHERE id = ?", (task_id,)).fetchone()[0] is None
    assert store.execute("SELECT worker_started_at FROM task_runs WHERE id = ?", (run_id,)).fetchone()[0] == "fixture-bound"


@pytest.mark.parametrize("case", ["retired", "no-result", "wrong-birth", "wrong-result-pid", "live-proxy", "live-bound"])
def test_legacy_birth_requires_ended_run_terminal_receipt_and_dead_workers(store, monkeypatch, case):
    from hermes_cli import kanban_db_dispatch as dispatch
    from hermes_cli.kanban_review_routing import route_review

    task_id = _pending_review(store)
    assert kb.claim_review_task(store, task_id)
    run_id = _stamp_worker(store, task_id)
    assert kb.schedule_task(store, task_id, reason="Partial native review", expected_run_id=run_id)
    with kb.write_txn(store):
        birth = "foreign-birth" if case == "wrong-birth" else "fixture-spawn"
        store.execute("UPDATE tasks SET worker_started_at = ? WHERE id = ?", (birth, task_id))
        store.execute("UPDATE task_runs SET worker_pid = NULL, worker_started_at = NULL WHERE id = ?", (run_id,))
        if case != "no-result":
            kb._append_event(store, task_id, "worker_result", {"pid": 789 if case == "wrong-result-pid" else 456, "exit_code": 0}, run_id=run_id)
    live = {"live-proxy": 123, "live-bound": 456}.get(case)
    monkeypatch.setattr(dispatch, "_worker_alive", lambda pid, _birth: pid == live)
    before = list(store.iterdump())
    ok, _reason = route_review(store, task_id, target="review", reviewer="second-reviewer", expected_event_id=_event_id(store, task_id))
    assert ok is (case == "retired")
    if ok:
        assert store.execute("SELECT worker_started_at FROM tasks WHERE id = ?", (task_id,)).fetchone()[0] is None
        payload = kb.list_events(store, task_id)[-1].payload
        assert payload is not None
        assert payload["retired_worker_started_at"] == "fixture-spawn"
        assert payload["retired_worker_run_id"] == run_id
    else:
        assert list(store.iterdump()) == before


def _pending_review(conn):
    task_id = kb.create_task(conn, title="Pending review", assignee="implementer")
    assert kb.request_review(conn, task_id, reviewer="first-reviewer", summary="Checkpoint")
    return task_id


def _prepare_refusal(conn, task_id, case):
    def implementation_claim():
        assert kb.reopen_review_task(conn, task_id)
        assert kb.claim_task(conn, task_id)

    def review_claim():
        assert kb.claim_review_task(conn, task_id)

    def orphan_run():
        review_claim()
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET status = 'review', claim_lock = NULL, claim_expires = NULL, current_run_id = NULL WHERE id = ?", (task_id,))

    def corrupt_ownership():
        with kb.write_txn(conn):
            run = kb.latest_run(conn, task_id)
            assert run is not None
            value = run.id if case == "current_run_id" else 1
            conn.execute(f"UPDATE tasks SET {case} = ? WHERE id = ?", (value, task_id))

    def corrupt_provenance():
        payload = {} if case == "missing_implementer" else {"implementer": "../invalid"}
        with kb.write_txn(conn):
            conn.execute("UPDATE task_events SET payload = ? WHERE task_id = ? AND kind = 'review_requested'", (json.dumps(payload), task_id))

    def unrelated_schedule():
        assert kb.reopen_review_task(conn, task_id)
        assert kb.schedule_task(conn, task_id, reason="Unrelated implementation delay")

    def stale_event():
        expected = _event_id(conn, task_id)
        kb.add_comment(conn, task_id, "operator", "Changed after observation")
        return expected

    def unfinished_parent():
        parent = kb.create_task(conn, title="Unfinished parent")
        kb.link_tasks(conn, parent, task_id)

    def stale_origin():
        assert kb.reopen_review_task(conn, task_id)
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET status = 'review' WHERE id = ?", (task_id,))

    actions = {
        "running_implementation": implementation_claim, "running_review": review_claim,
        "orphan_run": orphan_run, "unrelated_schedule": unrelated_schedule,
        "stale_event": stale_event, "parents": unfinished_parent, "stale_review_origin": stale_origin,
        **dict.fromkeys(("claim_lock", "claim_expires", "worker_pid", "worker_started_at", "current_run_id"), corrupt_ownership),
        **dict.fromkeys(("missing_implementer", "invalid_implementer"), corrupt_provenance),
    }
    action = actions.get(case)
    return action() if action else None


@pytest.mark.parametrize("case", [
    "running_implementation", "running_review", "claim_lock", "claim_expires", "worker_pid",
    "worker_started_at", "current_run_id", "missing_implementer", "unrelated_schedule",
    "stale_event", "parents", "self_review", "missing_reviewer", "stale_review_origin",
    "invalid_implementer", "invalid_reviewer", "orphan_run",
])
def test_refusals_leave_native_store_unchanged(store, case):
    from hermes_cli.kanban_review_routing import route_review

    conn = store
    task_id = _pending_review(conn)
    reviewer = {"self_review": "IMPLEMENTER", "missing_reviewer": None,
                "invalid_reviewer": "../invalid"}.get(case, "second-reviewer")
    expected = _prepare_refusal(conn, task_id, case)
    before = list(conn.iterdump())
    ok, reason = route_review(
        conn, task_id, target="review", reviewer=reviewer,
        expected_event_id=expected or _event_id(conn, task_id),
    )
    assert not ok, case
    assert reason
    assert list(conn.iterdump()) == before


def test_parked_review_requires_explicit_routing_not_generic_unblock(store):
    from hermes_cli.kanban_review_routing import route_review

    task_id = _pending_review(store)
    assert route_review(store, task_id, target="scheduled", expected_event_id=_event_id(store, task_id))[0]
    before = list(store.iterdump())
    assert kb.unblock_task(store, task_id) is False
    assert list(store.iterdump()) == before
    legacy = kb.create_task(store, title="Implementation delay", assignee="implementer")
    assert kb.schedule_task(store, legacy)
    assert kb.unblock_task(store, legacy)
    assert _task(store, legacy).status == "ready"


def test_cli_route_review_roundtrip_and_mutation_fence(store, monkeypatch, capsys):
    from hermes_cli import kanban as cli
    from agent.delegation_context import DELEGATED_CHILD_ENV_MARKER

    task_id = _pending_review(store)
    parser = argparse.ArgumentParser()
    cli.build_parser(parser.add_subparsers(dest="command"))

    def invoke(target, *extra):
        capsys.readouterr()
        assert cli.kanban_command(parser.parse_args(["kanban", "show", task_id, "--json"])) == 0
        observed = json.loads(capsys.readouterr().out)
        assert observed["latest_event_id"] == _event_id(store, task_id)
        return cli.kanban_command(parser.parse_args([
            "kanban", "route-review", task_id, "--target", target,
            "--expected-event-id", str(observed["latest_event_id"]), *extra,
        ]))

    assert invoke("scheduled", "--summary", "Parked checkpoint", "--metadata", '{"head_sha":"abc"}') == 0
    assert _task(store, task_id).status == "scheduled"
    assert invoke("review", "--reviewer", "next-reviewer") == 0
    assert _task(store, task_id).assignee == "next-reviewer"
    monkeypatch.setenv(DELEGATED_CHILD_ENV_MARKER, str(kb.kanban_home()))
    before = list(store.iterdump())
    assert invoke("scheduled") != 0
    assert list(store.iterdump()) == before


def test_show_event_token_and_task_share_a_read_snapshot(store, monkeypatch, capsys):
    from hermes_cli import kanban as cli
    from hermes_cli.kanban_review_routing import route_review

    task_id = _pending_review(store)
    observed_event = _event_id(store, task_id)
    read_task = kb.get_task

    def read_then_route(conn, tid):
        task = read_task(conn, tid)
        with kbc.connect() as writer:
            assert route_review(writer, tid, target="scheduled", expected_event_id=observed_event)[0]
        return task

    monkeypatch.setattr(kb, "get_task", read_then_route)
    parser = argparse.ArgumentParser()
    cli.build_parser(parser.add_subparsers(dest="command"))
    assert cli.kanban_command(parser.parse_args(["kanban", "show", task_id, "--json"])) == 0
    shown = json.loads(capsys.readouterr().out)
    assert shown["task"]["status"] == "review"
    assert shown["latest_event_id"] == observed_event
    assert _event_id(store, task_id) > observed_event
    assert not route_review(store, task_id, target="review", reviewer="second-reviewer", expected_event_id=observed_event)[0]


def _route_until_boundary(task_id, target, event_id, boundary, pipe, hold):
    from hermes_cli.kanban_review_routing import route_review

    native_txn = kb.write_txn

    @contextlib.contextmanager
    def barrier(conn):
        with native_txn(conn):
            yield conn
            if boundary == "before_commit":
                pipe.send((boundary, conn.in_transaction))
                hold.wait()
        if boundary == "after_commit":
            pipe.send((boundary, conn.in_transaction))
            hold.wait()

    kb.write_txn = barrier
    try:
        with kbc.connect() as conn:
            route_review(conn, task_id, target=target, reviewer="second-reviewer", expected_event_id=event_id)
    except BaseException as exc:
        pipe.send(("error", repr(exc)))
        raise


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("boundary", ["before_commit", "after_commit"])
@pytest.mark.parametrize("target", ["scheduled", "review"])
def test_process_kill_preserves_entire_old_or_final_state(store, monkeypatch, boundary, target):
    from hermes_cli.kanban_review_routing import route_review
    from hermes_cli import kanban_db_dispatch as dispatch

    task_id = _pending_review(store)
    if target == "review":
        assert route_review(store, task_id, target="scheduled", expected_event_id=_event_id(store, task_id))[0]
    before = list(store.iterdump())
    old_status = _task(store, task_id).status
    old_runs = len(kb.list_runs(store, task_id))
    ctx = multiprocessing.get_context("fork")
    receiver, sender = ctx.Pipe(duplex=False)
    process = ctx.Process(target=_route_until_boundary, args=(
        task_id, target, _event_id(store, task_id), boundary, sender, ctx.Event(),
    ))
    process.start()
    sender.close()
    try:
        assert receiver.poll(20), "routing child did not reach the transaction boundary"
        assert receiver.recv() == (boundary, boundary == "before_commit")
        # This is the actual dispatcher's lane selector, on the independent reader.
        assert not dispatch._lane_rows(store, "ready")
        visible = _task(store, task_id)
        assert visible.status == (old_status if boundary == "before_commit" else target)
    finally:
        process.kill()
        process.join(20)
        receiver.close()
    assert not process.is_alive()
    assert process.exitcode == -9
    if boundary == "before_commit":
        assert list(store.iterdump()) == before
    else:
        task = _task(store, task_id)
        assert (task.status, task.assignee) == (target, "second-reviewer" if target == "review" else None)
        assert len(kb.list_runs(store, task_id)) == old_runs + 1
        run = kb.latest_run(store, task_id)
        assert run is not None
        assert run.summary == "Checkpoint"
        assert run.ended_at is not None
        assert kb.list_events(store, task_id)[-1].run_id == run.id
    monkeypatch.setattr(dispatch, "_profile_exists_fn", lambda: lambda name: True)
    monkeypatch.setattr(dispatch, "review_dispatch_enabled", lambda: True)
    monkeypatch.setattr(dispatch, "_memory_pressure_level", lambda: "normal")
    result = dispatch.dispatch_once(store, dry_run=True, reconcile_orphans=False, default_assignee="implementer")
    assert all(profile != "implementer" for _, profile, _ in result.spawned)
    expected_profile = "first-reviewer" if boundary == "before_commit" else "second-reviewer"
    final_status = old_status if boundary == "before_commit" else target
    assert [profile for _, profile, _ in result.spawned] == ([expected_profile] if final_status == "review" else [])
