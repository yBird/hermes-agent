"""Remote dispatch must not replay a cell whose acknowledgement was lost."""
import json
from unittest.mock import Mock

import pytest

from tools import code_execution_tool as execution
from tools import code_kernel_remote as remote


@pytest.fixture
def remote_env(monkeypatch):
    env = Mock()
    env.execute.return_value = {"output": "OK\n", "returncode": 0}
    monkeypatch.setattr(execution, "_get_or_create_env", lambda task: (env, "ssh"))
    monkeypatch.setattr(execution, "_load_config", lambda: {"timeout": 30, "max_tool_calls": 5})
    return env


@pytest.mark.parametrize("failure", [OSError, RuntimeError, TimeoutError])
@pytest.mark.parametrize("reset", [False, True])
def test_committed_cell_exception_never_replays(tmp_path, monkeypatch, remote_env, failure, reset, caplog):
    counter = tmp_path / "counter"
    code = f"with open({str(counter)!r}, 'a') as stream:\n    stream.write('x')\n"
    secret = "exception-detail-must-not-leak"

    def commit_then_lose_result(source, **kwargs):
        exec(source, {})
        raise failure(secret + source)

    def replay(env, env_type, source, *args, **kwargs):
        exec(source, {})
        return json.dumps({"status": "success"})

    kernel = Mock(side_effect=commit_then_lose_result)
    fallback = Mock(side_effect=replay)
    monkeypatch.setattr(remote, "execute_in_remote_kernel", kernel)
    monkeypatch.setattr(execution, "_run_remote_per_call", fallback)
    result = json.loads(execution._execute_remote(code, "task", ["read_file"], reset=reset))

    assert counter.read_text() == "x"
    fallback.assert_not_called()
    kernel.assert_called_once()
    assert kernel.call_args.kwargs["reset"] is reset
    assert result["status"] == "error"
    assert result["outcome_uncertain"] is True
    assert result["tool_calls_made"] is None
    assert result["retry_policy"] == "never automatically replay"
    assert result["tool_calls_log"] == []
    assert "DO NOT REPLAY" in result["error"]
    assert isinstance(result["duration_seconds"], (int, float))
    assert result["duration_seconds"] >= 0
    assert secret not in json.dumps(result)
    assert code not in json.dumps(result)
    assert secret not in caplog.text
    assert code not in caplog.text


@pytest.mark.parametrize("outcome", ["unavailable", "success", "error", "timeout", "interrupt"])
@pytest.mark.parametrize("reset", [False, True])
def test_safe_unavailability_and_known_results_preserved(monkeypatch, remote_env, outcome, reset):
    fallback = Mock(return_value=json.dumps({"status": "success", "output": "fallback"}))
    monkeypatch.setattr(execution, "_run_remote_per_call", fallback)
    info = {"remote": True, "reused": False, "state_reset": reset}
    raw = {"status": outcome, "stdout": "kernel output", "tool_calls_made": 2,
           "kernel": info, "error": "known error"}
    acquire = Mock(return_value=(None, False, False, False))
    dispatch = Mock(side_effect=AssertionError("unavailable kernel dispatched a cell"))
    kernel = Mock(return_value=raw)
    if outcome == "unavailable":
        # Exercise the real helper's only None return: acquisition failed BEFORE dispatch.
        monkeypatch.setattr(remote, "_acquire_remote_kernel", acquire)
        monkeypatch.setattr(remote, "_run_attached_cell", dispatch)
    else:
        if outcome == "interrupt":
            kernel.side_effect = KeyboardInterrupt
        monkeypatch.setattr(remote, "execute_in_remote_kernel", kernel)

    if outcome == "interrupt":
        with pytest.raises(KeyboardInterrupt):
            execution._execute_remote("print(1)", "task", ["read_file"], reset=reset)
        fallback.assert_not_called()
        return
    result = json.loads(execution._execute_remote("print(1)", "task", ["read_file"], reset=reset))
    if outcome == "unavailable":
        acquire.assert_called_once()
        assert acquire.call_args.kwargs["reset"] is reset
        dispatch.assert_not_called()
        fallback.assert_called_once()
        assert result["output"] == "fallback"
        return
    fallback.assert_not_called()
    assert kernel.call_args.kwargs["reset"] is reset
    assert result["status"] == outcome
    assert result["tool_calls_made"] == 2
    assert result["kernel"] == info
    assert "kernel output" in result["output"]
    assert "outcome_uncertain" not in result
    if outcome == "timeout":
        assert "timed out after 30s" in result["error"]
        assert "state was lost" in result["output"]
    elif outcome == "error":
        assert result["error"] == "known error"
