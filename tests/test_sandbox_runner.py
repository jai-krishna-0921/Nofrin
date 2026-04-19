"""
tests/test_sandbox_runner.py

Unit tests for agents/sandbox_runner.py.

Tests sandbox execution: happy path, package install failure, code execution
failure, missing output file, cleanup on success/failure, missing env vars,
timeout scenarios (both pip and code exec).
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agents.sandbox_runner import SandboxExecutionError, execute_in_sandbox


# ---------------------------------------------------------------------------
# Test fixtures and helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Set required environment variables for sandbox."""
    monkeypatch.setenv("MSB_SERVER_URL", "http://localhost:5555")
    monkeypatch.setenv("MSB_API_KEY", "test-key")


def make_mock_sandbox(
    pip_exit_code: int = 0,
    pip_stderr: str = "",
    python_exit_code: int = 0,
    python_stderr: str = "",
    output_bytes: bytes | None = b"fake-bytes",
    read_raises: Exception | None = None,
) -> AsyncMock:
    """Create a mock Sandbox instance with configurable behavior."""
    mock_sb = AsyncMock()
    
    # Mock exec results
    mock_pip_result = MagicMock()
    mock_pip_result.exit_code = pip_exit_code
    mock_pip_result.stderr_text = pip_stderr
    
    mock_python_result = MagicMock()
    mock_python_result.exit_code = python_exit_code
    mock_python_result.stderr_text = python_stderr
    
    # exec returns different results based on command
    async def exec_side_effect(cmd: str, args: list[str]) -> MagicMock:
        if cmd == "pip":
            return mock_pip_result
        elif cmd == "python":
            return mock_python_result
        raise ValueError(f"Unexpected command: {cmd}")
    
    mock_sb.exec.side_effect = exec_side_effect
    
    # Mock file system read
    if read_raises is not None:
        mock_sb.fs.read.side_effect = read_raises
    else:
        mock_sb.fs.read.return_value = output_bytes
    
    # Mock stop_and_wait
    mock_sb.stop_and_wait = AsyncMock()
    
    return mock_sb


# ---------------------------------------------------------------------------
# Test suite
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_success_returns_bytes(mock_env: None) -> None:
    """Test #1: happy path — sandbox returns bytes from /tmp/out."""
    mock_sb = make_mock_sandbox()
    
    with patch("agents.sandbox_runner.Sandbox") as MockSandbox:
        MockSandbox.create = AsyncMock(return_value=mock_sb)
        
        result = await execute_in_sandbox(
            code='with open("/tmp/out", "wb") as f: f.write(b"hello")',
            packages=["python-docx"],
        )
    
    assert result == b"fake-bytes"
    assert mock_sb.stop_and_wait.await_count == 1
    MockSandbox.create.assert_awaited_once()


@pytest.mark.asyncio
async def test_package_install_failure_raises(mock_env: None) -> None:
    """Test #2: pip install failure raises SandboxExecutionError, NOT retried."""
    mock_sb = make_mock_sandbox(pip_exit_code=1, pip_stderr="package not found")
    
    with patch("agents.sandbox_runner.Sandbox") as MockSandbox:
        MockSandbox.create = AsyncMock(return_value=mock_sb)
        
        with pytest.raises(SandboxExecutionError) as exc_info:
            await execute_in_sandbox(
                code='print("hello")',
                packages=["nonexistent-package"],
            )
    
    # SandboxExecutionError is NOT retried — create called exactly once
    MockSandbox.create.assert_awaited_once()
    assert exc_info.value.exit_code == 1
    assert "Package installation failed" in str(exc_info.value)
    assert mock_sb.stop_and_wait.await_count == 1


@pytest.mark.asyncio
async def test_code_execution_failure_raises(mock_env: None) -> None:
    """Test #3: python exec failure raises SandboxExecutionError."""
    mock_sb = make_mock_sandbox(python_exit_code=1, python_stderr="syntax error")
    
    with patch("agents.sandbox_runner.Sandbox") as MockSandbox:
        MockSandbox.create = AsyncMock(return_value=mock_sb)
        
        with pytest.raises(SandboxExecutionError) as exc_info:
            await execute_in_sandbox(
                code='invalid python syntax',
                packages=["python-docx"],
            )
    
    MockSandbox.create.assert_awaited_once()
    assert exc_info.value.exit_code == 1
    assert "Code execution failed" in str(exc_info.value)
    assert mock_sb.stop_and_wait.await_count == 1


@pytest.mark.asyncio
async def test_missing_output_file_raises(mock_env: None) -> None:
    """Test #4: fs.read raises → SandboxExecutionError with 'not found'."""
    mock_sb = make_mock_sandbox(read_raises=FileNotFoundError("file not found"))
    
    with patch("agents.sandbox_runner.Sandbox") as MockSandbox:
        MockSandbox.create = AsyncMock(return_value=mock_sb)
        
        with pytest.raises(SandboxExecutionError) as exc_info:
            await execute_in_sandbox(
                code='print("no file written")',
                packages=[],
            )
    
    assert "not found" in str(exc_info.value)
    assert mock_sb.stop_and_wait.await_count == 1


@pytest.mark.asyncio
async def test_cleanup_called_on_success(mock_env: None) -> None:
    """Test #5: stop_and_wait called exactly once on success."""
    mock_sb = make_mock_sandbox()
    
    with patch("agents.sandbox_runner.Sandbox") as MockSandbox:
        MockSandbox.create = AsyncMock(return_value=mock_sb)
        
        await execute_in_sandbox(code='print("ok")', packages=[])
    
    assert mock_sb.stop_and_wait.await_count == 1


@pytest.mark.asyncio
async def test_cleanup_called_on_failure(mock_env: None) -> None:
    """Test #6: stop_and_wait called even when python exec fails."""
    mock_sb = make_mock_sandbox(python_exit_code=1)
    
    with patch("agents.sandbox_runner.Sandbox") as MockSandbox:
        MockSandbox.create = AsyncMock(return_value=mock_sb)
        
        with pytest.raises(SandboxExecutionError):
            await execute_in_sandbox(code='raise ValueError()', packages=[])
    
    # Cleanup still happens
    assert mock_sb.stop_and_wait.await_count == 1


@pytest.mark.asyncio
async def test_sandbox_create_called_regardless_of_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test #7: MSB_SERVER_URL/MSB_API_KEY are optional — SDK reads them itself.

    The SDK works in local mode without server env vars; Sandbox.create is
    always attempted (not gated by our code).
    """
    monkeypatch.delenv("MSB_SERVER_URL", raising=False)
    monkeypatch.delenv("MSB_API_KEY", raising=False)
    mock_sb = make_mock_sandbox()

    with patch("agents.sandbox_runner.Sandbox") as MockSandbox:
        MockSandbox.create = AsyncMock(return_value=mock_sb)
        result = await execute_in_sandbox(code="pass", packages=[])
        MockSandbox.create.assert_called_once()
        assert result == b"fake-bytes"


@pytest.mark.asyncio
async def test_exec_timeout_raises_sandbox_execution_error(mock_env: None) -> None:
    """Test #8: asyncio.TimeoutError during python exec → SandboxExecutionError."""
    mock_sb = make_mock_sandbox()
    
    # Mock wait_for to raise TimeoutError on second call (python exec)
    call_count = 0

    async def wait_for_side_effect(coro: Any, timeout: float) -> Any:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            # First call is pip install — succeed
            return await coro
        else:
            # Second call is python exec — timeout
            raise asyncio.TimeoutError()
    
    with patch("agents.sandbox_runner.Sandbox") as MockSandbox, \
         patch("agents.sandbox_runner.asyncio.wait_for", side_effect=wait_for_side_effect):
        MockSandbox.create = AsyncMock(return_value=mock_sb)
        
        with pytest.raises(SandboxExecutionError) as exc_info:
            await execute_in_sandbox(code='import time; time.sleep(999)', packages=["pkg"])
    
    assert "timed out" in str(exc_info.value)
    assert mock_sb.stop_and_wait.await_count == 1


@pytest.mark.asyncio
async def test_pip_install_timeout_raises_sandbox_execution_error(mock_env: None) -> None:
    """Test #9: asyncio.TimeoutError during pip install → SandboxExecutionError."""
    mock_sb = make_mock_sandbox()
    
    async def wait_for_side_effect(coro: Any, timeout: float) -> Any:
        # First call is pip install — timeout immediately
        raise asyncio.TimeoutError()
    
    with patch("agents.sandbox_runner.Sandbox") as MockSandbox, \
         patch("agents.sandbox_runner.asyncio.wait_for", side_effect=wait_for_side_effect):
        MockSandbox.create = AsyncMock(return_value=mock_sb)
        
        with pytest.raises(SandboxExecutionError) as exc_info:
            await execute_in_sandbox(code='print("ok")', packages=["huge-package"])
    
    assert "timed out" in str(exc_info.value)
    assert mock_sb.stop_and_wait.await_count == 1
