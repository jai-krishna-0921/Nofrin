"""
agents/sandbox_runner.py

Isolated sandbox execution for document rendering.
MSB_SERVER_URL and MSB_API_KEY are read from environment by the microsandbox
SDK automatically — no kwargs required on Sandbox.create().
"""

from __future__ import annotations

import asyncio
import os
from typing import Final

from microsandbox import Network, Sandbox
from microsandbox.errors import ExecTimeoutError, MicrosandboxError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

_OUTPUT_PATH: Final[str] = "/tmp/out"
_SANDBOX_IMAGE: Final[str] = "python"
_SANDBOX_CPUS: Final[int] = 1
_SANDBOX_MEMORY_MB: Final[int] = 512
_PACKAGE_INSTALL_TIMEOUT_SECS: Final[float] = 120.0
_CODE_EXECUTION_TIMEOUT_SECS: Final[float] = 60.0


class SandboxExecutionError(Exception):
    """Raised when sandbox code execution fails or the sandbox is unreachable."""

    def __init__(
        self,
        message: str,
        exit_code: int | None = None,
        stderr: str = "",
        cause: Exception | None = None,
    ) -> None:
        super().__init__(message)
        self.exit_code = exit_code
        self.stderr = stderr
        self.cause = cause


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type((MicrosandboxError, OSError, ConnectionError)),
    reraise=True,
)
async def execute_in_sandbox(code: str, packages: list[str]) -> bytes:
    """Execute Python code in an isolated, airgapped microsandbox container.

    MSB_SERVER_URL and MSB_API_KEY must be set in the environment; the
    microsandbox SDK reads them automatically — no kwargs required.

    The code MUST write its output to /tmp/out as raw bytes.

    Args:
        code: Python code string. Must write output to /tmp/out.
        packages: pip packages to install before execution.

    Returns:
        Raw bytes read from /tmp/out inside the sandbox.

    Raises:
        SandboxExecutionError: On non-zero exit, timeout, or missing output file.
        MicrosandboxError: On infra failure after 3 retries (re-raised by tenacity).
    """
    if not os.environ.get("MSB_SERVER_URL"):
        raise ValueError("MSB_SERVER_URL environment variable is not set.")
    if not os.environ.get("MSB_API_KEY"):
        raise ValueError("MSB_API_KEY environment variable is not set.")

    sb: Sandbox | None = None
    try:
        sb = await Sandbox.create(
            "docgen",
            image=_SANDBOX_IMAGE,
            cpus=_SANDBOX_CPUS,
            memory=_SANDBOX_MEMORY_MB,
            network=Network.none(),  # airgap: no outbound or inbound connections
        )

        if packages:
            try:
                pip_result = await asyncio.wait_for(
                    sb.exec("pip", ["install", "--quiet"] + packages),
                    timeout=_PACKAGE_INSTALL_TIMEOUT_SECS,
                )
            except asyncio.TimeoutError as e:
                raise SandboxExecutionError(
                    f"pip install timed out after {_PACKAGE_INSTALL_TIMEOUT_SECS}s.",
                    cause=e,
                ) from e
            if pip_result.exit_code != 0:
                raise SandboxExecutionError(
                    f"Package installation failed: {packages}",
                    exit_code=pip_result.exit_code,
                    stderr=pip_result.stderr_text,
                )

        try:
            exec_result = await asyncio.wait_for(
                sb.exec("python", ["-c", code]),
                timeout=_CODE_EXECUTION_TIMEOUT_SECS,
            )
        except asyncio.TimeoutError as e:
            raise SandboxExecutionError(
                f"Code execution timed out after {_CODE_EXECUTION_TIMEOUT_SECS}s.",
                cause=e,
            ) from e

        if exec_result.exit_code != 0:
            raise SandboxExecutionError(
                "Code execution failed",
                exit_code=exec_result.exit_code,
                stderr=exec_result.stderr_text,
            )

        try:
            output_bytes: bytes = await sb.fs.read(_OUTPUT_PATH)
        except Exception as e:
            raise SandboxExecutionError(
                f"Output file not found at {_OUTPUT_PATH}. "
                "Ensure code writes to this path.",
                cause=e,
            ) from e

        return output_bytes

    except SandboxExecutionError:
        raise
    except ExecTimeoutError as e:
        raise SandboxExecutionError("Sandbox execution timed out.", cause=e) from e
    except MicrosandboxError:
        raise  # let tenacity retry on infra errors
    except Exception as e:
        raise SandboxExecutionError(f"Sandbox execution failed: {e}", cause=e) from e
    finally:
        if sb is not None:
            await sb.stop_and_wait()


__all__ = ["SandboxExecutionError", "execute_in_sandbox"]
