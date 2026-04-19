# Microsandbox Integration Plan

## Problem Statement

The Deep Research Agent's delivery node currently raises `NotImplementedError` for DOCX, PDF, and PPTX output formats. These formats require document generation libraries (python-docx, python-pptx, WeasyPrint) that have heavy native dependencies and potential security concerns when running user-influenced code. Microsandbox provides isolated container execution to safely render these document formats without polluting the host environment or exposing the main process to dependency conflicts.

## API Facts (verified from v0.3.14 type stubs)

```python
# Exact signatures from _microsandbox.pyi:
async def Sandbox.create(name_or_config: str | dict, **kwargs) -> Sandbox
#   kwargs accepted: image, cpus, memory, network, ...
#   MSB_SERVER_URL and MSB_API_KEY are read from env AUTOMATICALLY — no kwargs needed.

async def sb.exec(cmd: str, args_or_options: list[str] | ExecOptions | None = None) -> ExecOutput
#   ExecOptions(args, cwd, user, env, timeout, stdin, tty, rlimits) — native timeout support.

# ExecOutput properties: .exit_code, .success, .stdout_text, .stderr_text, .stdout_bytes, .stderr_bytes

sb.fs  # -> SandboxFs (property, not async)
async def SandboxFs.read(path: str) -> bytes       # ✓ returns bytes
async def SandboxFs.read_text(path: str) -> str
async def SandboxFs.write(path: str, data: bytes) -> None

async def sb.stop_and_wait() -> tuple[int, bool]

Network.none() -> Network   # airgap — no outbound or inbound connections

# Real exceptions from microsandbox.errors:
ExecTimeoutError   # raised when ExecOptions.timeout exceeded
ExecFailedError    # raised on exec failure (depends on config)
MicrosandboxError  # base class
```

## Approach: Base64-encoded binary output stored in `final_output: str`

`ResearchAgentState.final_output` is typed as `Optional[str]`. Rather than changing the state schema (which would require updates across the entire pipeline), binary document bytes will be base64-encoded with a MIME-type prefix:

```
data:application/vnd.openxmlformats-officedocument.wordprocessingml.document;base64,UEsDB...
```

**Why this approach:**
1. No schema changes — avoids cascading TypedDict updates across graph/state.py, tests, all consumers
2. Self-describing — MIME prefix allows downstream code to detect format and decode
3. Reversible — trivial to decode: split on comma, base64.b64decode the second part

**Alternatives considered:**

| Alternative | Reason rejected |
|-------------|-----------------|
| Change `final_output` to `bytes \| str` | Breaks TypedDict contract; requires updates in router, API, tests |
| Write to temp file, store path | File lifecycle complexity; path may be invalid by consumption time |
| Separate `final_output_bytes: Optional[bytes]` field | Schema churn; unclear which field to read |

## Files

### New Files

| Path | Purpose |
|------|---------|
| `agents/sandbox_runner.py` | Sandbox execution wrapper with `SandboxExecutionError` and `execute_in_sandbox()` |
| `tests/test_sandbox_runner.py` | Unit tests for sandbox_runner with mocked Sandbox |

### Modified Files

| Path | Changes |
|------|---------|
| `agents/delivery.py` | Replace stub `_render_docx`, `_render_pdf`, `_render_pptx` with async sandbox-based implementations |
| `tests/test_delivery.py` | Add tests for binary format rendering (mocked sandbox); update 3 existing NotImplementedError tests |
| `requirements.txt` | Add `microsandbox>=0.3.14`, `tenacity>=8.2` |

### Environment Variables (add to `.env` and `.env.example`)

```bash
# Microsandbox — read automatically by microsandbox SDK (no kwargs needed)
MSB_SERVER_URL=http://localhost:5555
MSB_API_KEY=your-api-key-here
```

---

## Code Snippets

### 1. `agents/sandbox_runner.py` — Full API Shape

```python
"""
agents/sandbox_runner.py

Isolated sandbox execution for document rendering.
MSB_SERVER_URL and MSB_API_KEY are read from env by microsandbox SDK automatically.
"""
from __future__ import annotations

import os
from typing import Final

from microsandbox import Network, Sandbox
from microsandbox.errors import ExecTimeoutError, MicrosandboxError
from microsandbox.types import ExecOptions
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


def _is_transient(exc: BaseException) -> bool:
    """Return True for errors worth retrying (connection, infra), False for logic errors."""
    # SandboxExecutionError wrapping a non-zero exit is a code/logic error — don't retry.
    if isinstance(exc, SandboxExecutionError) and exc.exit_code is not None:
        return False
    # MicrosandboxError without an exit_code is infra (server unreachable, OOM, etc.)
    return isinstance(exc, (MicrosandboxError, OSError, ConnectionError))


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
        SandboxExecutionError: On non-zero exit, timeout, or missing output.
        MicrosandboxError: On infra failure after 3 retries (re-raised by tenacity).
    """
    sb: Sandbox | None = None
    try:
        sb = await Sandbox.create(
            "docgen",
            image=_SANDBOX_IMAGE,
            cpus=_SANDBOX_CPUS,
            memory=_SANDBOX_MEMORY_MB,
            network=Network.none(),   # airgap: no outbound or inbound connections
        )

        if packages:
            pip_opts = ExecOptions(
                args=["install", "--quiet"] + packages,
                timeout=_PACKAGE_INSTALL_TIMEOUT_SECS,
            )
            pip_result = await sb.exec("pip", pip_opts)
            if pip_result.exit_code != 0:
                raise SandboxExecutionError(
                    f"Package installation failed: {packages}",
                    exit_code=pip_result.exit_code,
                    stderr=pip_result.stderr_text,
                )

        exec_opts = ExecOptions(
            args=["-c", code],
            timeout=_CODE_EXECUTION_TIMEOUT_SECS,
        )
        exec_result = await sb.exec("python", exec_opts)
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
        raise SandboxExecutionError(
            "Sandbox execution timed out.", cause=e
        ) from e
    except MicrosandboxError:
        raise  # let tenacity retry on infra errors
    except Exception as e:
        raise SandboxExecutionError(
            f"Sandbox execution failed: {e}", cause=e
        ) from e
    finally:
        if sb is not None:
            await sb.stop_and_wait()


__all__ = ["SandboxExecutionError", "execute_in_sandbox"]
```

**Notes on retry strategy:**
- Retries only on `MicrosandboxError`, `OSError`, `ConnectionError` (infra/transient)
- Does NOT retry on `SandboxExecutionError` with `exit_code != None` (code logic failures)
- Does NOT retry on `ExecTimeoutError` (60s timeout hit — not worth retrying)
- Max 3 attempts, exponential backoff 1s → 10s

### 2. `agents/delivery.py` — Changes

**New imports:**
```python
import base64
import json

from agents.sandbox_runner import SandboxExecutionError, execute_in_sandbox
```

**New module-level constant (typed, no bare dict):**
```python
_MIME_TYPES: dict[str, str] = {
    "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "pdf": "application/pdf",
    "pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
}
```

**New pure helper:**
```python
def _encode_binary_output(data: bytes, format_key: str) -> str:
    """Encode binary document bytes as a data URI string."""
    mime = _MIME_TYPES[format_key]
    encoded = base64.b64encode(data).decode("ascii")
    return f"data:{mime};base64,{encoded}"
```

**`_render_docx` shape (stubs become `async def`):**
```python
async def _render_docx(
    synthesis: SynthesisOutput,
    critic_output: CriticOutput | None,
    is_force_delivered: bool,
) -> str:
    # repr(json.dumps(...)) embeds synthesis data as a Python string literal.
    # json.dumps normalizes the data; repr() wraps it in quotes and escapes
    # any characters that could break the outer f-string. Synthesis content
    # originates from the coordinator LLM — not raw user input — but repr()
    # ensures no synthesis text can escape the string boundary regardless.
    doc_data_repr = repr(json.dumps({
        "topic": synthesis.topic,
        "executive_summary": synthesis.executive_summary,
        "findings": [{"heading": f.heading, "body": f.body} for f in synthesis.findings],
        "risks": synthesis.risks,
        "gaps": synthesis.gaps,
        "citations": [
            {"title": c.source_title, "url": c.source_url}
            for c in synthesis.citations
        ],
        "is_force_delivered": is_force_delivered,
        "quality_score": critic_output.final_quality_score if critic_output else None,
    }))
    code = f"""
import json
from docx import Document
data = json.loads({doc_data_repr})
doc = Document()
doc.add_heading(data["topic"], level=0)
# ... full document build (headings, findings, risks, gaps, citations) ...
doc.save("/tmp/out")
"""
    docx_bytes = await execute_in_sandbox(code=code, packages=["python-docx", "lxml"])
    return _encode_binary_output(docx_bytes, "docx")
```

`_render_pdf` and `_render_pptx` follow the same pattern (WeasyPrint / python-pptx).

**`delivery_node` — add `await` to three renderers (already `async def`, no signature change):**
```python
elif output_format == "docx":
    final_output = await _render_docx(synthesis, critic_output, force_delivered)
elif output_format == "pdf":
    final_output = await _render_pdf(synthesis, critic_output, force_delivered)
elif output_format == "pptx":
    final_output = await _render_pptx(synthesis, critic_output, force_delivered)
```

---

## Tests

### `tests/test_sandbox_runner.py` (9 tests)

| Test | What it covers |
|------|---------------|
| `test_success_returns_bytes` | Mock Sandbox; verify fs.read bytes returned |
| `test_package_install_failure_raises` | pip exit_code=1 → SandboxExecutionError (no retry) |
| `test_code_execution_failure_raises` | python exit_code=1 → SandboxExecutionError (no retry) |
| `test_missing_output_file_raises` | fs.read raises → SandboxExecutionError "not found" |
| `test_cleanup_called_on_success` | stop_and_wait() called after success |
| `test_cleanup_called_on_failure` | stop_and_wait() called even on failure |
| `test_missing_env_vars_raises_valueerror` | Unset MSB_SERVER_URL → ValueError (before Sandbox.create) |
| `test_exec_timeout_raises_sandbox_execution_error` | ExecTimeoutError → SandboxExecutionError |
| `test_pip_install_timeout_raises_sandbox_execution_error` | ExecTimeoutError during pip → SandboxExecutionError |

**Mock pattern:**
```python
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

@pytest.fixture
def mock_exec_output() -> MagicMock:
    out = MagicMock()
    out.exit_code = 0
    out.success = True
    out.stdout_text = ""
    out.stderr_text = ""
    return out

@pytest.fixture
def mock_sandbox(mock_exec_output: MagicMock) -> AsyncMock:
    sb = AsyncMock()
    sb.exec = AsyncMock(return_value=mock_exec_output)
    sb.fs = MagicMock()
    sb.fs.read = AsyncMock(return_value=b"fake-bytes")
    sb.stop_and_wait = AsyncMock()
    return sb
```

### Updates to `tests/test_delivery.py` (5 new tests)

| Test | What it covers |
|------|---------------|
| `test_render_docx_returns_data_uri` | Mock execute_in_sandbox; output starts with correct MIME prefix |
| `test_render_docx_base64_decodable` | Base64 portion decodes without exception |
| `test_render_pdf_returns_data_uri` | Same for PDF MIME type |
| `test_render_pptx_returns_data_uri` | Same for PPTX MIME type |
| `test_render_docx_sandbox_error_propagates` | SandboxExecutionError bubbles through delivery_node |

Update 3 existing `test_output_format_*_raises_not_implemented` tests to expect success (mock execute_in_sandbox returning bytes).

---

## Failure Modes

| Mode | Detection | Mitigation |
|------|-----------|------------|
| Sandbox startup failure | MicrosandboxError from Sandbox.create() | tenacity retries 3× with backoff; reraises |
| Package install failure | pip exit_code != 0 | Check before proceeding; SandboxExecutionError |
| pip install timeout | ExecTimeoutError (ExecOptions.timeout=120s) | → SandboxExecutionError; not retried |
| Code produces no output | fs.read() raises (PathNotFoundError) | → SandboxExecutionError "Output file not found" |
| Code exec timeout | ExecTimeoutError (ExecOptions.timeout=60s) | → SandboxExecutionError; not retried |
| Server unreachable | ConnectionError / OSError | tenacity retries 3×; reraises as MicrosandboxError |
| Memory exhaustion | Container OOM → non-zero exit_code | Caught as code execution failure; 512MB default |
| Cleanup skipped | Exception before finally | finally block is unconditional |
| Data exfiltration | Outbound network call from sandbox code | `network=Network.none()` blocks all connections |

---

## Future-Proofing: LLM-Generated Code (Design Note Only)

`execute_in_sandbox()` accepts arbitrary code strings. When LLM-generated code is later passed in:
1. `network=Network.none()` is already enforced — prevents data exfiltration
2. Output path is fixed at `/tmp/out` — code cannot write elsewhere by convention
3. CPU/memory/time limits enforced at container level
4. For high-security: pre-scan code for prohibited imports (os, subprocess, socket)
5. Log all executed code strings to LangSmith for audit trail

---

## Configuration

| Variable | Required | How used |
|----------|----------|----------|
| `MSB_SERVER_URL` | Yes | Read automatically by microsandbox SDK from env |
| `MSB_API_KEY` | Yes | Read automatically by microsandbox SDK from env |

Add to `requirements.txt`:
```
microsandbox>=0.3.14
tenacity>=8.2
```

Constants in `agents/sandbox_runner.py`:

| Constant | Value | Purpose |
|----------|-------|---------|
| `_OUTPUT_PATH` | `/tmp/out` | Fixed output file location in sandbox |
| `_SANDBOX_IMAGE` | `python` | Base container image |
| `_SANDBOX_CPUS` | `1` | CPU allocation |
| `_SANDBOX_MEMORY_MB` | `512` | Memory limit (MB) |
| `_PACKAGE_INSTALL_TIMEOUT_SECS` | `120.0` | pip install timeout via ExecOptions |
| `_CODE_EXECUTION_TIMEOUT_SECS` | `60.0` | code exec timeout via ExecOptions |

---

## Token Cost Impact

**Zero additional LLM calls.** All document rendering runs in containers. No change to `COST_CEILING_USD`.

---

## Implementation Checklist

- [ ] Create `agents/sandbox_runner.py`
- [ ] Create `tests/test_sandbox_runner.py` (9 tests)
- [ ] Update `agents/delivery.py` — replace stubs with async renderers, add imports + helpers
- [ ] Update `tests/test_delivery.py` — 5 new tests, update 3 existing
- [ ] Add `microsandbox>=0.3.14` and `tenacity>=8.2` to `requirements.txt`
- [ ] Add `MSB_SERVER_URL` and `MSB_API_KEY` to `.env.example`
- [ ] Run `mypy agents/ --strict`
- [ ] Run `pytest tests/ -v`
