"""
tools/file_ops.py — CYRAX 3.0 File System Tools (Patched)

Patches applied vs. v1:
  1. Root-targeting guard added to DeleteFileTool, MoveFileTool, RenameFileTool.
     Resolving to CYRAX_WORKSPACE itself is now a hard block — no
     recursive deletion or infinite-loop move of the sandbox root.
  2. CopyFileTool now handles directories via shutil.copytree()
     with dirs_exist_ok=overwrite, matching MoveFileTool's capability.

Tools:
    ReadFileTool    — Reads text content from a sandboxed file
    WriteFileTool   — Writes or appends text to a sandboxed file
    DeleteFileTool  — Permanently deletes a file or empty dir from sandbox
    CopyFileTool    — Copies a file or directory within the sandbox
    MoveFileTool    — Moves a file or directory within the sandbox
    RenameFileTool  — Renames a file within the sandbox
    ListDirTool     — Lists contents of a sandbox directory
    MakeDirTool     — Creates a directory within the sandbox

Sandbox:
    ALL operations are strictly confined to CYRAX_WORKSPACE.
    Path traversal attacks are blocked by resolve_secure_path().
    The workspace root itself is blocked as a modification target.
    No user input is ever passed to shell=True.

Security levels:
    ReadFileTool    — USER
    WriteFileTool   — USER
    DeleteFileTool  — ADMIN
    CopyFileTool    — USER
    MoveFileTool    — ADMIN
    RenameFileTool  — USER
    ListDirTool     — USER
    MakeDirTool     — USER
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

from pydantic import BaseModel, Field

from security.auth import SecurityLevel
from tools.registry import BaseTool

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# SANDBOX
# ══════════════════════════════════════════════════════════════════════════════

CYRAX_WORKSPACE: Path = Path.home() / "CYRAX_WORKSPACE"
CYRAX_WORKSPACE.mkdir(parents=True, exist_ok=True)

# Resolved once at module load — used in root-targeting guard comparisons.
_WORKSPACE_RESOLVED: Path = CYRAX_WORKSPACE.resolve()

# Maximum file size readable in a single operation (5 MB).
_MAX_READ_BYTES: int = 5 * 1024 * 1024

# Maximum characters writable per WriteFileTool call (1 MB of text).
_MAX_WRITE_CHARS: int = 1_000_000


def resolve_secure_path(user_path: str) -> Path:
    """
    Resolves a user-supplied path relative to CYRAX_WORKSPACE and
    verifies the result is strictly inside the sandbox.

    Raises:
        PermissionError: If the resolved path escapes the workspace.
        ValueError:      If the path component is empty or invalid.

    Never passes user_path to a shell. All resolution is done by
    pathlib — the OS never interprets the raw string as a command.
    """
    if not user_path or not user_path.strip():
        raise ValueError("Path cannot be empty.")

    # Strip leading separators so "/etc/passwd" becomes "etc/passwd"
    # and is resolved relative to the workspace, not the filesystem root.
    clean = user_path.lstrip("/\\").strip()

    if not clean:
        raise ValueError("Path resolves to an empty string after sanitisation.")

    resolved = (CYRAX_WORKSPACE / clean).resolve()

    try:
        resolved.relative_to(_WORKSPACE_RESOLVED)
    except ValueError:
        raise PermissionError(
            f"Path traversal blocked: '{user_path}' resolves outside "
            f"the CYRAX sandbox ({CYRAX_WORKSPACE})."
        )

    return resolved


def _guard_workspace_root(target: Path, operation: str) -> str | None:
    """
    Returns an error string if target is the workspace root directory.
    Returns None if the path is safe to proceed with.

    Prevents:
      - DeleteFileTool from recursively wiping the entire sandbox.
      - MoveFileTool from causing infinite-loop moves of the root.
      - RenameFileTool from renaming the sandbox root itself.
    """
    if target == _WORKSPACE_RESOLVED:
        return (
            f"Error: Cannot {operation} the workspace root directory. "
            f"Only files and subdirectories inside the workspace may be targeted."
        )
    return None


# ══════════════════════════════════════════════════════════════════════════════
# TOOL: READ FILE
# ══════════════════════════════════════════════════════════════════════════════

class ReadFileSchema(BaseModel):
    filepath: str = Field(
        ...,
        description="Relative path to the file inside the CYRAX workspace.",
    )
    encoding: str = Field(
        default="utf-8",
        description="Text encoding. Defaults to utf-8.",
    )


class ReadFileTool(BaseTool):
    name           = "READ_FILE"
    description    = (
        "Reads and returns the text content of a file inside the CYRAX workspace."
    )
    security_level = SecurityLevel.USER
    args_schema    = ReadFileSchema

    def execute(self, filepath: str, encoding: str = "utf-8") -> str:  # type: ignore[override]
        try:
            target = resolve_secure_path(filepath)
        except (PermissionError, ValueError) as exc:
            return f"Error: {exc}"

        if not target.exists():
            return f"Error: File '{filepath}' does not exist in the workspace."

        if not target.is_file():
            return f"Error: '{filepath}' is a directory, not a file."

        file_size = target.stat().st_size
        if file_size > _MAX_READ_BYTES:
            return (
                f"Error: File '{filepath}' is too large to read "
                f"({file_size / 1024 / 1024:.1f} MB). "
                f"Maximum allowed: {_MAX_READ_BYTES // 1024 // 1024} MB."
            )

        try:
            content = target.read_text(encoding=encoding, errors="replace")
            logger.info(f"[READ_FILE] Read {len(content)} chars from '{filepath}'.")
            return content if content else f"Success: '{filepath}' is empty."
        except LookupError:
            return f"Error: Unknown encoding '{encoding}'."
        except OSError as exc:
            logger.error(f"[READ_FILE] OS error reading '{filepath}': {exc}")
            return f"Error: Could not read '{filepath}' — {exc}"
        except Exception as exc:
            logger.exception(f"[READ_FILE] Unexpected error: {exc}")
            return f"Error: Unexpected error reading '{filepath}'."


# ══════════════════════════════════════════════════════════════════════════════
# TOOL: WRITE FILE
# ══════════════════════════════════════════════════════════════════════════════

class WriteFileSchema(BaseModel):
    filepath: str = Field(
        ...,
        description="Relative path of the file to write inside the CYRAX workspace.",
    )
    content: str = Field(
        ...,
        description="Text content to write to the file.",
    )
    append: bool = Field(
        default=False,
        description="If True, appends to the file instead of overwriting it.",
    )
    encoding: str = Field(
        default="utf-8",
        description="Text encoding. Defaults to utf-8.",
    )


class WriteFileTool(BaseTool):
    name           = "WRITE_FILE"
    description    = (
        "Writes text content to a file inside the CYRAX workspace. "
        "Creates the file and any missing parent directories if they do not exist."
    )
    security_level = SecurityLevel.USER
    args_schema    = WriteFileSchema

    def execute(  # type: ignore[override]
        self,
        filepath: str,
        content:  str,
        append:   bool = False,
        encoding: str  = "utf-8",
    ) -> str:
        try:
            target = resolve_secure_path(filepath)
        except (PermissionError, ValueError) as exc:
            return f"Error: {exc}"

        if len(content) > _MAX_WRITE_CHARS:
            return (
                f"Error: Content is too large to write "
                f"({len(content):,} chars). "
                f"Maximum allowed: {_MAX_WRITE_CHARS:,} chars."
            )

        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            mode = "a" if append else "w"
            with open(target, mode, encoding=encoding) as f:
                f.write(content)
            action = "Appended to" if append else "Written to"
            logger.info(
                f"[WRITE_FILE] {action} '{filepath}' ({len(content):,} chars)."
            )
            return f"Success: {action} '{filepath}' ({len(content):,} characters)."
        except LookupError:
            return f"Error: Unknown encoding '{encoding}'."
        except OSError as exc:
            logger.error(f"[WRITE_FILE] OS error writing '{filepath}': {exc}")
            return f"Error: Could not write '{filepath}' — {exc}"
        except Exception as exc:
            logger.exception(f"[WRITE_FILE] Unexpected error: {exc}")
            return f"Error: Unexpected error writing '{filepath}'."


# ══════════════════════════════════════════════════════════════════════════════
# TOOL: DELETE FILE
# ══════════════════════════════════════════════════════════════════════════════

class DeleteFileSchema(BaseModel):
    filepath: str = Field(
        ...,
        description=(
            "Relative path of the file or empty directory to delete "
            "inside the CYRAX workspace."
        ),
    )


class DeleteFileTool(BaseTool):
    name           = "DELETE_FILE"
    description    = (
        "Permanently deletes a file or empty directory from the CYRAX workspace. "
        "This action is irreversible."
    )
    security_level = SecurityLevel.ADMIN
    args_schema    = DeleteFileSchema

    def execute(self, filepath: str) -> str:  # type: ignore[override]
        try:
            target = resolve_secure_path(filepath)
        except (PermissionError, ValueError) as exc:
            return f"Error: {exc}"

        # ── Patch 1: Root-targeting guard ─────────────────────────────────────
        root_err = _guard_workspace_root(target, "delete")
        if root_err:
            return root_err

        if not target.exists():
            return f"Error: '{filepath}' does not exist in the workspace."

        try:
            if target.is_file() or target.is_symlink():
                target.unlink()
                logger.warning(f"[DELETE_FILE] Deleted file: '{filepath}'.")
                return f"Success: File '{filepath}' deleted."

            if target.is_dir():
                if any(target.iterdir()):
                    return (
                        f"Error: '{filepath}' is a non-empty directory. "
                        f"Empty it first, or use a recursive delete command."
                    )
                target.rmdir()
                logger.warning(
                    f"[DELETE_FILE] Deleted empty directory: '{filepath}'."
                )
                return f"Success: Empty directory '{filepath}' deleted."

            return f"Error: '{filepath}' is not a file or directory."

        except OSError as exc:
            logger.error(f"[DELETE_FILE] OS error deleting '{filepath}': {exc}")
            return f"Error: Could not delete '{filepath}' — {exc}"
        except Exception as exc:
            logger.exception(f"[DELETE_FILE] Unexpected error: {exc}")
            return f"Error: Unexpected error deleting '{filepath}'."


# ══════════════════════════════════════════════════════════════════════════════
# TOOL: COPY FILE OR DIRECTORY
# ══════════════════════════════════════════════════════════════════════════════

class CopyFileSchema(BaseModel):
    source: str = Field(
        ...,
        description="Relative path of the source file or directory inside the workspace.",
    )
    destination: str = Field(
        ...,
        description="Relative path of the destination inside the workspace.",
    )
    overwrite: bool = Field(
        default=False,
        description=(
            "If True, overwrites an existing destination file. "
            "For directories, merges into the existing destination tree."
        ),
    )


class CopyFileTool(BaseTool):
    name           = "COPY_FILE"
    description    = (
        "Copies a file or directory within the CYRAX workspace. "
        "For directories, recursively copies all contents."
    )
    security_level = SecurityLevel.USER
    args_schema    = CopyFileSchema

    def execute(  # type: ignore[override]
        self,
        source:      str,
        destination: str,
        overwrite:   bool = False,
    ) -> str:
        try:
            src  = resolve_secure_path(source)
            dest = resolve_secure_path(destination)
        except (PermissionError, ValueError) as exc:
            return f"Error: {exc}"

        if not src.exists():
            return f"Error: Source '{source}' does not exist."

        try:
            dest.parent.mkdir(parents=True, exist_ok=True)

            # ── Patch 2: Directory copy support ───────────────────────────────
            if src.is_dir():
                if dest.exists() and not overwrite:
                    return (
                        f"Error: Destination '{destination}' already exists. "
                        f"Set overwrite=true to merge into it."
                    )
                shutil.copytree(src, dest, dirs_exist_ok=overwrite)
                logger.info(
                    f"[COPY_FILE] Directory '{source}' → '{destination}'."
                )
                return f"Success: Directory '{source}' copied to '{destination}'."

            # File copy
            if dest.exists() and not overwrite:
                return (
                    f"Error: Destination '{destination}' already exists. "
                    f"Set overwrite=true to replace it."
                )
            shutil.copy2(src, dest)
            logger.info(f"[COPY_FILE] File '{source}' → '{destination}'.")
            return f"Success: Copied '{source}' to '{destination}'."

        except OSError as exc:
            logger.error(f"[COPY_FILE] OS error: {exc}")
            return f"Error: Copy failed — {exc}"
        except Exception as exc:
            logger.exception(f"[COPY_FILE] Unexpected error: {exc}")
            return f"Error: Unexpected error during copy."


# ══════════════════════════════════════════════════════════════════════════════
# TOOL: MOVE FILE
# ══════════════════════════════════════════════════════════════════════════════

class MoveFileSchema(BaseModel):
    source: str = Field(
        ...,
        description="Relative path of the file or directory to move inside the workspace.",
    )
    destination: str = Field(
        ...,
        description="Relative destination path inside the workspace.",
    )
    overwrite: bool = Field(
        default=False,
        description="If True, overwrites an existing destination file.",
    )


class MoveFileTool(BaseTool):
    name           = "MOVE_FILE"
    description    = (
        "Moves a file or directory to a new location within the CYRAX workspace. "
        "The source path is removed after a successful move."
    )
    security_level = SecurityLevel.ADMIN
    args_schema    = MoveFileSchema

    def execute(  # type: ignore[override]
        self,
        source:      str,
        destination: str,
        overwrite:   bool = False,
    ) -> str:
        try:
            src  = resolve_secure_path(source)
            dest = resolve_secure_path(destination)
        except (PermissionError, ValueError) as exc:
            return f"Error: {exc}"

        # ── Patch 1: Root-targeting guard ─────────────────────────────────────
        root_err = _guard_workspace_root(src, "move")
        if root_err:
            return root_err

        if not src.exists():
            return f"Error: Source '{source}' does not exist."

        if dest.exists() and not overwrite:
            return (
                f"Error: Destination '{destination}' already exists. "
                f"Set overwrite=true to replace it."
            )

        try:
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dest))
            logger.warning(f"[MOVE_FILE] '{source}' → '{destination}'.")
            return f"Success: Moved '{source}' to '{destination}'."
        except OSError as exc:
            logger.error(f"[MOVE_FILE] OS error: {exc}")
            return f"Error: Move failed — {exc}"
        except Exception as exc:
            logger.exception(f"[MOVE_FILE] Unexpected error: {exc}")
            return f"Error: Unexpected error during move."


# ══════════════════════════════════════════════════════════════════════════════
# TOOL: RENAME FILE
# ══════════════════════════════════════════════════════════════════════════════

class RenameFileSchema(BaseModel):
    filepath: str = Field(
        ...,
        description="Relative path of the file to rename inside the workspace.",
    )
    new_name: str = Field(
        ...,
        description=(
            "New filename only — not a path. "
            "Must not contain directory separators."
        ),
    )


class RenameFileTool(BaseTool):
    name           = "RENAME_FILE"
    description    = "Renames a file in its current directory within the CYRAX workspace."
    security_level = SecurityLevel.USER
    args_schema    = RenameFileSchema

    def execute(self, filepath: str, new_name: str) -> str:  # type: ignore[override]
        if "/" in new_name or "\\" in new_name:
            return (
                "Error: 'new_name' must be a plain filename, not a path. "
                "Use MOVE_FILE to relocate a file."
            )

        if not new_name.strip():
            return "Error: 'new_name' cannot be empty."

        try:
            target = resolve_secure_path(filepath)
        except (PermissionError, ValueError) as exc:
            return f"Error: {exc}"

        # ── Patch 1: Root-targeting guard ─────────────────────────────────────
        root_err = _guard_workspace_root(target, "rename")
        if root_err:
            return root_err

        if not target.exists():
            return f"Error: '{filepath}' does not exist in the workspace."

        new_path = target.with_name(new_name.strip())

        # Verify the renamed path also stays inside the sandbox.
        try:
            new_path.resolve().relative_to(_WORKSPACE_RESOLVED)
        except ValueError:
            return "Error: Rename would place the file outside the sandbox."

        if new_path.exists():
            return (
                f"Error: A file named '{new_name}' already exists "
                f"in the same directory."
            )

        try:
            target.rename(new_path)
            logger.info(f"[RENAME_FILE] '{filepath}' → '{new_name}'.")
            return f"Success: Renamed '{filepath}' to '{new_name}'."
        except OSError as exc:
            logger.error(f"[RENAME_FILE] OS error: {exc}")
            return f"Error: Rename failed — {exc}"
        except Exception as exc:
            logger.exception(f"[RENAME_FILE] Unexpected error: {exc}")
            return f"Error: Unexpected error during rename."


# ══════════════════════════════════════════════════════════════════════════════
# TOOL: LIST DIRECTORY
# ══════════════════════════════════════════════════════════════════════════════

class ListDirSchema(BaseModel):
    dirpath: str = Field(
        default=".",
        description=(
            "Relative path of the directory to list. "
            "Defaults to '.' (the workspace root)."
        ),
    )
    show_hidden: bool = Field(
        default=False,
        description="If True, includes hidden files (names starting with '.').",
    )


class ListDirTool(BaseTool):
    name           = "LIST_DIR"
    description    = (
        "Lists the contents of a directory inside the CYRAX workspace."
    )
    security_level = SecurityLevel.USER
    args_schema    = ListDirSchema

    def execute(  # type: ignore[override]
        self,
        dirpath:     str  = ".",
        show_hidden: bool = False,
    ) -> str:
        try:
            target = resolve_secure_path(dirpath)
        except (PermissionError, ValueError) as exc:
            return f"Error: {exc}"

        if not target.exists():
            return f"Error: Directory '{dirpath}' does not exist in the workspace."

        if not target.is_dir():
            return f"Error: '{dirpath}' is a file, not a directory."

        try:
            entries = sorted(
                target.iterdir(),
                key=lambda p: (p.is_file(), p.name.lower()),
            )
        except PermissionError:
            return f"Error: Permission denied reading directory '{dirpath}'."
        except OSError as exc:
            return f"Error: Could not list '{dirpath}' — {exc}"

        if not show_hidden:
            entries = [e for e in entries if not e.name.startswith(".")]

        if not entries:
            return f"Directory '{dirpath}' is empty."

        lines: list[str] = [f"Contents of '{dirpath}':"]
        for entry in entries:
            try:
                if entry.is_dir():
                    lines.append(f"  📁 {entry.name}/")
                else:
                    size     = entry.stat().st_size
                    size_str = _format_size(size)
                    lines.append(f"  📄 {entry.name}  ({size_str})")
            except OSError:
                lines.append(f"  ❓ {entry.name}  (unreadable)")

        lines.append(f"\n{len(entries)} item(s) total.")
        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# TOOL: MAKE DIRECTORY
# ══════════════════════════════════════════════════════════════════════════════

class MakeDirSchema(BaseModel):
    dirpath: str = Field(
        ...,
        description=(
            "Relative path of the directory to create inside the CYRAX workspace. "
            "Intermediate parent directories are created automatically."
        ),
    )


class MakeDirTool(BaseTool):
    name           = "MAKE_DIR"
    description    = (
        "Creates a directory (and any missing parent directories) "
        "inside the CYRAX workspace."
    )
    security_level = SecurityLevel.USER
    args_schema    = MakeDirSchema

    def execute(self, dirpath: str) -> str:  # type: ignore[override]
        try:
            target = resolve_secure_path(dirpath)
        except (PermissionError, ValueError) as exc:
            return f"Error: {exc}"

        if target.exists():
            if target.is_dir():
                return f"Success: Directory '{dirpath}' already exists."
            return (
                f"Error: '{dirpath}' exists but is a file, not a directory."
            )

        try:
            target.mkdir(parents=True, exist_ok=True)
            logger.info(f"[MAKE_DIR] Created directory: '{dirpath}'.")
            return f"Success: Directory '{dirpath}' created."
        except OSError as exc:
            logger.error(f"[MAKE_DIR] OS error: {exc}")
            return f"Error: Could not create directory '{dirpath}' — {exc}"
        except Exception as exc:
            logger.exception(f"[MAKE_DIR] Unexpected error: {exc}")
            return f"Error: Unexpected error creating directory."


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _format_size(size_bytes: int) -> str:
    """Formats a byte count into a human-readable string."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    if size_bytes < 1024 ** 2:
        return f"{size_bytes / 1024:.1f} KB"
    if size_bytes < 1024 ** 3:
        return f"{size_bytes / 1024 ** 2:.1f} MB"
    return f"{size_bytes / 1024 ** 3:.1f} GB"