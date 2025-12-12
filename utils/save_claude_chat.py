#!/usr/bin/env python3
"""
Save and restore Claude Code chats to/from an archive directory.

Usage:
    # Save the most recent chat
    python utils/save_claude_chat.py --chat-name my-chat-name

    # Restore from archive (copies back to .claude/projects)
    python utils/save_claude_chat.py --restore my-chat-name

    # Restore and automatically resume
    python utils/save_claude_chat.py --restore my-chat-name --exec
"""

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path


def get_most_recent_chat(claude_dir: Path) -> Path:
    """Find the most recent chat file (excluding agent logs)."""
    # Get all .jsonl files that aren't agent logs
    chat_files = [
        f for f in claude_dir.glob("*.jsonl")
        if not f.name.startswith("agent-")
    ]

    if not chat_files:
        raise FileNotFoundError(f"No chat files found in {claude_dir}")

    # Sort by modification time (most recent first)
    most_recent = max(chat_files, key=lambda f: f.stat().st_mtime)
    return most_recent


def save_claude_chat(chat_name: str, source_dir: Path, dest_base: Path, include_agents: bool = False) -> str:
    """Copy the most recent chat to the archive directory. Returns session ID."""
    # Find most recent chat
    most_recent = get_most_recent_chat(source_dir)
    session_id = most_recent.stem  # UUID without .jsonl extension

    # Create destination directory
    dest_dir = dest_base / chat_name
    dest_dir.mkdir(parents=True, exist_ok=True)

    # Copy main chat file
    dest_file = dest_dir / most_recent.name
    shutil.copy2(most_recent, dest_file)

    print(f"✓ Saved chat to: {dest_file}")
    print(f"  Session ID: {session_id}")

    # Optionally copy agent files that reference this session
    if include_agents:
        agent_files = [f for f in source_dir.glob("agent-*.jsonl")]
        copied_agents = []

        for agent_file in agent_files:
            # Read first line to check if it references this session
            try:
                with open(agent_file) as f:
                    first_line = f.readline()
                    data = json.loads(first_line)
                    if data.get("sessionId") == session_id:
                        dest_agent = dest_dir / agent_file.name
                        shutil.copy2(agent_file, dest_agent)
                        copied_agents.append(agent_file.name)
            except (json.JSONDecodeError, KeyError):
                continue

        if copied_agents:
            print(f"  + {len(copied_agents)} agent files: {', '.join(copied_agents)}")
        else:
            print("  (no agent files found for this session)")

    print(f"\nTo restore later: python utils/save_claude_chat.py --restore {chat_name}")
    return session_id


def restore_chat(chat_name: str, source_base: Path, dest_dir: Path, exec_resume: bool = False) -> str:
    """Restore a chat from archive back to .claude/projects. Returns session ID."""
    # Find chat directory in archive
    chat_dir = source_base / chat_name
    if not chat_dir.exists():
        raise FileNotFoundError(f"Chat archive not found: {chat_dir}")

    # Find chat file (should be only one non-agent file)
    chat_files = [f for f in chat_dir.glob("*.jsonl") if not f.name.startswith("agent-")]
    if not chat_files:
        raise FileNotFoundError(f"No chat file found in {chat_dir}")
    if len(chat_files) > 1:
        print(f"Warning: Multiple chat files found, using most recent")

    chat_file = max(chat_files, key=lambda f: f.stat().st_mtime)
    session_id = chat_file.stem

    # Copy main chat file
    dest_file = dest_dir / chat_file.name
    if dest_file.exists():
        print(f"Warning: Chat already exists in projects directory, overwriting...")
    shutil.copy2(chat_file, dest_file)

    print(f"✓ Restored chat from: {chat_dir}")
    print(f"  Session ID: {session_id}")

    # Copy agent files
    agent_files = [f for f in chat_dir.glob("agent-*.jsonl")]
    if agent_files:
        for agent_file in agent_files:
            dest_agent = dest_dir / agent_file.name
            shutil.copy2(agent_file, dest_agent)
        print(f"  + {len(agent_files)} agent files")

    # Print or execute resume command
    resume_cmd = f"claude -r {session_id}"
    if exec_resume:
        print(f"\nExecuting: {resume_cmd}")
        subprocess.run(resume_cmd, shell=True)
    else:
        print(f"\nTo resume: {resume_cmd}")

    return session_id


def main():
    parser = argparse.ArgumentParser(
        description="Save and restore Claude Code chats to/from an archive directory"
    )
    parser.add_argument(
        "--chat-name",
        help="Name for the chat (creates a subdirectory with this name)"
    )
    parser.add_argument(
        "--restore",
        metavar="CHAT_NAME",
        help="Restore a chat from archive (specify the chat name to restore)"
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path.home() / ".claude/projects/-Users-ewern-Desktop-code-trait-stuff-trait-interp",
        help="Claude projects directory (default: ~/.claude/projects/...)"
    )
    parser.add_argument(
        "--archive-dir",
        type=Path,
        default=Path.home() / "code/claude-code-chats-saved",
        help="Archive base directory (default: ~/code/claude-code-chats-saved)"
    )
    parser.add_argument(
        "--include-agents",
        action="store_true",
        help="[Save mode] Also save agent files that were spawned during this chat"
    )
    parser.add_argument(
        "--exec",
        action="store_true",
        help="[Restore mode] Automatically execute 'claude -r SESSION_ID' after restoring"
    )

    args = parser.parse_args()

    # Determine mode: save or restore
    if args.restore:
        # Restore mode
        if not args.archive_dir.exists():
            raise FileNotFoundError(f"Archive directory not found: {args.archive_dir}")
        if not args.source_dir.exists():
            raise FileNotFoundError(f"Claude projects directory not found: {args.source_dir}")

        restore_chat(args.restore, args.archive_dir, args.source_dir, args.exec)

    elif args.chat_name:
        # Save mode
        if not args.source_dir.exists():
            raise FileNotFoundError(f"Source directory not found: {args.source_dir}")

        save_claude_chat(args.chat_name, args.source_dir, args.archive_dir, args.include_agents)

    else:
        parser.error("Must specify either --chat-name (save mode) or --restore CHAT_NAME (restore mode)")


if __name__ == "__main__":
    main()
