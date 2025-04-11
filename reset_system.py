#!/usr/bin/env python
"""
reset_system.py - Reset the agentic learning system by wiping archives and scripts
"""

import os
import shutil
import argparse
from pathlib import Path


def reset_system(confirm=False):
    """Reset the system by deleting archives and scripts"""
    # Directories to reset
    archive_dir = Path("archive")
    scripts_dir = Path("scripts")

    # Check if directories exist
    if not archive_dir.exists() and not scripts_dir.exists():
        print(
            "Nothing to reset - archive and scripts directories don't exist.")
        return

    # Ask for confirmation if not already provided
    if not confirm:
        response = input(
            "Are you sure you want to reset the system? This will delete all archives and scripts. (y/n): "
        )
        if response.lower() != 'y':
            print("Reset cancelled.")
            return

    # Delete archive directory and its contents
    if archive_dir.exists():
        try:
            shutil.rmtree(archive_dir)
            print(f"Deleted archive directory: {archive_dir}")
        except Exception as e:
            print(f"Error deleting archive directory: {e}")

    # Delete scripts directory and its contents
    if scripts_dir.exists():
        try:
            shutil.rmtree(scripts_dir)
            print(f"Deleted scripts directory: {scripts_dir}")
        except Exception as e:
            print(f"Error deleting scripts directory: {e}")

    # Recreate empty directories
    try:
        archive_dir.mkdir(exist_ok=True)
        scripts_dir.mkdir(exist_ok=True)
        print("Created fresh archive and scripts directories.")
    except Exception as e:
        print(f"Error creating directories: {e}")

    print(
        "System reset completed. The system will start from iteration 0 on next run."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Reset the agentic learning system")
    parser.add_argument("--yes",
                        "-y",
                        action="store_true",
                        help="Skip confirmation prompt and reset immediately")

    args = parser.parse_args()
    reset_system(confirm=args.yes)
