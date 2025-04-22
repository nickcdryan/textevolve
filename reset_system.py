#!/usr/bin/env python
"""
reset_system.py - Reset the agentic learning system by wiping archives, scripts, and learnings
"""

import os
import shutil
import argparse
from pathlib import Path


def reset_system(confirm=False):
    """Reset the system by deleting archives, scripts, and learnings file"""
    # Directories and files to reset
    archive_dir = Path("archive")
    scripts_dir = Path("scripts")
    viz_dir = Path("viz")
    learnings_file = Path("learnings.txt")

    # Check if directories or learnings file exist
    if not archive_dir.exists() and not scripts_dir.exists(
    ) and not learnings_file.exists() and not viz_dir.exists():
        print(
            "Nothing to reset - archive and scripts directories and learnings.txt don't exist."
        )
        return

    # Ask for confirmation if not already provided
    if not confirm:
        response = input(
            "Are you sure you want to reset the system? This will delete all archives, scripts, and the learnings file. (y/n): "
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

    # Delete viz directory and its contents
    if viz_dir.exists():
        try:
            shutil.rmtree(viz_dir)
            print(f"Deleted viz directory: {viz_dir}")
        except Exception as e:
            print(f"Error deleting viz directory: {e}")

    # Delete learnings file
    if learnings_file.exists():
        try:
            os.remove(learnings_file)
            print(f"Deleted learnings file: {learnings_file}")
        except Exception as e:
            print(f"Error deleting learnings file: {e}")

    # Recreate empty directories
    try:
        archive_dir.mkdir(exist_ok=True)
        scripts_dir.mkdir(exist_ok=True)
        viz_dir.mkdir(exist_ok=True)
        print("Created fresh archive, viz, and scripts directories.")
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
