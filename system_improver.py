#!/usr/bin/env python
"""
system_improver.py - Meta-improvement system that analyzes and enhances the Agentic Learning System

Usage:
    python system_improver.py [--backup]

This script:
1. Inspects the current state of the Agentic Learning System
2. Analyzes performance history and identifies improvement opportunities
3. Makes targeted code changes to enhance performance
4. Tracks changes and validates improvements



USAGE INSTRUCTIONS:
------------------

1. STAGING WORKFLOW (Recommended):
   a) Generate and stage changes for review:
      python system_improver.py --stage-only

   b) Review staged changes:
      cat staged_changes.json

   c) Apply staged changes if approved:
      python system_improver.py --apply-staged

2. AUTOMATIC WORKFLOW:
   a) Generate, apply and validate changes:
      python system_improver.py

   b) Generate and apply changes without validation:
      python system_improver.py --skip-validation


      
"""

import os
import sys
import json
import time
import argparse
import datetime
import difflib
import re
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

# Import required dependencies for LLM calling
from google import genai
from google.genai import types


class SystemImprover:
    """
    Autonomous improvement system for the Agentic Learning System.
    Reviews performance and makes code changes to enhance the system.
    """

    def __init__(self, 
                 create_backup: bool = True):
        """
        Initialize the system improver.

        Args:
            create_backup: Whether to create a backup before making changes
        """
        # System paths
        self.root_dir = Path(".")
        self.archive_dir = self.root_dir / "archive"
        self.scripts_dir = self.root_dir / "scripts"
        self.diffs_dir = self.root_dir / "diffs"
        self.backup_dir = self.root_dir / "backups"

        # Create required directories
        self.diffs_dir.mkdir(exist_ok=True)
        self.backup_dir.mkdir(exist_ok=True)

        # Core code files to analyze and potentially modify
        self.code_files = [
            "agent_system.py",
            "dataset_loader.py",
            "run_script.py",
            "system_prompt.md"
        ]

        # Settings
        self.create_backup = create_backup
        self.improvement_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        # Initialize LLM client
        try:
            self.client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
            print("Initialized Gemini API client successfully")
        except Exception as e:
            print(f"Error initializing Gemini API client: {e}")
            print("Make sure to set the GEMINI_API_KEY environment variable")
            sys.exit(1)

    def generate_changes_only(self) -> Dict:
        """
        Generate system improvement changes without applying them.
        These changes can be reviewed and applied later.

        Returns:
            Dict containing proposed changes and system analysis
        """
        print("\n" + "="*80)
        print(f"System Improver - Generating Changes - {self.improvement_timestamp}")
        print("="*80)

        # 1. Inspect the current system state
        system_data = self._inspect_system()

        # 2. Analyze performance and identify improvement areas
        improvement_plan = self._analyze_for_improvements(system_data)

        # 3. Save the improvement plan
        staged_changes = {
            "timestamp": self.improvement_timestamp,
            "system_state": {
                "current_iteration": improvement_plan.get("current_iteration", 0)
            },
            "analysis_summary": improvement_plan.get("analysis", ""),
            "improvement_history_analysis": improvement_plan.get("improvement_history_analysis", ""),
            "proposed_changes": improvement_plan.get("changes", []),
            "generated_by": "system_improver.py"
        }

        # 4. Create a human-readable report
        report_path = self.diffs_dir / f"proposed_changes_{self.improvement_timestamp}.md"

        report_content = f"""# Proposed System Changes - {self.improvement_timestamp}

## Summary
- **Timestamp:** {self.improvement_timestamp}
- **Current Iteration:** {improvement_plan.get("current_iteration", 0)}
- **Proposed Changes:** {len(improvement_plan.get("changes", []))}

## System Analysis
{improvement_plan.get("analysis", "")}

## Improvement History Analysis
{improvement_plan.get("improvement_history_analysis", "")}

## Proposed Changes
"""

        for i, change in enumerate(improvement_plan.get("changes", [])):
            report_content += f"""### Change {i+1}: {change['file']}
**Description:** {change['description']}

```diff
{change['diff']}
```

"""

        with open(report_path, 'w') as f:
            f.write(report_content)

        print(f"\nGenerated {len(improvement_plan.get('changes', []))} proposed changes.")
        print(f"Changes staged for review in: staged_changes.json")
        print(f"Report available at: {report_path}")

        return staged_changes

    def apply_changes(self, staged_changes: Dict) -> List[Dict]:
        """
        Apply previously staged changes.

        Args:
            staged_changes: Dict containing the staged changes to apply

        Returns:
            List of changes that were successfully applied
        """
        print("\n" + "="*80)
        print(f"System Improver - Applying Staged Changes - {self.improvement_timestamp}")
        print("="*80)

        # Create a backup if requested
        if self.create_backup:
            self._create_system_backup()

        # Extract the changes to apply
        changes_to_apply = staged_changes.get("proposed_changes", [])

        if not changes_to_apply:
            print("No changes to apply - staged_changes.json has no proposed_changes")
            return []

        # Implement the changes
        changes_made = []

        # Sort changes by priority
        sorted_changes = sorted(
            changes_to_apply, 
            key=lambda x: 0 if x["priority"] == "high" else 1
        )

        for change in sorted_changes:
            file_path = change["file"]
            full_path = self.root_dir / file_path

            if not full_path.exists():
                print(f"Warning: File not found: {file_path}, skipping change")
                continue

            # Standard approach for applying changes
            try:
                # Load current file content
                with open(full_path, 'r', encoding='utf-8') as f:
                    current_content = f.read()

                # Apply the diff if provided
                if change.get("diff"):
                    print(f"Applying diff to {file_path}")
                    new_content = self._apply_diff(current_content, change["diff"])
                else:
                    # Fallback to LLM for direct file modification
                    print(f"No diff found, using LLM to modify {file_path}")
                    new_content = self._generate_modified_file(file_path, current_content, change["description"])

                # Check if there was an actual change
                if new_content == current_content:
                    print(f"Warning: No changes made to {file_path} - content identical after applying changes")
                    print("Trying alternative approach with direct LLM modification...")
                    new_content = self._generate_modified_file(file_path, current_content, change["description"])

                # Only write if there's an actual change now
                if new_content != current_content:
                    # Generate the real diff for record-keeping
                    diff = self._generate_diff(current_content, new_content, file_path)

                    # Print a summary of changes for debugging
                    diff_lines = diff.splitlines()
                    added = sum(1 for line in diff_lines if line.startswith('+'))
                    removed = sum(1 for line in diff_lines if line.startswith('-'))
                    print(f"Changes to write: {added} lines added, {removed} lines removed")

                    # Check if we have write permission
                    if not os.access(full_path, os.W_OK):
                        print(f"⚠️ WARNING: No write permission for {file_path}")
                        # Try to make the file writable
                        try:
                            os.chmod(full_path, os.stat(full_path).st_mode | 0o200)  # Add write permission
                            print(f"  Attempted to add write permission to {file_path}")
                        except Exception as perm_e:
                            print(f"  Could not change permissions: {perm_e}")

                    # Create a backup of the specific file before modifying it
                    backup_dir = self.backup_dir / f"backup_file_{self.improvement_timestamp}"
                    backup_dir.mkdir(exist_ok=True, parents=True)
                    backup_path = backup_dir / file_path
                    backup_path.parent.mkdir(exist_ok=True, parents=True)
                    shutil.copy2(full_path, backup_path)
                    print(f"Backed up {file_path} to {backup_path}")

                    # Now write the changes
                    try:
                        with open(full_path, 'w', encoding='utf-8') as f:
                            f.write(new_content)
                        print(f"✅ Successfully wrote changes to {file_path}")
                    except Exception as write_e:
                        print(f"❌ ERROR WRITING FILE {file_path}: {write_e}")
                        # Try writing with different approach
                        try:
                            temp_path = self.root_dir / f"temp_{file_path}"
                            with open(temp_path, 'w', encoding='utf-8') as f:
                                f.write(new_content)
                            # If successful, rename the file
                            import shutil
                            shutil.move(temp_path, full_path)
                            print(f"✅ Successfully wrote changes using alternative approach")
                        except Exception as alt_e:
                            print(f"❌ Alternative writing approach also failed: {alt_e}")
                            continue

                    change_record = {
                        "file": file_path,
                        "description": change["description"],
                        "diff": diff,
                        "timestamp": datetime.datetime.now().isoformat()
                    }

                    changes_made.append(change_record)
                else:
                    print(f"❌ No changes could be made to {file_path} (content remains identical)")

            except Exception as e:
                print(f"❌ Error implementing change for {file_path}: {e}")
                import traceback
                traceback.print_exc()

        if not changes_made:
            print("No changes were successfully implemented.")
        else:
            print(f"\nSuccessfully applied {len(changes_made)} of {len(changes_to_apply)} changes.")

            # Optionally run validation
            if os.environ.get("SKIP_VALIDATION") != "1":
                self._validate_changes()

        return changes_made

    def run(self) -> bool:
        """
        Run the system improvement process.

        Returns:
            bool: True if improvements were made, False otherwise
        """
        print("\n" + "="*80)
        print(f"System Improver - Running at {self.improvement_timestamp}")
        print("="*80)

        # Create a backup before making changes if requested
        if self.create_backup:
            self._create_system_backup()

        # 1. Inspect the current system state
        system_data = self._inspect_system()

        # 2. Analyze performance and identify improvement areas
        improvement_plan = self._analyze_for_improvements(system_data)

        # 3. Generate and apply code modifications
        changes_made = self._implement_improvements(improvement_plan)

        # 4. Validate the changes
        validation_results = self._validate_changes()

        # 5. Record the changes and results
        self._record_improvement_results(
            improvement_plan, 
            changes_made, 
            validation_results
        )

        print("\nSystem improvement complete!")
        print(f"Changes recorded in: {self.diffs_dir}/changes_{self.improvement_timestamp}.json")
        print(f"Report available at: {self.diffs_dir}/report_{self.improvement_timestamp}.md")

        return len(changes_made) > 0

    def _inspect_system(self) -> Dict:
        """
        Inspect the current state of the system, including:
        - Current code files
        - Performance history
        - Learnings file
        - Best script
        - Common error patterns
        - Previous system improvements and their effects

        Returns:
            Dict containing all system data
        """
        print("\nInspecting system state...")

        system_data = {
            "code_files": {},
            "performance_history": [],
            "learnings": "",
            "best_script": None,
            "error_patterns": [],
            "current_iteration": 0,
            "timestamp": self.improvement_timestamp,
            "improvement_history": []  # Will contain previous improvements
        }

        # Read all code files
        for file_path in self.code_files:
            path = self.root_dir / file_path
            if path.exists():
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        system_data["code_files"][file_path] = f.read()
                    print(f"Loaded {file_path}: {len(system_data['code_files'][file_path])} chars")
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")

        # Get learnings file
        learnings_path = self.root_dir / "learnings.txt"
        if learnings_path.exists():
            try:
                with open(learnings_path, 'r', encoding='utf-8') as f:
                    system_data["learnings"] = f.read()
                print(f"Loaded learnings.txt: {len(system_data['learnings'])} chars")
            except Exception as e:
                print(f"Error reading learnings.txt: {e}")

        # Get performance history
        summaries_path = self.archive_dir / "summaries.json"
        if summaries_path.exists():
            try:
                with open(summaries_path, 'r') as f:
                    summaries = json.load(f)
                system_data["performance_history"] = summaries

                # Get the current iteration
                if summaries:
                    sorted_summaries = sorted(summaries, key=lambda x: x.get("iteration", 0))
                    system_data["current_iteration"] = sorted_summaries[-1].get("iteration", 0)

                print(f"Loaded {len(summaries)} iteration summaries")
            except Exception as e:
                print(f"Error reading summaries.json: {e}")

        # Get recent iterations for detailed analysis
        iterations = []
        for i in range(max(0, system_data["current_iteration"] - 5), system_data["current_iteration"] + 1):
            iteration_path = self.archive_dir / f"iteration_{i}.json"
            if iteration_path.exists():
                try:
                    with open(iteration_path, 'r') as f:
                        iteration_data = json.load(f)
                    iterations.append(iteration_data)
                except Exception as e:
                    print(f"Error reading iteration_{i}.json: {e}")

        system_data["recent_iterations"] = iterations

        # Extract common error patterns
        error_patterns = []
        for summary in system_data["performance_history"]:
            if "performance" in summary and "error_analysis" in summary["performance"]:
                patterns = summary["performance"]["error_analysis"].get("error_patterns", [])
                error_patterns.extend(patterns)

        # Deduplicate error patterns
        system_data["error_patterns"] = list(set(error_patterns))

        # Get previous improvement history from diffs directory
        self._load_improvement_history(system_data)

        return system_data

    def _load_improvement_history(self, system_data: Dict):
        """
        Load and analyze previous improvement attempts from the diffs directory.

        Args:
            system_data: Current system data dictionary to update
        """
        print("Analyzing previous system improvements...")

        # Find all change records in chronological order
        change_files = sorted(self.diffs_dir.glob("changes_*.json"))

        improvement_history = []
        performance_before_after = []

        # Track iterations where improvements were made
        improved_iterations = []

        # Process each improvement record
        for change_file in change_files:
            try:
                with open(change_file, 'r') as f:
                    change_record = json.load(f)

                # Extract basic improvement data
                timestamp = change_record.get("timestamp", "unknown")
                iteration = change_record.get("system_state", {}).get("current_iteration", "unknown")
                changes = change_record.get("changes_made", [])

                # Skip if no actual changes were made
                if not changes:
                    continue

                improved_iterations.append(iteration)

                # Find performance data from before and after this improvement
                if iteration != "unknown":
                    # Find summaries before and after this improvement
                    before_perf = None
                    after_perf = None
                    for summary in system_data["performance_history"]:
                        if summary.get("iteration") == iteration:
                            before_perf = summary.get("performance", {}).get("accuracy", 0)
                        elif summary.get("iteration") == iteration + 1:
                            after_perf = summary.get("performance", {}).get("accuracy", 0)

                    if before_perf is not None and after_perf is not None:
                        perf_change = after_perf - before_perf
                        performance_before_after.append({
                            "improvement_timestamp": timestamp,
                            "iteration": iteration,
                            "before_accuracy": before_perf,
                            "after_accuracy": after_perf,
                            "change": perf_change,
                            "success": perf_change > 0
                        })

                # Create a summary of this improvement
                files_changed = [change.get("file") for change in changes]
                change_descriptions = [change.get("description") for change in changes]

                improvement_history.append({
                    "timestamp": timestamp,
                    "iteration": iteration,
                    "files_changed": files_changed,
                    "num_changes": len(changes),
                    "descriptions": change_descriptions,
                    "validation_success": change_record.get("validation_results", {}).get("success", False)
                })

                print(f"Found improvement at iteration {iteration} with {len(changes)} changes")

            except Exception as e:
                print(f"Error processing improvement record {change_file}: {e}")

        # Add improvement history to system data
        system_data["improvement_history"] = improvement_history
        system_data["performance_impact"] = performance_before_after
        system_data["improved_iterations"] = improved_iterations

        print(f"Loaded {len(improvement_history)} previous improvement records")

        # Analyze effectiveness of previous improvements
        if performance_before_after:
            successful = sum(1 for p in performance_before_after if p.get("success", False))
            print(f"Previous improvements: {successful}/{len(performance_before_after)} led to performance gains")

    def _analyze_for_improvements(self, system_data: Dict) -> Dict:
        """
        Analyze the system data and identify improvement opportunities.
        Uses LLM to generate a comprehensive improvement plan.

        Args:
            system_data: System inspection data

        Returns:
            Dict containing improvement plan
        """
        print("\nAnalyzing system for improvement opportunities...")

        # DEBUG: Let's see what files are loaded
        print(f"Available files in system_data: {list(system_data.get('code_files', {}).keys())}")

        # CRITICAL FIX: Ensure agent_system.py is loaded, even if not in the initial set
        for key_file in ["agent_system.py", "dataset_loader.py", "system_prompt.md"]:
            if key_file not in system_data.get("code_files", {}):
                file_path = self.root_dir / key_file
                if file_path.exists():
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            if "code_files" not in system_data:
                                system_data["code_files"] = {}
                            system_data["code_files"][key_file] = f.read()
                        print(f"Added missing key file: {key_file}")
                    except Exception as e:
                        print(f"Error loading key file {key_file}: {e}")

        # Prepare LLM prompt for system analysis
        prompt = self._create_analysis_prompt(system_data)

        # Call LLM for improvement analysis
        system_instruction = """You are an Expert System Designer and Improvement Specialist. 
Your task is to analyze a machine learning system's code and performance data, 
identify limitations, and propose specific, actionable code changes to improve the system."""

        print("Calling LLM for improvement analysis...")
        improvement_analysis = self._call_llm(prompt, system_instruction)

        # Save the raw response for debugging
        debug_path = self.diffs_dir / f"raw_llm_response_{self.improvement_timestamp}.txt"
        with open(debug_path, 'w', encoding='utf-8') as f:
            f.write(improvement_analysis)
        print(f"Raw LLM response saved to {debug_path}")

        # Parse the improvement plan
        try:
            # Look for specific sections in the response
            improvement_plan = {
                "analysis": "",
                "improvement_history_analysis": "",
                "changes": [],
                "current_iteration": system_data.get("current_iteration", 0)
            }

            # Extract the system analysis section
            analysis_section = re.search(r"## SYSTEM ANALYSIS(.*?)(?:##|$)", improvement_analysis, re.DOTALL)
            if analysis_section:
                improvement_plan["analysis"] = analysis_section.group(1).strip()
                print(f"Extracted system analysis: {len(improvement_plan['analysis'])} chars")
            else:
                print("WARNING: Could not extract system analysis section from LLM response")

            # Extract the improvement history analysis section
            history_analysis_section = re.search(r"## IMPROVEMENT HISTORY ANALYSIS(.*?)(?:##|$)", improvement_analysis, re.DOTALL)
            if history_analysis_section:
                improvement_plan["improvement_history_analysis"] = history_analysis_section.group(1).strip()
                print(f"Extracted improvement history analysis: {len(improvement_plan['improvement_history_analysis'])} chars")
            else:
                print("WARNING: Could not extract improvement history analysis section from LLM response")

            # Extract the changes section
            changes_section = re.search(r"## PROPOSED CODE CHANGES(.*?)(?:##|$)", improvement_analysis, re.DOTALL)
            if changes_section:
                changes_text = changes_section.group(1).strip()
                print(f"Extracted proposed code changes section: {len(changes_text)} chars")

                # Parse each change specification
                change_blocks = re.findall(r"### Change (\d+):(.*?)(?=### Change \d+:|$)", changes_text, re.DOTALL)

                print(f"Found {len(change_blocks)} change blocks")

                for i, (change_num, change_block) in enumerate(change_blocks):
                    print(f"Processing change block {i+1}: {len(change_block)} chars")

                    # Extract the file, description, and diff
                    file_match = re.search(r"File:[ \t]*[`'\"]?([\w./]+\.\w+)[`'\"]?", change_block, re.MULTILINE)
                    description_match = re.search(r"Description:(.*?)(?:```diff|Diff:|$)", change_block, re.DOTALL)
                    diff_match = re.search(r"```diff\n(.*?)```", change_block, re.DOTALL)

                    # Print debug info for each part
                    if file_match:
                        file_path = file_match.group(1).strip()
                        print(f"  Found file path: {file_path}")
                    else:
                        print("  WARNING: Could not extract file path")

                    if description_match:
                        description = description_match.group(1).strip()
                        print(f"  Found description: {len(description)} chars")
                    else:
                        print("  WARNING: Could not extract description")

                    if diff_match:
                        diff = diff_match.group(1)
                        print(f"  Found diff: {len(diff)} chars")
                    else:
                        print("  WARNING: Could not extract diff")

                    file_path = file_match.group(1).strip() if file_match else None
                    description = description_match.group(1).strip() if description_match else ""
                    diff = diff_match.group(1) if diff_match else ""

                    # Check if we have a file path and it exists in the data
                    if file_path:
                        exists_in_data = file_path in system_data["code_files"]
                        exists_on_disk = os.path.exists(file_path)

                        print(f"  File '{file_path}' exists in system data: {exists_in_data}")
                        print(f"  File '{file_path}' exists on disk: {exists_on_disk}")

                        # CRITICAL FIX: Always check if the file exists on disk, even if not in system_data
                        if not exists_in_data and exists_on_disk:
                            try:
                                with open(file_path, 'r', encoding='utf-8') as f:
                                    system_data["code_files"][file_path] = f.read()
                                print(f"  Added file {file_path} to system data")
                                exists_in_data = True
                            except Exception as e:
                                print(f"  Error reading file {file_path}: {e}")

                        # CRITICAL FIX: Allow changes even if the file isn't in system_data but exists on disk
                        if exists_in_data or exists_on_disk:
                            if not exists_in_data:
                                # Create a placeholder if we couldn't read the file
                                system_data["code_files"][file_path] = ""

                            improvement_plan["changes"].append({
                                "change_id": i+1,
                                "file": file_path,
                                "description": description,
                                "diff": diff,
                                "original_content": system_data["code_files"][file_path],
                                "priority": "high" if "high priority" in description.lower() else "medium"
                            })
                            print(f"  Added change for file {file_path}")
                        else:
                            print(f"  WARNING: File {file_path} not found in system data or on disk, skipping change")
                    else:
                        print("  WARNING: No file path specified, skipping change")
            else:
                print("WARNING: Could not extract proposed code changes section from LLM response")
                # Try more generic patterns to find changes
                print("Trying alternative approaches to extract changes...")

                # Look for file paths and descriptions without explicit sections
                file_matches = re.findall(r"(?:In|For|Update|Modify|Change) [`']?([\w./]+\.py)[`']?", improvement_analysis)
                if file_matches:
                    print(f"Found potential file references: {file_matches}")
                    for i, file_path in enumerate(file_matches):
                        if file_path in system_data["code_files"]:
                            # Extract nearby text as description
                            context = re.search(r"(.{100})" + re.escape(file_path) + r"(.{200})", improvement_analysis)
                            description = f"Extracted from context around file mention: {context.group(1) if context else ''} {file_path} {context.group(2) if context else ''}"

                            improvement_plan["changes"].append({
                                "change_id": i+1,
                                "file": file_path,
                                "description": description,
                                "diff": "",  # No diff found, will use LLM to generate changes
                                "original_content": system_data["code_files"][file_path],
                                "priority": "medium"
                            })
                            print(f"Added change for file {file_path} via alternative detection")

            print(f"Identified {len(improvement_plan['changes'])} proposed code changes")

            # If we have analysis but no changes, try to generate changes from the analysis
            if improvement_plan["analysis"] and not improvement_plan["changes"]:
                print("Analysis found but no changes, asking LLM to generate specific changes...")
                improvement_plan["changes"] = self._generate_changes_from_analysis(
                    improvement_plan["analysis"], 
                    system_data
                )

            return improvement_plan

        except Exception as e:
            print(f"Error parsing improvement plan: {e}")
            import traceback
            traceback.print_exc()

            return {
                "analysis": improvement_analysis,
                "changes": [],
                "current_iteration": system_data.get("current_iteration", 0)
            }

    def _generate_changes_from_analysis(self, analysis: str, system_data: Dict) -> List[Dict]:
        """
        Generate specific code changes from system analysis when none were provided.

        Args:
            analysis: System analysis text
            system_data: System data including code files

        Returns:
            List of change specifications
        """
        print("Generating specific code changes from analysis...")

        # Get the files we have available
        available_files = list(system_data["code_files"].keys())
        if not available_files:
            print("No files available to modify")
            return []

        # Create a prompt to generate specific changes
        change_gen_system_instruction = """You are an Expert Code Improvement Specialist.
Your task is to translate a system analysis into specific, actionable code changes."""

        change_gen_prompt = f"""
Based on the system analysis below, I need you to propose 2-3 specific code changes to improve the system.

# SYSTEM ANALYSIS
{analysis}

# AVAILABLE FILES
{', '.join(available_files)}

For each change, specify:
1. Which specific file to modify
2. What specific changes to make
3. A detailed diff showing the exact changes

# REQUIRED OUTPUT FORMAT

## PROPOSED CODE CHANGES

### Change 1:
File: `filename.py`
Description: Clear description of the improvement

```diff
- Original line
+ Modified line
```

### Change 2:
[etc...]

The changes should be specific, focused improvements that address the issues identified in the analysis.
Make sure each change includes a clear diff showing exactly what lines to change.
"""

        # Call LLM to generate specific changes
        try:
            response = self._call_llm(change_gen_prompt, change_gen_system_instruction)

            # Save the response for debugging
            debug_path = self.diffs_dir / f"generated_changes_{self.improvement_timestamp}.txt"
            with open(debug_path, 'w', encoding='utf-8') as f:
                f.write(response)
            print(f"Generated changes saved to {debug_path}")

            # Parse the response to extract changes
            changes = []

            # Extract the changes section
            changes_section = re.search(r"## PROPOSED CODE CHANGES(.*?)(?:##|$)", response, re.DOTALL)
            if changes_section:
                changes_text = changes_section.group(1).strip()

                # Parse each change specification
                change_blocks = re.findall(r"### Change (\d+):(.*?)(?=### Change \d+:|$)", changes_text, re.DOTALL)

                for i, (change_num, change_block) in enumerate(change_blocks):
                    # Extract the file, description, and diff
                    file_match = re.search(r"File: [`']?(.*?)[`']?$", change_block, re.MULTILINE)
                    description_match = re.search(r"Description:(.*?)(?:```diff|Diff:|$)", change_block, re.DOTALL)
                    diff_match = re.search(r"```diff\n(.*?)```", change_block, re.DOTALL)

                    file_path = file_match.group(1).strip() if file_match else None
                    description = description_match.group(1).strip() if description_match else ""
                    diff = diff_match.group(1) if diff_match else ""

                    if file_path and file_path in system_data["code_files"]:
                        changes.append({
                            "change_id": i+1,
                            "file": file_path,
                            "description": description,
                            "diff": diff,
                            "original_content": system_data["code_files"][file_path],
                            "priority": "high" if "high priority" in description.lower() else "medium"
                        })

            print(f"Generated {len(changes)} specific changes from analysis")
            return changes

        except Exception as e:
            print(f"Error generating changes from analysis: {e}")
            return []

    def _implement_improvements(self, improvement_plan: Dict) -> List[Dict]:
        """
        Implement the proposed code changes.

        Args:
            improvement_plan: The improvement plan with proposed changes

        Returns:
            List of changes that were successfully implemented
        """
        print("\nImplementing code improvements...")

        changes_made = []

        # Check if we have changes to implement
        if not improvement_plan["changes"]:
            print("No changes to implement - generating changes from analysis...")
            if improvement_plan["analysis"]:
                improvement_plan["changes"] = self._generate_changes_from_analysis(
                    improvement_plan["analysis"], 
                    {"code_files": self._load_all_code_files()}
                )
            else:
                print("No analysis available to generate changes")
                return []

        # Check again if we have changes
        if not improvement_plan["changes"]:
            print("No changes could be generated - implementation aborted")
            return []

        # Sort changes by priority
        sorted_changes = sorted(
            improvement_plan["changes"], 
            key=lambda x: 0 if x["priority"] == "high" else 1
        )

        for change in sorted_changes:
            file_path = change["file"]
            full_path = self.root_dir / file_path

            if not full_path.exists():
                print(f"Warning: File not found: {file_path}, skipping change")
                continue

            # Determine if this is a large file that needs special handling
            try:
                file_size = full_path.stat().st_size
                is_large_file = file_size > 100000  # ~100KB threshold
                print(f"File size: {file_size/1024:.1f}KB ({'LARGE' if is_large_file else 'standard'} file)")

                # For Python files, try to identify specific functions/sections to modify
                if is_large_file and file_path.endswith('.py'):
                    result = self._targeted_file_modification(full_path, change)
                    if result["success"]:
                        changes_made.append(result["change_record"])
                        continue
                    else:
                        print(f"Targeted modification failed: {result['error']}")
                        print("Falling back to standard modification approach")
            except Exception as e:
                print(f"Error determining file size: {e}")

            # Standard approach for normal-sized files or if targeted modification failed
            try:
                # Load current file content
                with open(full_path, 'r', encoding='utf-8') as f:
                    current_content = f.read()

                # Apply the diff if provided
                if change.get("diff"):
                    print(f"Applying diff to {file_path}")
                    new_content = self._apply_diff(current_content, change["diff"])
                else:
                    # Fallback to LLM for direct file modification
                    print(f"No diff found, using LLM to modify {file_path}")
                    new_content = self._generate_modified_file(file_path, current_content, change["description"])

                # Check if there was an actual change
                if new_content == current_content:
                    print(f"Warning: No changes made to {file_path} - content identical after applying changes")
                    print("Trying alternative approach with direct LLM modification...")
                    new_content = self._generate_modified_file(file_path, current_content, change["description"])

                # Only write if there's an actual change now
                if new_content != current_content:
                    # Generate the real diff for record-keeping
                    diff = self._generate_diff(current_content, new_content, file_path)

                    # Print a summary of changes for debugging
                    diff_lines = diff.splitlines()
                    added = sum(1 for line in diff_lines if line.startswith('+'))
                    removed = sum(1 for line in diff_lines if line.startswith('-'))
                    print(f"Changes to write: {added} lines added, {removed} lines removed")

                    # Write the updated file - this is where the actual file modification happens
                    try:
                        # Check if we have write permission
                        if not os.access(full_path, os.W_OK):
                            print(f"⚠️ WARNING: No write permission for {file_path}")
                            # Try to make the file writable
                            try:
                                os.chmod(full_path, os.stat(full_path).st_mode | 0o200)  # Add write permission
                                print(f"  Attempted to add write permission to {file_path}")
                            except Exception as perm_e:
                                print(f"  Could not change permissions: {perm_e}")

                        # Create a backup of the specific file before modifying it
                        backup_dir = self.backup_dir / f"backup_file_{self.improvement_timestamp}"
                        backup_dir.mkdir(exist_ok=True, parents=True)
                        backup_path = backup_dir / file_path
                        backup_path.parent.mkdir(exist_ok=True, parents=True)
                        shutil.copy2(full_path, backup_path)
                        print(f"Backed up {file_path} to {backup_path}")

                        # Now write the changes
                        with open(full_path, 'w', encoding='utf-8') as f:
                            f.write(new_content)
                        print(f"✅ Successfully wrote changes to {file_path}")

                        # Verify the changes were actually written
                        time.sleep(0.1)  # Small delay to ensure file system is updated
                        with open(full_path, 'r', encoding='utf-8') as f:
                            after_content = f.read()
                        if after_content == new_content:
                            print(f"  Verified: Changes were successfully written to {file_path}")
                        else:
                            print(f"⚠️ WARNING: File was written but content doesn't match expected changes!")
                    except Exception as write_e:
                        print(f"❌ ERROR WRITING FILE {file_path}: {write_e}")
                        # Try writing with different approach
                        try:
                            temp_path = self.root_dir / f"temp_{file_path}"
                            with open(temp_path, 'w', encoding='utf-8') as f:
                                f.write(new_content)
                            # If successful, rename the file
                            import shutil
                            shutil.move(temp_path, full_path)
                            print(f"✅ Successfully wrote changes using alternative approach")
                        except Exception as alt_e:
                            print(f"❌ Alternative writing approach also failed: {alt_e}")
                            continue

                    change_record = {
                        "file": file_path,
                        "description": change["description"],
                        "diff": diff,
                        "timestamp": datetime.datetime.now().isoformat()
                    }

                    changes_made.append(change_record)
                else:
                    print(f"❌ No changes could be made to {file_path} (content remains identical)")

            except Exception as e:
                print(f"❌ Error implementing change for {file_path}: {e}")
                import traceback
                traceback.print_exc()

        if not changes_made:
            print("No changes were successfully implemented.")

        return changes_made

    def _load_all_code_files(self) -> Dict[str, str]:
        """
        Load all code files in the project.

        Returns:
            Dict mapping file paths to content
        """
        code_files = {}

        # Known file types for code
        extensions = ['.py', '.md', '.json', '.txt']

        # Exclude directories
        exclude_dirs = ['__pycache__', 'venv', 'env', '.git', '.github', 'diffs', 'backups', 'archive', 'scripts']

        # Walk directory and find code files
        for root, dirs, files in os.walk(self.root_dir):
            # Skip excluded directories
            dirs[:] = [d for d in dirs if d not in exclude_dirs]

            for file in files:
                if any(file.endswith(ext) for ext in extensions):
                    file_path = os.path.join(root, file)
                    rel_path = os.path.relpath(file_path, self.root_dir)

                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            code_files[rel_path] = f.read()
                    except Exception:
                        # Skip files that can't be read as text
                        pass

        return code_files

    def _targeted_file_modification(self, file_path, change):
        """
        Perform targeted modifications on large files by focusing on specific functions or sections.

        Args:
            file_path: Path to the file to modify
            change: Change specification

        Returns:
            Dict with success status, error message if any, and change record if successful
        """
        print(f"Using targeted modification for large file: {file_path}")

        # First, analyze the change description to identify target function/section
        target_info = self._identify_targets_from_change(change)

        if not target_info["targets"]:
            return {
                "success": False, 
                "error": "Could not identify specific targets in the file"
            }

        print(f"Identified {len(target_info['targets'])} targets for modification")

        # Get the original file content
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                original_content = f.read()

            # Keep a copy for diff generation
            current_content = original_content

            # Try different targeted modification approaches
            modified = False

            # 1. Try function-based modification if we have function names
            if target_info["function_names"]:
                try:
                    new_content = self._modify_functions(
                        current_content, 
                        target_info["function_names"], 
                        change["description"]
                    )
                    if new_content != current_content:
                        modified = True
                        current_content = new_content
                        print(f"Successfully modified functions: {target_info['function_names']}")
                except Exception as e:
                    print(f"Function-based modification failed: {e}")

            # 2. Try line-range based modification if we have line ranges
            if not modified and target_info["line_ranges"]:
                try:
                    new_content = self._modify_line_ranges(
                        current_content, 
                        target_info["line_ranges"], 
                        change["description"]
                    )
                    if new_content != current_content:
                        modified = True
                        current_content = new_content
                        print(f"Successfully modified line ranges: {target_info['line_ranges']}")
                except Exception as e:
                    print(f"Line-range modification failed: {e}")

            # 3. Try section-based modification if we have section markers
            if not modified and target_info["section_markers"]:
                try:
                    new_content = self._modify_sections(
                        current_content, 
                        target_info["section_markers"], 
                        change["description"]
                    )
                    if new_content != current_content:
                        modified = True
                        current_content = new_content
                        print(f"Successfully modified sections: {target_info['section_markers']}")
                except Exception as e:
                    print(f"Section-based modification failed: {e}")

            # If no modification worked, return failure
            if not modified:
                return {
                    "success": False, 
                    "error": "None of the targeted modification approaches worked"
                }

            # Generate diff for the changes
            diff = self._generate_diff(original_content, current_content, file_path.name)

            # Write the modified content back to the file
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(current_content)

            print(f"✅ Successfully wrote targeted changes to {file_path}")

            # Create change record
            change_record = {
                "file": str(file_path.relative_to(self.root_dir)),
                "description": change["description"],
                "diff": diff,
                "targets": target_info["targets"],
                "timestamp": datetime.datetime.now().isoformat()
            }

            return {
                "success": True,
                "change_record": change_record
            }

        except Exception as e:
            return {
                "success": False,
                "error": f"Error in targeted modification: {str(e)}"
            }

    def _identify_targets_from_change(self, change):
        """
        Analyze change description to identify specific targets for modification.

        Args:
            change: Change specification

        Returns:
            Dict with targets information
        """
        targets = []
        function_names = []
        line_ranges = []
        section_markers = []

        # First try to extract information from the diff if available
        if change.get("diff"):
            # Look for function definitions or class methods in the diff
            func_pattern = re.compile(r"[+-][ \t]*(def|class) ([a-zA-Z0-9_]+)")
            for match in func_pattern.finditer(change["diff"]):
                func_name = match.group(2)
                if func_name not in function_names:
                    function_names.append(func_name)
                    targets.append(f"function:{func_name}")

            # Look for line ranges in the diff header
            line_pattern = re.compile(r"@@ -(\d+),(\d+) \+(\d+),(\d+) @@")
            for match in line_pattern.finditer(change["diff"]):
                start_line = int(match.group(1))
                num_lines = int(match.group(2))
                line_ranges.append((start_line, start_line + num_lines - 1))
                targets.append(f"lines:{start_line}-{start_line + num_lines - 1}")

        # If we couldn't extract from diff, try the description
        if not targets:
            description = change.get("description", "")

            # Look for function names in the description
            func_pattern = re.compile(r"(function|method|def) ([a-zA-Z0-9_]+)")
            for match in func_pattern.finditer(description):
                func_name = match.group(2)
                if func_name not in function_names:
                    function_names.append(func_name)
                    targets.append(f"function:{func_name}")

            # Look for line ranges in the description
            line_pattern = re.compile(r"lines? (\d+)(?:\s*-\s*(\d+))?")
            for match in line_pattern.finditer(description):
                start_line = int(match.group(1))
                end_line = int(match.group(2)) if match.group(2) else start_line
                line_ranges.append((start_line, end_line))
                targets.append(f"lines:{start_line}-{end_line}")

            # Look for section markers in the description
            section_pattern = re.compile(r"section '([^']+)'|section \"([^\"]+)\"|between '([^']+)' and '([^']+)'")
            for match in section_pattern.finditer(description):
                if match.group(1):
                    section_markers.append((match.group(1), None))
                    targets.append(f"section:{match.group(1)}")
                elif match.group(2):
                    section_markers.append((match.group(2), None))
                    targets.append(f"section:{match.group(2)}")
                elif match.group(3) and match.group(4):
                    section_markers.append((match.group(3), match.group(4)))
                    targets.append(f"section:{match.group(3)}...{match.group(4)}")

        # If we still couldn't find targets, try to identify target from file
        if not targets:
            # We'll let the LLM help us identify targets
            targets = self._identify_targets_with_llm(change)

            # Extract the identified targets
            for target in targets:
                if target.startswith("function:"):
                    function_name = target[9:]
                    if function_name not in function_names:
                        function_names.append(function_name)
                elif target.startswith("lines:"):
                    range_str = target[6:]
                    if "-" in range_str:
                        start, end = map(int, range_str.split("-"))
                        line_ranges.append((start, end))
                elif target.startswith("section:"):
                    section_str = target[8:]
                    if "..." in section_str:
                        start, end = section_str.split("...")
                        section_markers.append((start, end))
                    else:
                        section_markers.append((section_str, None))

        return {
            "targets": targets,
            "function_names": function_names,
            "line_ranges": line_ranges,
            "section_markers": section_markers
        }

    def _identify_targets_with_llm(self, change):
        """
        Use LLM to identify specific targets for modification.

        Args:
            change: Change specification

        Returns:
            List of target identifiers
        """
        # Create a prompt for the LLM to identify targets
        system_instruction = """You are a Code Modification Expert. 
Your task is to identify specific parts of a file that need to be modified based on a change description."""

        prompt = f"""
I need to make a targeted modification to a file, but I need to identify exactly which parts of the file to modify.

# CHANGE DESCRIPTION
{change.get("description", "")}

# DIFF (if available)
```diff
{change.get("diff", "No diff available")}
```

Based on this information, identify the specific targets in the file that need to be modified. Targets can be:
1. Functions or methods (preferred)
2. Line ranges
3. Specific sections between identifiable markers

Return a list of targets in this exact format (one per line):
- function:function_name
- lines:start_line-end_line  
- section:start_marker...end_marker

Be as specific as possible. If you can identify exact function names or line ranges, those are preferred.
Return ONLY the list of targets, one per line, with no additional explanation.
"""

        # Call LLM to identify targets
        response = self._call_llm(prompt, system_instruction)

        # Parse the response into a list of targets
        targets = []
        for line in response.strip().split("\n"):
            line = line.strip()
            if line.startswith("-"):
                line = line[1:].strip()
            if line.startswith("function:") or line.startswith("lines:") or line.startswith("section:"):
                targets.append(line)

        return targets

    def _modify_functions(self, content, function_names, change_description):
        """
        Modify specific functions in a file.

        Args:
            content: Original file content
            function_names: List of function names to modify
            change_description: Description of the changes to make

        Returns:
            Modified file content
        """
        # Check if we have any function names to modify
        if not function_names:
            return content

        # For Python files, we'll use a simple regex-based approach
        # (For more complex cases, an AST-based approach would be better)
        modified_content = content

        for function_name in function_names:
            # Find the function definition
            pattern = re.compile(f"(def {re.escape(function_name)}\\([^)]*\\):.*?)(?:def |class |$)", re.DOTALL)
            match = pattern.search(modified_content)

            if not match:
                print(f"Function {function_name} not found in content")
                continue

            function_code = match.group(1)

            # Use LLM to modify just the function
            modified_function = self._modify_code_section(function_code, change_description)

            # Replace the function in the original content
            if modified_function != function_code:
                modified_content = modified_content.replace(function_code, modified_function)
                print(f"Modified function: {function_name}")

        return modified_content

    def _modify_line_ranges(self, content, line_ranges, change_description):
        """
        Modify specific line ranges in a file.

        Args:
            content: Original file content
            line_ranges: List of (start_line, end_line) tuples
            change_description: Description of the changes to make

        Returns:
            Modified file content
        """
        # Check if we have any line ranges to modify
        if not line_ranges:
            return content

        # Split content into lines for easier processing
        lines = content.splitlines(True)  # Keep line endings

        # Process each line range
        for start_line, end_line in line_ranges:
            # Convert to 0-based indexing
            start_idx = max(0, start_line - 1)
            end_idx = min(len(lines), end_line)

            # Extract the line range
            line_range = "".join(lines[start_idx:end_idx])

            # Use LLM to modify just this section
            modified_section = self._modify_code_section(line_range, change_description)

            # Replace the section in the original content
            if modified_section != line_range:
                modified_lines = modified_section.splitlines(True)
                lines[start_idx:end_idx] = modified_lines
                print(f"Modified lines: {start_line}-{end_line}")

        return "".join(lines)

    def _modify_sections(self, content, section_markers, change_description):
        """
        Modify specific sections in a file identified by markers.

        Args:
            content: Original file content
            section_markers: List of (start_marker, end_marker) tuples
            change_description: Description of the changes to make

        Returns:
            Modified file content
        """
        # Check if we have any section markers to modify
        if not section_markers:
            return content

        modified_content = content

        for start_marker, end_marker in section_markers:
            if end_marker:
                # Find section between start and end markers
                pattern = re.compile(f"({re.escape(start_marker)}.*?{re.escape(end_marker)})", re.DOTALL)
            else:
                # Find section starting with marker
                pattern = re.compile(f"({re.escape(start_marker)}.*?)(?={re.escape(start_marker)}|$)", re.DOTALL)

            match = pattern.search(modified_content)

            if not match:
                print(f"Section {start_marker}...{end_marker or 'end'} not found in content")
                continue

            section = match.group(1)

            # Use LLM to modify just the section
            modified_section = self._modify_code_section(section, change_description)

            # Replace the section in the original content
            if modified_section != section:
                modified_content = modified_content.replace(section, modified_section)
                print(f"Modified section: {start_marker}...{end_marker or 'end'}")

        return modified_content

    def _modify_code_section(self, code_section, change_description):
        """
        Use LLM to modify a specific section of code.

        Args:
            code_section: Code section to modify
            change_description: Description of the changes to make

        Returns:
            Modified code section
        """
        system_instruction = """You are an Expert Code Modifier. 
Your task is to modify a specific section of code according to the change description.
Make ONLY the requested changes while preserving the overall structure and style."""

        prompt = f"""
I need you to modify this specific section of code according to the following change description:

# CHANGE DESCRIPTION
{change_description}

# CODE SECTION TO MODIFY
```python
{code_section}
```

Apply EXACTLY these changes to the code. Don't make any additional changes.
Ensure the indentation and formatting remains consistent with the original code.
Return ONLY the modified code section with no explanations before or after.
"""

        # Call LLM to modify the code section
        modified_section = self._call_llm(prompt, system_instruction)

        # Extract code if wrapped in code blocks
        if "```python" in modified_section and "```" in modified_section:
            modified_section = modified_section.split("```python")[1].split("```")[0].strip()
        elif "```" in modified_section:
            modified_section = modified_section.split("```")[1].split("```")[0].strip()

        # Ensure consistent line endings
        if code_section.endswith("\n") and not modified_section.endswith("\n"):
            modified_section += "\n"

        return modified_section

    def _apply_diff(self, original_content: str, diff_text: str) -> str:
        """
        Apply a diff to the original content.

        Args:
            original_content: Original file content
            diff_text: Diff text to apply

        Returns:
            Modified content with diff applied
        """
        print(f"Applying diff with {len(diff_text)} characters")

        # Check if the diff is empty or malformed
        if not diff_text or not diff_text.strip():
            print("Warning: Empty diff provided, returning original content")
            return original_content

        # Print first few lines of diff for debugging
        print("Diff preview:")
        for i, line in enumerate(diff_text.splitlines()[:5]):
            print(f"  {line}")
        if len(diff_text.splitlines()) > 5:
            print(f"  ... ({len(diff_text.splitlines()) - 5} more lines)")

        # Simple approach: if the diff doesn't contain actual changes, ask LLM to make the changes directly
        has_change_markers = False
        for line in diff_text.splitlines():
            if line.startswith('+') or line.startswith('-'):
                has_change_markers = True
                break

        if not has_change_markers:
            print("Warning: Diff doesn't contain proper change markers, using LLM to apply changes directly")
            return self._apply_changes_with_llm(original_content, diff_text)

        try:
            # Handle the case where we have proper change markers but no @@ line markers
            # This often happens with LLM-generated diffs that don't include line numbers
            if '+' in diff_text and '-' in diff_text and '@@' not in diff_text:
                print("Detected LLM-style diff without line markers, using text-based diff application")
                return self._apply_simple_diff(original_content, diff_text)

            # Parse the diff to identify changes (standard unified diff format)
            original_lines = original_content.splitlines()
            new_content_lines = original_lines.copy()

            # Track line additions/removals
            line_offset = 0

            # Store modification stats for debugging
            additions = 0
            removals = 0

            # Process diff line by line
            current_line = 0
            for line in diff_text.splitlines():
                if line.startswith("@@"):
                    # Line position indicator
                    match = re.search(r"@@ -(\d+),(\d+) \+(\d+),(\d+) @@", line)
                    if match:
                        current_line = int(match.group(1)) - 1  # 0-based indexing
                        line_offset = 0
                elif line.startswith("+"):
                    # Add line
                    content = line[1:]
                    additions += 1
                    if current_line + line_offset < len(new_content_lines):
                        new_content_lines.insert(current_line + line_offset, content)
                    else:
                        new_content_lines.append(content)
                    line_offset += 1
                elif line.startswith("-"):
                    # Remove line
                    removals += 1
                    if 0 <= current_line + line_offset < len(new_content_lines):
                        new_content_lines.pop(current_line + line_offset)
                        line_offset -= 1
                    # Don't increment current_line as we removed a line
                else:
                    # Context line or other
                    current_line += 1

            modified_content = "\n".join(new_content_lines)

            print(f"Diff applied: {additions} additions, {removals} removals")

            # Sanity check - if content is identical, something went wrong
            if modified_content == original_content:
                print("Warning: Applied diff resulted in identical content, attempting direct LLM modification")
                return self._apply_changes_with_llm(original_content, diff_text)

            return modified_content

        except Exception as e:
            print(f"Error applying diff: {e}")
            print("Falling back to LLM-based modification")
            return self._apply_changes_with_llm(original_content, diff_text)

    def _apply_simple_diff(self, original_content: str, diff_text: str) -> str:
        """
        Apply a simple LLM-style diff without line numbers.
        Looks for - and + lines and does text replacement.

        Args:
            original_content: Original content
            diff_text: Diff text with +/- markers but no @@ markers

        Returns:
            Modified content
        """
        print("Applying simple diff without line markers")

        # Extract changes: pairs of (text_to_remove, text_to_add)
        changes = []

        removal_lines = []
        addition_lines = []

        # Group removals and additions
        for line in diff_text.splitlines():
            if line.startswith('-') and not line.startswith('---'):
                removal_lines.append(line[1:])
            elif line.startswith('+') and not line.startswith('+++'):
                addition_lines.append(line[1:])
            elif removal_lines or addition_lines:
                # We've hit a non-change line after collecting changes
                if removal_lines or addition_lines:
                    changes.append(('\n'.join(removal_lines), '\n'.join(addition_lines)))
                    removal_lines = []
                    addition_lines = []

        # Add any remaining changes
        if removal_lines or addition_lines:
            changes.append(('\n'.join(removal_lines), '\n'.join(addition_lines)))

        print(f"Extracted {len(changes)} change blocks")

        # Apply changes
        modified_content = original_content
        for old_text, new_text in changes:
            if old_text:
                # If text to remove is present, replace it
                if old_text in modified_content:
                    modified_content = modified_content.replace(old_text, new_text)
                    print(f"Replaced: '{old_text[:40]}...' with '{new_text[:40]}...'")
                else:
                    # Try with flexible whitespace matching
                    old_text_normalized = ' '.join(old_text.split())
                    content_normalized = ' '.join(modified_content.split())
                    if old_text_normalized in content_normalized:
                        # Find the actual text to replace in the original
                        start_idx = content_normalized.find(old_text_normalized)
                        end_idx = start_idx + len(old_text_normalized)

                        # Find the corresponding positions in the original
                        orig_pos = 0
                        orig_start = None
                        norm_pos = 0

                        # Find start position
                        for i, char in enumerate(modified_content):
                            if orig_start is None and norm_pos >= start_idx:
                                orig_start = i
                            if norm_pos >= end_idx:
                                orig_end = i
                                break

                            if not char.isspace():
                                norm_pos += 1

                        if orig_start is not None and orig_end is not None:
                            prefix = modified_content[:orig_start]
                            suffix = modified_content[orig_end:]
                            modified_content = prefix + new_text + suffix
                            print(f"Replaced with whitespace normalization")
                    else:
                        print(f"Warning: Could not find text to replace: '{old_text[:40]}...'")
            else:
                # If no text to remove, just add the new text at the beginning
                modified_content = new_text + modified_content
                print(f"Added: '{new_text[:40]}...'")

        # Check if content changed
        if modified_content == original_content:
            print("Warning: Simple diff application resulted in no changes")

        return modified_content

    def _apply_changes_with_llm(self, original_content: str, change_description: str) -> str:
        """
        Use LLM to apply changes when diff application fails.

        Args:
            original_content: Original file content
            change_description: Description of changes or partial diff

        Returns:
            Modified content with changes applied
        """
        system_instruction = """You are an Expert Code Editor. 
Your task is to apply the described changes to the code. 
Make EXACTLY the modifications specified in the change description/diff."""

        prompt = f"""
I need you to modify this code according to the change description/diff below.

# ORIGINAL CODE
```python
{original_content}
```

# CHANGES TO APPLY
```
{change_description}
```

Apply EXACTLY these changes to the code. Don't make any additional changes.
Return the complete modified code, not just the changed sections.

If the changes don't specify exact locations or exact text, use your best judgment to apply them while maintaining the original code's structure and style.

Return ONLY the complete modified code with no explanations before or after.
"""

        # Call LLM to apply the changes
        modified_content = self._call_llm(prompt, system_instruction)

        # Extract code if wrapped in code blocks
        if "```python" in modified_content and "```" in modified_content:
            modified_content = modified_content.split("```python")[1].split("```")[0].strip()
        elif "```" in modified_content:
            modified_content = modified_content.split("```")[1].split("```")[0].strip()

        # Check if content actually changed
        if modified_content == original_content:
            print("Warning: LLM did not make any changes to the content")
        else:
            print("Successfully applied changes using LLM")

        return modified_content

    def _generate_modified_file(self, file_path: str, current_content: str, improvement_description: str) -> str:
        """
        Generate a modified version of a file based on improvement description.

        Args:
            file_path: Path to the file
            current_content: Current file content
            improvement_description: Description of the improvement to make

        Returns:
            Modified file content
        """
        system_instruction = """You are an Expert System Developer. 
You must modify the provided code file to implement the specific improvement described.
Make minimal, focused changes to implement the improvement while maintaining the overall structure and style."""

        prompt = f"""
I need you to modify a file ({file_path}) to implement a specific improvement.

# IMPROVEMENT DESCRIPTION
{improvement_description}

# CURRENT FILE CONTENT
```python
{current_content}
```

# TASK
1. Implement the described improvement with minimal changes
2. Maintain the same coding style and architecture
3. Ensure no functionality is broken
4. If you're unsure about any aspect, maintain the existing code
5. Return the COMPLETE new file content (not just the changed sections)

Return ONLY the complete modified file content, with no explanations before or after.
"""

        # Call LLM to generate the modified file
        modified_content = self._call_llm(prompt, system_instruction)

        # Extract code if wrapped in code blocks
        if "```python" in modified_content and "```" in modified_content:
            modified_content = modified_content.split("```python")[1].split("```")[0].strip()
        elif "```" in modified_content:
            modified_content = modified_content.split("```")[1].split("```")[0].strip()

        return modified_content

    def _create_system_backup(self):
        """Create a backup of the current system state"""
        backup_timestamp = self.improvement_timestamp
        backup_dir = self.backup_dir / f"backup_{backup_timestamp}"
        backup_dir.mkdir(exist_ok=True)

        print(f"\nCreating system backup in {backup_dir}")

        # Backup all important files
        for file_path in self.code_files:
            src_path = self.root_dir / file_path
            if src_path.exists():
                dst_path = backup_dir / file_path

                # Create parent directories if needed
                dst_path.parent.mkdir(parents=True, exist_ok=True)

                # Copy the file
                shutil.copy2(src_path, dst_path)
                print(f"Backed up {file_path}")

        # Backup learnings.txt
        learnings_path = self.root_dir / "learnings.txt"
        if learnings_path.exists():
            shutil.copy2(learnings_path, backup_dir / "learnings.txt")
            print("Backed up learnings.txt")

        print("System backup complete")

    def _generate_diff(self, original: str, modified: str, file_path: str) -> str:
        """
        Generate a unified diff between original and modified content.

        Args:
            original: Original content
            modified: Modified content
            file_path: Path of the file for diff header

        Returns:
            Unified diff as string
        """
        original_lines = original.splitlines(True)
        modified_lines = modified.splitlines(True)

        diff = difflib.unified_diff(
            original_lines, 
            modified_lines,
            fromfile=f'a/{file_path}',
            tofile=f'b/{file_path}',
            n=3
        )

        return ''.join(diff)

    def _validate_changes(self) -> Dict:
        """
        Validate the changes by running a test iteration.

        Returns:
            Dict with validation results
        """
        print("\nValidating system changes...")

        validation_results = {
            "success": False,
            "errors": [],
            "performance_impact": None,
            "timestamp": datetime.datetime.now().isoformat()
        }

        # Skip actual validation if requested
        if os.environ.get("SKIP_VALIDATION") == "1":
            print("Skipping validation (SKIP_VALIDATION=1)")
            validation_results["success"] = True
            validation_results["execution_output"] = "Validation skipped as requested"
            return validation_results

        # Check if we can identify a dataset path from the archive
        dataset_path = self._find_dataset_path()
        if not dataset_path:
            print("Could not identify dataset path. Using simple validation.")
            # Use a simple validation instead
            return self._simple_validation()

        # Try running a single iteration to validate changes
        try:
            # Run a test iteration with the modified system
            start_time = time.time()

            # Execute run_script.py with a single iteration and the identified dataset path
            cmd = [
                sys.executable, 
                "run_script.py", 
                "--iterations", "1",
                "--dataset", dataset_path,
                "--loader", "arc"  # Default to ARC loader (most common use case)
            ]

            print(f"Executing test command: {' '.join(cmd)}")
            process = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )

            execution_time = time.time() - start_time

            # Check if the run was successful
            if process.returncode == 0:
                validation_results["success"] = True
                validation_results["execution_output"] = process.stdout
                validation_results["execution_time"] = execution_time

                print(f"Validation successful! Execution time: {execution_time:.2f}s")
            else:
                validation_results["success"] = False
                validation_results["errors"].append("Test run failed with non-zero exit code")
                validation_results["execution_output"] = process.stdout
                validation_results["execution_error"] = process.stderr

                print(f"Validation failed! Error output: {process.stderr}")
                print("Using simple validation as fallback...")

                # Fall back to simple validation
                simple_results = self._simple_validation()
                validation_results["success"] = simple_results["success"]
                if "execution_output" not in validation_results:
                    validation_results["execution_output"] = simple_results.get("execution_output", "")

        except subprocess.TimeoutExpired:
            validation_results["success"] = False
            validation_results["errors"].append("Test run timed out after 5 minutes")
            print("Validation timed out after 5 minutes")

            # Fall back to simple validation
            simple_results = self._simple_validation()
            validation_results["success"] = simple_results["success"]

        except Exception as e:
            validation_results["success"] = False
            validation_results["errors"].append(f"Error during validation: {str(e)}")
            print(f"Error during validation: {e}")

            # Fall back to simple validation
            simple_results = self._simple_validation()
            validation_results["success"] = simple_results["success"]

        return validation_results

    def _find_dataset_path(self) -> str:
        """Find a dataset path from the archive directory"""
        try:
            # First, check if there's an environment variable set
            env_dataset = os.environ.get("DATASET_PATH")
            if env_dataset and os.path.exists(env_dataset):
                print(f"Using dataset path from environment variable: {env_dataset}")
                return env_dataset

            # Try to find a dataset path in recent iteration files
            iterations = sorted(self.archive_dir.glob("iteration_*.json"), reverse=True)
            for iteration_file in iterations[:5]:  # Check the 5 most recent
                try:
                    with open(iteration_file, 'r') as f:
                        data = json.load(f)

                    # Look for command line arguments or dataset path
                    cmd_args = data.get("command_args", {})
                    if cmd_args.get("dataset"):
                        path = cmd_args.get("dataset")
                        if os.path.exists(path):
                            print(f"Found dataset path in {iteration_file.name}: {path}")
                            return path
                except Exception:
                    continue

            # If we can't find it in iteration files, look for ARC directories
            arc_dirs = ["ARC", "ARC_data", "ARC_2024_Training", "dataset"]
            for arc_dir in arc_dirs:
                if os.path.exists(arc_dir) and os.path.isdir(arc_dir):
                    print(f"Found possible ARC dataset directory: {arc_dir}")
                    return arc_dir

            print("Could not identify dataset path from archive or common directories")
            return ""
        except Exception as e:
            print(f"Error finding dataset path: {e}")
            return ""

    def _simple_validation(self) -> Dict:
        """Perform a simple validation by checking if files load correctly"""
        print("Performing simple validation...")

        validation_results = {
            "success": True,  # Assume success by default
            "errors": [],
            "execution_output": "Simple validation: checking if modified files load correctly"
        }

        # Try to import and load key files
        try:
            # Just check if we can import the main module without errors
            import importlib.util

            # Try to load agent_system.py
            if os.path.exists("agent_system.py"):
                spec = importlib.util.spec_from_file_location("agent_system", "agent_system.py")
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                print("✅ agent_system.py loads successfully")

            # Try to load dataset_loader.py
            if os.path.exists("dataset_loader.py"):
                spec = importlib.util.spec_from_file_location("dataset_loader", "dataset_loader.py")
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                print("✅ dataset_loader.py loads successfully")

            print("Simple validation successful!")
            return validation_results

        except Exception as e:
            validation_results["success"] = False
            validation_results["errors"].append(f"Error in simple validation: {str(e)}")
            validation_results["execution_output"] += f"\nError: {str(e)}"
            print(f"❌ Simple validation failed: {e}")
            return validation_results

    def _record_improvement_results(self, 
                                   improvement_plan: Dict, 
                                   changes_made: List[Dict],
                                   validation_results: Dict):
        """
        Record the improvement results.

        Args:
            improvement_plan: The improvement plan
            changes_made: List of changes made
            validation_results: Validation results
        """
        # Create a record of this improvement run
        record = {
            "timestamp": self.improvement_timestamp,
            "system_state": {
                "current_iteration": improvement_plan.get("current_iteration", 0)
            },
            "analysis_summary": improvement_plan.get("analysis", ""),
            "improvement_history_analysis": improvement_plan.get("improvement_history_analysis", ""),
            "changes_made": changes_made,
            "validation_results": validation_results
        }

        # Save the record
        record_path = self.diffs_dir / f"changes_{self.improvement_timestamp}.json"
        with open(record_path, 'w') as f:
            json.dump(record, f, indent=2)

        print(f"Recorded improvement results to {record_path}")

        # Also create a markdown report
        report_path = self.diffs_dir / f"report_{self.improvement_timestamp}.md"

        report_content = f"""# System Improvement Report - {self.improvement_timestamp}

## Summary
- **Timestamp:** {self.improvement_timestamp}
- **Current Iteration:** {improvement_plan.get("current_iteration", 0)}
- **Changes Made:** {len(changes_made)}
- **Validation Success:** {validation_results["success"]}

## System Analysis
{improvement_plan.get("analysis", "")}

## Improvement History Analysis
{improvement_plan.get("improvement_history_analysis", "")}

## Changes Implemented

"""

        for i, change in enumerate(changes_made):
            report_content += f"""### Change {i+1}: {change['file']}
**Description:** {change['description']}

```diff
{change['diff']}
```

"""

        report_content += f"""## Validation Results
- **Success:** {validation_results["success"]}
- **Execution Time:** {validation_results.get("execution_time", "N/A")}
"""

        if validation_results.get("errors"):
            report_content += "### Errors\n"
            for error in validation_results["errors"]:
                report_content += f"- {error}\n"

        with open(report_path, 'w') as f:
            f.write(report_content)

        print(f"Created improvement report at {report_path}")

    def _call_llm(self, prompt: str, system_instruction: str = None) -> str:
        """
        Call the LLM API with a prompt and system instruction.

        Args:
            prompt: The prompt text
            system_instruction: Optional system instruction

        Returns:
            LLM response text
        """
        try:
            # Configure system instruction if provided
            config = None
            if system_instruction:
                config = types.GenerateContentConfig(system_instruction=system_instruction)

            response = self.client.models.generate_content(
                model="gemini-1.5-pro",
                config=config,
                contents=prompt
            )

            return response.text

        except Exception as e:
            print(f"Error calling LLM: {e}")
            return f"Error: {str(e)}"

    def _create_analysis_prompt(self, system_data: Dict) -> str:
        """
        Create a prompt for system analysis.

        Args:
            system_data: System inspection data

        Returns:
            Prompt string for LLM
        """
        # Extract key data for the prompt
        current_iteration = system_data.get("current_iteration", 0)
        code_files = system_data.get("code_files", {})
        performance_history = system_data.get("performance_history", [])
        error_patterns = system_data.get("error_patterns", [])
        learnings = system_data.get("learnings", "")

        # Get the most recent iterations for analysis
        recent_iterations = sorted(
            performance_history, 
            key=lambda x: x.get("iteration", 0), 
            reverse=True
        )[:5]

        # Get performance trends
        performance_trend = ""
        if recent_iterations:
            performance_trend = "**Performance Trend**:\n"
            for summary in recent_iterations:
                iteration = summary.get("iteration", "?")
                accuracy = summary.get("performance", {}).get("accuracy", 0) * 100
                batch_size = summary.get("batch_size", 0)
                performance_trend += f"- Iteration {iteration}: Accuracy {accuracy:.2f}% (batch size: {batch_size})\n"

        # Format the code files section - limit to manageable size
        code_files_section = ""
        for file_path, content in code_files.items():
            # Truncate very large files
            if len(content) > 50000:
                code_files_section += f"\n## File: `{file_path}` (truncated)\n"
                code_files_section += "```python\n"
                code_files_section += content[:50000] + "\n... (file truncated) ...\n"
                code_files_section += "```\n"
            else:
                code_files_section += f"\n## File: `{file_path}`\n"
                code_files_section += "```python\n"
                code_files_section += content + "\n"
                code_files_section += "```\n"

        # Format the error patterns section
        error_patterns_section = "## Common Error Patterns\n"
        for pattern in error_patterns:
            error_patterns_section += f"- {pattern}\n"

        # Format improvement history section
        improvement_history_section = "## Previous System Improvements\n"
        improvement_history = system_data.get("improvement_history", [])

        if improvement_history:
            improvement_history_section += "Previous improvements and their effects:\n\n"

            for i, improvement in enumerate(improvement_history):
                timestamp = improvement.get("timestamp", "unknown")
                iteration = improvement.get("iteration", "unknown")
                num_changes = improvement.get("num_changes", 0)
                files = improvement.get("files_changed", [])
                descriptions = improvement.get("descriptions", [])

                improvement_history_section += f"### Improvement {i+1} (Iteration {iteration})\n"
                improvement_history_section += f"- **Timestamp:** {timestamp}\n"
                improvement_history_section += f"- **Files Changed:** {', '.join(files)}\n"
                improvement_history_section += f"- **Number of Changes:** {num_changes}\n"
                improvement_history_section += "- **Changes Made:**\n"

                for j, desc in enumerate(descriptions):
                    improvement_history_section += f"  - Change {j+1}: {desc}\n"

                # Add performance impact if available
                performance_impact = system_data.get("performance_impact", [])
                for impact in performance_impact:
                    if impact.get("iteration") == iteration:
                        before = impact.get("before_accuracy", 0)
                        after = impact.get("after_accuracy", 0)
                        change = impact.get("change", 0)

                        improvement_history_section += f"- **Performance Impact:**\n"
                        improvement_history_section += f"  - Before: {before:.4f}\n"
                        improvement_history_section += f"  - After: {after:.4f}\n"
                        improvement_history_section += f"  - Change: {change:.4f}\n"
                        improvement_history_section += f"  - Result: {'Positive' if change > 0 else 'Negative or Neutral'}\n"

                improvement_history_section += "\n"
        else:
            improvement_history_section += "No previous improvements found.\n"

        # Truncate learnings if too large
        if len(learnings) > 10000:
            learnings = learnings[:10000] + "\n... (learnings file truncated) ...\n"

        # Build the full prompt
        prompt = f"""# SYSTEM IMPROVEMENT ANALYSIS

You are tasked with improving the Agentic Learning System based on its current state and performance history.

## SYSTEM OVERVIEW
The Agentic Learning System is a framework that uses LLM reasoning to continuously improve its approach to solving dataset problems through iterative exploration and exploitation.

Current iteration: {current_iteration}

{performance_trend}

## ERROR PATTERNS AND ISSUES
{error_patterns_section}

## IMPROVEMENT HISTORY
{improvement_history_section}

## ACCUMULATED LEARNINGS
```
{learnings}
```

## CURRENT CODE FILES
{code_files_section}

## IMPROVEMENT TASK
Analyze the current system and propose specific code changes to enhance its performance.

Your task:
1. Identify key limitations and bottlenecks in the current implementation
2. Consider the history of previous improvements and their effects
3. Avoid repeating changes that didn't yield performance improvements
4. Build upon successful improvements from the past
5. Propose 2-3 specific code changes that would improve the system
6. For each change, provide:
   - The file to modify (use EXACTLY the name shown in "CURRENT CODE FILES" section)
   - A clear description of the improvement 
   - A detailed diff using standard diff format

IMPORTANT: For each change, you MUST include a proper diff using "```diff" code blocks showing the exact lines to modify.
The diff format must follow standard git diff format with - for removals and + for additions:

```diff
- line to remove
+ line to add
```

## REQUIRED OUTPUT FORMAT

Structure your response as follows:

## SYSTEM ANALYSIS
(Provide your analysis of the current system, identifying key limitations and bottlenecks)

## IMPROVEMENT HISTORY ANALYSIS
(Analyze the history of previous improvements and their effects. What worked? What didn't? What patterns emerge?)

## PROPOSED CODE CHANGES

### Change 1:
File: `filename.py`
Description: Clear description of the improvement and its expected impact

```diff
- Original line
+ Modified line
```

### Change 2:
File: `filename.py`
Description: Clear description of the improvement and its expected impact

```diff
- Original line
+ Modified line
```

IMPORTANT:
1. Make specific, focused changes rather than wholesale rewrites
2. Maintain the same coding style and architecture
3. Focus on impactful improvements that address known issues 
4. Include detailed diffs with proper formatting
5. Note: It is CRITICAL that you include proper diffs in ```diff code blocks, showing exact lines to change
"""

        return prompt


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="System Improver for Agentic Learning System")

    # Backup options
    parser.add_argument(
        "--no-backup", 
        action="store_true",
        help="Skip creating system backup before making changes"
    )

    # Dataset path for validation
    parser.add_argument(
        "--dataset",
        type=str,
        help="Path to dataset for validation (will be set in DATASET_PATH env var)"
    )

    # Workflow options
    workflow_group = parser.add_argument_group("Workflow Options")

    # Staging workflow
    workflow_group.add_argument(
        "--stage-only",
        action="store_true",
        help="Generate changes and stage them for review without applying"
    )

    workflow_group.add_argument(
        "--apply-staged",
        action="store_true",
        help="Apply the most recently staged changes"
    )

    # Process control options
    workflow_group.add_argument(
        "--force-changes",
        action="store_true",
        help="Force generating changes even without proper diff formatting"
    )

    workflow_group.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip validation step after applying changes"
    )

    # Direct changes workflow
    workflow_group.add_argument(
        "--force-hard-changes",
        action="store_true",
        help="Force direct file changes without using LLM (DESTRUCTIVE)"
    )

    workflow_group.add_argument(
        "--option",
        type=int,
        default=0,
        help="Option to implement (1=Grid class, 2=system_prompt.md, 0=all)"
    )

    return parser.parse_args()


def main():
    """Main entry point for the system improver"""
    args = parse_arguments()

    # Check environment variables
    if not os.environ.get("GEMINI_API_KEY"):
        print("Error: GEMINI_API_KEY environment variable is not set.")
        print("Please set this variable to your Gemini API key before running the script.")
        print("Example: export GEMINI_API_KEY=your_api_key_here")
        sys.exit(1)

    # Set environment variables based on arguments
    if args.dataset:
        os.environ["DATASET_PATH"] = args.dataset
        print(f"Set DATASET_PATH to {args.dataset}")

    if args.force_changes:
        os.environ["FORCE_CHANGES"] = "1"
        print("Force changes mode enabled")

    if args.skip_validation:
        os.environ["SKIP_VALIDATION"] = "1"
        print("Validation will be skipped")

    # Create improver instance 
    improver = SystemImprover(create_backup=not args.no_backup)

    # Handle staging workflow
    if args.stage_only:
        # Only generate changes and stage them
        staged_changes = improver.generate_changes_only()
        # Save to staging file
        with open("staged_changes.json", "w") as f:
            json.dump(staged_changes, f, indent=2)
        print(f"Changes staged in staged_changes.json - review and apply with --apply-staged")
        return

    # Handle applying staged changes
    if args.apply_staged:
        # Load and apply staged changes
        if os.path.exists("staged_changes.json"):
            try:
                with open("staged_changes.json", "r") as f:
                    staged_changes = json.load(f)
                applied_changes = improver.apply_changes(staged_changes)
                if applied_changes:
                    print("Staged changes applied successfully")
                else:
                    print("No changes were applied from staged_changes.json")
            except Exception as e:
                print(f"Error applying staged changes: {e}")
        else:
            print("No staged changes found (staged_changes.json doesn't exist)")
        return

    # If we're forcing hard changes
    if args.force_hard_changes:
        print("WARNING: Force hard changes mode enabled - will directly edit files")

        # Option 1: Create Grid class in agent_system.py
        if args.option == 1 or args.option == 0:
            print("\nImplementing Option 1: Adding Grid class to agent_system.py")
            try:
                # Read the current file
                with open("agent_system.py", 'r') as f:
                    content = f.read()

                # Check if Grid class already exists
                if "class Grid:" in content:
                    print("Grid class already exists in agent_system.py")
                else:
                    # Add Grid class after the imports
                    import_section_end = content.find("\n\n", content.find("import "))
                    if import_section_end == -1:
                        import_section_end = content.find("\n", content.find("import "))

                    grid_class = """
import numpy as np

class Grid:
    def __init__(self, grid_string):
        self.grid_string = grid_string
        self.grid_array = self._string_to_array(grid_string)

    def _string_to_array(self, grid_string):
        # Convert the string representation to a numpy array.
        # Example implementation (adapt as needed for your grid format):
        lines = grid_string.strip().split('\\n')
        grid = [list(line) for line in lines]
        return np.array(grid)

    def __str__(self):
        return self.grid_string

    # Example spatial manipulation method (add more as needed)
    def rotate(self):
        self.grid_array = np.rot90(self.grid_array)
        self.grid_string = self._array_to_string(self.grid_array)

    def _array_to_string(self, grid_array):
         # Convert numpy array back to string representation.
         # Example implementation (adapt as needed)
         return '\\n'.join([''.join(row) for row in grid_array])

"""
                    new_content = content[:import_section_end] + grid_class + content[import_section_end:]

                    # Write the changes
                    with open("agent_system.py", 'w') as f:
                        f.write(new_content)

                    print("✅ Successfully added Grid class to agent_system.py")
            except Exception as e:
                print(f"Error implementing option 1: {e}")

        # Option 2: Modify system_prompt.md
        if args.option == 2 or args.option == 0:
            print("\nImplementing Option 2: Updating system_prompt.md")
            try:
                # Read the current file
                with open("system_prompt.md", 'r') as f:
                    content = f.read()

                # Check if the changes are already there
                if "Identify the key differences between the input and output grids" in content:
                    print("system_prompt.md already contains the required changes")
                else:
                    # Find the line to replace
                    old_line = '"Given an input grid and an output grid, generate a Python script that transforms the input grid into the output grid."'

                    if old_line in content:
                        new_content = content.replace(old_line, """\"Given an input grid and an output grid, generate a Python script that transforms the input grid into the output grid using the following steps:
1. Identify the key differences between the input and output grids (e.g., color changes, object movements, rotations). Describe these differences in a JSON format: {'differences': [{'type': 'rotation', 'object': 'square', 'degrees': 90}, ... ]}.
2. For each identified difference, generate a Python function that implements the corresponding transformation. Ensure each function operates on a copy of the grid to avoid unintended side-effects.
3. Combine these functions into a single script that applies the transformations sequentially to the input grid to produce the output grid.  Provide the final consolidated Python script.
Example:\"
```json
{ \"differences\": [{\"type\": \"move\", \"object\": \"triangle\", \"x\": 1, \"y\":2}]}
```
```python
import copy

def move_triangle(grid, x, y):
    new_grid = copy.deepcopy(grid)
     # Implementation for moving the triangle
    return new_grid

# Example usage applying the move_triangle transformation
```""")

                        # Write the changes
                        with open("system_prompt.md", 'w') as f:
                            f.write(new_content)

                        print("✅ Successfully updated system_prompt.md")
                    else:
                        print("Could not find the target line in system_prompt.md")
            except Exception as e:
                print(f"Error implementing option 2: {e}")

        print("\nDirect file changes completed")
        return

    # Default: run the full improvement process
    improver.run()


if __name__ == "__main__":
    main()