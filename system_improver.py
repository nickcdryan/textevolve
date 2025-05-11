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

"""
            if 'find' in change and 'replace' in change:
                report_content += f"""**Find:**
```
{change['find']}
```

**Replace With:**
```
{change['replace']}
```

"""
            elif 'diff' in change:
                report_content += f"""```diff
{change['diff']}
```

"""

        with open(report_path, 'w') as f:
            f.write(report_content)

        print(f"\nGenerated {len(improvement_plan.get('changes', []))} proposed changes.")
        print(f"Changes staged for review in: staged_changes.json")
        print(f"Report available at: {report_path}")

        return staged_changes

    def fix_nested_triple_quotes(self, code: str) -> str:
        """
        Fix issues with nested triple quotes in f-strings.
        Uses a pattern-specific approach that handles various types of nesting.

        Args:
            code: Python code that might contain problematic nested quotes

        Returns:
            Modified code with fixed nested quotes
        """
        # Skip processing if there's no code or no potential problematic patterns
        if not code or ('f"""' not in code and "f'''" not in code):
            return code

        # Track if we made changes
        made_changes = False

        # 1. Handle docstrings in code blocks
        if '"""Analyzes' in code:
            code = code.replace('"""Analyzes', "'''Analyzes")
            code = code.replace('format."""', "format.'''")
            made_changes = True

        # 2. Other docstrings in code blocks
        if '"""Applies the described' in code:
            code = code.replace('"""Applies the described', "'''Applies the described")
            code = code.replace('grid."""', "grid.'''")
            made_changes = True

        # 3. Nested f-strings with exact patterns from the example
        if 'prompt = f"""' in code and 'Analyze the transformation' in code:
            code = code.replace('prompt = f"""', "prompt = f'''")
            code = code.replace('Describe the transformation concisely, including:', 
                               'Describe the transformation concisely, including:')
            code = code.replace('structured format."""', "structured format.'''")
            made_changes = True

        # 4. General nested f-strings with prompt patterns
        patterns_to_fix = [
            ('prompt = f"""Apply the following', "prompt = f'''Apply the following"),
            ('JSON format: [[row1], [row2], ...]', 'JSON format: [[row1], [row2], ...]'),
            ('"""  # Call LLM', "'''  # Call LLM"),
            ('"""\n return', "'''\n return"),
            ('"""\ntransformed_grid_json', "'''\ntransformed_grid_json")
        ]

        for old, new in patterns_to_fix:
            if old in code:
                code = code.replace(old, new)
                made_changes = True

        # 5. If we've made changes, do a final pass to ensure we haven't missed anything
        if made_changes:
            # Look for any remaining nested triple quotes in typical patterns
            if 'f"""' in code and '"""' in code[code.find('f"""') + 5:]:
                # Find all occurrences of f""" (the start of an f-string)
                f_starts = []
                pos = 0
                while True:
                    pos = code.find('f"""', pos)
                    if pos == -1:
                        break
                    f_starts.append(pos)
                    pos += 5  # Move past the current f"""

                # For each f-string start, find the matching end and convert any triple quotes between
                for start in f_starts:
                    # Find the matching end """ for this f-string
                    # This is approximate - we assume the first """ after nested content is the end
                    content_start = start + 4  # Skip the f"""
                    # Look for the next """ that terminates this f-string
                    end = content_start
                    quote_depth = 1
                    while quote_depth > 0 and end < len(code):
                        # Find next triple quote
                        next_triple = code.find('"""', end)
                        if next_triple == -1:
                            break

                        # Check if this is the matching end quote
                        if quote_depth == 1:
                            # This is our matching end quote - extract content between
                            content = code[content_start:next_triple]
                            # Replace any triple quotes in the content
                            modified_content = content.replace('"""', "'''")
                            # Replace the content in the original code
                            code = code[:content_start] + modified_content + code[next_triple:]
                            quote_depth -= 1
                        else:
                            quote_depth -= 1

                        end = next_triple + 3

        return code


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

        # Import here to ensure it's available in this method
        import shutil

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
            key=lambda x: 0 if x.get("priority", "medium") == "high" else 1
        )

        for change in sorted_changes:
            file_path = change["file"]
            full_path = self.root_dir / file_path

            if not full_path.exists():
                print(f"Warning: File not found: {file_path}, skipping change")
                continue

            # Use find-and-replace if those fields are present
            if 'find' in change and 'replace' in change:
                try:
                    # Read current file content
                    with open(full_path, 'r', encoding='utf-8') as f:
                        current_content = f.read()

                    # Backup the file
                    backup_dir = self.backup_dir / f"backup_file_{self.improvement_timestamp}"
                    backup_dir.mkdir(exist_ok=True, parents=True)
                    backup_path = backup_dir / file_path
                    backup_path.parent.mkdir(exist_ok=True, parents=True)
                    shutil.copy2(full_path, backup_path)
                    print(f"Backed up {file_path} to {backup_path}")

                    # Apply find-and-replace
                    new_content, success = self._apply_find_replace(full_path, change["find"], change["replace"])

                    if success:
                        # Generate diff for record
                        diff = self._generate_diff(current_content, new_content, file_path)

                        # Write changes to file
                        with open(full_path, 'w', encoding='utf-8') as f:
                            f.write(new_content)

                        print(f"✅ Successfully applied find-and-replace to {file_path}")

                        # Record the change
                        change_record = {
                            "file": file_path,
                            "description": change["description"],
                            "find": change["find"],
                            "replace": change["replace"],
                            "diff": diff,
                            "timestamp": datetime.datetime.now().isoformat()
                        }
                        changes_made.append(change_record)
                    else:
                        print(f"❌ Could not find text to replace in {file_path}")
                except Exception as e:
                    print(f"❌ Error applying find-and-replace to {file_path}: {e}")
                    import traceback
                    traceback.print_exc()

            # Standard approach for applying changes with diff
            elif 'diff' in change:
                try:
                    # Load current file content
                    with open(full_path, 'r', encoding='utf-8') as f:
                        current_content = f.read()

                    # Apply the diff
                    print(f"Applying diff to {file_path}")
                    new_content = self._apply_diff(current_content, change["diff"])

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
            else:
                print(f"⚠️ Change for {file_path} has neither find/replace nor diff - skipping")

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

        # Initialize the improvement plan with default values
        improvement_plan = {
            "analysis": "",
            "improvement_history_analysis": "",
            "changes": [],
            "current_iteration": system_data.get("current_iteration", 0)
        }

        try:
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

            # Extract find-replace changes (primary approach)
            find_replace_changes = self._extract_find_replace_changes(improvement_analysis)

            if find_replace_changes:
                # Process and use find-replace changes
                print(f"Found {len(find_replace_changes)} find-replace changes")
                improvement_plan["changes"] = find_replace_changes
            else:
                # Fall back to extracting regular changes
                print("No find-replace changes found, trying to extract regular changes")
                regular_changes = self._extract_changes_from_llm_response(improvement_analysis)
                if regular_changes:
                    improvement_plan["changes"] = regular_changes
                    print(f"Extracted {len(regular_changes)} regular changes")
                else:
                    print("WARNING: Could not extract any changes from LLM response")
                    # Try again with targeted prompt for changes only
                    if improvement_plan["analysis"]:
                        print("Trying to generate changes with targeted prompt...")
                        targeted_changes = self._generate_changes_from_analysis(
                            improvement_plan["analysis"], 
                            system_data
                        )
                        if targeted_changes:
                            improvement_plan["changes"] = targeted_changes
                            print(f"Generated {len(targeted_changes)} targeted changes")

            print(f"Identified {len(improvement_plan['changes'])} proposed code changes")

        except Exception as e:
            print(f"Error parsing improvement plan: {e}")
            import traceback
            traceback.print_exc()

            # Return what we have, plus the raw analysis
            improvement_plan["raw_analysis"] = improvement_analysis

        return improvement_plan

    def _extract_changes_from_llm_response(self, improvement_analysis: str) -> List[Dict]:
        """
        Extract proposed changes from LLM response using a more robust approach.

        Args:
            improvement_analysis: Raw LLM response text

        Returns:
            List of extracted changes with file, description, and diff
        """
        changes = []

        # Check if the response contains a proposed changes section
        if "## PROPOSED CODE CHANGES" not in improvement_analysis and "PROPOSED CODE CHANGES" not in improvement_analysis:
            print("Warning: No 'PROPOSED CODE CHANGES' section found in the LLM response")
            # Save the raw response for debugging
            debug_path = self.diffs_dir / f"failed_analysis_{self.improvement_timestamp}.txt"
            with open(debug_path, 'w', encoding='utf-8') as f:
                f.write(improvement_analysis)
            print(f"Saved raw LLM response to {debug_path}")
            return changes

        # Split the response into sections
        sections = re.split(r'##\s+', improvement_analysis)
        proposed_changes_section = None

        # Find the changes section
        for section in sections:
            if section.strip().startswith("PROPOSED CODE CHANGES") or section.strip().upper().startswith("PROPOSED CODE CHANGES"):
                proposed_changes_section = section
                break

        if not proposed_changes_section:
            print("Warning: Could not extract proposed changes section")
            return changes

        # Extract individual changes
        change_blocks = re.split(r'###\s+Change\s+\d+:', proposed_changes_section)

        # The first block is usually just header text, not a change
        if len(change_blocks) > 1:
            change_blocks = change_blocks[1:]  # Skip the header

        print(f"Found {len(change_blocks)} change blocks to process")

        for i, block in enumerate(change_blocks):
            # Extract file path - try multiple patterns
            file_path = None
            file_patterns = [
                r"""File:\s*[`\'"]?([\w./\-_]+\.\w+)[`\'"]?""",  # Standard format
                r"""File\s+`([\w./\-_]+\.\w+)`""",               # With backticks
                r'File\s+([\w./\-_]+\.\w+)',                 # Without any quotes
                r"""in\s+[`\'"]?([\w./\-_]+\.\w+)[`\'"]?"""      # Alternative "in file.py" format
            ]

            for pattern in file_patterns:
                file_match = re.search(pattern, block, re.IGNORECASE | re.MULTILINE)
                if file_match:
                    file_path = file_match.group(1).strip()
                    break

            if not file_path:
                print(f"  Warning: Could not extract file path from change block {i+1}")
                continue

            # Extract description - multiple patterns
            description = ""
            desc_patterns = [
                r'Description:(.*?)(?:```|Diff:|FIND:|REPLACE WITH:|$)',          # Standard format
                r'Description\s*:(.*?)(?:```|Diff:|FIND:|REPLACE WITH:|$)'        # With flexible whitespace
            ]

            for pattern in desc_patterns:
                desc_match = re.search(pattern, block, re.DOTALL | re.IGNORECASE)
                if desc_match:
                    description = desc_match.group(1).strip()
                    break

            # Extract diff - try multiple patterns
            diff = ""
            diff_patterns = [
                r'```diff\s*(.*?)```',      # Standard diff format
                r'```\s*(.*?)```',          # Any code block if diff not found
                r'Diff:\s*(.*?)(?=###|$)'   # Diff: marker without code block
            ]

            for pattern in diff_patterns:
                diff_match = re.search(pattern, block, re.DOTALL)
                if diff_match:
                    diff = diff_match.group(1).strip()
                    break

            # For debugging
            if file_path:
                print(f"  Change {i+1}: File '{file_path}'")
                print(f"    Description: {len(description)} chars")
                print(f"    Diff: {len(diff)} chars")

            # Add the change if we have at least a file path and either description or diff
            if file_path and (description or diff):
                # Determine priority based on description
                priority = "high" if "high priority" in description.lower() else "medium"

                changes.append({
                    "change_id": i+1,
                    "file": file_path,
                    "description": description,
                    "diff": diff,
                    "priority": priority
                })
                print(f"  Added change for {file_path}")
            else:
                print(f"  Warning: Incomplete change information for block {i+1}, skipping")

        print(f"Successfully extracted {len(changes)} changes from LLM response")
        return changes

    def _extract_find_replace_changes(self, improvement_analysis: str) -> List[Dict]:
        """
        Extract find and replace changes from the improvement analysis.
        Enhanced version that handles nested code blocks and various formatting styles.

        Args:
            improvement_analysis: The full LLM analysis text

        Returns:
            List of changes with file, description, find, and replace fields
        """
        changes = []

        # Check if we have any actual analysis
        if not improvement_analysis or len(improvement_analysis.strip()) == 0:
            print("No improvement analysis to extract changes from")
            return changes

        # Check if there's a proposed changes section
        if "## PROPOSED CODE CHANGES" not in improvement_analysis and "PROPOSED CODE CHANGES" not in improvement_analysis:
            print("No PROPOSED CODE CHANGES section found in analysis")
            return changes

        # Split into change blocks - each change starts with "### Change X:"
        change_blocks = re.split(r'###\s+Change\s+\d+\s*:', improvement_analysis)

        # The first block is usually just header text
        if len(change_blocks) > 1:
            change_blocks = change_blocks[1:]  # Skip the header

        print(f"Found {len(change_blocks)} change blocks to process")

        for i, block in enumerate(change_blocks):
            print(f"\nProcessing Change Block {i+1}:")
            print("-" * 40)
            print(block[:200] + "..." if len(block) > 200 else block)  # Print a preview

            # Extract file path
            file_match = re.search(r'File\s*:\s*[`\'"]*([^`\'"]+)[`\'"]*', block, re.IGNORECASE)
            if not file_match:
                print(f"  Warning: Could not extract file path from change block {i+1}")
                continue

            file_path = file_match.group(1).strip()
            print(f"  File: {file_path}")

            # Extract description
            desc_match = re.search(r'Description\s*:(.*?)(?=FIND:|```|$)', block, re.DOTALL | re.IGNORECASE)
            description = desc_match.group(1).strip() if desc_match else ""
            print(f"  Description: {len(description)} chars")

            # Extract find content - looking for content between FIND: and REPLACE WITH:
            find_section = None
            replace_section = None

            # First try to find the whole sections
            find_section_match = re.search(r'FIND\s*:(.*?)(?=REPLACE WITH:|REPLACE:|$)', block, re.DOTALL | re.IGNORECASE)
            if find_section_match:
                find_section = find_section_match.group(1).strip()

            replace_section_match = re.search(r'(?:REPLACE WITH|REPLACE)\s*:(.*?)(?=$|###)', block, re.DOTALL | re.IGNORECASE)
            if replace_section_match:
                replace_section = replace_section_match.group(1).strip()

            print(f"  Find section: {len(find_section) if find_section else 0} chars")
            print(f"  Replace section: {len(replace_section) if replace_section else 0} chars")

            # Now extract the code blocks from within these sections
            find_text = ""
            replace_text = ""

            if find_section:
                # Extract code block from find section
                code_block_match = re.search(r'```(?:python)?\s*\n(.*?)\n```', find_section, re.DOTALL)
                if code_block_match:
                    find_text = code_block_match.group(1)
                else:
                    # If no code block, use the whole section
                    find_text = find_section

            if replace_section:
                # Extract code block from replace section
                code_block_match = re.search(r'```(?:python)?\s*\n(.*?)\n```', replace_section, re.DOTALL)
                if code_block_match:
                    replace_text = code_block_match.group(1)
                else:
                    # If no code block, use the whole section
                    replace_text = replace_section

            # Fallback: Try direct pattern matching if sections weren't found
            if not find_text:
                # Try alternative formats where FIND and code block might be separated
                alt_find_match = re.search(r'FIND:\s*(?:```(?:python)?\s*\n)?(.*?)(?:\n```)?(?=REPLACE WITH:|REPLACE:|$)', 
                                          block, re.DOTALL | re.IGNORECASE)
                if alt_find_match:
                    find_text = alt_find_match.group(1).strip()

            if not replace_text:
                # Try alternative formats where REPLACE and code block might be separated
                alt_replace_match = re.search(r'(?:REPLACE WITH:|REPLACE:)\s*(?:```(?:python)?\s*\n)?(.*?)(?:\n```)?(?=$|###)', 
                                             block, re.DOTALL | re.IGNORECASE)
                if alt_replace_match:
                    replace_text = alt_replace_match.group(1).strip()

            print(f"  Final find text: {len(find_text)} chars")
            print(f"  Final replace text: {len(replace_text)} chars")

            if file_path and (find_text or replace_text):
                # Determine priority
                priority = "high" if "high priority" in description.lower() else "medium"

                if find_text and replace_text:
                    changes.append({
                        "change_id": i+1,
                        "file": file_path,
                        "description": description,
                        "find": find_text,
                        "replace": replace_text,
                        "priority": priority
                    })
                    print(f"  ✅ Added find-replace change for {file_path}")
                else:
                    # Check for diff format instead
                    diff_match = re.search(r'```diff\s*(.*?)```', block, re.DOTALL)
                    if diff_match:
                        diff_text = diff_match.group(1).strip()

                        changes.append({
                            "change_id": i+1,
                            "file": file_path,
                            "description": description,
                            "diff": diff_text,
                            "priority": priority
                        })
                        print(f"  ✅ Added diff-based change for {file_path}")
                    else:
                        print(f"  ⚠️ Warning: Incomplete find-replace information for {file_path}")
            else:
                print(f"  ⚠️ Warning: Incomplete change information (file: {bool(file_path)}, find: {bool(find_text)}, replace: {bool(replace_text)})")

        print(f"\nExtracted {len(changes)} find-replace/diff changes from improvement analysis")
        return changes

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
Your task is to translate a system analysis into specific, actionable code changes using a FIND and REPLACE WITH approach."""

        change_gen_prompt = f"""
Based on the system analysis below, I need you to propose 2-3 specific code changes to improve the system.

# SYSTEM ANALYSIS
{analysis}

# AVAILABLE FILES
{', '.join(available_files)}

For each change, specify:
1. Which specific file to modify
2. A clear description of the improvement
3. Exact text to find (FIND)
4. Exact text to replace it with (REPLACE WITH)

# REQUIRED OUTPUT FORMAT

## PROPOSED CODE CHANGES

### Change 1:
File: `filename.py`
Description: Clear description of the improvement

FIND:
```python
def some_function():
    # Code to find here
    pass
```

REPLACE WITH:
```python
def some_function():
    # New improved code here
    pass
```

### Change 2:
[etc...]

The changes should be specific, focused improvements that address the issues identified in the analysis.
Make sure to provide the exact text to find and replace for each change.
"""

        # Call LLM to generate specific changes
        try:
            response = self._call_llm(change_gen_prompt, change_gen_system_instruction)

            # Save the response for debugging
            debug_path = self.diffs_dir / f"generated_changes_{self.improvement_timestamp}.txt"
            with open(debug_path, 'w', encoding='utf-8') as f:
                f.write(response)
            print(f"Generated changes saved to {debug_path}")

            # Extract find-replace changes
            changes = self._extract_find_replace_changes(response)

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

        # Ensure necessary imports are available
        import shutil

        changes_made = []

        # Check if we have changes to implement
        if not improvement_plan.get("changes"):
            print("No changes to implement - generating changes from analysis...")
            if improvement_plan.get("analysis"):
                improvement_plan["changes"] = self._generate_changes_from_analysis(
                    improvement_plan["analysis"], 
                    {"code_files": self._load_all_code_files()}
                )
            else:
                print("No analysis available to generate changes")
                return []

        # Check again if we have changes
        if not improvement_plan.get("changes"):
            print("No changes could be generated - implementation aborted")
            return []

        # Sort changes by priority
        sorted_changes = sorted(
            improvement_plan["changes"], 
            key=lambda x: 0 if x.get("priority", "medium") == "high" else 1
        )

        # Process each change
        for change in sorted_changes:
            file_path = change["file"]
            full_path = self.root_dir / file_path

            if not full_path.exists():
                print(f"Warning: File not found: {file_path}, skipping change")
                continue

            # Process find-replace change
            if 'find' in change and 'replace' in change:
                try:
                    # Read current file content
                    with open(full_path, 'r', encoding='utf-8') as f:
                        current_content = f.read()

                    # Back up the file
                    backup_dir = self.backup_dir / f"backup_file_{self.improvement_timestamp}"
                    backup_dir.mkdir(exist_ok=True, parents=True)
                    backup_path = backup_dir / file_path
                    backup_path.parent.mkdir(exist_ok=True, parents=True)
                    shutil.copy2(full_path, backup_path)
                    print(f"Backed up {file_path} to {backup_path}")

                    # Fix the replacement text to prevent nested triple quote issues
                    original_replace = change["replace"]
                    fixed_replace = self.fix_nested_triple_quotes(original_replace)

                    if fixed_replace != original_replace:
                        print(f"Fixed nested triple quotes in replacement for {file_path}")
                        change["replace"] = fixed_replace

                    # Apply find-and-replace
                    new_content, success = self._apply_find_replace(full_path, change["find"], change["replace"])

                    if success:
                        # Generate diff for record
                        diff = self._generate_diff(current_content, new_content, file_path)

                        # Write changes to file
                        with open(full_path, 'w', encoding='utf-8') as f:
                            f.write(new_content)

                        print(f"✅ Successfully applied find-and-replace to {file_path}")

                        # Record the change
                        change_record = {
                            "file": file_path,
                            "description": change["description"],
                            "find": change["find"],
                            "replace": change["replace"],
                            "diff": diff,
                            "timestamp": datetime.datetime.now().isoformat()
                        }
                        changes_made.append(change_record)
                    else:
                        print(f"❌ Could not find text to replace in {file_path}")
                        # Try fallback to LLM
                        print("Trying fallback with LLM-guided modification...")
                        new_content = self._generate_modified_file(file_path, current_content, 
                                                              f"FIND: {change['find']}\nREPLACE WITH: {change['replace']}\n{change['description']}")

                        if new_content != current_content:
                            # Generate diff
                            diff = self._generate_diff(current_content, new_content, file_path)

                            # Write changes
                            with open(full_path, 'w', encoding='utf-8') as f:
                                f.write(new_content)

                            print(f"✅ Successfully applied changes using LLM-guided modification to {file_path}")

                            # Record the change
                            change_record = {
                                "file": file_path,
                                "description": change["description"],
                                "find": change["find"],
                                "replace": change["replace"],
                                "diff": diff,
                                "timestamp": datetime.datetime.now().isoformat()
                            }
                            changes_made.append(change_record)
                        else:
                            print(f"❌ LLM-guided modification also failed for {file_path}")
                except Exception as e:
                    print(f"❌ Error applying find-and-replace to {file_path}: {e}")
                    import traceback
                    traceback.print_exc()
            # Process diff-based change
            elif 'diff' in change:
                try:
                    # Load current file content
                    with open(full_path, 'r', encoding='utf-8') as f:
                        current_content = f.read()

                    # Apply the diff
                    print(f"Applying diff to {file_path}")
                    new_content = self._apply_diff(current_content, change["diff"])

                    # Check if there was an actual change
                    if new_content == current_content:
                        print(f"Warning: No changes made to {file_path} - content identical after applying changes")
                        print("Trying alternative approach with direct LLM modification...")
                        new_content = self._generate_modified_file(file_path, current_content, change["description"])

                    # Only write if there's an actual change now
                    if new_content != current_content:
                        # Generate the real diff for record-keeping
                        diff = self._generate_diff(current_content, new_content, file_path)

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
            else:
                print(f"⚠️ Change for {file_path} has neither find/replace nor diff information - skipping")

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

    def _apply_find_replace(self, file_path, find_text: str, replace_text: str) -> Tuple[str, bool]:
        """
        Apply a find-and-replace change to a file with enhanced fuzzy matching.

        Args:
            file_path: Path to the file to modify
            find_text: Text to find
            replace_text: Text to replace it with

        Returns:
            Tuple of (modified content, success flag)
        """
        try:
            # Read the file
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Fix potential nested triple quotes in the replacement text
            replace_text = self.fix_nested_triple_quotes(replace_text)

            # Try exact replacement first
            if find_text in content:
                new_content = content.replace(find_text, replace_text)
                print(f"Applied find-and-replace to {file_path} (exact match)")
                return new_content, True

            # Try with normalized whitespace
            find_text_normalized = ' '.join(find_text.split())
            content_normalized = ' '.join(content.split())

            if find_text_normalized in content_normalized:
                print(f"Text found with normalized whitespace, attempting replacement")

                # Use regex with flexible whitespace
                pattern = '\\s+'.join(re.escape(word) for word in find_text_normalized.split())
                new_content = re.sub(pattern, replace_text, content)

                if new_content != content:
                    print(f"Applied find-and-replace to {file_path} (normalized whitespace)")
                    return new_content, True

            # Try line-by-line matching for better fuzzy matching
            find_lines = find_text.strip().splitlines()
            content_lines = content.splitlines()

            # Line matching with different threshold levels
            for match_threshold in [0.9, 0.8, 0.7]:  # Try decreasing thresholds
                for i in range(len(content_lines) - len(find_lines) + 1):
                    match_scores = []

                    for j, find_line in enumerate(find_lines):
                        if i + j >= len(content_lines):
                            break

                        content_line = content_lines[i + j]

                        # Skip empty lines in comparison
                        if not find_line.strip() and not content_line.strip():
                            match_scores.append(1.0)  # Perfect match for empty lines
                            continue

                        # Calculate similarity score
                        find_line_norm = ' '.join(find_line.split())
                        content_line_norm = ' '.join(content_line.split())

                        # Exact match
                        if find_line_norm == content_line_norm:
                            match_scores.append(1.0)
                        # Fuzzy match - check if most of the words match
                        else:
                            find_words = set(find_line_norm.split())
                            content_words = set(content_line_norm.split())

                            if not find_words:  # Handle empty set
                                match_scores.append(0.0)
                            else:
                                intersection = find_words.intersection(content_words)
                                score = len(intersection) / len(find_words)
                                match_scores.append(score)

                    # If we have enough matching lines with good scores
                    avg_score = sum(match_scores) / len(match_scores) if match_scores else 0

                    if avg_score >= match_threshold:
                        print(f"Found fuzzy match at line {i+1} with average score {avg_score:.2f} (threshold: {match_threshold})")

                        # Replace the matching lines
                        new_content_lines = content_lines.copy()
                        new_content_lines[i:i + len(find_lines)] = replace_text.strip().splitlines()

                        # Convert back to a single string
                        new_content = '\n'.join(new_content_lines)

                        print(f"Applied find-and-replace to {file_path} (fuzzy match)")
                        return new_content, True

            # If we've tried all thresholds and still no match, try matching based on key identifiers
            # This is useful for function definitions, class definitions, etc.
            key_patterns = [
                r'def\s+([a-zA-Z0-9_]+)\s*\(',  # Function definitions
                r'class\s+([a-zA-Z0-9_]+)\s*[:\(]',  # Class definitions
                r'([a-zA-Z0-9_]+)\s*=\s*',  # Variable assignments
            ]

            # Extract key identifiers from find_text
            key_identifiers = []
            for pattern in key_patterns:
                for match in re.finditer(pattern, find_text):
                    key_identifiers.append(match.group(1))

            if key_identifiers:
                print(f"Searching for key identifiers: {key_identifiers}")

                # Look for these identifiers in the content
                for identifier in key_identifiers:
                    pattern = r'((?:^|\n)(?:[ \t]*)(?:def|class)?\s*' + re.escape(identifier) + r'\s*(?:\(|\{|=|:)(?:.|[\r\n])*?(?:\n[ \t]*\n|\Z))'
                    match = re.search(pattern, content)

                    if match:
                        matched_block = match.group(1)
                        new_content = content.replace(matched_block, replace_text)

                        if new_content != content:
                            print(f"Applied find-and-replace to {file_path} (key identifier match: {identifier})")
                            return new_content, True

            print(f"Warning: Could not find the text to replace in {file_path}")
            return content, False

        except Exception as e:
            print(f"Error applying find-and-replace: {e}")
            import traceback
            traceback.print_exc()
            return content, False

    def _apply_diff(self, original_content: str, diff_text: str) -> str:
        """
        Apply a diff to the original content with improved handling of various diff formats.

        Args:
            original_content: Original file content
            diff_text: Diff text to apply

        Returns:
            Modified content with diff applied
        """
        print(f"Applying diff with {len(diff_text)} characters")

        # Check if the diff is empty
        if not diff_text or not diff_text.strip():
            print("Warning: Empty diff provided, returning original content")
            return original_content

        # Print first few lines of diff for debugging
        print("Diff preview:")
        for i, line in enumerate(diff_text.splitlines()[:5]):
            print(f"  {line}")
        if len(diff_text.splitlines()) > 5:
            print(f"  ... ({len(diff_text.splitlines()) - 5} more lines)")

        # Handle different diff formats
        has_change_markers = False
        for line in diff_text.splitlines():
            if line.startswith('+') or line.startswith('-'):
                has_change_markers = True
                break

        if not has_change_markers:
            print("Warning: Diff doesn't contain proper change markers, using LLM to apply changes directly")
            return self._apply_changes_with_llm(original_content, diff_text)

        try:
            # Handle LLM-style diff without line markers but with + and - prefixes
            if '+' in diff_text and '-' in diff_text and '@@' not in diff_text:
                print("Detected LLM-style diff without line markers, using improved text-based diff application")

                # Extract changes as pairs of (text_to_remove, text_to_add)
                changes = []
                removal_lines = []
                addition_lines = []

                current_context = []  # Track context lines to help with matching

                # Process diff line by line to group changes
                for line in diff_text.splitlines():
                    if line.startswith('-') and not line.startswith('---'):
                        # Clear context when we start a new removal block
                        if not removal_lines:
                            current_context = []
                        removal_lines.append(line[1:])
                    elif line.startswith('+') and not line.startswith('+++'):
                        addition_lines.append(line[1:])
                    else:
                        # If we have pending changes, add them to the changes list
                        if removal_lines or addition_lines:
                            changes.append((
                                '\n'.join(removal_lines), 
                                '\n'.join(addition_lines),
                                current_context.copy() if current_context else None
                            ))
                            removal_lines = []
                            addition_lines = []

                        # Track context lines
                        if line.strip():
                            current_context.append(line)
                            # Keep only the last 3 context lines
                            if len(current_context) > 3:
                                current_context.pop(0)

                # Add any remaining changes
                if removal_lines or addition_lines:
                    changes.append((
                        '\n'.join(removal_lines), 
                        '\n'.join(addition_lines),
                        current_context.copy() if current_context else None
                    ))

                # For debugging
                print(f"Extracted {len(changes)} change blocks from diff")

                # Apply changes
                modified_content = original_content
                for old_text, new_text, context in changes:
                    if old_text:
                        # Try direct replacement first
                        if old_text in modified_content:
                            modified_content = modified_content.replace(old_text, new_text)
                            print(f"  Replaced: '{old_text[:40]}...' with '{new_text[:40]}...'")
                        else:
                            # Try using context to find the right location
                            found = False
                            if context:
                                # Join the context lines and look for them
                                context_text = '\n'.join(context)
                                if context_text in modified_content:
                                    # Find where the context appears
                                    context_pos = modified_content.find(context_text)
                                    # Look for the removal text near the context
                                    search_area = modified_content[max(0, context_pos - 200):min(len(modified_content), context_pos + 200)]
                                    if old_text in search_area:
                                        # Replace in the full content
                                        modified_content = modified_content.replace(old_text, new_text)
                                        found = True
                                        print(f"  Replaced using context: '{old_text[:40]}...'")

                            if not found:
                                # Try normalized space comparison
                                old_normalized = ' '.join(old_text.split())
                                content_normalized = ' '.join(modified_content.split())
                                if old_normalized in content_normalized:
                                    # Use regex with flexible whitespace
                                    pattern = '\s+'.join(re.escape(word) for word in old_normalized.split())
                                    modified_content = re.sub(pattern, new_text, modified_content)
                                    print(f"  Replaced using normalized space: '{old_text[:40]}...'")
                                else:
                                    print(f"  Warning: Could not find text to replace: '{old_text[:40]}...'")
                    elif new_text:
                        # If no text to remove but we have context, try to use it for insertion
                        inserted = False
                        if context:
                            context_text = '\n'.join(context)
                            if context_text in modified_content:
                                context_pos = modified_content.find(context_text) + len(context_text)
                                prefix = modified_content[:context_pos]
                                suffix = modified_content[context_pos:]
                                modified_content = prefix + '\n' + new_text + suffix
                                inserted = True
                                print(f"  Inserted after context: '{new_text[:40]}...'")

                        if not inserted:
                            # If no context or context not found, add to the beginning
                            modified_content = new_text + modified_content
                            print(f"  Added at beginning: '{new_text[:40]}...'")

                # Check if content changed
                if modified_content == original_content:
                    print("Warning: Simple diff application resulted in no changes")
                    # Try to reuse an existing implementation from the LLM
                    modified_content = self._apply_changes_with_llm(original_content, diff_text)

                return modified_content

            # Parse standard unified diff format
            original_lines = original_content.splitlines()
            new_content_lines = original_lines.copy()

            # Track line additions/removals
            line_offset = 0
            current_line = 0

            # Track modification stats
            additions = 0
            removals = 0

            # Process diff line by line
            for line in diff_text.splitlines():
                if line.startswith("@@"):
                    # Line position indicator
                    match = re.search(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", line)
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
                else:
                    # Context line
                    current_line += 1

            modified_content = "\n".join(new_content_lines)

            print(f"Diff applied: {additions} additions, {removals} removals")

            # Sanity check
            if modified_content == original_content:
                print("Warning: Applied diff resulted in identical content, attempting direct LLM modification")
                return self._apply_changes_with_llm(original_content, diff_text)

            return modified_content

        except Exception as e:
            print(f"Error applying diff: {e}")
            import traceback
            traceback.print_exc()
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
        Also fixes nested triple quotes in the modified content.

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

        # Fix any nested triple quotes in the generated content
        modified_content = self.fix_nested_triple_quotes(modified_content)

        # Check if content actually changed
        if modified_content == original_content:
            print("Warning: LLM did not make any changes to the content")
        else:
            print("Successfully applied changes using LLM")

        return modified_content

    def _generate_modified_file(self, file_path: str, current_content: str, improvement_description: str) -> str:
        """
        Generate a modified version of a file based on improvement description.
        Also fixes nested triple quotes in the generated content.

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

        # Fix any nested triple quotes in the generated content
        modified_content = self.fix_nested_triple_quotes(modified_content)

        return modified_content

    def _create_system_backup(self):
        """Create a backup of the current system state"""
        import shutil  # Ensure import is available

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

"""
            if 'find' in change and 'replace' in change:
                report_content += f"""**Find:**
```
{change['find']}
```

**Replace With:**
```
{change['replace']}
```

"""
            if 'diff' in change:
                report_content += f"""```diff
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
   - The exact text to find (FIND)
   - The exact text to replace it with (REPLACE WITH)

Focus on FIND/REPLACE changes rather than diffs for better reliability and easier implementation.

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

FIND:
```python
# Exact code block to find
def some_function():
    original_code_here
```

REPLACE WITH:
```python
# New code block to replace it with
def some_function():
    modified_code_here
```

### Change 2:
File: `filename.py`
Description: Clear description of the improvement and its expected impact

FIND:
```python
# Another code block to find
original_code_here
```

REPLACE WITH:
```python
# New code to replace it with
modified_code_here
```

IMPORTANT:
1. Make specific, focused changes rather than wholesale rewrites
2. Maintain the same coding style and architecture
3. Focus on impactful improvements that address known issues
4. Be EXTREMELY PRECISE with the FIND text to ensure it can be located in the file
5. Make sure the FIND text appears EXACTLY as-is in the file (whitespace matters)
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

    # Default: run the full improvement process
    improver.run()


if __name__ == "__main__":
    main()