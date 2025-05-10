#!/usr/bin/env python
"""
enhance_system_improver.py - Script to enhance the system_improver.py file

This script:
1. Improves the change extraction logic in system_improver.py
2. Adds more robust diff handling
3. Enhances the logging and error reporting
"""

import os
import re
import shutil
from pathlib import Path

def enhance_system_improver():
    """Enhance the system_improver.py file with better extraction logic"""
    file_path = "system_improver.py"

    if not Path(file_path).exists():
        print(f"Error: {file_path} not found")
        return False

    # Create a backup
    backup_path = f"{file_path}.bak"
    shutil.copy2(file_path, backup_path)
    print(f"Created backup at {backup_path}")

    # Read the current file
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Enhanced extraction function
    enhanced_extraction_code = """
    def _extract_changes_from_llm_response(self, improvement_analysis: str) -> List[Dict]:
        \"\"\"
        Extract proposed changes from LLM response using a more robust approach.

        Args:
            improvement_analysis: Raw LLM response text

        Returns:
            List of extracted changes with file, description, and diff
        \"\"\"
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
                r'File:\s*[`\'"]?([\w./\-_]+\.\w+)[`\'"]?',  # Standard format
                r'File\s+`([\w./\-_]+\.\w+)`',                # With backticks
                r'File\s+([\w./\-_]+\.\w+)',                  # Without any quotes
                r'in\s+[`\'"]?([\w./\-_]+\.\w+)[`\'"]?'       # Alternative "in file.py" format
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
                r'Description:(.*?)(?:```|Diff:|$)',          # Standard format
                r'Description\s*:(.*?)(?:```|Diff:|$)'        # With flexible whitespace
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
    """

    # Improved _apply_diff function code
    improved_apply_diff_code = """
    def _apply_diff(self, original_content: str, diff_text: str) -> str:
        \"\"\"
        Apply a diff to the original content with improved handling of various diff formats.

        Args:
            original_content: Original file content
            diff_text: Diff text to apply

        Returns:
            Modified content with diff applied
        \"\"\"
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
                                '\\n'.join(removal_lines), 
                                '\\n'.join(addition_lines),
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
                        '\\n'.join(removal_lines), 
                        '\\n'.join(addition_lines),
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
                                context_text = '\\n'.join(context)
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
                                    pattern = '\\s+'.join(re.escape(word) for word in old_normalized.split())
                                    modified_content = re.sub(pattern, new_text, modified_content)
                                    print(f"  Replaced using normalized space: '{old_text[:40]}...'")
                                else:
                                    print(f"  Warning: Could not find text to replace: '{old_text[:40]}...'")
                    elif new_text:
                        # If no text to remove but we have context, try to use it for insertion
                        inserted = False
                        if context:
                            context_text = '\\n'.join(context)
                            if context_text in modified_content:
                                context_pos = modified_content.find(context_text) + len(context_text)
                                prefix = modified_content[:context_pos]
                                suffix = modified_content[context_pos:]
                                modified_content = prefix + '\\n' + new_text + suffix
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

            modified_content = "\\n".join(new_content_lines)

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
    """

    # Update the _analyze_for_improvements method to use our new extraction function
    # Find the method
    analyze_method_match = re.search(r'def _analyze_for_improvements\(self, system_data: Dict\).*?(?=def |$)', content, re.DOTALL)

    if not analyze_method_match:
        print("Error: Could not find _analyze_for_improvements method")
        return False

    analyze_method = analyze_method_match.group(0)

    # Find the part where change extraction happens
    extraction_section_start = analyze_method.find("# Extract the changes section")
    if extraction_section_start == -1:
        extraction_section_start = analyze_method.find("changes_section = re.search")

    if extraction_section_start == -1:
        print("Error: Could not find change extraction section in _analyze_for_improvements")
        return False

    # Find the end of the extraction section
    extraction_section_end = analyze_method.find("print(f\"Identified {len(improvement_plan['changes'])} proposed code changes\")", extraction_section_start)

    if extraction_section_end == -1:
        extraction_section_end = analyze_method.find("# If we have analysis but no changes", extraction_section_start)

    if extraction_section_end == -1:
        print("Error: Could not find end of extraction section")
        return False

    # Replace the extraction section with a call to our new function
    new_extraction_section = """
            # Extract the changes section using the robust extractor
            improvement_plan["changes"] = self._extract_changes_from_llm_response(improvement_analysis)

            print(f"Identified {len(improvement_plan['changes'])} proposed code changes")
    """

    # Create the updated method
    updated_analyze_method = analyze_method[:extraction_section_start] + new_extraction_section + analyze_method[extraction_section_end:]

    # Replace the old method with the updated one
    new_content = content.replace(analyze_method, updated_analyze_method)

    # Add our new extraction function after the _analyze_for_improvements method
    new_content = new_content.replace(updated_analyze_method, updated_analyze_method + enhanced_extraction_code)

    # Also replace the _apply_diff method
    apply_diff_method_match = re.search(r'def _apply_diff\(self, original_content: str, diff_text: str\).*?(?=def |$)', new_content, re.DOTALL)

    if apply_diff_method_match:
        old_apply_diff = apply_diff_method_match.group(0)
        new_content = new_content.replace(old_apply_diff, improved_apply_diff_code)
        print("Replaced _apply_diff method with improved version")
    else:
        print("Warning: Could not find _apply_diff method to replace")

    # Write the updated file
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(new_content)

    print(f"Successfully enhanced {file_path}")
    print("The system_improver.py now has more robust change extraction and diff handling")
    return True

if __name__ == "__main__":
    enhance_system_improver()