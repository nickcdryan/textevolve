#!/usr/bin/env python3
"""
Script to convert iteration JSON files to human-readable text summaries.
"""

import json
import os
import sys
import textwrap
from pathlib import Path

# Global configuration - Change this to adjust text wrapping width
TEXT_WIDTH = 100  # Set to 100, 120, or any width you prefer


def wrap_text(text, width=TEXT_WIDTH, preserve_paragraphs=True):
    """
    Wrap text to specified width while preserving paragraph structure and indentation.
    
    Args:
        text (str): Text to wrap
        width (int): Maximum line width
        preserve_paragraphs (bool): Whether to preserve paragraph breaks
        
    Returns:
        str: Wrapped text
    """
    if not text:
        return text
    
    if preserve_paragraphs:
        # Split by double newlines to preserve paragraphs
        paragraphs = text.split('\n\n')
        wrapped_paragraphs = []
        
        for paragraph in paragraphs:
            if paragraph.strip():
                wrapped_paragraphs.append(wrap_paragraph(paragraph, width))
            else:
                wrapped_paragraphs.append(paragraph)
        
        return '\n\n'.join(wrapped_paragraphs)
    else:
        return textwrap.fill(text, width=width)


def wrap_paragraph(paragraph, width=TEXT_WIDTH):
    """
    Wrap a single paragraph while preserving indentation for lists and structured content.
    """
    lines = paragraph.split('\n')
    wrapped_lines = []
    
    for line in lines:
        if not line.strip():
            wrapped_lines.append(line)
            continue
            
        # Detect indentation
        indent = len(line) - len(line.lstrip())
        content = line.strip()
        
        # Check if this is a list item (numbered, bulleted, or dashed)
        if is_list_item(content):
            # For list items, wrap with hanging indent
            wrapped = wrap_list_item(line, width, indent)
            wrapped_lines.extend(wrapped)
        elif indent > 0:
            # For other indented content, preserve the indentation but still wrap
            indent_str = line[:indent]
            wrapped = textwrap.fill(content, width=width-indent, 
                                  initial_indent=indent_str,
                                  subsequent_indent=indent_str)
            wrapped_lines.append(wrapped)
        else:
            # Regular text
            wrapped = textwrap.fill(content, width=width)
            wrapped_lines.append(wrapped)
    
    return '\n'.join(wrapped_lines)


def is_list_item(content):
    """Check if content is a list item (numbered, bulleted, or dashed)."""
    import re
    # Check for numbered lists (1., 2., a., etc.), bullets (-, *, •), or other markers
    list_patterns = [
        r'^\d+\.',           # 1. 2. 3.
        r'^[a-zA-Z]\.',      # a. b. c.
        r'^[-*•]',           # - * •
        r'^\d+\)',           # 1) 2) 3)
        r'^[a-zA-Z]\)',      # a) b) c)
    ]
    
    for pattern in list_patterns:
        if re.match(pattern, content.strip()):
            return True
    return False


def wrap_list_item(line, width=TEXT_WIDTH, base_indent=0):
    """
    Wrap a list item with proper hanging indent.
    """
    # Find the content after the list marker
    content = line.strip()
    indent_str = line[:base_indent]
    
    # Find where the actual content starts (after the list marker)
    import re
    match = re.match(r'^(\s*(?:\d+\.|\d+\)|[a-zA-Z]\.|[a-zA-Z]\)|[-*•])\s*)', content)
    if match:
        marker = match.group(1)
        text_content = content[len(marker):].strip()
        
        if text_content:
            # Calculate hanging indent (marker width + base indent)
            hanging_indent = base_indent + len(marker)
            hanging_indent_str = ' ' * hanging_indent
            
            # Wrap with hanging indent
            wrapped = textwrap.fill(text_content, width=width-hanging_indent,
                                  initial_indent=indent_str + marker,
                                  subsequent_indent=hanging_indent_str)
            return wrapped.split('\n')
        else:
            # Just the marker, no content
            return [line]
    else:
        # Fallback to regular indented wrapping
        wrapped = textwrap.fill(content, width=width-base_indent,
                              initial_indent=indent_str,
                              subsequent_indent=indent_str)
        return [wrapped]


def format_customer_email(sample_meta):
    """Extract and format the original customer email from sample metadata."""
    if not sample_meta:
        return "No customer email available"
    
    email = sample_meta.get('original_customer_email', 'N/A')
    subject = sample_meta.get('original_subject', 'N/A')
    body = sample_meta.get('original_body', 'N/A')
    timestamp = sample_meta.get('original_timestamp', 'N/A')
    
    return f"""Customer Email: {email}
Subject: {subject}
Timestamp: {timestamp}

Message Body:
{wrap_text(body)}"""


def format_json_response(json_str):
    """Format JSON string for better readability with proper text wrapping."""
    try:
        # Parse and re-format the JSON for better indentation
        parsed = json.loads(json_str)
        formatted = json.dumps(parsed, indent=2)
        return wrap_json_content(formatted)
    except (json.JSONDecodeError, TypeError):
        # If it's not valid JSON or not a string, return as-is
        return str(json_str)


def wrap_json_content(json_text, width=TEXT_WIDTH):
    """
    Wrap JSON content while preserving structure and indentation.
    """
    lines = json_text.split('\n')
    wrapped_lines = []
    
    for line in lines:
        if not line.strip():
            wrapped_lines.append(line)
            continue
            
        # Detect the indentation level
        indent = len(line) - len(line.lstrip())
        content = line.strip()
        
        # Skip lines that are just structural JSON (braces, brackets)
        if content in ['{', '}', '[', ']', '{,', '},', '[,', '],']:
            wrapped_lines.append(line)
            continue
            
        # Check if this is a JSON key-value pair
        if ':' in content and content.endswith(','):
            # This is a key-value pair ending with comma
            key_value = content[:-1]  # Remove trailing comma
            wrapped = wrap_json_key_value(key_value, indent, width)
            if wrapped.endswith(','):
                wrapped_lines.append(wrapped + ',')
            else:
                wrapped_lines.append(wrapped + ',')
        elif ':' in content and not content.endswith(','):
            # This is a key-value pair without trailing comma
            wrapped = wrap_json_key_value(content, indent, width)
            wrapped_lines.append(wrapped)
        else:
            # Other JSON content (like array items)
            indent_str = ' ' * indent
            if len(line) > width:
                # Wrap while preserving indentation
                wrapped = textwrap.fill(content, width=width-indent,
                                      initial_indent=indent_str,
                                      subsequent_indent=indent_str)
                wrapped_lines.append(wrapped)
            else:
                wrapped_lines.append(line)
    
    return '\n'.join(wrapped_lines)


def wrap_json_key_value(key_value, indent, width=TEXT_WIDTH):
    """
    Wrap a JSON key-value pair while preserving the structure.
    """
    indent_str = ' ' * indent
    
    # Split on the first colon to separate key and value
    if ':' in key_value:
        parts = key_value.split(':', 1)
        key = parts[0].strip()
        value = parts[1].strip()
        
        # Calculate how much space we have for the value
        key_part = f"{key}: "
        key_length = len(indent_str + key_part)
        
        # If the whole line fits, return as-is
        if key_length + len(value) <= width:
            return indent_str + key_part + value
        
        # If the value is a long string, wrap it
        if value.startswith('"') and value.endswith('"'):
            # This is a string value
            string_content = value[1:-1]  # Remove quotes
            
            # Calculate available width for the string content
            available_width = width - key_length - 2  # -2 for quotes
            
            if len(string_content) > available_width:
                # Wrap the string content
                wrapped_string = textwrap.fill(string_content, 
                                             width=available_width,
                                             subsequent_indent=' ' * key_length)
                return indent_str + key_part + '"' + wrapped_string + '"'
        
        # For other long values, just wrap with hanging indent
        if len(value) > (width - key_length):
            wrapped_value = textwrap.fill(value,
                                        width=width - key_length,
                                        subsequent_indent=' ' * key_length)
            return indent_str + key_part + wrapped_value
    
    # Fallback: return with basic indentation
    return indent_str + key_value


def convert_iteration_to_text(json_file_path, output_file_path):
    """Convert a single iteration JSON file to human-readable text."""
    
    # Read the JSON file
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Start building the text output
    lines = []
    
    # Header
    lines.append("=" * 80)
    lines.append(f"ITERATION {data.get('iteration', 'N/A')} SUMMARY")
    lines.append("=" * 80)
    lines.append("")
    
    # Basic info
    lines.append(f"Timestamp: {data.get('timestamp', 'N/A')}")
    lines.append(f"Strategy: {data.get('strategy', 'N/A')}")
    lines.append(f"Sample Count: {data.get('sample_count', 'N/A')}")
    lines.append(f"Batch Size: {data.get('batch_size', 'N/A')}")
    lines.append("")
    
    # Approach Summary
    lines.append("APPROACH SUMMARY")
    lines.append("-" * 40)
    approach_summary = data.get('approach_summary', 'No approach summary available')
    lines.append(wrap_text(approach_summary))
    lines.append("")
    
    # Performance Summary
    performance = data.get('performance', {})
    lines.append("PERFORMANCE SUMMARY")
    lines.append("-" * 40)
    lines.append(f"Accuracy: {performance.get('accuracy', 'N/A')} ({performance.get('correct_count', 0)}/{performance.get('total_count', 0)})")
    lines.append("")
    
    # Samples
    samples = data.get('samples', [])
    results = data.get('results', [])
    samples_metadata = data.get('samples_metadata', [])
    
    for i, sample in enumerate(samples):
        lines.append("=" * 60)
        lines.append(f"SAMPLE {i + 1}")
        lines.append("=" * 60)
        lines.append("")
        
        # Customer Email (from metadata if available)
        if i < len(samples_metadata):
            lines.append("CUSTOMER EMAIL:")
            lines.append("-" * 20)
            lines.append(format_customer_email(samples_metadata[i]))
            lines.append("")
        
        # Expected Answer
        lines.append("EXPECTED ANSWER:")
        lines.append("-" * 20)
        expected_answer = sample.get('answer', 'No expected answer available')
        lines.append(format_json_response(expected_answer))
        lines.append("")
        
        # System Output
        if i < len(results):
            result = results[i]
            lines.append("SYSTEM OUTPUT:")
            lines.append("-" * 20)
            
            if result.get('success', False):
                system_answer = result.get('answer', 'No system answer available')
                lines.append(format_json_response(system_answer))
            else:
                lines.append("EXECUTION FAILED")
                error_msg = result.get('error', 'Unknown error')
                output_msg = result.get('output', 'No output available')
                lines.append(f"Error: {error_msg}")
                lines.append(f"Output: {output_msg}")
            lines.append("")
            
            # Match Result
            lines.append("EVALUATION:")
            lines.append("-" * 20)
            is_match = result.get('match', False)
            lines.append(f"Match: {'✓ YES' if is_match else '✗ NO'}")
            
            evaluation = result.get('evaluation', {})
            if evaluation:
                confidence = evaluation.get('confidence', 'N/A')
                explanation = evaluation.get('explanation', 'No explanation available')
                lines.append(f"Confidence: {confidence}")
                lines.append(f"Reasoning: {wrap_text(explanation, width=TEXT_WIDTH-10)}")
            lines.append("")
    
    # Error Analysis
    performance = data.get('performance', {})
    error_analysis = performance.get('error_analysis', {})
    error_text = None
    
    # Check for error analysis text in multiple possible locations
    if error_analysis and error_analysis.get('text_report'):
        error_text = error_analysis['text_report']
    elif performance.get('error_analysis_text'):
        error_text = performance.get('error_analysis_text')
    elif data.get('error_analysis_text'):
        error_text = data.get('error_analysis_text')
    
    if error_text:
        lines.append("=" * 80)
        lines.append("ERROR ANALYSIS")
        lines.append("=" * 80)
        lines.append("")
        lines.append(wrap_text(error_text))
        lines.append("")
    
    # Capability Report
    capability_report = performance.get('capability_report', {})
    capability_text = None
    
    # Check for capability report text in multiple possible locations
    if capability_report and capability_report.get('text_report') and capability_report['text_report'] != "No report available":
        capability_text = capability_report['text_report']
    elif performance.get('capability_report_text'):
        capability_text = performance.get('capability_report_text')
    elif data.get('capability_report_text'):
        capability_text = data.get('capability_report_text')
    
    if capability_text:
        lines.append("=" * 80)
        lines.append("CAPABILITY REPORT")
        lines.append("=" * 80)
        lines.append("")
        lines.append(wrap_text(capability_text))
        lines.append("")
    
    # Write the output file
    with open(output_file_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    
    print(f"Converted {json_file_path} -> {output_file_path}")


def convert_single_iteration(json_file_path):
    """
    Convert a single iteration JSON file to human-readable text format.
    
    Args:
        json_file_path (str or Path): Path to the iteration JSON file
        
    Returns:
        str: Path to the generated text file
    """
    json_file_path = Path(json_file_path)
    
    # Generate output filename by replacing .json with .txt
    txt_file_path = json_file_path.with_suffix('.txt')
    
    # Convert the file
    convert_iteration_to_text(json_file_path, txt_file_path)
    
    return str(txt_file_path)


def main():
    """Main function to process all iteration files in the archive."""
    
    # Get the script directory and find the archive folder
    script_dir = Path(__file__).parent
    archive_dir = script_dir / "archive"
    
    if not archive_dir.exists():
        print(f"Error: Archive directory not found at {archive_dir}")
        sys.exit(1)
    
    # Find all iteration JSON files
    iteration_files = sorted(archive_dir.glob("iteration_*.json"))
    
    if not iteration_files:
        print("No iteration files found in the archive directory.")
        sys.exit(1)
    
    print(f"Found {len(iteration_files)} iteration files to process...")
    
    # Process each file
    for json_file in iteration_files:
        # Extract iteration number from filename
        filename = json_file.stem  # e.g., "iteration_8"
        txt_filename = f"{filename}.txt"
        txt_file_path = archive_dir / txt_filename
        
        try:
            convert_iteration_to_text(json_file, txt_file_path)
        except Exception as e:
            print(f"Error processing {json_file}: {e}")
    
    print("Conversion complete!")


if __name__ == "__main__":
    main() 