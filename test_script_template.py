import sys
import traceback
import os
import json
import datetime
import inspect
import functools
import importlib.util

# Add the scripts directory to the path
sys.path.append("{scripts_dir}")

# Configure tracing
trace_file = "{trace_file}"
os.makedirs(os.path.dirname(trace_file), exist_ok=True)

def call_llm(prompt, system_instruction=None):
    try:
        from google import genai
        from google.genai import types
        import os  # Import the os module

        # Initialize the Gemini client
        client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

        # Call the API with system instruction if provided
        if system_instruction:
            response = client.models.generate_content(
                model="gemini-2.0-flash", 
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    #thinking_config=types.ThinkingConfig(thinking_budget=0) # Disables thinking
                ),
                contents=prompt
            )
        else:
            response = client.models.generate_content(
                model="gemini-2.0-flash",
                #thinking_config=types.ThinkingConfig(thinking_budget=0), # Disables thinking
                contents=prompt
            )

        return response.text
    except Exception as e:
        print("Error calling Gemini API: " + str(e))
        return "Error: " + str(e)

def call_database(db_path, sql_query):
    """Execute SQL queries against a SQLite database with proper error handling"""
    import sqlite3
    import os
    from pathlib import Path
    
    print(f"  [DATABASE] Executing query on {{db_path}}: {{sql_query[:100]}}...")
    
    try:
        # Handle path resolution for sandbox environment
        if not os.path.isabs(db_path):
            # If relative path, assume it's in the workspace
            if '/workspace' in os.getcwd() or os.path.exists('/workspace'):
                workspace_path = Path('/workspace') / db_path
                if workspace_path.exists():
                    db_path = str(workspace_path)
                else:
                    # Try current working directory
                    current_path = Path(db_path)
                    if current_path.exists():
                        db_path = str(current_path)
                    else:
                        return f"Error: Database file not found at {{db_path}} (checked workspace and current directory)"
            else:
                # Not in sandbox, use relative to current directory
                current_path = Path(db_path)
                if current_path.exists():
                    db_path = str(current_path)
                else:
                    return f"Error: Database file not found at {{db_path}}"
        
        # Check if database file exists
        if not os.path.exists(db_path):
            return f"Error: Database file not found at {{db_path}}"
        
        # Connect to database
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row  # Enable column access by name
        cursor = conn.cursor()
        
        try:
            # Execute the query
            cursor.execute(sql_query)
            
            # Handle different types of queries
            if sql_query.strip().upper().startswith(('SELECT', 'WITH', 'PRAGMA')):
                # Query returns results
                results = cursor.fetchall()
                if not results:
                    return "Query executed successfully. No results returned."
                
                # Convert results to a readable format
                columns = [description[0] for description in cursor.description]
                result_text = f"Results ({{len(results)}} rows):\n"
                result_text += " | ".join(columns) + "\n"
                result_text += "-" * (len(" | ".join(columns))) + "\n"
                
                for row in results[:50]:  # Limit to first 50 rows to avoid overwhelming output
                    row_values = [str(row[col]) if row[col] is not None else 'NULL' for col in columns]
                    result_text += " | ".join(row_values) + "\n"
                
                if len(results) > 50:
                    result_text += f"... and {{len(results) - 50}} more rows\n"
                
                return result_text
            else:
                # Query modifies data (INSERT, UPDATE, DELETE, etc.)
                conn.commit()
                rows_affected = cursor.rowcount
                return f"Query executed successfully. {{rows_affected}} rows affected."
                
        except sqlite3.Error as e:
            return f"SQL Error: {{str(e)}}"
        finally:
            cursor.close()
            conn.close()
            
    except Exception as e:
        return f"Database Error: {{str(e)}}"

def read_file(filepath, start_line=None, end_line=None):
    """Read entire files or specific line ranges with proper error handling"""
    import os
    from pathlib import Path
    
    print(f"  [FILE] Reading file: {{filepath}}" + (f" (lines {{start_line}}-{{end_line}})" if start_line is not None else ""))
    
    try:
        # Handle path resolution for sandbox environment
        if not os.path.isabs(filepath):
            # If relative path, check workspace first then current directory
            if '/workspace' in os.getcwd() or os.path.exists('/workspace'):
                workspace_path = Path('/workspace') / filepath
                if workspace_path.exists():
                    filepath = str(workspace_path)
                else:
                    # Try current working directory
                    current_path = Path(filepath)
                    if current_path.exists():
                        filepath = str(current_path)
                    else:
                        return f"Error: File not found at {{filepath}} (checked workspace and current directory)"
            else:
                # Not in sandbox, use relative to current directory
                current_path = Path(filepath)
                if current_path.exists():
                    filepath = str(current_path)
                else:
                    return f"Error: File not found at {{filepath}}"
        
        # Check if file exists
        if not os.path.exists(filepath):
            return f"Error: File not found at {{filepath}}"
        
        # Check if it's actually a file (not a directory)
        if not os.path.isfile(filepath):
            return f"Error: {{filepath}} is not a file"
        
        # Read the file
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # Handle line range selection
            if start_line is not None or end_line is not None:
                total_lines = len(lines)
                
                # Convert to 0-based indexing
                start_idx = (start_line - 1) if start_line is not None else 0
                end_idx = end_line if end_line is not None else total_lines
                
                # Validate line numbers
                if start_idx < 0:
                    start_idx = 0
                if end_idx > total_lines:
                    end_idx = total_lines
                if start_idx >= end_idx:
                    return f"Error: Invalid line range. File has {{total_lines}} lines."
                
                lines = lines[start_idx:end_idx]
                
                result = f"File content ({{filepath}}) - Lines {{start_idx + 1}} to {{end_idx}} of {{total_lines}}:\n"
                result += "=" * 50 + "\n"
                
                for i, line in enumerate(lines, start=start_idx + 1):
                    result += f"{{i:4d}}: {{line.rstrip()}}\n"
                
                return result
            else:
                # Return entire file
                total_lines = len(lines)
                if total_lines > 200:
                    # For very large files, show warning
                    result = f"File content ({{filepath}}) - {{total_lines}} lines (showing all):\n"
                    result += "WARNING: Large file - consider using line ranges for better performance\n"
                    result += "=" * 50 + "\n"
                else:
                    result = f"File content ({{filepath}}) - {{total_lines}} lines:\n"
                    result += "=" * 50 + "\n"
                
                for i, line in enumerate(lines, start=1):
                    result += f"{{i:4d}}: {{line.rstrip()}}\n"
                
                return result
                
        except UnicodeDecodeError:
            return f"Error: Unable to read {{filepath}} - file appears to be binary"
        except PermissionError:
            return f"Error: Permission denied reading {{filepath}}"
            
    except Exception as e:
        return f"File Error: {{str(e)}}"

def search_file(filepath, pattern, case_sensitive=True, show_line_numbers=True, context_lines=0, max_results=50):
    """Search for patterns in files with grep-like functionality"""
    import os
    import re
    from pathlib import Path
    
    print(f"  [SEARCH] Searching in {{filepath}} for pattern: '{{pattern}}'" + 
          ("" if case_sensitive else " (case-insensitive)"))
    
    try:
        # Handle path resolution for sandbox environment
        if not os.path.isabs(filepath):
            # If relative path, check workspace first then current directory
            if '/workspace' in os.getcwd() or os.path.exists('/workspace'):
                workspace_path = Path('/workspace') / filepath
                if workspace_path.exists():
                    filepath = str(workspace_path)
                else:
                    # Try current working directory
                    current_path = Path(filepath)
                    if current_path.exists():
                        filepath = str(current_path)
                    else:
                        return f"Error: File not found at {{filepath}} (checked workspace and current directory)"
            else:
                # Not in sandbox, use relative to current directory
                current_path = Path(filepath)
                if current_path.exists():
                    filepath = str(current_path)
                else:
                    return f"Error: File not found at {{filepath}}"
        
        # Check if file exists
        if not os.path.exists(filepath):
            return f"Error: File not found at {{filepath}}"
        
        # Check if it's actually a file (not a directory)
        if not os.path.isfile(filepath):
            return f"Error: {{filepath}} is not a file"
        
        # Read the file
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        except UnicodeDecodeError:
            return f"Error: Unable to read {{filepath}} - file appears to be binary"
        except PermissionError:
            return f"Error: Permission denied reading {{filepath}}"
        
        # Prepare regex pattern
        regex_flags = 0 if case_sensitive else re.IGNORECASE
        try:
            compiled_pattern = re.compile(pattern, regex_flags)
        except re.error as e:
            return f"Error: Invalid regex pattern '{{pattern}}': {{str(e)}}"
        
        # Find matches
        matches = []
        for line_num, line in enumerate(lines, start=1):
            if compiled_pattern.search(line):
                matches.append(line_num)
                if len(matches) >= max_results:
                    break
        
        if not matches:
            return f"No matches found for pattern '{{pattern}}' in {{filepath}}"
        
        # Build result with context
        result = f"Search results for '{{pattern}}' in {{filepath}}:\n"
        if len(matches) >= max_results:
            result += f"Found {{len(matches)}}+ matches (showing first {{max_results}}):\n"
        else:
            result += f"Found {{len(matches)}} matches:\n"
        result += "=" * 50 + "\n"
        
        displayed_lines = set()
        
        for match_line in matches:
            # Calculate context range
            start_context = max(1, match_line - context_lines)
            end_context = min(len(lines), match_line + context_lines)
            
            # Add separator if we're not continuing from previous context
            if displayed_lines and start_context > max(displayed_lines) + 1:
                result += "\n" + "-" * 20 + "\n"
            
            # Show context lines
            for line_num in range(start_context, end_context + 1):
                if line_num in displayed_lines:
                    continue
                
                line_content = lines[line_num - 1].rstrip()
                
                if line_num == match_line:
                    # Highlight the matching line
                    prefix = ">>> " if show_line_numbers else ">>> "
                    line_display = f"{{line_num:4d}}: {{line_content}}" if show_line_numbers else line_content
                    result += f"{{prefix}}{{line_display}}\n"
                else:
                    # Context line
                    prefix = "    " if show_line_numbers else ""
                    line_display = f"{{line_num:4d}}: {{line_content}}" if show_line_numbers else line_content
                    result += f"{{prefix}}{{line_display}}\n"
                
                displayed_lines.add(line_num)
        
        return result
        
    except Exception as e:
        return f"Search Error: {{str(e)}}"

def execute_code(code_str, timeout=10):
    """Execute Python code with automatic package installation and proper scoping"""
    import sys
    import re
    import subprocess
    from io import StringIO

    print("  [SYSTEM] Auto-installing execute_code() with scope fix")

    # Clean markdown formatting
    patterns = [
        r'```python\s*\n(.*?)\n```',
        r'```python\s*(.*?)```', 
        r'```\s*\n(.*?)\n```',
        r'```\s*(.*?)```'
    ]

    cleaned_code = code_str.strip()
    for pattern in patterns:
        match = re.search(pattern, code_str, re.DOTALL | re.IGNORECASE)
        if match:
            cleaned_code = match.group(1).strip()
            print("  [CLEANING] Removed markdown")
            break

    # Function to install a package
    def install_package(package_name):
        try:
            print("  [INSTALLING] Installing " + package_name + "...")
            result = subprocess.run([
                sys.executable, "-m", "pip", "install", package_name
            ], capture_output=True, text=True, timeout=30)

            if result.returncode == 0:
                print("  [SUCCESS] " + package_name + " installed successfully")
                return True
            else:
                print("  [FAILED] Could not install " + package_name + ": " + result.stderr)
                return False
        except Exception as e:
            print("  [ERROR] Installation error: " + str(e))
            return False

    # Execute with proper scoping and auto-installation retry
    max_install_attempts = 3
    attempt = 0

    while attempt <= max_install_attempts:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        stdout_capture = StringIO()
        stderr_capture = StringIO()

        try:
            sys.stdout = stdout_capture
            sys.stderr = stderr_capture

            # CRITICAL FIX: Provide explicit globals and locals
            # This ensures imports are available to functions defined in the code
            exec_namespace = {{}}
            exec(cleaned_code, exec_namespace, exec_namespace)

            # Success!
            sys.stdout = old_stdout
            sys.stderr = old_stderr

            output = stdout_capture.getvalue().strip()
            return output if output else "Code executed successfully"

        except ModuleNotFoundError as e:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

            # Extract the missing module name
            module_name = str(e).split("'")[1] if "'" in str(e) else None

            if module_name and attempt < max_install_attempts:
                print("  [MISSING] Module '" + module_name + "' not found, attempting to install...")

                # Try to install the missing package
                if install_package(module_name):
                    attempt += 1
                    print("  [RETRY] Retrying code execution (attempt " + str(attempt + 1) + ")...")
                    continue
                else:
                    return "Error: Could not install required package '" + module_name + "'"
            else:
                return "Error: " + str(e)

        except Exception as e:
            sys.stdout = old_stdout  
            sys.stderr = old_stderr
            return "Error: " + str(e)

        attempt += 1

    return "Error: Maximum installation attempts exceeded"

# Trace entry for execution start
with open(trace_file, 'a', encoding='utf-8') as f:
    start_entry = {{
        "timestamp": datetime.datetime.now().isoformat(),
        "event": "execution_start",
        "iteration": {current_iteration},
        "sample_id": "{sample_id}",
        "question": {question_repr}
    }}
    f.write(json.dumps(start_entry) + "\n")

# More reliable method for getting caller information
def get_real_caller():
    """Get information about the caller, skipping intermediate functions like wrappers and decorators."""
    frames = inspect.stack()
    # Skip first 2 frames (this function and immediate caller)
    for frame_info in frames[2:]:
        # Get the frame's module
        frame_module = frame_info.frame.f_globals.get('__name__', '')
        # If this frame is from our module (not from system libraries)
        if frame_module == 'current_script_{current_iteration}':
            # Check if it's not the call_llm function itself
            if frame_info.function != 'call_llm' and 'wrapper' not in frame_info.function:
                return {{
                    "function": frame_info.function,
                    "filename": frame_info.filename,
                    "lineno": frame_info.lineno
                }}
    # Fallback if we can't find a suitable caller
    return {{"function": "unknown", "filename": "unknown", "lineno": 0}}

# Create a tracing decorator for call_llm
def trace_call_llm(func):
    @functools.wraps(func)
    def wrapper(prompt, system_instruction=None):
        # Get caller information using our improved method
        caller_info = get_real_caller()

        # Create trace entry with caller information
        trace_entry = {{
            "timestamp": datetime.datetime.now().isoformat(),
            "event": "llm_call",
            "iteration": {current_iteration},
            "sample_id": "{sample_id}",
            "function": "call_llm",
            "caller": caller_info,
            "input": {{
                "prompt": prompt,
                "system_instruction": system_instruction
            }}
        }}

        # Call the original function
        try:
            result = func(prompt, system_instruction)

            # Log successful response
            trace_entry["output"] = result
            trace_entry["status"] = "success"

            with open(trace_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(trace_entry) + "\n")

            return result

        except Exception as e:
            # Log error
            trace_entry["error"] = str(e)
            trace_entry["status"] = "error"
            trace_entry["traceback"] = traceback.format_exc()

            with open(trace_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(trace_entry) + "\n")

            raise

    return wrapper

try:
    # Import the script as a module
    spec = importlib.util.spec_from_file_location(
        "current_script_{current_iteration}", 
        "{script_path}"
    )
    module = importlib.util.module_from_spec(spec)
    
    # Try to execute the module, handling system_tools import failures gracefully
    try:
        spec.loader.exec_module(module)
    except ModuleNotFoundError as e:
        if 'system_tools' in str(e):
            print("  [INFO] system_tools import failed as expected - functions will be injected")
            # Continue execution - we'll inject the functions below
            # Re-execute without the problematic import by modifying the script
            import re
            with open("{script_path}", 'r') as f:
                script_content = f.read()
            
            # Comment out the system_tools import line
            modified_content = re.sub(
                r'^from system_tools import.*$', 
                '# from system_tools import... # (functions injected by system)', 
                script_content, 
                flags=re.MULTILINE
            )
            
            # Write the modified script to a temp location and load it
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as temp_file:
                temp_file.write(modified_content)
                temp_path = temp_file.name
            
            try:
                temp_spec = importlib.util.spec_from_file_location(
                    "current_script_{current_iteration}_fixed", 
                    temp_path
                )
                module = importlib.util.module_from_spec(temp_spec)
                temp_spec.loader.exec_module(module)
            finally:
                import os
                os.unlink(temp_path)
        else:
            # Re-raise other import errors
            raise

    # INJECT ALL FUNCTIONS
    module.execute_code = execute_code
    module.call_llm = call_llm
    module.call_database = call_database
    module.read_file = read_file
    module.search_file = search_file

    # Patch call_llm function if it exists
    if hasattr(module, 'call_llm'):
        original_call_llm = module.call_llm
        module.call_llm = trace_call_llm(original_call_llm)

    # Also patch any other functions that might call LLM directly
    for name, obj in inspect.getmembers(module):
        if inspect.isfunction(obj) and obj.__module__ == module.__name__:
            try:
                source = inspect.getsource(obj)
                if 'generate_content' in source and obj is not getattr(module, 'call_llm', None):
                    setattr(module, name, trace_call_llm(obj))
            except:
                pass

    # Execute the main function with the question string
    question = {question_repr}

    # Call the main function and get the answer
    answer = module.main(question)

    # Log execution completion
    with open(trace_file, 'a', encoding='utf-8') as f:
        end_entry = {{
            "timestamp": datetime.datetime.now().isoformat(),
            "event": "execution_complete",
            "iteration": {current_iteration},
            "sample_id": "{sample_id}",
            "answer": str(answer)
        }}
        f.write(json.dumps(end_entry) + "\n")

    # Print the answer for capture
    print("ANSWER_START")
    print(answer)
    print("ANSWER_END")

except Exception as e:
    # Log the error
    with open(trace_file, 'a', encoding='utf-8') as f:
        error_entry = {{
            "timestamp": datetime.datetime.now().isoformat(),
            "event": "execution_error",
            "iteration": {current_iteration},
            "sample_id": "{sample_id}",
            "error": str(e),
            "traceback": traceback.format_exc()
        }}
        f.write(json.dumps(error_entry) + "\n")

    print("ERROR_START")
    print(str(e))
    print(traceback.format_exc())
    print("ERROR_END")