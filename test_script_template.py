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

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from clients import GeminiClient, OpenAIClient

provider_type = "{provider_type}"
model_name = "{model_name}"

if provider_type == "gemini":
    client = GeminiClient(model_name)
elif provider_type == "openai":
    client = OpenAIClient(model_name)
else:
    raise ValueError(f"Unknown provider: {provider_type}")

def call_llm(prompt, system_instruction=""):
    return client.call_llm(prompt, system_instruction)


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
    spec.loader.exec_module(module)

    # INJECT BOTH FUNCTIONS
    module.execute_code = execute_code
    module.call_llm = call_llm

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