import sys
import traceback
import os
import json
import datetime
import inspect
import functools
import importlib.util
import openai
from openai import OpenAI


# Add the scripts directory to the path
sys.path.append("scripts")

# Ensure the Gemini API key is available to the script
os.environ["GEMINI_API_KEY"] = "AIzaSyD_DWppm-TR9CN7xTTVmrW5ngTax7xsLDA"

# Configure tracing
trace_file = "archive/trace_iteration_8.jsonl"
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
                    system_instruction=system_instruction
                ),
                contents=prompt
            )
        else:
            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=prompt
            )

        return response.text
    except Exception as e:
        print(f"Error calling Gemini API: {str(e)}")
        return f"Error: {str(e)}"

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
            print(f"  [INSTALLING] Installing {package_name}...")
            result = subprocess.run([
                sys.executable, "-m", "pip", "install", package_name
            ], capture_output=True, text=True, timeout=30)

            if result.returncode == 0:
                print(f"  [SUCCESS] {package_name} installed successfully")
                return True
            else:
                print(f"  [FAILED] Could not install {package_name}: {result.stderr}")
                return False
        except Exception as e:
            print(f"  [ERROR] Installation error: {str(e)}")
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
            exec_namespace = {}
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
                print(f"  [MISSING] Module '{module_name}' not found, attempting to install...")

                # Try to install the missing package
                if install_package(module_name):
                    attempt += 1
                    print(f"  [RETRY] Retrying code execution (attempt {attempt + 1})...")
                    continue
                else:
                    return f"Error: Could not install required package '{module_name}'"
            else:
                return f"Error: {str(e)}"

        except Exception as e:
            sys.stdout = old_stdout  
            sys.stderr = old_stderr
            return f"Error: {str(e)}"

        attempt += 1

    return "Error: Maximum installation attempts exceeded"


# Trace entry for execution start
with open(trace_file, 'a', encoding='utf-8') as f:
    start_entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "event": "execution_start",
        "iteration": 8,
        "sample_id": "cdf79bfa-ba57-4c8d-9622-4f34d14128ee",
        "question": "PASSAGE: Hoping to rebound from their divisional road loss to the Giants, the Redskins played their Week 2 home opener against the New Orleans Saints.  In the first quarter, the 'Skins took the early lead when Shaun Suisham kicked a 22-yard field goal. In the second quarter, Washington increased its lead with Suisham's 36-yard field goal.  The Saints would respond with RB Pierre Thomas's 1-yard TD run and kicker Mart&#237;n Gram&#225;tica's 49-yard field goal.  This score marked the 269th consecutive game in which the Redskins allowed a point, tying the Cincinnati Bengals (1989-2006) for the lonest longest streak ever. In the third quarter, New Orleans increased its lead with QB Drew Brees completing a 19-yard TD pass to WR Robert Meachem.  Afterwards, the 'Skins responded when RB Clinton Portis scored on a 9-yard TD run (with a failed 2-point conversion).  The Saints responded with a Reggie Bush 55-yard punt return for a touchdown. The Redskins were able to mount a fourth quarter comeback by rallying for 14 points in the final stanza. The comeback started with a Clinton Portis 8-yard TD run, and was completed when QB Jason Campbell threw a 67-yard TD pass to WR Santana Moss. With the come from behind win, the Redskins improved to 1-1.\n\nQUESTION: How many yards longer was the longest touchdown pass compared with the shortest touchdown pass?"
    }
    f.write(json.dumps(start_entry) + "\n")

# More reliable method for getting caller information
def get_real_caller():
    #Get information about the caller, skipping intermediate functions like wrappers and decorators.
    frames = inspect.stack()
    # Skip first 2 frames (this function and immediate caller)
    for frame_info in frames[2:]:
        # Get the frame's module
        frame_module = frame_info.frame.f_globals.get('__name__', '')
        # If this frame is from our module (not from system libraries)
        if frame_module == 'current_script_8':
            # Check if it's not the call_llm function itself
            if frame_info.function != 'call_llm' and 'wrapper' not in frame_info.function:
                return {
                    "function": frame_info.function,
                    "filename": frame_info.filename,
                    "lineno": frame_info.lineno
                }
    # Fallback if we can't find a suitable caller
    return {"function": "unknown", "filename": "unknown", "lineno": 0}

# Create a tracing decorator for call_llm
def trace_call_llm(func):
    @functools.wraps(func)
    def wrapper(prompt, system_instruction=None):
        # Get caller information using our improved method
        caller_info = get_real_caller()

        # Create trace entry with caller information
        trace_entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "event": "llm_call",
            "iteration": 8,
            "sample_id": "cdf79bfa-ba57-4c8d-9622-4f34d14128ee",
            "function": "call_llm",
            "caller": caller_info,
            "input": {
                "prompt": prompt,
                "system_instruction": system_instruction
            }
        }

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
        "current_script_8", 
        "scripts/current_script_8.py"
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
    question = "PASSAGE: Hoping to rebound from their divisional road loss to the Giants, the Redskins played their Week 2 home opener against the New Orleans Saints.  In the first quarter, the 'Skins took the early lead when Shaun Suisham kicked a 22-yard field goal. In the second quarter, Washington increased its lead with Suisham's 36-yard field goal.  The Saints would respond with RB Pierre Thomas's 1-yard TD run and kicker Mart&#237;n Gram&#225;tica's 49-yard field goal.  This score marked the 269th consecutive game in which the Redskins allowed a point, tying the Cincinnati Bengals (1989-2006) for the lonest longest streak ever. In the third quarter, New Orleans increased its lead with QB Drew Brees completing a 19-yard TD pass to WR Robert Meachem.  Afterwards, the 'Skins responded when RB Clinton Portis scored on a 9-yard TD run (with a failed 2-point conversion).  The Saints responded with a Reggie Bush 55-yard punt return for a touchdown. The Redskins were able to mount a fourth quarter comeback by rallying for 14 points in the final stanza. The comeback started with a Clinton Portis 8-yard TD run, and was completed when QB Jason Campbell threw a 67-yard TD pass to WR Santana Moss. With the come from behind win, the Redskins improved to 1-1.\n\nQUESTION: How many yards longer was the longest touchdown pass compared with the shortest touchdown pass?"

    # Call the main function and get the answer
    answer = module.main(question)

    # Log execution completion
    with open(trace_file, 'a', encoding='utf-8') as f:
        end_entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "event": "execution_complete",
            "iteration": 8,
            "sample_id": "cdf79bfa-ba57-4c8d-9622-4f34d14128ee",
            "answer": str(answer)
        }
        f.write(json.dumps(end_entry) + "\n")

    # Print the answer for capture
    print("ANSWER_START")
    print(answer)
    print("ANSWER_END")

except Exception as e:
    # Log the error
    with open(trace_file, 'a', encoding='utf-8') as f:
        error_entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "event": "execution_error",
            "iteration": 8,
            "sample_id": "cdf79bfa-ba57-4c8d-9622-4f34d14128ee",
            "error": str(e),
            "traceback": traceback.format_exc()
        }
        f.write(json.dumps(error_entry) + "\n")

    print("ERROR_START")
    print(str(e))
    print(traceback.format_exc())
    print("ERROR_END")
