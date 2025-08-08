import sys
import traceback
import os
import json
import datetime
import inspect
import functools
import importlib.util

# Import system tool functions
from system_tools import call_llm, call_database, read_file, search_file, execute_code

# Add the scripts directory to the path
sys.path.append("{scripts_dir}")

# Configure tracing
trace_file = "{trace_file}"
os.makedirs(os.path.dirname(trace_file), exist_ok=True)

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
            
            # Comment out the system_tools import line while preserving indentation
            def preserve_indentation(match):
                line = match.group(0)
                # Get leading whitespace from the original line
                leading_whitespace = re.match(r'^(\s*)', line).group(1)
                return leading_whitespace + '# from system_tools import... # (functions injected by system)'
            
            modified_content = re.sub(
                r'^(\s*)from system_tools import.*$', 
                preserve_indentation, 
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