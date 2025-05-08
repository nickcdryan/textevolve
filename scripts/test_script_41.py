import sys
import traceback
import os
import json
import datetime
import inspect
import functools
import importlib.util

# Add the scripts directory to the path
sys.path.append("scripts")

# Ensure the Gemini API key is available to the script
os.environ["GEMINI_API_KEY"] = "AIzaSyD_DWppm-TR9CN7xTTVmrW5ngTax7xsLDA"

# Configure tracing
trace_file = "archive/trace_iteration_41.jsonl"
os.makedirs(os.path.dirname(trace_file), exist_ok=True)

# Trace entry for execution start
with open(trace_file, 'a', encoding='utf-8') as f:
    start_entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "event": "execution_start",
        "iteration": 41,
        "sample_id": "example_134",
        "question": 'Grid Transformation Task\n\n=== TRAINING EXAMPLES ===\n\nExample 1:\nInput Grid:\n[\n  [3, 0, 0, 7, 0, 0, 9, 7, 0]\n  [8, 4, 0, 6, 6, 0, 4, 8, 4]\n  [1, 7, 0, 0, 0, 0, 4, 0, 0]\n  [1, 1, 0, 9, 1, 0, 7, 0, 0]\n  [0, 0, 0, 0, 7, 7, 0, 0, 0]\n  [8, 0, 0, 1, 7, 0, 8, 4, 0]\n  [0, 7, 0, 9, 9, 2, 1, 0, 0]\n  [0, 0, 0, 0, 0, 0, 5, 0, 0]\n  [0, 0, 0, 2, 4, 0, 8, 0, 0]\n]\n\nOutput Grid:\n[\n  [9, 7, 0]\n  [4, 8, 4]\n  [4, 0, 0]\n]\nExample 2:\nInput Grid:\n[\n  [9, 0, 0, 0, 0, 0, 0, 6, 0]\n  [0, 4, 0, 7, 0, 5, 0, 8, 1]\n  [0, 2, 0, 0, 7, 1, 4, 4, 5]\n  [0, 6, 0, 0, 4, 0, 0, 0, 0]\n  [8, 3, 0, 4, 2, 0, 0, 9, 7]\n  [0, 0, 2, 3, 0, 2, 0, 6, 7]\n  [4, 0, 4, 0, 3, 4, 7, 0, 7]\n  [7, 1, 0, 0, 0, 0, 3, 0, 0]\n  [3, 2, 0, 0, 4, 0, 0, 0, 0]\n]\n\nOutput Grid:\n[\n  [0, 6, 0]\n  [0, 8, 1]\n  [4, 4, 5]\n]\nExample 3:\nInput Grid:\n[\n  [2, 5, 0, 0, 6, 0, 0, 0, 0]\n  [2, 5, 5, 7, 0, 0, 6, 0, 1]\n  [0, 3, 0, 0, 0, 1, 9, 4, 0]\n  [0, 7, 0, 6, 0, 0, 0, 0, 0]\n  [0, 9, 0, 0, 0, 1, 0, 0, 8]\n  [0, 0, 0, 0, 0, 0, 0, 0, 0]\n  [0, 0, 4, 0, 0, 0, 0, 0, 0]\n  [0, 0, 0, 1, 0, 0, 0, 0, 4]\n  [0, 5, 0, 0, 0, 0, 0, 0, 0]\n]\n\nOutput Grid:\n[\n  [0, 0, 0]\n  [6, 0, 1]\n  [9, 4, 0]\n]\nExample 4:\nInput Grid:\n[\n  [0, 5, 0, 0, 8, 0, 0, 0, 4]\n  [0, 0, 0, 0, 0, 0, 3, 0, 0]\n  [0, 0, 0, 0, 2, 1, 0, 0, 3]\n  [0, 1, 0, 0, 0, 0, 3, 0, 0]\n  [1, 0, 0, 1, 0, 0, 0, 0, 0]\n  [0, 0, 0, 0, 0, 0, 0, 8, 0]\n  [0, 0, 0, 0, 0, 0, 0, 0, 0]\n  [0, 0, 9, 4, 0, 0, 0, 0, 0]\n  [3, 0, 7, 0, 0, 2, 0, 0, 6]\n]\n\nOutput Grid:\n[\n  [0, 0, 4]\n  [3, 0, 0]\n  [0, 0, 3]\n]\n\n=== TEST INPUT ===\n[\n  [6, 9, 0, 0, 1, 0, 5, 8, 9]\n  [2, 9, 0, 6, 0, 8, 0, 9, 0]\n  [0, 0, 0, 0, 0, 9, 9, 2, 0]\n  [9, 2, 6, 0, 0, 8, 0, 6, 8]\n  [7, 7, 4, 0, 7, 0, 9, 0, 0]\n  [0, 0, 7, 0, 0, 1, 5, 7, 4]\n  [4, 1, 0, 0, 7, 5, 0, 0, 9]\n  [9, 9, 0, 0, 0, 0, 1, 0, 0]\n  [4, 9, 2, 0, 0, 0, 8, 4, 0]\n]\n\nTransform the test input according to the pattern shown in the training examples.'
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
        if frame_module == 'current_script_41':
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
            "iteration": 41,
            "sample_id": "example_134",
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
        "current_script_41", 
        "scripts/current_script_41.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

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
    question = 'Grid Transformation Task\n\n=== TRAINING EXAMPLES ===\n\nExample 1:\nInput Grid:\n[\n  [3, 0, 0, 7, 0, 0, 9, 7, 0]\n  [8, 4, 0, 6, 6, 0, 4, 8, 4]\n  [1, 7, 0, 0, 0, 0, 4, 0, 0]\n  [1, 1, 0, 9, 1, 0, 7, 0, 0]\n  [0, 0, 0, 0, 7, 7, 0, 0, 0]\n  [8, 0, 0, 1, 7, 0, 8, 4, 0]\n  [0, 7, 0, 9, 9, 2, 1, 0, 0]\n  [0, 0, 0, 0, 0, 0, 5, 0, 0]\n  [0, 0, 0, 2, 4, 0, 8, 0, 0]\n]\n\nOutput Grid:\n[\n  [9, 7, 0]\n  [4, 8, 4]\n  [4, 0, 0]\n]\nExample 2:\nInput Grid:\n[\n  [9, 0, 0, 0, 0, 0, 0, 6, 0]\n  [0, 4, 0, 7, 0, 5, 0, 8, 1]\n  [0, 2, 0, 0, 7, 1, 4, 4, 5]\n  [0, 6, 0, 0, 4, 0, 0, 0, 0]\n  [8, 3, 0, 4, 2, 0, 0, 9, 7]\n  [0, 0, 2, 3, 0, 2, 0, 6, 7]\n  [4, 0, 4, 0, 3, 4, 7, 0, 7]\n  [7, 1, 0, 0, 0, 0, 3, 0, 0]\n  [3, 2, 0, 0, 4, 0, 0, 0, 0]\n]\n\nOutput Grid:\n[\n  [0, 6, 0]\n  [0, 8, 1]\n  [4, 4, 5]\n]\nExample 3:\nInput Grid:\n[\n  [2, 5, 0, 0, 6, 0, 0, 0, 0]\n  [2, 5, 5, 7, 0, 0, 6, 0, 1]\n  [0, 3, 0, 0, 0, 1, 9, 4, 0]\n  [0, 7, 0, 6, 0, 0, 0, 0, 0]\n  [0, 9, 0, 0, 0, 1, 0, 0, 8]\n  [0, 0, 0, 0, 0, 0, 0, 0, 0]\n  [0, 0, 4, 0, 0, 0, 0, 0, 0]\n  [0, 0, 0, 1, 0, 0, 0, 0, 4]\n  [0, 5, 0, 0, 0, 0, 0, 0, 0]\n]\n\nOutput Grid:\n[\n  [0, 0, 0]\n  [6, 0, 1]\n  [9, 4, 0]\n]\nExample 4:\nInput Grid:\n[\n  [0, 5, 0, 0, 8, 0, 0, 0, 4]\n  [0, 0, 0, 0, 0, 0, 3, 0, 0]\n  [0, 0, 0, 0, 2, 1, 0, 0, 3]\n  [0, 1, 0, 0, 0, 0, 3, 0, 0]\n  [1, 0, 0, 1, 0, 0, 0, 0, 0]\n  [0, 0, 0, 0, 0, 0, 0, 8, 0]\n  [0, 0, 0, 0, 0, 0, 0, 0, 0]\n  [0, 0, 9, 4, 0, 0, 0, 0, 0]\n  [3, 0, 7, 0, 0, 2, 0, 0, 6]\n]\n\nOutput Grid:\n[\n  [0, 0, 4]\n  [3, 0, 0]\n  [0, 0, 3]\n]\n\n=== TEST INPUT ===\n[\n  [6, 9, 0, 0, 1, 0, 5, 8, 9]\n  [2, 9, 0, 6, 0, 8, 0, 9, 0]\n  [0, 0, 0, 0, 0, 9, 9, 2, 0]\n  [9, 2, 6, 0, 0, 8, 0, 6, 8]\n  [7, 7, 4, 0, 7, 0, 9, 0, 0]\n  [0, 0, 7, 0, 0, 1, 5, 7, 4]\n  [4, 1, 0, 0, 7, 5, 0, 0, 9]\n  [9, 9, 0, 0, 0, 0, 1, 0, 0]\n  [4, 9, 2, 0, 0, 0, 8, 4, 0]\n]\n\nTransform the test input according to the pattern shown in the training examples.'

    # Call the main function and get the answer
    answer = module.main(question)

    # Log execution completion
    with open(trace_file, 'a', encoding='utf-8') as f:
        end_entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "event": "execution_complete",
            "iteration": 41,
            "sample_id": "example_134",
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
            "iteration": 41,
            "sample_id": "example_134",
            "error": str(e),
            "traceback": traceback.format_exc()
        }
        f.write(json.dumps(error_entry) + "\n")

    print("ERROR_START")
    print(str(e))
    print(traceback.format_exc())
    print("ERROR_END")
