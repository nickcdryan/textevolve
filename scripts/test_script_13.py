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
trace_file = "archive/trace_iteration_13.jsonl"
os.makedirs(os.path.dirname(trace_file), exist_ok=True)

# Trace entry for execution start
with open(trace_file, 'a', encoding='utf-8') as f:
    start_entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "event": "execution_start",
        "iteration": 13,
        "sample_id": "example_120",
        "question": 'PASSAGE: At the start of World War II, the 24th Infantry was stationed at Fort Benning as school troops for the Infantry School. They participated in the Carolina Maneuvers of October - December 1941. During World War II, the 24th Infantry fought in the South Pacific Theatre as a separate regiment. Deploying on 4 April 1942 from the San Francisco Port of Embarkation, the regiment arrived in the New Hebrides Islands on 4 May 1942. The 24th moved to Guadalcanal on 28 August 1943, and was assigned to the XIV Corps. 1st Battalion deployed to Bougainville, attached to the 37th Infantry Division, from March to May 1944 for perimeter defense duty. The regiment departed Guadalcanal on 8 December 1944, and landed on Saipan and Tinian on 19 December 1944 for Garrison Duty that included mopping up the remaining Japanese forces that had yet to surrender. The regiment was assigned to the Pacific Ocean Area Command on 15 March 1945, and then to the Central Pacific Base Command on 15 May 1945, and to the Western pacific Base Command on 22 June 1945. The regiment departed Saipan and Tinian on 9 July 1945, and arrived on the Kerama Islands off Okinawa on 29 July 1945. At the end of the war, the 24th took the surrender of forces on the island of Aka-shima, the first formal surrender of a Japanese Imperial Army garrison. The regiment remained on Okinawa through 1946.\n\nQUESTION: How many months did the Carolina Manuevers last during 1941?'
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
        if frame_module == 'current_script_13':
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
            "iteration": 13,
            "sample_id": "example_120",
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
        "current_script_13", 
        "scripts/current_script_13.py"
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
    question = 'PASSAGE: At the start of World War II, the 24th Infantry was stationed at Fort Benning as school troops for the Infantry School. They participated in the Carolina Maneuvers of October - December 1941. During World War II, the 24th Infantry fought in the South Pacific Theatre as a separate regiment. Deploying on 4 April 1942 from the San Francisco Port of Embarkation, the regiment arrived in the New Hebrides Islands on 4 May 1942. The 24th moved to Guadalcanal on 28 August 1943, and was assigned to the XIV Corps. 1st Battalion deployed to Bougainville, attached to the 37th Infantry Division, from March to May 1944 for perimeter defense duty. The regiment departed Guadalcanal on 8 December 1944, and landed on Saipan and Tinian on 19 December 1944 for Garrison Duty that included mopping up the remaining Japanese forces that had yet to surrender. The regiment was assigned to the Pacific Ocean Area Command on 15 March 1945, and then to the Central Pacific Base Command on 15 May 1945, and to the Western pacific Base Command on 22 June 1945. The regiment departed Saipan and Tinian on 9 July 1945, and arrived on the Kerama Islands off Okinawa on 29 July 1945. At the end of the war, the 24th took the surrender of forces on the island of Aka-shima, the first formal surrender of a Japanese Imperial Army garrison. The regiment remained on Okinawa through 1946.\n\nQUESTION: How many months did the Carolina Manuevers last during 1941?'

    # Call the main function and get the answer
    answer = module.main(question)

    # Log execution completion
    with open(trace_file, 'a', encoding='utf-8') as f:
        end_entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "event": "execution_complete",
            "iteration": 13,
            "sample_id": "example_120",
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
            "iteration": 13,
            "sample_id": "example_120",
            "error": str(e),
            "traceback": traceback.format_exc()
        }
        f.write(json.dumps(error_entry) + "\n")

    print("ERROR_START")
    print(str(e))
    print(traceback.format_exc())
    print("ERROR_END")
