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
trace_file = "archive/trace_iteration_35.jsonl"
os.makedirs(os.path.dirname(trace_file), exist_ok=True)

# Trace entry for execution start
with open(trace_file, 'a', encoding='utf-8') as f:
    start_entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "event": "execution_start",
        "iteration": 35,
        "sample_id": "a6d5feed-0d8b-420a-8a75-c2ea9f6d57ee",
        "question": "PASSAGE: In week 5 the Lions hosted the St. Louis Rams. The Lions started the scoring early with a 30-yard Jason Hanson field goal. The Rams tied it up at the end of the first quarter with a 28-yard field goal by Josh Brown. To start the second quarter, the Lions took the lead with a 105-yard kickoff return by Stefan Logan, the longest touchdown run in the NFL this season. The Lions added to their lead a few minutes later with a 1-yard TD catch by Calvin Johnson. The Rams kicked another 28-yard field goal a few minutes later. The Lions made it 24-6 just before halftime with a 3-yard TD catch by Brandon Pettigrew. The Lions' defense shut out the Rams in the second half. The only score of the third quarter was a 26-yard TD catch by Nate Burleson. In the fourth quarter the Lions kicked 2 field goals: from 48 then from 47. The Lions capped off their victory with a 42-yard interception return TD by Alphonso Smith.  With the win, not only did the Lions improve to 1-4, but it was their largest margin of victory since 1995 and their first win since November 22, 2009.\n\nQUESTION: Who scored the second longest touchdown?"
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
        if frame_module == 'current_script_35':
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
            "iteration": 35,
            "sample_id": "a6d5feed-0d8b-420a-8a75-c2ea9f6d57ee",
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
        "current_script_35", 
        "scripts/current_script_35.py"
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
    question = "PASSAGE: In week 5 the Lions hosted the St. Louis Rams. The Lions started the scoring early with a 30-yard Jason Hanson field goal. The Rams tied it up at the end of the first quarter with a 28-yard field goal by Josh Brown. To start the second quarter, the Lions took the lead with a 105-yard kickoff return by Stefan Logan, the longest touchdown run in the NFL this season. The Lions added to their lead a few minutes later with a 1-yard TD catch by Calvin Johnson. The Rams kicked another 28-yard field goal a few minutes later. The Lions made it 24-6 just before halftime with a 3-yard TD catch by Brandon Pettigrew. The Lions' defense shut out the Rams in the second half. The only score of the third quarter was a 26-yard TD catch by Nate Burleson. In the fourth quarter the Lions kicked 2 field goals: from 48 then from 47. The Lions capped off their victory with a 42-yard interception return TD by Alphonso Smith.  With the win, not only did the Lions improve to 1-4, but it was their largest margin of victory since 1995 and their first win since November 22, 2009.\n\nQUESTION: Who scored the second longest touchdown?"

    # Call the main function and get the answer
    answer = module.main(question)

    # Log execution completion
    with open(trace_file, 'a', encoding='utf-8') as f:
        end_entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "event": "execution_complete",
            "iteration": 35,
            "sample_id": "a6d5feed-0d8b-420a-8a75-c2ea9f6d57ee",
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
            "iteration": 35,
            "sample_id": "a6d5feed-0d8b-420a-8a75-c2ea9f6d57ee",
            "error": str(e),
            "traceback": traceback.format_exc()
        }
        f.write(json.dumps(error_entry) + "\n")

    print("ERROR_START")
    print(str(e))
    print(traceback.format_exc())
    print("ERROR_END")
