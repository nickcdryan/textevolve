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
trace_file = "archive/trace_iteration_5.jsonl"
os.makedirs(os.path.dirname(trace_file), exist_ok=True)

# Trace entry for execution start
with open(trace_file, 'a', encoding='utf-8') as f:
    start_entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "event": "execution_start",
        "iteration": 5,
        "sample_id": "example_40",
        "question": 'PASSAGE: Trying to snap a five-game losing streak, the Rams stayed at home for a Week 13 interconference duel with the Miami Dolphins. The Rams would welcome back Steven Jackson to the lineup after weeks of sitting out with a thigh injury. Jackson gave the Rams a good enough boost to strike first with a first possession field goal by Josh Brown from 23 yards. Brown would kick a 51-yard field goal to give the Rams a 6-0 lead. In the second quarter, the Dolphins responded as RB Ronnie Brown got a 3-yard TD run. The Rams would answer with Brown making a 33-yard field goal, but Miami replied with kicker Dan Carpenter getting a 37-yard field goal. In the third quarter, the Dolphins increased their lead as Carpenter got a 47-yard field goal. In the fourth quarter, St. Louis tried to keep up as Brown made a 38-yard field goal, yet Miami answered right back with Carpenter nailing a 42-yard field goal. The Rams tried to come back, but a late-game interception shattered any hope of a comeback.\n\nQUESTION: How many points got the Rams on the board?'
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
        if frame_module == 'current_script_5':
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
            "iteration": 5,
            "sample_id": "example_40",
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
        "current_script_5", 
        "scripts/current_script_5.py"
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
    question = 'PASSAGE: Trying to snap a five-game losing streak, the Rams stayed at home for a Week 13 interconference duel with the Miami Dolphins. The Rams would welcome back Steven Jackson to the lineup after weeks of sitting out with a thigh injury. Jackson gave the Rams a good enough boost to strike first with a first possession field goal by Josh Brown from 23 yards. Brown would kick a 51-yard field goal to give the Rams a 6-0 lead. In the second quarter, the Dolphins responded as RB Ronnie Brown got a 3-yard TD run. The Rams would answer with Brown making a 33-yard field goal, but Miami replied with kicker Dan Carpenter getting a 37-yard field goal. In the third quarter, the Dolphins increased their lead as Carpenter got a 47-yard field goal. In the fourth quarter, St. Louis tried to keep up as Brown made a 38-yard field goal, yet Miami answered right back with Carpenter nailing a 42-yard field goal. The Rams tried to come back, but a late-game interception shattered any hope of a comeback.\n\nQUESTION: How many points got the Rams on the board?'

    # Call the main function and get the answer
    answer = module.main(question)

    # Log execution completion
    with open(trace_file, 'a', encoding='utf-8') as f:
        end_entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "event": "execution_complete",
            "iteration": 5,
            "sample_id": "example_40",
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
            "iteration": 5,
            "sample_id": "example_40",
            "error": str(e),
            "traceback": traceback.format_exc()
        }
        f.write(json.dumps(error_entry) + "\n")

    print("ERROR_START")
    print(str(e))
    print(traceback.format_exc())
    print("ERROR_END")
