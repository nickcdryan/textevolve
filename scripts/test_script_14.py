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
trace_file = "archive/trace_iteration_14.jsonl"
os.makedirs(os.path.dirname(trace_file), exist_ok=True)

# Trace entry for execution start
with open(trace_file, 'a', encoding='utf-8') as f:
    start_entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "event": "execution_start",
        "iteration": 14,
        "sample_id": "71a0cca2-3fb4-4b42-a015-3ad78081902f",
        "question": "PASSAGE: Trying to rebound from their road loss to the Packers, the Redskins went home for a Week 7 match-up against the Arizona Cardinals. In the first quarter, Washington took the early lead with running back Clinton Portis getting a 2-yard touchdown run for the only score of the quarter. In the second quarter, the Redskins increased its lead with linebacker London Fletcher returning an interception 27 yards for a touchdown. The Cardinals would get a touchdown as quarterback Kurt Warner completed a 2-yard touchdown pass to wide receiver Anquan Boldin (with a failed PAT). In the third quarter, Washington increased its lead with Portis getting a 1-yard touchdown run for the only score of the quarter. In the fourth quarter, the Cardinals managed to get within striking distance as quarterback Warner and wide receiver Boldin hooked up on a 10-yard touchdown pass. Afterwards, the Cardinals got within two points with quarterback Tim Rattay completing a 1-yard touchdown pass to tight end Leonard Pope (with a failed 2-point conversion). Later, the Cardinals managed to recover its onside kick and managed to set up a game-winning 55-yard field goal. Fortunately for the Redskins: the Cardinals' kick missed wide left&#8212;securing the victory.\n\nQUESTION: How many field goals were made in the game?"
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
        if frame_module == 'current_script_14':
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
            "iteration": 14,
            "sample_id": "71a0cca2-3fb4-4b42-a015-3ad78081902f",
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
        "current_script_14", 
        "scripts/current_script_14.py"
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
    question = "PASSAGE: Trying to rebound from their road loss to the Packers, the Redskins went home for a Week 7 match-up against the Arizona Cardinals. In the first quarter, Washington took the early lead with running back Clinton Portis getting a 2-yard touchdown run for the only score of the quarter. In the second quarter, the Redskins increased its lead with linebacker London Fletcher returning an interception 27 yards for a touchdown. The Cardinals would get a touchdown as quarterback Kurt Warner completed a 2-yard touchdown pass to wide receiver Anquan Boldin (with a failed PAT). In the third quarter, Washington increased its lead with Portis getting a 1-yard touchdown run for the only score of the quarter. In the fourth quarter, the Cardinals managed to get within striking distance as quarterback Warner and wide receiver Boldin hooked up on a 10-yard touchdown pass. Afterwards, the Cardinals got within two points with quarterback Tim Rattay completing a 1-yard touchdown pass to tight end Leonard Pope (with a failed 2-point conversion). Later, the Cardinals managed to recover its onside kick and managed to set up a game-winning 55-yard field goal. Fortunately for the Redskins: the Cardinals' kick missed wide left&#8212;securing the victory.\n\nQUESTION: How many field goals were made in the game?"

    # Call the main function and get the answer
    answer = module.main(question)

    # Log execution completion
    with open(trace_file, 'a', encoding='utf-8') as f:
        end_entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "event": "execution_complete",
            "iteration": 14,
            "sample_id": "71a0cca2-3fb4-4b42-a015-3ad78081902f",
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
            "iteration": 14,
            "sample_id": "71a0cca2-3fb4-4b42-a015-3ad78081902f",
            "error": str(e),
            "traceback": traceback.format_exc()
        }
        f.write(json.dumps(error_entry) + "\n")

    print("ERROR_START")
    print(str(e))
    print(traceback.format_exc())
    print("ERROR_END")
