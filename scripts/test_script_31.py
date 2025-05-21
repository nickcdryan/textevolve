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
trace_file = "archive/trace_iteration_31.jsonl"
os.makedirs(os.path.dirname(trace_file), exist_ok=True)

# Trace entry for execution start
with open(trace_file, 'a', encoding='utf-8') as f:
    start_entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "event": "execution_start",
        "iteration": 31,
        "sample_id": "348d3cac-4da4-4517-a0d2-6530503d8601",
        "question": "PASSAGE: The Cardinals began their 2007 campaign on the road against their NFC West foe, the San Francisco 49ers.  In the first quarter, Arizona trailed early as 49ers RB Frank Gore got a 6-yard TD run for the only score of the period.  In the second quarter, the Cardinals took the lead with kicker Neil Rackers getting a 35-yard field goal, while RB Edgerrin James got a 7-yard TD run.  San Francisco would tie the game with kicker Joe Nedney getting a 33-yard field goal. In the third quarter, the 49ers regained the lead with Nedney kicking a 30-yard field goal for the only score of the period.  In the fourth quarter, the Cardinals retook the lead with QB Matt Leinart completing a 5-yard TD pass to WR Anquan Boldin.  However, late in the game, the Cards' defense failed to hold off San Francisco's ensuing drive, which ended with WR Arnaz Battle getting a 1-yard TD run.  With just over&#160;:20 seconds left in the game, Arizona had one final chance to save the game. Leinart's pass to WR Larry Fitzgerald was intercepted by 49ers CB Shawntae Spencer. With the heartbreaking loss, the Cardinals began their season at 0-1. Q1 - SF - 11:24 - Frank Gore 6-yard TD run (Joe Nedney kick) (SF 7-0) Q2 - ARI - 12:55 - Neil Rackers 35-yard FG (SF 7-3) Q2 - ARI - 9:15 - Edgerrin James 7-yard TD run (Rackers kick) (ARI 10-7) Q2 - SF - 3:40 - Joe Nedney 33-yard FG (10-10) Q3 - SF - 11:20 - Joe Nedney 30-yard FG (SF 13-10) Q4 - ARI - 6:46 - 5-yard TD pass from Matt Leinart to Anquan Boldin (Rackers kick) (ARI 17-13) Q4 - SF - 0:26 - Arnaz Battle 1-yard TD run (Nedney kick) (SF 20-17)\n\nQUESTION: How many rushing tds did Edgerrin James have in the game?"
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
        if frame_module == 'current_script_31':
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
            "iteration": 31,
            "sample_id": "348d3cac-4da4-4517-a0d2-6530503d8601",
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
        "current_script_31", 
        "scripts/current_script_31.py"
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
    question = "PASSAGE: The Cardinals began their 2007 campaign on the road against their NFC West foe, the San Francisco 49ers.  In the first quarter, Arizona trailed early as 49ers RB Frank Gore got a 6-yard TD run for the only score of the period.  In the second quarter, the Cardinals took the lead with kicker Neil Rackers getting a 35-yard field goal, while RB Edgerrin James got a 7-yard TD run.  San Francisco would tie the game with kicker Joe Nedney getting a 33-yard field goal. In the third quarter, the 49ers regained the lead with Nedney kicking a 30-yard field goal for the only score of the period.  In the fourth quarter, the Cardinals retook the lead with QB Matt Leinart completing a 5-yard TD pass to WR Anquan Boldin.  However, late in the game, the Cards' defense failed to hold off San Francisco's ensuing drive, which ended with WR Arnaz Battle getting a 1-yard TD run.  With just over&#160;:20 seconds left in the game, Arizona had one final chance to save the game. Leinart's pass to WR Larry Fitzgerald was intercepted by 49ers CB Shawntae Spencer. With the heartbreaking loss, the Cardinals began their season at 0-1. Q1 - SF - 11:24 - Frank Gore 6-yard TD run (Joe Nedney kick) (SF 7-0) Q2 - ARI - 12:55 - Neil Rackers 35-yard FG (SF 7-3) Q2 - ARI - 9:15 - Edgerrin James 7-yard TD run (Rackers kick) (ARI 10-7) Q2 - SF - 3:40 - Joe Nedney 33-yard FG (10-10) Q3 - SF - 11:20 - Joe Nedney 30-yard FG (SF 13-10) Q4 - ARI - 6:46 - 5-yard TD pass from Matt Leinart to Anquan Boldin (Rackers kick) (ARI 17-13) Q4 - SF - 0:26 - Arnaz Battle 1-yard TD run (Nedney kick) (SF 20-17)\n\nQUESTION: How many rushing tds did Edgerrin James have in the game?"

    # Call the main function and get the answer
    answer = module.main(question)

    # Log execution completion
    with open(trace_file, 'a', encoding='utf-8') as f:
        end_entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "event": "execution_complete",
            "iteration": 31,
            "sample_id": "348d3cac-4da4-4517-a0d2-6530503d8601",
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
            "iteration": 31,
            "sample_id": "348d3cac-4da4-4517-a0d2-6530503d8601",
            "error": str(e),
            "traceback": traceback.format_exc()
        }
        f.write(json.dumps(error_entry) + "\n")

    print("ERROR_START")
    print(str(e))
    print(traceback.format_exc())
    print("ERROR_END")
