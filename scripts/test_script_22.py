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
trace_file = "archive/trace_iteration_22.jsonl"
os.makedirs(os.path.dirname(trace_file), exist_ok=True)

# Trace entry for execution start
with open(trace_file, 'a', encoding='utf-8') as f:
    start_entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "event": "execution_start",
        "iteration": 22,
        "sample_id": "636528d3-8d19-4ef6-80e9-1d48e2bee046",
        "question": "PASSAGE: Coming off their season-sweeping road win over the Giants, the Cowboys went home for a Week 11 NFC East duel with the Washington Redskins, a longtime rival of the Cowboys.  In the first quarter, Dallas trailed early as Redskins QB Jason Campbell completed a 19-yard TD pass to TE Chris Cooley for the only score of the period.  In the second quarter, the 'Boys got on the board as QB Tony Romo completed a 4-yard TD pass to WR Terrell Owens.  Washington would take the lead into halftime as kicker Shaun Suisham managed to get a 45-yard field goal. In the third quarter, the Cowboys took the lead with Romo and Owens hooking up with each other again on a 31-yard TD pass.  The Redskins got close as Suisham kicked a 39-yard field goal.  Fortunately, in the fourth quarter, the 'Boys started to run away with the lead as Romo and Owens hooked up with each other for a third time on a 46-yard TD pass.  Washington tried to come back as Suisham nailed a 44-yard field goal, yet Dallas continued to roll as Romo hooked up with Owens for the fourth time on a 52-yard TD pass.  The Redskins tried to come back as Campbell completed a 5-yard TD pass to WR Santana Moss.  Fortunately, the Dallas defense held on for the victory. Terrell Owens, with his 4 touchdown receptions, tied Bob Hayes for the most touchdown receptions by a Cowboys receiver in one game.\n\nQUESTION: How many yards was the longest touchdown pass of the game?"
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
        if frame_module == 'current_script_22':
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
            "iteration": 22,
            "sample_id": "636528d3-8d19-4ef6-80e9-1d48e2bee046",
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
        "current_script_22", 
        "scripts/current_script_22.py"
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
    question = "PASSAGE: Coming off their season-sweeping road win over the Giants, the Cowboys went home for a Week 11 NFC East duel with the Washington Redskins, a longtime rival of the Cowboys.  In the first quarter, Dallas trailed early as Redskins QB Jason Campbell completed a 19-yard TD pass to TE Chris Cooley for the only score of the period.  In the second quarter, the 'Boys got on the board as QB Tony Romo completed a 4-yard TD pass to WR Terrell Owens.  Washington would take the lead into halftime as kicker Shaun Suisham managed to get a 45-yard field goal. In the third quarter, the Cowboys took the lead with Romo and Owens hooking up with each other again on a 31-yard TD pass.  The Redskins got close as Suisham kicked a 39-yard field goal.  Fortunately, in the fourth quarter, the 'Boys started to run away with the lead as Romo and Owens hooked up with each other for a third time on a 46-yard TD pass.  Washington tried to come back as Suisham nailed a 44-yard field goal, yet Dallas continued to roll as Romo hooked up with Owens for the fourth time on a 52-yard TD pass.  The Redskins tried to come back as Campbell completed a 5-yard TD pass to WR Santana Moss.  Fortunately, the Dallas defense held on for the victory. Terrell Owens, with his 4 touchdown receptions, tied Bob Hayes for the most touchdown receptions by a Cowboys receiver in one game.\n\nQUESTION: How many yards was the longest touchdown pass of the game?"

    # Call the main function and get the answer
    answer = module.main(question)

    # Log execution completion
    with open(trace_file, 'a', encoding='utf-8') as f:
        end_entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "event": "execution_complete",
            "iteration": 22,
            "sample_id": "636528d3-8d19-4ef6-80e9-1d48e2bee046",
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
            "iteration": 22,
            "sample_id": "636528d3-8d19-4ef6-80e9-1d48e2bee046",
            "error": str(e),
            "traceback": traceback.format_exc()
        }
        f.write(json.dumps(error_entry) + "\n")

    print("ERROR_START")
    print(str(e))
    print(traceback.format_exc())
    print("ERROR_END")
