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
trace_file = "archive/trace_iteration_4.jsonl"
os.makedirs(os.path.dirname(trace_file), exist_ok=True)

# Trace entry for execution start
with open(trace_file, 'a', encoding='utf-8') as f:
    start_entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "event": "execution_start",
        "iteration": 4,
        "sample_id": "0a87e21c-a1d0-42b1-9b62-62fa67d0a7f0",
        "question": "PASSAGE: Still searching for their first win of the year, the Dolphins returned home for a Week 7 AFC East duel with the undefeated New England Patriots.  In the first quarter, Miami's struggles continued as Patriots QB Tom Brady completed a 30-yard TD pass to WR Dont&#233; Stallworth and a 2-yard TD pass to TE Kyle Brady.  In the second quarter, the Dolphins got their only score of the half with QB Cleo Lemon getting a 4-yard TD run.  Afterwards, New England continued its pounding with CB Willie Andrews returning a kickoff 77 yards for a touchdown, along with Brady and WR Randy Moss hooking up with each other on a 35-yard and a 50-yard TD pass.  The Patriots would end the half with Brady hooking up with WR Wes Welker on a 14-yard TD pass. After a scoreless third quarter, Miami tried to make a comeback in the fourth quarter.  It began with RB Patrick Cobbs getting a 1-yard TD run, along with DE Jason Taylor returning an interception 36 yards for a touchdown.  However, New England made sure that they sealed their victory with Brady hooking up with Welker again on a 16-yard TD pass.  Afterwards, the 'Fins ended the game with RB Jesse Chatman getting a 7-yard TD run. With the loss, not only did the Dolphins fall to 0-7 for the first time in franchise history, but they also lost RB Ronnie Brown for the year, as he tore his ACL in the third quarter while trying to stop an interception return.\n\nQUESTION: How many TD passes did Tom Brady complete?"
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
        if frame_module == 'current_script_4':
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
            "iteration": 4,
            "sample_id": "0a87e21c-a1d0-42b1-9b62-62fa67d0a7f0",
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
        "current_script_4", 
        "scripts/current_script_4.py"
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
    question = "PASSAGE: Still searching for their first win of the year, the Dolphins returned home for a Week 7 AFC East duel with the undefeated New England Patriots.  In the first quarter, Miami's struggles continued as Patriots QB Tom Brady completed a 30-yard TD pass to WR Dont&#233; Stallworth and a 2-yard TD pass to TE Kyle Brady.  In the second quarter, the Dolphins got their only score of the half with QB Cleo Lemon getting a 4-yard TD run.  Afterwards, New England continued its pounding with CB Willie Andrews returning a kickoff 77 yards for a touchdown, along with Brady and WR Randy Moss hooking up with each other on a 35-yard and a 50-yard TD pass.  The Patriots would end the half with Brady hooking up with WR Wes Welker on a 14-yard TD pass. After a scoreless third quarter, Miami tried to make a comeback in the fourth quarter.  It began with RB Patrick Cobbs getting a 1-yard TD run, along with DE Jason Taylor returning an interception 36 yards for a touchdown.  However, New England made sure that they sealed their victory with Brady hooking up with Welker again on a 16-yard TD pass.  Afterwards, the 'Fins ended the game with RB Jesse Chatman getting a 7-yard TD run. With the loss, not only did the Dolphins fall to 0-7 for the first time in franchise history, but they also lost RB Ronnie Brown for the year, as he tore his ACL in the third quarter while trying to stop an interception return.\n\nQUESTION: How many TD passes did Tom Brady complete?"

    # Call the main function and get the answer
    answer = module.main(question)

    # Log execution completion
    with open(trace_file, 'a', encoding='utf-8') as f:
        end_entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "event": "execution_complete",
            "iteration": 4,
            "sample_id": "0a87e21c-a1d0-42b1-9b62-62fa67d0a7f0",
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
            "iteration": 4,
            "sample_id": "0a87e21c-a1d0-42b1-9b62-62fa67d0a7f0",
            "error": str(e),
            "traceback": traceback.format_exc()
        }
        f.write(json.dumps(error_entry) + "\n")

    print("ERROR_START")
    print(str(e))
    print(traceback.format_exc())
    print("ERROR_END")
