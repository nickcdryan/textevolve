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
trace_file = "archive/trace_iteration_0.jsonl"
os.makedirs(os.path.dirname(trace_file), exist_ok=True)

# Trace entry for execution start
with open(trace_file, 'a', encoding='utf-8') as f:
    start_entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "event": "execution_start",
        "iteration": 0,
        "sample_id": "example_7",
        "question": "PASSAGE: Coming off their easy road win over the Rams, the Vikings went home for a Week 6 inter-conference duel with the Baltimore Ravens. Minnesota got off to a fast start in the first quarter with quarterback Brett Favre completing a 19-yard touchdown pass to tight end Visanthe Shiancoe and a 4-yard touchdown pass to wide receiver Bernard Berrian. Afterwards, the Ravens got the only points of the second quarter as kicker Steven Hauschka getting a 29-yard field goal. In the third quarter, the Vikings picked up where they left off with a 40-yard field goal from kicker Ryan Longwell. Baltimore responded with a 22-yard touchdown run from running back Ray Rice, yet Longwell helped out Minnesota by nailing a 22-yard field goal. Afterwards, an action-packed fourth quarter ensued. Minnesota increased its lead with Favre hooking up with Shiancoe again on a 1-yard touchdown pass, but the Ravens continued to hang around as quarterback Joe Flacco found wide receiver Mark Clayton on a 32-yard touchdown pass. The Vikings replied with Longwell's 29-yard field goal, but Baltimore took lead for the first time in the game as Flacco hooked up with wide receiver Derrick Mason on a 12-yard touchdown pass and Rice running 33 yards for a touchdown. Minnesota then regained the lead as Longwell booted a 31-yard field goal after a 58-yard pass from quarterback Brett Favre to wide receiver Sidney Rice. The Ravens got a last-minute drive into scoring range, but Hauschka's 44-yard field goal attempt went wide left, preserving the Vikings' perfect season. With the win, the Vikings acquired their first 6-0 start since 2003 (unfortunately that team did not make the playoffs). Also, dating back to Week 17 of the 2008 season, Minnesota has won seven-straight regular season games for the first time since 2000.\n\nQUESTION: Who threw the second longest touchdown pass?"
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
        if frame_module == 'current_script_0':
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
            "iteration": 0,
            "sample_id": "example_7",
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
        "current_script_0", 
        "scripts/current_script_0.py"
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
    question = "PASSAGE: Coming off their easy road win over the Rams, the Vikings went home for a Week 6 inter-conference duel with the Baltimore Ravens. Minnesota got off to a fast start in the first quarter with quarterback Brett Favre completing a 19-yard touchdown pass to tight end Visanthe Shiancoe and a 4-yard touchdown pass to wide receiver Bernard Berrian. Afterwards, the Ravens got the only points of the second quarter as kicker Steven Hauschka getting a 29-yard field goal. In the third quarter, the Vikings picked up where they left off with a 40-yard field goal from kicker Ryan Longwell. Baltimore responded with a 22-yard touchdown run from running back Ray Rice, yet Longwell helped out Minnesota by nailing a 22-yard field goal. Afterwards, an action-packed fourth quarter ensued. Minnesota increased its lead with Favre hooking up with Shiancoe again on a 1-yard touchdown pass, but the Ravens continued to hang around as quarterback Joe Flacco found wide receiver Mark Clayton on a 32-yard touchdown pass. The Vikings replied with Longwell's 29-yard field goal, but Baltimore took lead for the first time in the game as Flacco hooked up with wide receiver Derrick Mason on a 12-yard touchdown pass and Rice running 33 yards for a touchdown. Minnesota then regained the lead as Longwell booted a 31-yard field goal after a 58-yard pass from quarterback Brett Favre to wide receiver Sidney Rice. The Ravens got a last-minute drive into scoring range, but Hauschka's 44-yard field goal attempt went wide left, preserving the Vikings' perfect season. With the win, the Vikings acquired their first 6-0 start since 2003 (unfortunately that team did not make the playoffs). Also, dating back to Week 17 of the 2008 season, Minnesota has won seven-straight regular season games for the first time since 2000.\n\nQUESTION: Who threw the second longest touchdown pass?"

    # Call the main function and get the answer
    answer = module.main(question)

    # Log execution completion
    with open(trace_file, 'a', encoding='utf-8') as f:
        end_entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "event": "execution_complete",
            "iteration": 0,
            "sample_id": "example_7",
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
            "iteration": 0,
            "sample_id": "example_7",
            "error": str(e),
            "traceback": traceback.format_exc()
        }
        f.write(json.dumps(error_entry) + "\n")

    print("ERROR_START")
    print(str(e))
    print(traceback.format_exc())
    print("ERROR_END")
