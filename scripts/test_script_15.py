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
trace_file = "archive/trace_iteration_15.jsonl"
os.makedirs(os.path.dirname(trace_file), exist_ok=True)

# Trace entry for execution start
with open(trace_file, 'a', encoding='utf-8') as f:
    start_entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "event": "execution_start",
        "iteration": 15,
        "sample_id": "4cbe93b3-effb-4007-9fd3-c31396fcad84",
        "question": "PASSAGE: Hoping to rebound from their loss to the Rams, the Broncos returned home for an AFC duel against the Miami Dolphins &#8212; the Broncos' first home game in exactly one month. The Dolphins grabbed the early lead, with running back Daniel Thomas rushing for a 3-yard touchdown. The Broncos subsequently got on the scoreboard, with a 38-yard field goal by placekicker Brandon McManus. The Dolphins added to their lead early in the second quarter, with quarterback Ryan Tannehill rushing for a 1-yard touchdown. The Broncos cut into the Dolphins' lead, with quarterback Peyton Manning connecting on a 5-yard touchdown pass to wide receiver Demaryius Thomas, however, the Dolphins responded, with Tannehill throwing a 10-yard touchdown pass to wide receiver Mike Wallace. Another touchdown pass from Manning to Demaryius Thomas &#8212; from 14 yards out &#8212; narrowed the Dolphins' lead to 21-17 just before halftime. The Broncos drove to as far as the Dolphins' 7-yard line on the initial possession of the second half, but after Manning was sacked by Dolphins' linebacker Jelani Jenkins for an 8-yard loss, McManus missed on a 33-yard field goal attempt. After the Broncos' defense forced a Dolphins' punt, return specialist Isaiah Burse fumbled deep in Broncos' territory, giving the football back to the Dolphins. Four plays later, Tannehill connected with wide receiver Jarvis Landry on a 5-yard touchdown pass to give the Dolphins a 28-17 lead at the 2:15 mark of the third quarter. The Broncos then reeled off 22 unanswered points. Early in the fourth quarter, Manning threw his third touchdown pass of the game to Demaryius Thomas &#8212; from 5 yards out &#8212; coupled with a two-point conversion pass to wide receiver Emmanuel Sanders. After forcing a Dolphins' punt, the Broncos grabbed their first lead of the game at the 5:05 mark of the fourth quarter, with running back C. J. Anderson rushing for a 10-yard touchdown. On the third play of the Dolphins' next possession, Tannehill was intercepted by Broncos' safety T. J. Ward, who advanced the football all the way to the Dolphins' 8-yard line. Two plays later, Manning threw a 2-yard touchdown pass to wide receiver Wes Welker to increase the Broncos' lead to 39-28 with 3:17 remaining in the fourth quarter. The Dolphins attempted a rally, with Tannehill connecting on a 1-yard touchdown pass to Landry, coupled with running back Lamar Miller rushing for a two-point conversion, with 1:37 remaining in the game. However, the Dolphins' onside kick attempt was unsuccessful, and the Broncos subsequently ran out the clock. Demaryius Thomas' streak of 100-yard receiving games ended at seven games, one shy of the NFL record that was set by Detroit Lions' wide receiver Calvin Johnson in 2012.\n\nQUESTION: What quarter did Manning throw his first touchdown pass in?"
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
        if frame_module == 'current_script_15':
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
            "iteration": 15,
            "sample_id": "4cbe93b3-effb-4007-9fd3-c31396fcad84",
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
        "current_script_15", 
        "scripts/current_script_15.py"
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
    question = "PASSAGE: Hoping to rebound from their loss to the Rams, the Broncos returned home for an AFC duel against the Miami Dolphins &#8212; the Broncos' first home game in exactly one month. The Dolphins grabbed the early lead, with running back Daniel Thomas rushing for a 3-yard touchdown. The Broncos subsequently got on the scoreboard, with a 38-yard field goal by placekicker Brandon McManus. The Dolphins added to their lead early in the second quarter, with quarterback Ryan Tannehill rushing for a 1-yard touchdown. The Broncos cut into the Dolphins' lead, with quarterback Peyton Manning connecting on a 5-yard touchdown pass to wide receiver Demaryius Thomas, however, the Dolphins responded, with Tannehill throwing a 10-yard touchdown pass to wide receiver Mike Wallace. Another touchdown pass from Manning to Demaryius Thomas &#8212; from 14 yards out &#8212; narrowed the Dolphins' lead to 21-17 just before halftime. The Broncos drove to as far as the Dolphins' 7-yard line on the initial possession of the second half, but after Manning was sacked by Dolphins' linebacker Jelani Jenkins for an 8-yard loss, McManus missed on a 33-yard field goal attempt. After the Broncos' defense forced a Dolphins' punt, return specialist Isaiah Burse fumbled deep in Broncos' territory, giving the football back to the Dolphins. Four plays later, Tannehill connected with wide receiver Jarvis Landry on a 5-yard touchdown pass to give the Dolphins a 28-17 lead at the 2:15 mark of the third quarter. The Broncos then reeled off 22 unanswered points. Early in the fourth quarter, Manning threw his third touchdown pass of the game to Demaryius Thomas &#8212; from 5 yards out &#8212; coupled with a two-point conversion pass to wide receiver Emmanuel Sanders. After forcing a Dolphins' punt, the Broncos grabbed their first lead of the game at the 5:05 mark of the fourth quarter, with running back C. J. Anderson rushing for a 10-yard touchdown. On the third play of the Dolphins' next possession, Tannehill was intercepted by Broncos' safety T. J. Ward, who advanced the football all the way to the Dolphins' 8-yard line. Two plays later, Manning threw a 2-yard touchdown pass to wide receiver Wes Welker to increase the Broncos' lead to 39-28 with 3:17 remaining in the fourth quarter. The Dolphins attempted a rally, with Tannehill connecting on a 1-yard touchdown pass to Landry, coupled with running back Lamar Miller rushing for a two-point conversion, with 1:37 remaining in the game. However, the Dolphins' onside kick attempt was unsuccessful, and the Broncos subsequently ran out the clock. Demaryius Thomas' streak of 100-yard receiving games ended at seven games, one shy of the NFL record that was set by Detroit Lions' wide receiver Calvin Johnson in 2012.\n\nQUESTION: What quarter did Manning throw his first touchdown pass in?"

    # Call the main function and get the answer
    answer = module.main(question)

    # Log execution completion
    with open(trace_file, 'a', encoding='utf-8') as f:
        end_entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "event": "execution_complete",
            "iteration": 15,
            "sample_id": "4cbe93b3-effb-4007-9fd3-c31396fcad84",
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
            "iteration": 15,
            "sample_id": "4cbe93b3-effb-4007-9fd3-c31396fcad84",
            "error": str(e),
            "traceback": traceback.format_exc()
        }
        f.write(json.dumps(error_entry) + "\n")

    print("ERROR_START")
    print(str(e))
    print(traceback.format_exc())
    print("ERROR_END")
