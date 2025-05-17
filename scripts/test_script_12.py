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
trace_file = "archive/trace_iteration_12.jsonl"
os.makedirs(os.path.dirname(trace_file), exist_ok=True)

# Trace entry for execution start
with open(trace_file, 'a', encoding='utf-8') as f:
    start_entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "event": "execution_start",
        "iteration": 12,
        "sample_id": "1f777fdb-92f3-4f84-a846-d3f32b4d6c5a",
        "question": "PASSAGE: The Browns traveled to Baltimore to take on the Ravens on Thursday Night Football. After a scoreless first quarter, the Ravens scored first in the second quarter as Joe Flacco found Torrey Smith on an 18-yard touchdown pass (with a failed PAT) for a 6-0 lead. The team increased their lead as Justin Tucker made a 45-yard field goal to make the score 9-0. Finally, the Browns scored not long before halftime when Trent Richardson ran for a 2-yard touchdown, making the halftime score 9-7. After the break, the Ravens went right back to work as Flacco used a QB sneak 1-yard run for a 16-7 lead. However, The Browns drew within 6 points as Phil Dawson nailed a 51-yard field goal, making the score 16-10. But then, the Ravens pulled away as Cary Williams picked off Weeden and returned the ball 63 yards for a touchdown, making the score 23-10. In the fourth quarter, the Browns tried to come back as Dawson nailed two field goals from 50 and 52 yards out making the score 23-13 and then 23-16, respectively. However, the Ravens took control of the game and the Browns' record dropped to 0-4 on the season, losing their 13th straight game against a division rival and their 9th straight game against the Ravens.\n\nQUESTION: How many field goals did Phil Dawson have from over 49 yards?"
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
        if frame_module == 'current_script_12':
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
            "iteration": 12,
            "sample_id": "1f777fdb-92f3-4f84-a846-d3f32b4d6c5a",
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
        "current_script_12", 
        "scripts/current_script_12.py"
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
    question = "PASSAGE: The Browns traveled to Baltimore to take on the Ravens on Thursday Night Football. After a scoreless first quarter, the Ravens scored first in the second quarter as Joe Flacco found Torrey Smith on an 18-yard touchdown pass (with a failed PAT) for a 6-0 lead. The team increased their lead as Justin Tucker made a 45-yard field goal to make the score 9-0. Finally, the Browns scored not long before halftime when Trent Richardson ran for a 2-yard touchdown, making the halftime score 9-7. After the break, the Ravens went right back to work as Flacco used a QB sneak 1-yard run for a 16-7 lead. However, The Browns drew within 6 points as Phil Dawson nailed a 51-yard field goal, making the score 16-10. But then, the Ravens pulled away as Cary Williams picked off Weeden and returned the ball 63 yards for a touchdown, making the score 23-10. In the fourth quarter, the Browns tried to come back as Dawson nailed two field goals from 50 and 52 yards out making the score 23-13 and then 23-16, respectively. However, the Ravens took control of the game and the Browns' record dropped to 0-4 on the season, losing their 13th straight game against a division rival and their 9th straight game against the Ravens.\n\nQUESTION: How many field goals did Phil Dawson have from over 49 yards?"

    # Call the main function and get the answer
    answer = module.main(question)

    # Log execution completion
    with open(trace_file, 'a', encoding='utf-8') as f:
        end_entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "event": "execution_complete",
            "iteration": 12,
            "sample_id": "1f777fdb-92f3-4f84-a846-d3f32b4d6c5a",
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
            "iteration": 12,
            "sample_id": "1f777fdb-92f3-4f84-a846-d3f32b4d6c5a",
            "error": str(e),
            "traceback": traceback.format_exc()
        }
        f.write(json.dumps(error_entry) + "\n")

    print("ERROR_START")
    print(str(e))
    print(traceback.format_exc())
    print("ERROR_END")
