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
trace_file = "archive/trace_iteration_17.jsonl"
os.makedirs(os.path.dirname(trace_file), exist_ok=True)

# Trace entry for execution start
with open(trace_file, 'a', encoding='utf-8') as f:
    start_entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "event": "execution_start",
        "iteration": 17,
        "sample_id": "3c825795-6584-4a05-8534-7db17776c0ab",
        "question": "PASSAGE: Coming off their win over the Redskins, the Dolphins stayed at home for a Week 11 AFC East showdown with the Buffalo Bills.  Miami trailed early in the first quarter as Bills kicker Dave Rayner made a 30-yard field goal, yet the Dolphins immediately answered with quarterback Matt Moore finding tight end Anthony Fasano on a 1-yard touchdown pass, followed by a 5-yard touchdown run from running back Reggie Bush.  Miami would add onto their lead in the second quarter with Moore connecting with fullback Charles Clay on a 12-yard touchdown pass, followed by wide receiver Davone Bess on a 4-yard touchdown pass.  Buffalo would close out the half with Rayner booting a 56-yard field goal. The Dolphins continued their dominating day in the third quarter with running back Lex Hilliard recovering a blocked punt in the endzone for a touchdown (the team's first TD off of a blocked punt since 1990).  The Bills tried to rally as linebacker Kelvin Sheppard tackled rookie running back Daniel Thomas in the endzone for a safety, but Miami's lead proved to be too much to overcome. This game also marked the first time since 1990 that Miami allowed 10 points or fewer in three-straight games.They have now become the 3rd team in NFL History to win 3 straight after starting season at least 0-7 ('78 Cardinals & '86 Colts).\n\nQUESTION: Which player had the shortest touchdown scoring play?"
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
        if frame_module == 'current_script_17':
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
            "iteration": 17,
            "sample_id": "3c825795-6584-4a05-8534-7db17776c0ab",
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
        "current_script_17", 
        "scripts/current_script_17.py"
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
    question = "PASSAGE: Coming off their win over the Redskins, the Dolphins stayed at home for a Week 11 AFC East showdown with the Buffalo Bills.  Miami trailed early in the first quarter as Bills kicker Dave Rayner made a 30-yard field goal, yet the Dolphins immediately answered with quarterback Matt Moore finding tight end Anthony Fasano on a 1-yard touchdown pass, followed by a 5-yard touchdown run from running back Reggie Bush.  Miami would add onto their lead in the second quarter with Moore connecting with fullback Charles Clay on a 12-yard touchdown pass, followed by wide receiver Davone Bess on a 4-yard touchdown pass.  Buffalo would close out the half with Rayner booting a 56-yard field goal. The Dolphins continued their dominating day in the third quarter with running back Lex Hilliard recovering a blocked punt in the endzone for a touchdown (the team's first TD off of a blocked punt since 1990).  The Bills tried to rally as linebacker Kelvin Sheppard tackled rookie running back Daniel Thomas in the endzone for a safety, but Miami's lead proved to be too much to overcome. This game also marked the first time since 1990 that Miami allowed 10 points or fewer in three-straight games.They have now become the 3rd team in NFL History to win 3 straight after starting season at least 0-7 ('78 Cardinals & '86 Colts).\n\nQUESTION: Which player had the shortest touchdown scoring play?"

    # Call the main function and get the answer
    answer = module.main(question)

    # Log execution completion
    with open(trace_file, 'a', encoding='utf-8') as f:
        end_entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "event": "execution_complete",
            "iteration": 17,
            "sample_id": "3c825795-6584-4a05-8534-7db17776c0ab",
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
            "iteration": 17,
            "sample_id": "3c825795-6584-4a05-8534-7db17776c0ab",
            "error": str(e),
            "traceback": traceback.format_exc()
        }
        f.write(json.dumps(error_entry) + "\n")

    print("ERROR_START")
    print(str(e))
    print(traceback.format_exc())
    print("ERROR_END")
