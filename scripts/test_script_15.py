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
        "sample_id": "c483f44a-3d89-4237-9e5a-6d5d6a32f432",
        "question": "PASSAGE: Coming off their road win over the Raiders, the Chargers would play their Week 2 home opener against the Baltimore Ravens.  San Diego would deliver the opening strike in the first quarter as quarterback Philip Rivers completed an 81-yard touchdown pass to running back Darren Sproles.  The Ravens would respond with running back Willis McGahee getting a 5-yard touchdown run, yet the Chargers would regain the lead as kicker Nate Kaeding got a 29-yard field goal.  Baltimore would take the lead in the second quarter as McGahee got a 3-yard touchdown run.  San Diego would pull within one as Kaeding made a 22-yard field goal, but the Ravens answered with quarterback Joe Flacco completing a 27-yard touchdown pass to wide receiver Kelley Washington.  The Chargers would end the half as Kaeding would make a 23-yard field goal. In the third quarter, Baltimore would add onto their lead as Flacco completed a 9-yard touchdown pass to tight end Todd Heap.  San Diego would stay close as Rivers completed a 35-yard touchdown pass to wide receiver Vincent Jackson.  In the fourth quarter, the Chargers got closer as Kaeding kicked a 25-yard field goal, but the Ravens would answer with kicker Steve Hauschka nailing a 33-yard field goal.  San Diego would manage to get a late drive all the way to the Ravens' 15-yard line, but on 4th-&-2, Sproles was tackled behind the line of scrimmage by an unblocked Ray Lewis, ending any hope of a comeback.\n\nQUESTION: How many more yards of touchdown passes were there compared to touchdown runs?"
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
            "sample_id": "c483f44a-3d89-4237-9e5a-6d5d6a32f432",
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
    question = "PASSAGE: Coming off their road win over the Raiders, the Chargers would play their Week 2 home opener against the Baltimore Ravens.  San Diego would deliver the opening strike in the first quarter as quarterback Philip Rivers completed an 81-yard touchdown pass to running back Darren Sproles.  The Ravens would respond with running back Willis McGahee getting a 5-yard touchdown run, yet the Chargers would regain the lead as kicker Nate Kaeding got a 29-yard field goal.  Baltimore would take the lead in the second quarter as McGahee got a 3-yard touchdown run.  San Diego would pull within one as Kaeding made a 22-yard field goal, but the Ravens answered with quarterback Joe Flacco completing a 27-yard touchdown pass to wide receiver Kelley Washington.  The Chargers would end the half as Kaeding would make a 23-yard field goal. In the third quarter, Baltimore would add onto their lead as Flacco completed a 9-yard touchdown pass to tight end Todd Heap.  San Diego would stay close as Rivers completed a 35-yard touchdown pass to wide receiver Vincent Jackson.  In the fourth quarter, the Chargers got closer as Kaeding kicked a 25-yard field goal, but the Ravens would answer with kicker Steve Hauschka nailing a 33-yard field goal.  San Diego would manage to get a late drive all the way to the Ravens' 15-yard line, but on 4th-&-2, Sproles was tackled behind the line of scrimmage by an unblocked Ray Lewis, ending any hope of a comeback.\n\nQUESTION: How many more yards of touchdown passes were there compared to touchdown runs?"

    # Call the main function and get the answer
    answer = module.main(question)

    # Log execution completion
    with open(trace_file, 'a', encoding='utf-8') as f:
        end_entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "event": "execution_complete",
            "iteration": 15,
            "sample_id": "c483f44a-3d89-4237-9e5a-6d5d6a32f432",
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
            "sample_id": "c483f44a-3d89-4237-9e5a-6d5d6a32f432",
            "error": str(e),
            "traceback": traceback.format_exc()
        }
        f.write(json.dumps(error_entry) + "\n")

    print("ERROR_START")
    print(str(e))
    print(traceback.format_exc())
    print("ERROR_END")
