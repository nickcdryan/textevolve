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
trace_file = "archive/trace_iteration_18.jsonl"
os.makedirs(os.path.dirname(trace_file), exist_ok=True)

# Trace entry for execution start
with open(trace_file, 'a', encoding='utf-8') as f:
    start_entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "event": "execution_start",
        "iteration": 18,
        "sample_id": "example_135",
        "question": 'PASSAGE: Cincinnati scored on its first possession, when Andy Dalton threw to A. J. Green for an 82-yard touchdown. The Lions tied it later in the first quarter on a 3-yard TD pass from Matthew Stafford to Brandon Pettigrew. A 36-yard field goal by David Akers gave the Lions a 10-7 second quarter lead.  But late in the quarter, Akers had a 34-yard field goal attempt blocked by Carlos Dunlap, which the Bengals returned all the way to the Lions 40 yard line despite fumbling during the return. That set up a 12-yard TD strike from Andy Dalton to Marvin Jones just before the first half closed, giving Cincinnati a 14-10 lead. The teams exchanged TD passes in the third quarter. First, Dalton hit Tyler Eifert for a 32-yard TD, and Stafford followed shortly after with a 27-yard TD toss to Calvin Johnson. Mike Nugent connected on a 48-yard field goal late in the third to put the Bengals up 24-17.  The Lions tied the game at 24 in the fourth quarter, when Calvin Johnson leaped up and beat three Bengals defenders in the end zone on a 50-yard pass from Matthew Stafford. After the game, Stafford called Johnson\'s play "one of the best catches I have ever seen." Late in the fourth quarter, a punt by the Bengals Kevin Huber pinned the Lions at their own 6 yard line. Detroit attempted to kill enough clock to get the game to overtime, but could only gain one first down and 17 yards. Detroit punter Sam Martin then shanked a punt that netted only 28 yards before going out of bounds at the Cincinnati 49 with 26 seconds left in the game. Three plays and 15 yards later, Mike Nugent boomed a 54-yard field goal as time expired to give the Bengals a 27-24 victory. The aerial attack for both teams produced big numbers.  Andy Dalton was 24-of-34 for 372 yards and 3 touchdowns, while Matthew Stafford was 28-of-51 for 357 yards and 3 scores.  A. J. Green of the Bengals and Calvin Johnson of the Lions both tallied 155 yards receiving on the day.\n\nQUESTION: Which quarterback had more incomplete passes?'
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
        if frame_module == 'current_script_18':
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
            "iteration": 18,
            "sample_id": "example_135",
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
        "current_script_18", 
        "scripts/current_script_18.py"
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
    question = 'PASSAGE: Cincinnati scored on its first possession, when Andy Dalton threw to A. J. Green for an 82-yard touchdown. The Lions tied it later in the first quarter on a 3-yard TD pass from Matthew Stafford to Brandon Pettigrew. A 36-yard field goal by David Akers gave the Lions a 10-7 second quarter lead.  But late in the quarter, Akers had a 34-yard field goal attempt blocked by Carlos Dunlap, which the Bengals returned all the way to the Lions 40 yard line despite fumbling during the return. That set up a 12-yard TD strike from Andy Dalton to Marvin Jones just before the first half closed, giving Cincinnati a 14-10 lead. The teams exchanged TD passes in the third quarter. First, Dalton hit Tyler Eifert for a 32-yard TD, and Stafford followed shortly after with a 27-yard TD toss to Calvin Johnson. Mike Nugent connected on a 48-yard field goal late in the third to put the Bengals up 24-17.  The Lions tied the game at 24 in the fourth quarter, when Calvin Johnson leaped up and beat three Bengals defenders in the end zone on a 50-yard pass from Matthew Stafford. After the game, Stafford called Johnson\'s play "one of the best catches I have ever seen." Late in the fourth quarter, a punt by the Bengals Kevin Huber pinned the Lions at their own 6 yard line. Detroit attempted to kill enough clock to get the game to overtime, but could only gain one first down and 17 yards. Detroit punter Sam Martin then shanked a punt that netted only 28 yards before going out of bounds at the Cincinnati 49 with 26 seconds left in the game. Three plays and 15 yards later, Mike Nugent boomed a 54-yard field goal as time expired to give the Bengals a 27-24 victory. The aerial attack for both teams produced big numbers.  Andy Dalton was 24-of-34 for 372 yards and 3 touchdowns, while Matthew Stafford was 28-of-51 for 357 yards and 3 scores.  A. J. Green of the Bengals and Calvin Johnson of the Lions both tallied 155 yards receiving on the day.\n\nQUESTION: Which quarterback had more incomplete passes?'

    # Call the main function and get the answer
    answer = module.main(question)

    # Log execution completion
    with open(trace_file, 'a', encoding='utf-8') as f:
        end_entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "event": "execution_complete",
            "iteration": 18,
            "sample_id": "example_135",
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
            "iteration": 18,
            "sample_id": "example_135",
            "error": str(e),
            "traceback": traceback.format_exc()
        }
        f.write(json.dumps(error_entry) + "\n")

    print("ERROR_START")
    print(str(e))
    print(traceback.format_exc())
    print("ERROR_END")
