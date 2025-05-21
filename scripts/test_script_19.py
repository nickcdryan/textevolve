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
trace_file = "archive/trace_iteration_19.jsonl"
os.makedirs(os.path.dirname(trace_file), exist_ok=True)

# Trace entry for execution start
with open(trace_file, 'a', encoding='utf-8') as f:
    start_entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "event": "execution_start",
        "iteration": 19,
        "sample_id": "5b8cccc1-f6ac-468d-87c0-e2f86f0b00e6",
        "question": "PASSAGE: Though Chad Pennington was active and in uniform for the Jets, the team erred on the side of caution due to injured right ankle, instead starting second-year backup Kellen Clemens, who was making his first career start. The Ravens' defense welcomed him rudely on his first drive with an interception by Ed Reed. The Ravens were able to attain good field position consistently throughout the first half, and quarterback Kyle Boller (who himself was starting in place of an injured starting quarterback, Steve McNair), capitalized with a two-yard touchdown to Willis McGahee late in the first quarter. The teams traded field goals to start the second quarter; Jets kicker Mike Nugent hit a 50-yard field goal, followed by Matt Stover hitting a 28-yard attempt for the Ravens. After Stover missed a 46-yard try, the Jets tried to respond with Nugent attempting a 52-yard field goal, but Nugent missed wide left, his first miss in twenty attempts dating back to last season. Boller once again took advantage of the short field provided and hit tight end Todd Heap on a four-yard touchdown with six seconds left in the half to extend the Ravens' lead to 17-3. Heap's catch was initially ruled incomplete, but the call was subjected to a booth review and reversed, as replays showed he was able to touch both feet within the end zone. After a quiet third quarter, Stover hit a 43-yard field goal to start the fourth quarter, and extended Baltimore's lead to seventeen. Baltimore's defense, which ranked as the best in the NFL in 2006, was able to shut down Clemens and the Jets for most of the game, but Clemens was able to rally the team in the fourth quarter. Using a no huddle offense, Clemens drove the team down to the Baltimore three-yard line, before the Jets settled for a 21-yard field goal. On the Jets' next possession, 44 and 24-yard strikes by Clemens to Jerricho Cotchery got the Jets to the Ravens' goal line, where he found tight end Chris Baker for a three-yard touchdown, cutting the Jets' deficit to seven. Though the Jets failed to convert the ensuing onside kick, poor clock management by Boller gave the Jets the ball back with 2:38 left in the game. Clemens immediately found Cotchery on a 50-yard catch-and-run, later followed by a 24-yard pass to Laveranues Coles that brought the Jets' to the Baltimore seven-yard line with just over a minute to go. Clemens passed to Justin McCareins for a potential touchdown, but the pass was dropped by McCareins. A second pass to McCareins in the end zone deflected off him and into the arms of Ravens linebacker Ray Lewis for the game-ending interception. The loss made the Jets 8-20 since 2002 in games not started by Chad Pennington.\n\nQUESTION: How many touchdowns did the Ravens score in the second quarter?"
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
        if frame_module == 'current_script_19':
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
            "iteration": 19,
            "sample_id": "5b8cccc1-f6ac-468d-87c0-e2f86f0b00e6",
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
        "current_script_19", 
        "scripts/current_script_19.py"
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
    question = "PASSAGE: Though Chad Pennington was active and in uniform for the Jets, the team erred on the side of caution due to injured right ankle, instead starting second-year backup Kellen Clemens, who was making his first career start. The Ravens' defense welcomed him rudely on his first drive with an interception by Ed Reed. The Ravens were able to attain good field position consistently throughout the first half, and quarterback Kyle Boller (who himself was starting in place of an injured starting quarterback, Steve McNair), capitalized with a two-yard touchdown to Willis McGahee late in the first quarter. The teams traded field goals to start the second quarter; Jets kicker Mike Nugent hit a 50-yard field goal, followed by Matt Stover hitting a 28-yard attempt for the Ravens. After Stover missed a 46-yard try, the Jets tried to respond with Nugent attempting a 52-yard field goal, but Nugent missed wide left, his first miss in twenty attempts dating back to last season. Boller once again took advantage of the short field provided and hit tight end Todd Heap on a four-yard touchdown with six seconds left in the half to extend the Ravens' lead to 17-3. Heap's catch was initially ruled incomplete, but the call was subjected to a booth review and reversed, as replays showed he was able to touch both feet within the end zone. After a quiet third quarter, Stover hit a 43-yard field goal to start the fourth quarter, and extended Baltimore's lead to seventeen. Baltimore's defense, which ranked as the best in the NFL in 2006, was able to shut down Clemens and the Jets for most of the game, but Clemens was able to rally the team in the fourth quarter. Using a no huddle offense, Clemens drove the team down to the Baltimore three-yard line, before the Jets settled for a 21-yard field goal. On the Jets' next possession, 44 and 24-yard strikes by Clemens to Jerricho Cotchery got the Jets to the Ravens' goal line, where he found tight end Chris Baker for a three-yard touchdown, cutting the Jets' deficit to seven. Though the Jets failed to convert the ensuing onside kick, poor clock management by Boller gave the Jets the ball back with 2:38 left in the game. Clemens immediately found Cotchery on a 50-yard catch-and-run, later followed by a 24-yard pass to Laveranues Coles that brought the Jets' to the Baltimore seven-yard line with just over a minute to go. Clemens passed to Justin McCareins for a potential touchdown, but the pass was dropped by McCareins. A second pass to McCareins in the end zone deflected off him and into the arms of Ravens linebacker Ray Lewis for the game-ending interception. The loss made the Jets 8-20 since 2002 in games not started by Chad Pennington.\n\nQUESTION: How many touchdowns did the Ravens score in the second quarter?"

    # Call the main function and get the answer
    answer = module.main(question)

    # Log execution completion
    with open(trace_file, 'a', encoding='utf-8') as f:
        end_entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "event": "execution_complete",
            "iteration": 19,
            "sample_id": "5b8cccc1-f6ac-468d-87c0-e2f86f0b00e6",
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
            "iteration": 19,
            "sample_id": "5b8cccc1-f6ac-468d-87c0-e2f86f0b00e6",
            "error": str(e),
            "traceback": traceback.format_exc()
        }
        f.write(json.dumps(error_entry) + "\n")

    print("ERROR_START")
    print(str(e))
    print(traceback.format_exc())
    print("ERROR_END")
