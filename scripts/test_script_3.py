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
trace_file = "archive/trace_iteration_3.jsonl"
os.makedirs(os.path.dirname(trace_file), exist_ok=True)

# Trace entry for execution start
with open(trace_file, 'a', encoding='utf-8') as f:
    start_entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "event": "execution_start",
        "iteration": 3,
        "sample_id": "371adf56-f493-41da-936e-dcdeceba6898",
        "question": "PASSAGE: Hoping to rebound from their loss to the Raiders, the Broncos traveled to Heinz Field to face the Pittsburgh Steelers. On the Broncos' first offensive possession, running back Ronnie Hillman committed a fumble deep in their own territory, and the Steelers capitalized two plays later, with a 2-yard touchdown run by running back DeAngelo Williams. The Broncos reeled off 20 unanswered points, with quarterback Brock Osweiler throwing a pair of touchdown passes - an 18-yarder to wide receiver Demaryius Thomas and a 61-yarder to wide receiver Emmanuel Sanders, followed in the second quarter by Osweiler scrambling for a 7-yard touchdown. The latter score came after the Broncos' defense intercepted Steelers' quarterback Ben Roethlisberger, and also had a missed extra-point attempt by placekicker Brandon McManus. The Steelers then marched down the field, but had to settle for a 24-yard field goal by placekicker Chris Boswell. The Broncos then increased their lead to 27-10, with Osweiler connecting with Thomas on a 6-yard touchdown pass at the 2-minute warning. However, it would be the Broncos' final scoring play of the game, as the offense was shut out in the second half for the third consecutive week. The Steelers pulled to within 27-13, with Boswell kicking a 41-yard field goal just before halftime. Midway through the third quarter, Roethlisberger connected with wide receiver Antonio Brown on a 9-yard touchdown. Later in the third quarter, Broncos' return specialist Jordan Norwood returned a punt 71 yards for a touchdown; however, the touchdown was nullified by an illegal substitution penalty. Following a Broncos' three-and-out, the Steelers tied the game early in the fourth quarter, with Roethlisberger throwing a 9-yard touchdown pass to wide receiver Markus Wheaton. Each team proceeded to trade punt on their next two possessions, and with just over five minutes remaining, Osweiler was intercepted by Steelers' linebacker Ryan Shazier at the Broncos' 37-yard line, and three plays later, the Steelers re-claimed the lead, with a 23-yard touchdown pass from Roethlisberger to Brown. The Broncos then marched down to the Steelers' 36-yard line with 2:17 remaining, but turned the football over on downs. Just before the two-minute warning, the Steelers were attempting to run out the clock, however, Roethlisberger chose to pass the football, and was intercepted by linebacker Brandon Marshall just before the two-minute warning, giving the Broncos one last possession. However, Osweiler threw four straight incompletions, and the Steelers ran out the clock.\n\nQUESTION: Who scored more points, the Steelers or the Broncos?"
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
        if frame_module == 'current_script_3':
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
            "iteration": 3,
            "sample_id": "371adf56-f493-41da-936e-dcdeceba6898",
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
        "current_script_3", 
        "scripts/current_script_3.py"
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
    question = "PASSAGE: Hoping to rebound from their loss to the Raiders, the Broncos traveled to Heinz Field to face the Pittsburgh Steelers. On the Broncos' first offensive possession, running back Ronnie Hillman committed a fumble deep in their own territory, and the Steelers capitalized two plays later, with a 2-yard touchdown run by running back DeAngelo Williams. The Broncos reeled off 20 unanswered points, with quarterback Brock Osweiler throwing a pair of touchdown passes - an 18-yarder to wide receiver Demaryius Thomas and a 61-yarder to wide receiver Emmanuel Sanders, followed in the second quarter by Osweiler scrambling for a 7-yard touchdown. The latter score came after the Broncos' defense intercepted Steelers' quarterback Ben Roethlisberger, and also had a missed extra-point attempt by placekicker Brandon McManus. The Steelers then marched down the field, but had to settle for a 24-yard field goal by placekicker Chris Boswell. The Broncos then increased their lead to 27-10, with Osweiler connecting with Thomas on a 6-yard touchdown pass at the 2-minute warning. However, it would be the Broncos' final scoring play of the game, as the offense was shut out in the second half for the third consecutive week. The Steelers pulled to within 27-13, with Boswell kicking a 41-yard field goal just before halftime. Midway through the third quarter, Roethlisberger connected with wide receiver Antonio Brown on a 9-yard touchdown. Later in the third quarter, Broncos' return specialist Jordan Norwood returned a punt 71 yards for a touchdown; however, the touchdown was nullified by an illegal substitution penalty. Following a Broncos' three-and-out, the Steelers tied the game early in the fourth quarter, with Roethlisberger throwing a 9-yard touchdown pass to wide receiver Markus Wheaton. Each team proceeded to trade punt on their next two possessions, and with just over five minutes remaining, Osweiler was intercepted by Steelers' linebacker Ryan Shazier at the Broncos' 37-yard line, and three plays later, the Steelers re-claimed the lead, with a 23-yard touchdown pass from Roethlisberger to Brown. The Broncos then marched down to the Steelers' 36-yard line with 2:17 remaining, but turned the football over on downs. Just before the two-minute warning, the Steelers were attempting to run out the clock, however, Roethlisberger chose to pass the football, and was intercepted by linebacker Brandon Marshall just before the two-minute warning, giving the Broncos one last possession. However, Osweiler threw four straight incompletions, and the Steelers ran out the clock.\n\nQUESTION: Who scored more points, the Steelers or the Broncos?"

    # Call the main function and get the answer
    answer = module.main(question)

    # Log execution completion
    with open(trace_file, 'a', encoding='utf-8') as f:
        end_entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "event": "execution_complete",
            "iteration": 3,
            "sample_id": "371adf56-f493-41da-936e-dcdeceba6898",
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
            "iteration": 3,
            "sample_id": "371adf56-f493-41da-936e-dcdeceba6898",
            "error": str(e),
            "traceback": traceback.format_exc()
        }
        f.write(json.dumps(error_entry) + "\n")

    print("ERROR_START")
    print(str(e))
    print(traceback.format_exc())
    print("ERROR_END")
