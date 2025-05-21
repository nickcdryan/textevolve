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
trace_file = "archive/trace_iteration_24.jsonl"
os.makedirs(os.path.dirname(trace_file), exist_ok=True)

# Trace entry for execution start
with open(trace_file, 'a', encoding='utf-8') as f:
    start_entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "event": "execution_start",
        "iteration": 24,
        "sample_id": "6ccd57c6-0108-451a-ad0f-186c3979aa14",
        "question": "PASSAGE: The Bills' WR Lee Evans and QB J. P. Losman were unstoppable in the first quarter, connecting for 205 yards and two 83-yard touchdowns. Evans' first-quarter performance was a Buffalo record for a period and just shy of the NFL record for yards receiving in a quarter of 210 by Baltimore's Qadry Ismail in 1999. The first touchdown came after Nate Clements' interception on the third play of the game. Both were after Evans got in front of Texans cornerback Demarcus Faggins for the easy score. The 83-yard TDs were career highs for both Losman and Evans and marked the first time in franchise history the Bills have had two 80-yard passes in a single game. David Carr opened 1 for 3 with an interception on his first pass before completing his next 22 throws. Carr finished 25 of 30 for 223 yards and no touchdowns. Carr tied the record held by Mark Brunell, who had 22 consecutive completions in Washington's 31-15 win over the Texans at Reliant Stadium on September 24. He tied Brunell's record on a short pass to Wali Lundy for no gain with 6:19 left. The streak was broken when his pass to Andre Johnson with 5:44 remaining fell short. Lundy cut the lead to 14-7 with a 17-yard run in the first quarter. That score was set up by a 17-yard reception by Eric Moulds on third-and-8. Samkon Gado made it 17-14 on a 1-yard run in the second quarter. Moulds also had a key third-down reception on that drive. Dunta Robinson gave the Texans the lead on a 9-yard interception return midway through the third quarter. Losman was throwing out of the end zone when Robinson intercepted the pass intended for Evans to score his first career touchdown and the Texans' first defensive touchdown since 2004. But after the offense failed to put the game away in the fourth quarter the Texans gave the Bills just enough time to beat them. Losman hit a diving Peerless Price in the back of the end zone for the 15-yard touchdown with 13 seconds left, giving Buffalo the 24-21 win.\n\nQUESTION: What was the final score of the game?"
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
        if frame_module == 'current_script_24':
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
            "iteration": 24,
            "sample_id": "6ccd57c6-0108-451a-ad0f-186c3979aa14",
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
        "current_script_24", 
        "scripts/current_script_24.py"
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
    question = "PASSAGE: The Bills' WR Lee Evans and QB J. P. Losman were unstoppable in the first quarter, connecting for 205 yards and two 83-yard touchdowns. Evans' first-quarter performance was a Buffalo record for a period and just shy of the NFL record for yards receiving in a quarter of 210 by Baltimore's Qadry Ismail in 1999. The first touchdown came after Nate Clements' interception on the third play of the game. Both were after Evans got in front of Texans cornerback Demarcus Faggins for the easy score. The 83-yard TDs were career highs for both Losman and Evans and marked the first time in franchise history the Bills have had two 80-yard passes in a single game. David Carr opened 1 for 3 with an interception on his first pass before completing his next 22 throws. Carr finished 25 of 30 for 223 yards and no touchdowns. Carr tied the record held by Mark Brunell, who had 22 consecutive completions in Washington's 31-15 win over the Texans at Reliant Stadium on September 24. He tied Brunell's record on a short pass to Wali Lundy for no gain with 6:19 left. The streak was broken when his pass to Andre Johnson with 5:44 remaining fell short. Lundy cut the lead to 14-7 with a 17-yard run in the first quarter. That score was set up by a 17-yard reception by Eric Moulds on third-and-8. Samkon Gado made it 17-14 on a 1-yard run in the second quarter. Moulds also had a key third-down reception on that drive. Dunta Robinson gave the Texans the lead on a 9-yard interception return midway through the third quarter. Losman was throwing out of the end zone when Robinson intercepted the pass intended for Evans to score his first career touchdown and the Texans' first defensive touchdown since 2004. But after the offense failed to put the game away in the fourth quarter the Texans gave the Bills just enough time to beat them. Losman hit a diving Peerless Price in the back of the end zone for the 15-yard touchdown with 13 seconds left, giving Buffalo the 24-21 win.\n\nQUESTION: What was the final score of the game?"

    # Call the main function and get the answer
    answer = module.main(question)

    # Log execution completion
    with open(trace_file, 'a', encoding='utf-8') as f:
        end_entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "event": "execution_complete",
            "iteration": 24,
            "sample_id": "6ccd57c6-0108-451a-ad0f-186c3979aa14",
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
            "iteration": 24,
            "sample_id": "6ccd57c6-0108-451a-ad0f-186c3979aa14",
            "error": str(e),
            "traceback": traceback.format_exc()
        }
        f.write(json.dumps(error_entry) + "\n")

    print("ERROR_START")
    print(str(e))
    print(traceback.format_exc())
    print("ERROR_END")
