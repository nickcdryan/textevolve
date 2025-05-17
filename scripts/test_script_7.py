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
trace_file = "archive/trace_iteration_7.jsonl"
os.makedirs(os.path.dirname(trace_file), exist_ok=True)

# Trace entry for execution start
with open(trace_file, 'a', encoding='utf-8') as f:
    start_entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "event": "execution_start",
        "iteration": 7,
        "sample_id": "0f852a81-05ea-4b12-9442-5f04defb00c9",
        "question": "PASSAGE: After the first three possessions of the game ended in punts, the Patriots marched 62 yards to the Bills 14-yard line, but the drive ended with Gostkowski missing a 36-yard field goal. On the third play of the Bills resulting possession, though, Kyle Orton was intercepted by Jamie Collins at the Patriots 39-yard line. The Patriots marched 61 yards in just five plays to take the lead on a 1-yard touchdown pass from Brady to Tim Wright. After both teams punted, the Bills lugged 67 yards in just under 5 minutes to tie the game 7-7 on a Kyle Orton touchdown pass to Robert Woods. After a Patriots three-and-out, Orton was strip-sacked by Chandler Jones with Jones recovering at the Bills 24. The Patriots didn't gain a single yard, however, but Gostkowski was successful on a 42-yard field goal try. With 0:06 seconds left in the half, the Bills were attempting to run out the clock, but McCourty stripped C. J. Spiller with Zach Moore recovering at the Bills 42. Brady hit Edelman on a quick 7 yards pass and Gostkowski kicked a 53-yarder for a 13-7 lead at the half. After receiving the opening kickoff of the second half the Patriots reached the Bills 43 in just five plays before Brady launched a bomb to Brian Tyms for a 43-yard touchdown, increasing the New England lead to 20-7. The Bills struck right back with a 13 play, 80-yard drive in just under 7 minutes, culminating in Fred Jackson scoring on a 1-yard touchdown run. The Patriots marched 56 yards on their next possession with Gostkowski adding a 40-yard field goal to increase the lead to 23-14. After a Bills punt Brady led the Patriots down the field and, with just over 6 minutes remaining, found his favorite target, Gronkowski, for a 17-yard touchdown, but the play was nullified for an offensive holding penalty on Jordan Devey. This would prove to just be a delay, because Brady threw an 18-yard touchdown pass to LaFell two plays later, capping a 12 play, 80-yard drive. The Patriots now led 30-14 and looked to be well on their way to victory. However, the Bills wouldn't go away quietly. Kyle Orton calmly engineered an 8 play, 80-yard drive, aided by a 35-yard completion on 4th-and-2, that ended in his 8-yard touchdown pass to Chris Hogan, with a successful two-point conversion, trimming the deficit to one possession, 30-22. Starting at the Patriots 7, Brady converted a 3rd-and-16 with a 17-yard completion to Gronkowski, and a few plays later found LaFell on a medium pass, who turned upfield and raced down the sideline for a 56-yard touchdown, increasing the Patriots lead to 37-22. The Bills reached their own 42 on their final drive, but a sack by Rob Ninkovich and an incomplete pass intended for Scott Chandler on 4th-and-9 officially sealed the deal. Brady completed 27 of 37 passes for 361 yards, with 4 touchdowns and no interceptions. Kyle Orton was equally very impressive, finishing the game  24 of 38 for 299 yards, with 2 TDs and 1 INT. With 4 catches for 97 yards and 2 touchdowns, LaFell continued to be the Patriots awaited deep-threat at wide receiver. Both teams struggled to run the football. Stevan Ridley ran for only 23 yards on 10 carries, while Fred Jackson ran for only 26 yards on 10 carries. Unfortunately, it was later learned the Patriots lost Stevan Ridley and linebacker Jerod Mayo to season ending injuries.\n\nQUESTION: What yard line did both teams score from?"
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
        if frame_module == 'current_script_7':
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
            "iteration": 7,
            "sample_id": "0f852a81-05ea-4b12-9442-5f04defb00c9",
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
        "current_script_7", 
        "scripts/current_script_7.py"
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
    question = "PASSAGE: After the first three possessions of the game ended in punts, the Patriots marched 62 yards to the Bills 14-yard line, but the drive ended with Gostkowski missing a 36-yard field goal. On the third play of the Bills resulting possession, though, Kyle Orton was intercepted by Jamie Collins at the Patriots 39-yard line. The Patriots marched 61 yards in just five plays to take the lead on a 1-yard touchdown pass from Brady to Tim Wright. After both teams punted, the Bills lugged 67 yards in just under 5 minutes to tie the game 7-7 on a Kyle Orton touchdown pass to Robert Woods. After a Patriots three-and-out, Orton was strip-sacked by Chandler Jones with Jones recovering at the Bills 24. The Patriots didn't gain a single yard, however, but Gostkowski was successful on a 42-yard field goal try. With 0:06 seconds left in the half, the Bills were attempting to run out the clock, but McCourty stripped C. J. Spiller with Zach Moore recovering at the Bills 42. Brady hit Edelman on a quick 7 yards pass and Gostkowski kicked a 53-yarder for a 13-7 lead at the half. After receiving the opening kickoff of the second half the Patriots reached the Bills 43 in just five plays before Brady launched a bomb to Brian Tyms for a 43-yard touchdown, increasing the New England lead to 20-7. The Bills struck right back with a 13 play, 80-yard drive in just under 7 minutes, culminating in Fred Jackson scoring on a 1-yard touchdown run. The Patriots marched 56 yards on their next possession with Gostkowski adding a 40-yard field goal to increase the lead to 23-14. After a Bills punt Brady led the Patriots down the field and, with just over 6 minutes remaining, found his favorite target, Gronkowski, for a 17-yard touchdown, but the play was nullified for an offensive holding penalty on Jordan Devey. This would prove to just be a delay, because Brady threw an 18-yard touchdown pass to LaFell two plays later, capping a 12 play, 80-yard drive. The Patriots now led 30-14 and looked to be well on their way to victory. However, the Bills wouldn't go away quietly. Kyle Orton calmly engineered an 8 play, 80-yard drive, aided by a 35-yard completion on 4th-and-2, that ended in his 8-yard touchdown pass to Chris Hogan, with a successful two-point conversion, trimming the deficit to one possession, 30-22. Starting at the Patriots 7, Brady converted a 3rd-and-16 with a 17-yard completion to Gronkowski, and a few plays later found LaFell on a medium pass, who turned upfield and raced down the sideline for a 56-yard touchdown, increasing the Patriots lead to 37-22. The Bills reached their own 42 on their final drive, but a sack by Rob Ninkovich and an incomplete pass intended for Scott Chandler on 4th-and-9 officially sealed the deal. Brady completed 27 of 37 passes for 361 yards, with 4 touchdowns and no interceptions. Kyle Orton was equally very impressive, finishing the game  24 of 38 for 299 yards, with 2 TDs and 1 INT. With 4 catches for 97 yards and 2 touchdowns, LaFell continued to be the Patriots awaited deep-threat at wide receiver. Both teams struggled to run the football. Stevan Ridley ran for only 23 yards on 10 carries, while Fred Jackson ran for only 26 yards on 10 carries. Unfortunately, it was later learned the Patriots lost Stevan Ridley and linebacker Jerod Mayo to season ending injuries.\n\nQUESTION: What yard line did both teams score from?"

    # Call the main function and get the answer
    answer = module.main(question)

    # Log execution completion
    with open(trace_file, 'a', encoding='utf-8') as f:
        end_entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "event": "execution_complete",
            "iteration": 7,
            "sample_id": "0f852a81-05ea-4b12-9442-5f04defb00c9",
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
            "iteration": 7,
            "sample_id": "0f852a81-05ea-4b12-9442-5f04defb00c9",
            "error": str(e),
            "traceback": traceback.format_exc()
        }
        f.write(json.dumps(error_entry) + "\n")

    print("ERROR_START")
    print(str(e))
    print(traceback.format_exc())
    print("ERROR_END")
