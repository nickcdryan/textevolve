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
trace_file = "archive/trace_iteration_9.jsonl"
os.makedirs(os.path.dirname(trace_file), exist_ok=True)

# Trace entry for execution start
with open(trace_file, 'a', encoding='utf-8') as f:
    start_entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "event": "execution_start",
        "iteration": 9,
        "sample_id": "951dc841-a092-488f-b4be-6ce65c33c6b9",
        "question": "PASSAGE: Coming off their win against Denver, the Patriots flew to Seattle to play their only game on the west coast of the season. After a Patriots punt, the Seahawks drove all the way to the Patriots 9, but ended up forcing to settle for a 34-yard field goal by Steven Hauschka. The Patriots responded, racing 82 yards in 6 plays, scoring on a 46-yard touchdown bomb from Brady to Welker. Seattle countered with a touchdown of their own, racing 85 yards in 7 plays, scoring on a 24-yard touchdown pass from rookie Russell Wilson to Doug Baldwin, taking a 10-7 lead. The Patriots countered again, going 80 yards in 15 plays, scoring on a yard touchdown pass to Hernandez, retaking the lead 14-10.  Seattle drove to the Patriots 48 on their next drive, but Chandler Jones strip-sacked Wilson with Ninkovich recovering at the Seahawks 47. After starting the first half with a 17-10 lead, the Patriots would only score two field goals in the second half. The Patriots drove to the Seahawks 6, but settled for a 24-yard field goal by Gostkowski, increasing their lead to 17-10. The Seahawks turned the ball over at their own 38 with less than a minute remaining in the half, and the Patriots drove to the Seahawks 3, but an intentional grounding penalty cost the Patriots points, sending the game to the half 17-10. After a Seahawks punt, the Patriots drove to the Seahawks 17, and Gostkowski increased the lead to 20-10 on a 35-yard field goal by Gostkowski. After a Seattle punt, the Patriots drove to the Seahawks 43, but Brady was intercepted by Richard Sherman at the Seahawks 20. After a Seahawks three-and-out, the Patriots drove to the Seahawks 6, but Brady was intercepted again at the 3, this time by Earl Thomas who returned it 20 yards to the Seahawks 23. On the third play of the Seahawks' next drive, Wilson hit Zach Miller for a 7-yard gain, but fumbled with Mayo recovering at the Patriots 30. The Patriots drove to the Seattle 17, settling for a 35-yard field goal by Gostkowski, extending the Patriots lead to 23-10 midway through the fourth quarter. On the first play of the Seahawks' next drive, Wilson hit Golden Tate for a 51-yard gain, with a 15-yard unnecessary roughness penalty on Spikes, moving the ball to the Patriots 17. Four plays later, Wilson threw a 10-yard touchdown pass to Braylon Edwards with 7:21 remaining, trimming the deficit to 23-17. Later in the fourth, after forcing the Patriots to go three and out, Leon Washington got the ball to the Seahawks 43 with a 25-yard return. On the fourth play of the Seahawks drive, Wilson hit Sidney Rice for a 46-yard touchdown bomb, giving the Seahawks a 24-23 lead with 1:18 remaining. On the second play of the Patriots next drive, Jason Jones sacked Brady for a 7-yard loss. Two plays later on 4th down, Brady completed a pass to Welker for 15 yards, but it was 2 yards short of a first down, giving the Seahawks the ball. Wilson kneed twice and the Seahawks got the surprising win. With the loss, the Patriots fell to 3-3.  The loss also left them 0-2 against the NFC West.\n\nQUESTION: Which team won the game?"
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
        if frame_module == 'current_script_9':
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
            "iteration": 9,
            "sample_id": "951dc841-a092-488f-b4be-6ce65c33c6b9",
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
        "current_script_9", 
        "scripts/current_script_9.py"
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
    question = "PASSAGE: Coming off their win against Denver, the Patriots flew to Seattle to play their only game on the west coast of the season. After a Patriots punt, the Seahawks drove all the way to the Patriots 9, but ended up forcing to settle for a 34-yard field goal by Steven Hauschka. The Patriots responded, racing 82 yards in 6 plays, scoring on a 46-yard touchdown bomb from Brady to Welker. Seattle countered with a touchdown of their own, racing 85 yards in 7 plays, scoring on a 24-yard touchdown pass from rookie Russell Wilson to Doug Baldwin, taking a 10-7 lead. The Patriots countered again, going 80 yards in 15 plays, scoring on a yard touchdown pass to Hernandez, retaking the lead 14-10.  Seattle drove to the Patriots 48 on their next drive, but Chandler Jones strip-sacked Wilson with Ninkovich recovering at the Seahawks 47. After starting the first half with a 17-10 lead, the Patriots would only score two field goals in the second half. The Patriots drove to the Seahawks 6, but settled for a 24-yard field goal by Gostkowski, increasing their lead to 17-10. The Seahawks turned the ball over at their own 38 with less than a minute remaining in the half, and the Patriots drove to the Seahawks 3, but an intentional grounding penalty cost the Patriots points, sending the game to the half 17-10. After a Seahawks punt, the Patriots drove to the Seahawks 17, and Gostkowski increased the lead to 20-10 on a 35-yard field goal by Gostkowski. After a Seattle punt, the Patriots drove to the Seahawks 43, but Brady was intercepted by Richard Sherman at the Seahawks 20. After a Seahawks three-and-out, the Patriots drove to the Seahawks 6, but Brady was intercepted again at the 3, this time by Earl Thomas who returned it 20 yards to the Seahawks 23. On the third play of the Seahawks' next drive, Wilson hit Zach Miller for a 7-yard gain, but fumbled with Mayo recovering at the Patriots 30. The Patriots drove to the Seattle 17, settling for a 35-yard field goal by Gostkowski, extending the Patriots lead to 23-10 midway through the fourth quarter. On the first play of the Seahawks' next drive, Wilson hit Golden Tate for a 51-yard gain, with a 15-yard unnecessary roughness penalty on Spikes, moving the ball to the Patriots 17. Four plays later, Wilson threw a 10-yard touchdown pass to Braylon Edwards with 7:21 remaining, trimming the deficit to 23-17. Later in the fourth, after forcing the Patriots to go three and out, Leon Washington got the ball to the Seahawks 43 with a 25-yard return. On the fourth play of the Seahawks drive, Wilson hit Sidney Rice for a 46-yard touchdown bomb, giving the Seahawks a 24-23 lead with 1:18 remaining. On the second play of the Patriots next drive, Jason Jones sacked Brady for a 7-yard loss. Two plays later on 4th down, Brady completed a pass to Welker for 15 yards, but it was 2 yards short of a first down, giving the Seahawks the ball. Wilson kneed twice and the Seahawks got the surprising win. With the loss, the Patriots fell to 3-3.  The loss also left them 0-2 against the NFC West.\n\nQUESTION: Which team won the game?"

    # Call the main function and get the answer
    answer = module.main(question)

    # Log execution completion
    with open(trace_file, 'a', encoding='utf-8') as f:
        end_entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "event": "execution_complete",
            "iteration": 9,
            "sample_id": "951dc841-a092-488f-b4be-6ce65c33c6b9",
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
            "iteration": 9,
            "sample_id": "951dc841-a092-488f-b4be-6ce65c33c6b9",
            "error": str(e),
            "traceback": traceback.format_exc()
        }
        f.write(json.dumps(error_entry) + "\n")

    print("ERROR_START")
    print(str(e))
    print(traceback.format_exc())
    print("ERROR_END")
