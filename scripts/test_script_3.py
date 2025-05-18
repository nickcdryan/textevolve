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
        "sample_id": "2436207d-a6bf-44f8-9ea5-eea148a5270f",
        "question": "PASSAGE: In September 1190 Richard and Philip arrived in Sicily. After the death of King William II of Sicily his cousin Tancred had seized power and had been crowned early in 1190 as King Tancred of Sicily, although the legal heir was William's aunt Constance, wife of the new Emperor Henry VI. Tancred had imprisoned William's widow, Queen Joan, who was Richard's sister and did not give her the money she had inherited in William's will. When Richard arrived he demanded that his sister be released and given her inheritance; she was freed on 28 September, but without the inheritance. The presence of foreign troops also caused unrest: in October, the people of Messina revolted, demanding that the foreigners leave. Richard attacked Messina, capturing it on 4 October 1190. After looting and burning the city Richard established his base there, but this created tension between Richard and Philip Augustus. He remained there until Tancred finally agreed to sign a treaty on 4 March 1191. The treaty was signed by Richard, Philip, and Tancred. Its main terms were:\n\nQUESTION: King Tancred of Sicily seized power from who?"
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
            "sample_id": "2436207d-a6bf-44f8-9ea5-eea148a5270f",
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
    question = "PASSAGE: In September 1190 Richard and Philip arrived in Sicily. After the death of King William II of Sicily his cousin Tancred had seized power and had been crowned early in 1190 as King Tancred of Sicily, although the legal heir was William's aunt Constance, wife of the new Emperor Henry VI. Tancred had imprisoned William's widow, Queen Joan, who was Richard's sister and did not give her the money she had inherited in William's will. When Richard arrived he demanded that his sister be released and given her inheritance; she was freed on 28 September, but without the inheritance. The presence of foreign troops also caused unrest: in October, the people of Messina revolted, demanding that the foreigners leave. Richard attacked Messina, capturing it on 4 October 1190. After looting and burning the city Richard established his base there, but this created tension between Richard and Philip Augustus. He remained there until Tancred finally agreed to sign a treaty on 4 March 1191. The treaty was signed by Richard, Philip, and Tancred. Its main terms were:\n\nQUESTION: King Tancred of Sicily seized power from who?"

    # Call the main function and get the answer
    answer = module.main(question)

    # Log execution completion
    with open(trace_file, 'a', encoding='utf-8') as f:
        end_entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "event": "execution_complete",
            "iteration": 3,
            "sample_id": "2436207d-a6bf-44f8-9ea5-eea148a5270f",
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
            "sample_id": "2436207d-a6bf-44f8-9ea5-eea148a5270f",
            "error": str(e),
            "traceback": traceback.format_exc()
        }
        f.write(json.dumps(error_entry) + "\n")

    print("ERROR_START")
    print(str(e))
    print(traceback.format_exc())
    print("ERROR_END")
