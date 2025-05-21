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
trace_file = "archive/trace_iteration_28.jsonl"
os.makedirs(os.path.dirname(trace_file), exist_ok=True)

# Trace entry for execution start
with open(trace_file, 'a', encoding='utf-8') as f:
    start_entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "event": "execution_start",
        "iteration": 28,
        "sample_id": "4312789b-0f97-4627-a640-9ff7b225ec9d",
        "question": 'PASSAGE: After the unsuccessful operation Athanasius returned to Himara for the preparations of another rebellion. The remaining Spaniards left the region, but Athanasius awaited this time an aid of 3-4,000 soldiers from the Spanish king. On August 23, 1596 he met with Albanian captains Michael Bua, Giovanni Golemi and Michael Papada. They each received a monthly pay of 50 ducats. They then went to Lecce to arm the Himariotes with 1,000 arquebuses, powder, lead, four drums and four royal banners . Athanasius then moved to Rome and had an audience with the Pope. The following 20 years he continued to visit various western European leaders to trigger their intervention against the Ottomans, but without success. On the other hand, Venetian Cypriot Hieronimo Combi discouraged Michael Bua and his companions. In the Sanjak of Herzegovina and Montenegro Vilayet, the Serbs rose up in 1596-97, but without foreign support the rebels were forced to capitulate. In 1600, a rebellion broke out in Thessaly.\n\nQUESTION: Who did Athanasius meet with first, the Albanian captains or the Pope?'
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
        if frame_module == 'current_script_28':
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
            "iteration": 28,
            "sample_id": "4312789b-0f97-4627-a640-9ff7b225ec9d",
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
        "current_script_28", 
        "scripts/current_script_28.py"
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
    question = 'PASSAGE: After the unsuccessful operation Athanasius returned to Himara for the preparations of another rebellion. The remaining Spaniards left the region, but Athanasius awaited this time an aid of 3-4,000 soldiers from the Spanish king. On August 23, 1596 he met with Albanian captains Michael Bua, Giovanni Golemi and Michael Papada. They each received a monthly pay of 50 ducats. They then went to Lecce to arm the Himariotes with 1,000 arquebuses, powder, lead, four drums and four royal banners . Athanasius then moved to Rome and had an audience with the Pope. The following 20 years he continued to visit various western European leaders to trigger their intervention against the Ottomans, but without success. On the other hand, Venetian Cypriot Hieronimo Combi discouraged Michael Bua and his companions. In the Sanjak of Herzegovina and Montenegro Vilayet, the Serbs rose up in 1596-97, but without foreign support the rebels were forced to capitulate. In 1600, a rebellion broke out in Thessaly.\n\nQUESTION: Who did Athanasius meet with first, the Albanian captains or the Pope?'

    # Call the main function and get the answer
    answer = module.main(question)

    # Log execution completion
    with open(trace_file, 'a', encoding='utf-8') as f:
        end_entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "event": "execution_complete",
            "iteration": 28,
            "sample_id": "4312789b-0f97-4627-a640-9ff7b225ec9d",
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
            "iteration": 28,
            "sample_id": "4312789b-0f97-4627-a640-9ff7b225ec9d",
            "error": str(e),
            "traceback": traceback.format_exc()
        }
        f.write(json.dumps(error_entry) + "\n")

    print("ERROR_START")
    print(str(e))
    print(traceback.format_exc())
    print("ERROR_END")
