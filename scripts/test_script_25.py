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
trace_file = "archive/trace_iteration_25.jsonl"
os.makedirs(os.path.dirname(trace_file), exist_ok=True)

# Trace entry for execution start
with open(trace_file, 'a', encoding='utf-8') as f:
    start_entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "event": "execution_start",
        "iteration": 25,
        "sample_id": "6f56c1ee-41bb-46f9-9334-3b9207350caa",
        "question": "PASSAGE: At the Corfu European Council, held on 24-25 June 1994, the EU officially confirmed that Cyprus would be included in the Union's next phase of enlargement. Two weeks later, on 5 July, the European Court of Justice imposed restrictions on the export of goods from Northern Cyprus into the European Union. Soon afterwards, in December, relations between the EU and Turkey were further damaged when Greece blocked the final implementation of a customs union. As a result, talks remained completely blocked throughout 1995 and 1996. In December 1996, the European Court of Human Rights  delivered a landmark ruling that declared that Turkey was an occupying power in Cyprus. The case - Loizidou v. Turkey - centred on Titina Loizidou, a refugee from Kyrenia, who was judged to have been unlawfully denied the control of her property by Turkey. The case also had severe financial implications as the Court later ruled that Turkey should pay Mrs Loizidou US$825,000 in compensation for the loss of use of her property. Ankara rejected the ruling as politically motivated. After twenty years of talks, a settlement seemed as far off as ever. However, the basic parameters of a settlement were by now internationally agreed. Cyprus would be a bi-zonal, bi-communal federation. A solution would also be expected to address the following issues:\n\nQUESTION: How many years were talks between the EU and Turkey blocked?"
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
        if frame_module == 'current_script_25':
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
            "iteration": 25,
            "sample_id": "6f56c1ee-41bb-46f9-9334-3b9207350caa",
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
        "current_script_25", 
        "scripts/current_script_25.py"
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
    question = "PASSAGE: At the Corfu European Council, held on 24-25 June 1994, the EU officially confirmed that Cyprus would be included in the Union's next phase of enlargement. Two weeks later, on 5 July, the European Court of Justice imposed restrictions on the export of goods from Northern Cyprus into the European Union. Soon afterwards, in December, relations between the EU and Turkey were further damaged when Greece blocked the final implementation of a customs union. As a result, talks remained completely blocked throughout 1995 and 1996. In December 1996, the European Court of Human Rights  delivered a landmark ruling that declared that Turkey was an occupying power in Cyprus. The case - Loizidou v. Turkey - centred on Titina Loizidou, a refugee from Kyrenia, who was judged to have been unlawfully denied the control of her property by Turkey. The case also had severe financial implications as the Court later ruled that Turkey should pay Mrs Loizidou US$825,000 in compensation for the loss of use of her property. Ankara rejected the ruling as politically motivated. After twenty years of talks, a settlement seemed as far off as ever. However, the basic parameters of a settlement were by now internationally agreed. Cyprus would be a bi-zonal, bi-communal federation. A solution would also be expected to address the following issues:\n\nQUESTION: How many years were talks between the EU and Turkey blocked?"

    # Call the main function and get the answer
    answer = module.main(question)

    # Log execution completion
    with open(trace_file, 'a', encoding='utf-8') as f:
        end_entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "event": "execution_complete",
            "iteration": 25,
            "sample_id": "6f56c1ee-41bb-46f9-9334-3b9207350caa",
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
            "iteration": 25,
            "sample_id": "6f56c1ee-41bb-46f9-9334-3b9207350caa",
            "error": str(e),
            "traceback": traceback.format_exc()
        }
        f.write(json.dumps(error_entry) + "\n")

    print("ERROR_START")
    print(str(e))
    print(traceback.format_exc())
    print("ERROR_END")
