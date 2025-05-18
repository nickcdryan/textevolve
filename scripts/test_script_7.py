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
        "sample_id": "example_65",
        "question": 'PASSAGE: The total population in sub-Saharan Africa is projected to increase to almost one billion people, making it the most populated region outside of South-Central Asia. According to the United Nations, the population of Nigeria will reach 411 million by 2050. Nigeria might then be the 3rd most populous country in the world. In 2100, the population of Nigeria may reach 794 million. While the overall population is expected to increase, the growth rate is estimated to decrease from 1.2 percent per year in 2010 to 0.4 percent per year in 2050. The birth rate is also projected to decrease from 20.7 to 13.7, while the death rate is projected to increase from 8.5 in 2010 to 9.8 in 2050. List of countries by life expectancy is all expected to increase from 67.0 years in 2010 to 75.2 years in 2050. By 2050 the percent of the population estimated to be living in urban areas is 69.6% compared to the 50.6% in 2010.\n\nQUESTION: How many millions of people is the population of Nigeria expected to grow by in 2100 compared to 2050?'
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
            "sample_id": "example_65",
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
    question = 'PASSAGE: The total population in sub-Saharan Africa is projected to increase to almost one billion people, making it the most populated region outside of South-Central Asia. According to the United Nations, the population of Nigeria will reach 411 million by 2050. Nigeria might then be the 3rd most populous country in the world. In 2100, the population of Nigeria may reach 794 million. While the overall population is expected to increase, the growth rate is estimated to decrease from 1.2 percent per year in 2010 to 0.4 percent per year in 2050. The birth rate is also projected to decrease from 20.7 to 13.7, while the death rate is projected to increase from 8.5 in 2010 to 9.8 in 2050. List of countries by life expectancy is all expected to increase from 67.0 years in 2010 to 75.2 years in 2050. By 2050 the percent of the population estimated to be living in urban areas is 69.6% compared to the 50.6% in 2010.\n\nQUESTION: How many millions of people is the population of Nigeria expected to grow by in 2100 compared to 2050?'

    # Call the main function and get the answer
    answer = module.main(question)

    # Log execution completion
    with open(trace_file, 'a', encoding='utf-8') as f:
        end_entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "event": "execution_complete",
            "iteration": 7,
            "sample_id": "example_65",
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
            "sample_id": "example_65",
            "error": str(e),
            "traceback": traceback.format_exc()
        }
        f.write(json.dumps(error_entry) + "\n")

    print("ERROR_START")
    print(str(e))
    print(traceback.format_exc())
    print("ERROR_END")
