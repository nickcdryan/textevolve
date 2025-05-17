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
trace_file = "archive/trace_iteration_10.jsonl"
os.makedirs(os.path.dirname(trace_file), exist_ok=True)

# Trace entry for execution start
with open(trace_file, 'a', encoding='utf-8') as f:
    start_entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "event": "execution_start",
        "iteration": 10,
        "sample_id": "d1f13dc6-2a84-4cb2-87ab-e94224e7c645",
        "question": 'PASSAGE: The Church of Ireland, at 2.7% of the population, is the second largest Christian denomination. Membership declined throughout the twentieth century, but experienced an increase early in the 21st century, as have other small Christian denominations. Significant Protestant denominations are the Presbyterian Church and Methodist Church. Immigration has contributed to a growth in Hindu and Muslim populations. In percentage terms, Orthodox Christianity and Islam were the fastest growing religions, with increases of 100% and 70% respectively. Ireland\'s patron saints are Saint Patrick, Saint Bridget and Saint Columba. Saint Patrick is the only one commonly recognised as the patron saint. Saint Patrick\'s Day is celebrated on 17 March in Ireland and abroad as the Irish national day, with parades and other celebrations. As with other predominantly Catholic European states, Ireland underwent a period of legal secularisation in the late twentieth century. In 1972, the article of the Constitution naming specific religious groups was deleted by the Fifth Amendment in a referendum. Article 44 remains in the Constitution: "The State acknowledges that the homage of public worship is due to Almighty God. It shall hold His Name in reverence, and shall respect and honour religion." The article also establishes freedom of religion, prohibits endowment of any religion, prohibits the state from religious discrimination, and requires the state to treat religious and non-religious schools in a non-prejudicial manner. Religious studies was introduced as an optional Junior Certificate subject in 2001. Although many schools are run by religious organisations, a secularist trend is occurring among younger generations.\n\nQUESTION: How many percentage points difference was there between Muslims and Christians?'
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
        if frame_module == 'current_script_10':
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
            "iteration": 10,
            "sample_id": "d1f13dc6-2a84-4cb2-87ab-e94224e7c645",
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
        "current_script_10", 
        "scripts/current_script_10.py"
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
    question = 'PASSAGE: The Church of Ireland, at 2.7% of the population, is the second largest Christian denomination. Membership declined throughout the twentieth century, but experienced an increase early in the 21st century, as have other small Christian denominations. Significant Protestant denominations are the Presbyterian Church and Methodist Church. Immigration has contributed to a growth in Hindu and Muslim populations. In percentage terms, Orthodox Christianity and Islam were the fastest growing religions, with increases of 100% and 70% respectively. Ireland\'s patron saints are Saint Patrick, Saint Bridget and Saint Columba. Saint Patrick is the only one commonly recognised as the patron saint. Saint Patrick\'s Day is celebrated on 17 March in Ireland and abroad as the Irish national day, with parades and other celebrations. As with other predominantly Catholic European states, Ireland underwent a period of legal secularisation in the late twentieth century. In 1972, the article of the Constitution naming specific religious groups was deleted by the Fifth Amendment in a referendum. Article 44 remains in the Constitution: "The State acknowledges that the homage of public worship is due to Almighty God. It shall hold His Name in reverence, and shall respect and honour religion." The article also establishes freedom of religion, prohibits endowment of any religion, prohibits the state from religious discrimination, and requires the state to treat religious and non-religious schools in a non-prejudicial manner. Religious studies was introduced as an optional Junior Certificate subject in 2001. Although many schools are run by religious organisations, a secularist trend is occurring among younger generations.\n\nQUESTION: How many percentage points difference was there between Muslims and Christians?'

    # Call the main function and get the answer
    answer = module.main(question)

    # Log execution completion
    with open(trace_file, 'a', encoding='utf-8') as f:
        end_entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "event": "execution_complete",
            "iteration": 10,
            "sample_id": "d1f13dc6-2a84-4cb2-87ab-e94224e7c645",
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
            "iteration": 10,
            "sample_id": "d1f13dc6-2a84-4cb2-87ab-e94224e7c645",
            "error": str(e),
            "traceback": traceback.format_exc()
        }
        f.write(json.dumps(error_entry) + "\n")

    print("ERROR_START")
    print(str(e))
    print(traceback.format_exc())
    print("ERROR_END")
