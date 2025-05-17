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
trace_file = "archive/trace_iteration_2.jsonl"
os.makedirs(os.path.dirname(trace_file), exist_ok=True)

# Trace entry for execution start
with open(trace_file, 'a', encoding='utf-8') as f:
    start_entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "event": "execution_start",
        "iteration": 2,
        "sample_id": "f893d369-9a1c-453e-95f5-6e671c06e55f",
        "question": "PASSAGE: He was the Speaker of the Parliament extraordinary held in Warsaw on 3 December to 24 December 1613. During the war with Sweden  and the  fought in Livonia winning to the Swedish army at Kroppenhof, Lixna, Slavskoye and Dyneburg. It is because of fighting with the Swedes until 1629, he was appointing Palatine-Governor of Smolensk region, but the nomination was already in 1625. Seeing the threat from Russia, Gosiewski immediately upon taking the governorate of Smolensk began to renovate the walls of the city. He personally oversaw the construction of Sigismund Fortress, which strengthened the eastern part of the stronghold. Intensively collected supplies of food and ammunition, and developed a business intelligence gathering valuable information about Moscow's war preparations. In the spring of 1632, he made review the fortifications in Dorogobuzh and other frontier forts. During the war with Russia, in the year 1632 and 1634 after a particularly famous defense of Smolensk - for ten months he defended the city against besieging forces led by Mikhail Shein, repelling all assaults, until the advent of the battle led by Prince Władysław. He fought at Vitebsk, Alder and Mstislav. He participated as a Commissioner in peace negotiations, topped the conclusion on 14 June 1634, in Treaty of Polanów. For his services, he received numerous goods in the province of Smolensk. He founded the Jesuits' College in Vitebsk and the female Monastery of the Holy Brigit at Brest-Litovsk.As Palatine-Governor, he commemorated the death of his longtime client - Jan Kunowski, who in 1640 wrote a series of poems dedicated to his late patron.\n\nQUESTION: How many years after being appointed Palatine-Governor of Smolensk region, did Gosiewski participated as a Commissioner in peace negotiations?"
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
        if frame_module == 'current_script_2':
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
            "iteration": 2,
            "sample_id": "f893d369-9a1c-453e-95f5-6e671c06e55f",
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
        "current_script_2", 
        "scripts/current_script_2.py"
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
    question = "PASSAGE: He was the Speaker of the Parliament extraordinary held in Warsaw on 3 December to 24 December 1613. During the war with Sweden  and the  fought in Livonia winning to the Swedish army at Kroppenhof, Lixna, Slavskoye and Dyneburg. It is because of fighting with the Swedes until 1629, he was appointing Palatine-Governor of Smolensk region, but the nomination was already in 1625. Seeing the threat from Russia, Gosiewski immediately upon taking the governorate of Smolensk began to renovate the walls of the city. He personally oversaw the construction of Sigismund Fortress, which strengthened the eastern part of the stronghold. Intensively collected supplies of food and ammunition, and developed a business intelligence gathering valuable information about Moscow's war preparations. In the spring of 1632, he made review the fortifications in Dorogobuzh and other frontier forts. During the war with Russia, in the year 1632 and 1634 after a particularly famous defense of Smolensk - for ten months he defended the city against besieging forces led by Mikhail Shein, repelling all assaults, until the advent of the battle led by Prince Władysław. He fought at Vitebsk, Alder and Mstislav. He participated as a Commissioner in peace negotiations, topped the conclusion on 14 June 1634, in Treaty of Polanów. For his services, he received numerous goods in the province of Smolensk. He founded the Jesuits' College in Vitebsk and the female Monastery of the Holy Brigit at Brest-Litovsk.As Palatine-Governor, he commemorated the death of his longtime client - Jan Kunowski, who in 1640 wrote a series of poems dedicated to his late patron.\n\nQUESTION: How many years after being appointed Palatine-Governor of Smolensk region, did Gosiewski participated as a Commissioner in peace negotiations?"

    # Call the main function and get the answer
    answer = module.main(question)

    # Log execution completion
    with open(trace_file, 'a', encoding='utf-8') as f:
        end_entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "event": "execution_complete",
            "iteration": 2,
            "sample_id": "f893d369-9a1c-453e-95f5-6e671c06e55f",
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
            "iteration": 2,
            "sample_id": "f893d369-9a1c-453e-95f5-6e671c06e55f",
            "error": str(e),
            "traceback": traceback.format_exc()
        }
        f.write(json.dumps(error_entry) + "\n")

    print("ERROR_START")
    print(str(e))
    print(traceback.format_exc())
    print("ERROR_END")
