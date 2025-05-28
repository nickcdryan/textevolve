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
trace_file = "archive/trace_iteration_6.jsonl"
os.makedirs(os.path.dirname(trace_file), exist_ok=True)

# Trace entry for execution start
with open(trace_file, 'a', encoding='utf-8') as f:
    start_entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "event": "execution_start",
        "iteration": 6,
        "sample_id": "math_38",
        "question": 'In the diagram, $AB,$ $BC,$ $CD,$ $DE,$ $EF,$ $FG,$ $GH,$ and $HK$ all have length $4,$ and all angles are right angles, with the exception of the angles at $D$ and $F.$\n\n[asy]\ndraw((0,0)--(0,4)--(4,4)--(4,8)--(6.8284,5.1716)--(9.6569,8)--(9.6569,4)--(13.6569,4)--(13.6569,0)--cycle,black+linewidth(1));\ndraw((0,0)--(0.5,0)--(0.5,0.5)--(0,0.5)--cycle,black+linewidth(1));\ndraw((0,4)--(0.5,4)--(0.5,3.5)--(0,3.5)--cycle,black+linewidth(1));\ndraw((4,4)--(4,4.5)--(3.5,4.5)--(3.5,4)--cycle,black+linewidth(1));\ndraw((6.8284,5.1716)--(7.0784,5.4216)--(6.8284,5.6716)--(6.5784,5.4216)--cycle,black+linewidth(1));\ndraw((9.6569,4)--(10.1569,4)--(10.1569,4.5)--(9.6569,4.5)--cycle,black+linewidth(1));\ndraw((13.6569,4)--(13.1569,4)--(13.1569,3.5)--(13.6569,3.5)--cycle,black+linewidth(1));\ndraw((13.6569,0)--(13.1569,0)--(13.1569,0.5)--(13.6569,0.5)--cycle,black+linewidth(1));\nlabel("$A$",(0,0),W);\nlabel("$B$",(0,4),NW);\nlabel("$C$",(4,4),S);\nlabel("$D$",(4,8),N);\nlabel("$E$",(6.8284,5.1716),S);\nlabel("$F$",(9.6569,8),N);\nlabel("$G$",(9.6569,4),S);\nlabel("$H$",(13.6569,4),NE);\nlabel("$K$",(13.6569,0),E);\n[/asy]\n\nDetermine the length of $DF.$\n\n[asy]\ndraw((0,0)--(2.8284,-2.8284)--(5.6568,0),black+linewidth(1));\ndraw((0,0)--(5.6568,0),black+linewidth(1)+dashed);\ndraw((2.8284,-2.8284)--(3.0784,-2.5784)--(2.8284,-2.3284)--(2.5784,-2.5784)--cycle,black+linewidth(1));\nlabel("$D$",(0,0),N);\nlabel("$E$",(2.8284,-2.8284),S);\nlabel("$F$",(5.6568,0),N);\n[/asy]'
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
        if frame_module == 'current_script_6':
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
            "iteration": 6,
            "sample_id": "math_38",
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
        "current_script_6", 
        "scripts/current_script_6.py"
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
    question = 'In the diagram, $AB,$ $BC,$ $CD,$ $DE,$ $EF,$ $FG,$ $GH,$ and $HK$ all have length $4,$ and all angles are right angles, with the exception of the angles at $D$ and $F.$\n\n[asy]\ndraw((0,0)--(0,4)--(4,4)--(4,8)--(6.8284,5.1716)--(9.6569,8)--(9.6569,4)--(13.6569,4)--(13.6569,0)--cycle,black+linewidth(1));\ndraw((0,0)--(0.5,0)--(0.5,0.5)--(0,0.5)--cycle,black+linewidth(1));\ndraw((0,4)--(0.5,4)--(0.5,3.5)--(0,3.5)--cycle,black+linewidth(1));\ndraw((4,4)--(4,4.5)--(3.5,4.5)--(3.5,4)--cycle,black+linewidth(1));\ndraw((6.8284,5.1716)--(7.0784,5.4216)--(6.8284,5.6716)--(6.5784,5.4216)--cycle,black+linewidth(1));\ndraw((9.6569,4)--(10.1569,4)--(10.1569,4.5)--(9.6569,4.5)--cycle,black+linewidth(1));\ndraw((13.6569,4)--(13.1569,4)--(13.1569,3.5)--(13.6569,3.5)--cycle,black+linewidth(1));\ndraw((13.6569,0)--(13.1569,0)--(13.1569,0.5)--(13.6569,0.5)--cycle,black+linewidth(1));\nlabel("$A$",(0,0),W);\nlabel("$B$",(0,4),NW);\nlabel("$C$",(4,4),S);\nlabel("$D$",(4,8),N);\nlabel("$E$",(6.8284,5.1716),S);\nlabel("$F$",(9.6569,8),N);\nlabel("$G$",(9.6569,4),S);\nlabel("$H$",(13.6569,4),NE);\nlabel("$K$",(13.6569,0),E);\n[/asy]\n\nDetermine the length of $DF.$\n\n[asy]\ndraw((0,0)--(2.8284,-2.8284)--(5.6568,0),black+linewidth(1));\ndraw((0,0)--(5.6568,0),black+linewidth(1)+dashed);\ndraw((2.8284,-2.8284)--(3.0784,-2.5784)--(2.8284,-2.3284)--(2.5784,-2.5784)--cycle,black+linewidth(1));\nlabel("$D$",(0,0),N);\nlabel("$E$",(2.8284,-2.8284),S);\nlabel("$F$",(5.6568,0),N);\n[/asy]'

    # Call the main function and get the answer
    answer = module.main(question)

    # Log execution completion
    with open(trace_file, 'a', encoding='utf-8') as f:
        end_entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "event": "execution_complete",
            "iteration": 6,
            "sample_id": "math_38",
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
            "iteration": 6,
            "sample_id": "math_38",
            "error": str(e),
            "traceback": traceback.format_exc()
        }
        f.write(json.dumps(error_entry) + "\n")

    print("ERROR_START")
    print(str(e))
    print(traceback.format_exc())
    print("ERROR_END")
