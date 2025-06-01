import sys
import traceback
import os
import json
import datetime
import inspect
import functools
import importlib.util
import openai
from openai import OpenAI


# Add the scripts directory to the path
sys.path.append("scripts")

# Configure tracing
trace_file = "archive/trace_iteration_2.jsonl"
os.makedirs(os.path.dirname(trace_file), exist_ok=True)



def call_llm(prompt, system_instruction=None):

    try:
        from google import genai
        from google.genai import types
        import os  # Import the os module

        # Initialize the Gemini client
        client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

        # Call the API with system instruction if provided
        if system_instruction:
            response = client.models.generate_content(
                model="gemini-2.0-flash", 
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction
                ),
                contents=prompt
            )
        else:
            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=prompt
            )

        return response.text
    except Exception as e:
        print(f"Error calling Gemini API: {str(e)}")
        return f"Error: {str(e)}"

def execute_code(code_str, timeout=10):
    """Execute Python code with automatic package installation and proper scoping"""
    import sys
    import re
    import subprocess
    from io import StringIO

    print("  [SYSTEM] Auto-installing execute_code() with scope fix")

    # Clean markdown formatting
    patterns = [
        r'```python\s*\n(.*?)\n```',
        r'```python\s*(.*?)```', 
        r'```\s*\n(.*?)\n```',
        r'```\s*(.*?)```'
    ]

    cleaned_code = code_str.strip()
    for pattern in patterns:
        match = re.search(pattern, code_str, re.DOTALL | re.IGNORECASE)
        if match:
            cleaned_code = match.group(1).strip()
            print("  [CLEANING] Removed markdown")
            break

    # Function to install a package
    def install_package(package_name):
        try:
            print(f"  [INSTALLING] Installing {package_name}...")
            result = subprocess.run([
                sys.executable, "-m", "pip", "install", package_name
            ], capture_output=True, text=True, timeout=30)

            if result.returncode == 0:
                print(f"  [SUCCESS] {package_name} installed successfully")
                return True
            else:
                print(f"  [FAILED] Could not install {package_name}: {result.stderr}")
                return False
        except Exception as e:
            print(f"  [ERROR] Installation error: {str(e)}")
            return False

    # Execute with proper scoping and auto-installation retry
    max_install_attempts = 3
    attempt = 0

    while attempt <= max_install_attempts:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        stdout_capture = StringIO()
        stderr_capture = StringIO()

        try:
            sys.stdout = stdout_capture
            sys.stderr = stderr_capture

            # CRITICAL FIX: Provide explicit globals and locals
            # This ensures imports are available to functions defined in the code
            exec_namespace = {}
            exec(cleaned_code, exec_namespace, exec_namespace)

            # Success!
            sys.stdout = old_stdout
            sys.stderr = old_stderr

            output = stdout_capture.getvalue().strip()
            return output if output else "Code executed successfully"

        except ModuleNotFoundError as e:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

            # Extract the missing module name
            module_name = str(e).split("'")[1] if "'" in str(e) else None

            if module_name and attempt < max_install_attempts:
                print(f"  [MISSING] Module '{module_name}' not found, attempting to install...")

                # Try to install the missing package
                if install_package(module_name):
                    attempt += 1
                    print(f"  [RETRY] Retrying code execution (attempt {attempt + 1})...")
                    continue
                else:
                    return f"Error: Could not install required package '{module_name}'"
            else:
                return f"Error: {str(e)}"

        except Exception as e:
            sys.stdout = old_stdout  
            sys.stderr = old_stderr
            return f"Error: {str(e)}"

        attempt += 1

    return "Error: Maximum installation attempts exceeded"


# Trace entry for execution start
with open(trace_file, 'a', encoding='utf-8') as f:
    start_entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "event": "execution_start",
        "iteration": 2,
        "sample_id": "example_22",
        "question": 'Multi-hop reasoning task:\n\nQuestion: What film was written and directed by Joby Harold with music written by Samuel Sim?\n\nSupporting Documents:\n=== Document 1: Here (1954 song) ===\n"Here" is a popular song, with music written by Harold Grant and lyrics by Dorcas Cochran, published in 1954. (Most sources show music and lyrics by both, but Cochran was a lyricist and Grant a composer.) The melody was adapted from the operatic aria, ""Caro nome,"" from the opera "Rigoletto" by Giuseppe Verdi. \n\n=== Document 2: Beautiful Young Minds ===\nBeautiful Young Minds was a documentary first shown at the BRITDOC Festival on 26 July 2007 and first broadcast on BBC 2 on 14 October 2007. The documentary follows the selection process and training for the U.K. team to compete in the 2006 International Mathematical Olympiad (IMO), as well as the actual event in Slovenia. Many of the young mathematicians featured in the film had a form of autism, which the documentary links to mathematical ability. The team goes on to win numerous medals at the IMO, including four silver and one bronze. It was directed by Morgan Matthews, edited by Joby Gee and featured music by Sam Hooper. It was also screened at the Bath Film Festival in October 2007. The documentary inspired the 2014 film X+Y, which was also directed by Morgan Matthews, based on IMO participant Daniel Lightwing. \n\n=== Document 3: Joby Talbot ===\nJoby Talbot (born 25 August 1971) is a British composer. He has written for a wide variety of purposes and an accordingly broad range of styles, including instrumental and vocal concert music, film and television scores, pop arrangements and works for dance. He is therefore known to sometimes disparate audiences for quite different works. \n\n=== Document 4: Robin Hood (2018 film) ===\nRobin Hood is an upcoming American action-adventure film directed by Otto Bathurst and written by Joby Harold, Peter Craig, and David James Kelly based on the tale of Robin Hood. The film stars Taron Egerton, Jamie Foxx, Eve Hewson, Ben Mendelsohn, Jamie Dornan, Tim Minchin, Björn Bengtsson, and Paul Anderson. It will be released by Lionsgate\'s Summit Entertainment in all IMAX theatres on September 21, 2018. \n\n=== Document 5: King Arthur: Legend of the Sword ===\nKing Arthur: Legend of the Sword is a 2017 epic fantasy film directed by Guy Ritchie and written by Ritchie, Joby Harold and Lionel Wigram, inspired by Arthurian legends. The film stars Charlie Hunnam as the eponymous character, with Jude Law, Àstrid Bergès-Frisbey, Djimon Hounsou, Aidan Gillen and Eric Bana in supporting roles. \n\n=== Document 6: Blog Wars ===\nBlog Wars is a 2006 documentary film about the rise of political blogging and its influence on the 2006 midterm Connecticut senate election. Original musical score is composed by Samuel Sim. \n\n=== Document 7: Awake (film) ===\nAwake is a 2007 American conspiracy thriller film written and directed by Joby Harold. It stars Hayden Christensen, Jessica Alba, Terrence Howard and Lena Olin. The film was released in the United States and Canada on November 30, 2007. \n\n=== Document 8: Samuel Sim ===\nSamuel Sim is a film and television composer. He first gained recognition with his award winning score for the BBC drama series "Dunkirk". Since then he has written the music for a wide variety of film and television productions, most recently scoring the film "Awake" for The Weinstein Company and the BBC/HBO drama series "House of Saddam". His most recent acclaimed music is the soundtrack for Home Fires. Home Fires (Music from the Television Series) released May 6, 2016 by Sony Classical Records. \n\n=== Document 9: Gidget Goes to Rome ===\nGidget Goes to Rome is a 1963 Columbia Pictures Eastmancolor feature film starring Cindy Carol as the archetypal high school teen surfer girl originally created by Sandra Dee in the 1959 film "Gidget". The film is the third of three Gidget films directed by Paul Wendkos and expands upon Gidget\'s romance with boyfriend Moondoggie. The screenplay was written by Ruth Brooks Flippen based on characters created by Frederick Kohner. Veterans of previous Gidget films making appearances include James Darren as "Moondoggie", Joby Baker, and Jean "Jeff" Donnell as Gidget\'s mom, Mrs. Lawrence. The film has been released to VHS and DVD. \n\n=== Document 10: By the Beautiful Sea (song) ===\n"By the Beautiful Sea" is a popular song published in 1914, with music written by Harry Carroll and lyrics written by Harold R. Atteridge. The sheet music was published by Shapiro, Bernstein & Co. \n\n\nProvide your answer based on the information in the supporting documents.'
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
            "sample_id": "example_22",
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

    # INJECT BOTH FUNCTIONS
    module.execute_code = execute_code
    module.call_llm = call_llm

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
    question = 'Multi-hop reasoning task:\n\nQuestion: What film was written and directed by Joby Harold with music written by Samuel Sim?\n\nSupporting Documents:\n=== Document 1: Here (1954 song) ===\n"Here" is a popular song, with music written by Harold Grant and lyrics by Dorcas Cochran, published in 1954. (Most sources show music and lyrics by both, but Cochran was a lyricist and Grant a composer.) The melody was adapted from the operatic aria, ""Caro nome,"" from the opera "Rigoletto" by Giuseppe Verdi. \n\n=== Document 2: Beautiful Young Minds ===\nBeautiful Young Minds was a documentary first shown at the BRITDOC Festival on 26 July 2007 and first broadcast on BBC 2 on 14 October 2007. The documentary follows the selection process and training for the U.K. team to compete in the 2006 International Mathematical Olympiad (IMO), as well as the actual event in Slovenia. Many of the young mathematicians featured in the film had a form of autism, which the documentary links to mathematical ability. The team goes on to win numerous medals at the IMO, including four silver and one bronze. It was directed by Morgan Matthews, edited by Joby Gee and featured music by Sam Hooper. It was also screened at the Bath Film Festival in October 2007. The documentary inspired the 2014 film X+Y, which was also directed by Morgan Matthews, based on IMO participant Daniel Lightwing. \n\n=== Document 3: Joby Talbot ===\nJoby Talbot (born 25 August 1971) is a British composer. He has written for a wide variety of purposes and an accordingly broad range of styles, including instrumental and vocal concert music, film and television scores, pop arrangements and works for dance. He is therefore known to sometimes disparate audiences for quite different works. \n\n=== Document 4: Robin Hood (2018 film) ===\nRobin Hood is an upcoming American action-adventure film directed by Otto Bathurst and written by Joby Harold, Peter Craig, and David James Kelly based on the tale of Robin Hood. The film stars Taron Egerton, Jamie Foxx, Eve Hewson, Ben Mendelsohn, Jamie Dornan, Tim Minchin, Björn Bengtsson, and Paul Anderson. It will be released by Lionsgate\'s Summit Entertainment in all IMAX theatres on September 21, 2018. \n\n=== Document 5: King Arthur: Legend of the Sword ===\nKing Arthur: Legend of the Sword is a 2017 epic fantasy film directed by Guy Ritchie and written by Ritchie, Joby Harold and Lionel Wigram, inspired by Arthurian legends. The film stars Charlie Hunnam as the eponymous character, with Jude Law, Àstrid Bergès-Frisbey, Djimon Hounsou, Aidan Gillen and Eric Bana in supporting roles. \n\n=== Document 6: Blog Wars ===\nBlog Wars is a 2006 documentary film about the rise of political blogging and its influence on the 2006 midterm Connecticut senate election. Original musical score is composed by Samuel Sim. \n\n=== Document 7: Awake (film) ===\nAwake is a 2007 American conspiracy thriller film written and directed by Joby Harold. It stars Hayden Christensen, Jessica Alba, Terrence Howard and Lena Olin. The film was released in the United States and Canada on November 30, 2007. \n\n=== Document 8: Samuel Sim ===\nSamuel Sim is a film and television composer. He first gained recognition with his award winning score for the BBC drama series "Dunkirk". Since then he has written the music for a wide variety of film and television productions, most recently scoring the film "Awake" for The Weinstein Company and the BBC/HBO drama series "House of Saddam". His most recent acclaimed music is the soundtrack for Home Fires. Home Fires (Music from the Television Series) released May 6, 2016 by Sony Classical Records. \n\n=== Document 9: Gidget Goes to Rome ===\nGidget Goes to Rome is a 1963 Columbia Pictures Eastmancolor feature film starring Cindy Carol as the archetypal high school teen surfer girl originally created by Sandra Dee in the 1959 film "Gidget". The film is the third of three Gidget films directed by Paul Wendkos and expands upon Gidget\'s romance with boyfriend Moondoggie. The screenplay was written by Ruth Brooks Flippen based on characters created by Frederick Kohner. Veterans of previous Gidget films making appearances include James Darren as "Moondoggie", Joby Baker, and Jean "Jeff" Donnell as Gidget\'s mom, Mrs. Lawrence. The film has been released to VHS and DVD. \n\n=== Document 10: By the Beautiful Sea (song) ===\n"By the Beautiful Sea" is a popular song published in 1914, with music written by Harry Carroll and lyrics written by Harold R. Atteridge. The sheet music was published by Shapiro, Bernstein & Co. \n\n\nProvide your answer based on the information in the supporting documents.'

    # Call the main function and get the answer
    answer = module.main(question)

    # Log execution completion
    with open(trace_file, 'a', encoding='utf-8') as f:
        end_entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "event": "execution_complete",
            "iteration": 2,
            "sample_id": "example_22",
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
            "sample_id": "example_22",
            "error": str(e),
            "traceback": traceback.format_exc()
        }
        f.write(json.dumps(error_entry) + "\n")

    print("ERROR_START")
    print(str(e))
    print(traceback.format_exc())
    print("ERROR_END")
