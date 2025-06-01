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
trace_file = "archive/trace_iteration_0.jsonl"
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
        "iteration": 0,
        "sample_id": "5a88d745554299206df2b378",
        "question": 'Multi-hop reasoning task:\n\nQuestion: What occupation was shared by David Yates and Pietro Germi?\n\nSupporting Documents:\n=== Document 1: Divorce Italian Style ===\nDivorce Italian Style (Italian: "Divorzio all\'italiana" ) is a 1961 Italian comedy film directed by Pietro Germi. The screenplay was written by Ennio De Concini, Pietro Germi, Alfredo Giannetti, and Agenore Incrocci; based on the novel "Un delitto d\'onore" ("Honour Killing") by Giovanni Arpino. It stars Marcello Mastroianni, Daniela Rocca, Stefania Sandrelli, Lando Buzzanca, and Leopoldo Trieste. The movie won the Academy Award for Best Writing, Story and Screenplay - Written Directly for the Screen; Mastroianni was nominated for Best Actor in a Leading Role (Marcello Mastroianni) and Germi for Best Director. \n\n=== Document 2: Ottavio Alessi ===\nBorn in Cammarata, Province of Agrigento, Alessi entered the film industry in 1940 as an assistant director. In 1945 he started an intense career as a screenwriter, alternating between genre films and art films and collaborating with Pietro Germi, Franco Rossi, Folco Quilici and Luciano Salce, among others. He also directed two films in the 1960s. \n\n=== Document 3: Serafino (film) ===\nSerafino (also known as "Serafino ou L\'amour aux champs" in France) is a 1968 Italian film directed by Pietro Germi. \n\n=== Document 4: Black 13 ===\nBlack 13 is a 1953 British crime drama film directed by Ken Hughes and starring Peter Reynolds, Rona Anderson, Patrick Barr and John Le Mesurier. The film is a remake of the 1948 Italian film "Gioventù perduta" (a.k.a. "Lost Youth") by Pietro Germi. It was made by Vandyke Productions. \n\n=== Document 5: David Yates ===\nDavid Yates (born (1963--)08 1963 ) is an English filmmaker who has directed feature films, short films, and television productions. \n\n=== Document 6: The Testimony (1946 film) ===\nThe Testimony (Italian:Il testimone) is 1946 Italian crime film directed by Pietro Germi and starring Roldano Lupi, Marina Berti and Ernesto Almirante. The film was made at the Cines Studios in Rome. It is one of several films regarded as an antecedent of the later giallo thrillers. \n\n=== Document 7: Cineriz ===\nCineriz was an Italian media company, involved primarily in the production and distribution of films, founded in the early 50s by the businessman Angelo Rizzoli. The company catalogue counts also many movies directed by Federico Fellini, Gillo Pontecorvo, Luchino Visconti, Michelangelo Antonioni, Pier Paolo Pasolini, Pietro Germi, Roberto Rossellini and Vittorio De Sica. \n\n=== Document 8: Pietro Germi ===\nPietro Germi (] ; 14 September 1914 – 5 December 1974) was an Italian actor, screenwriter, and director. Germi was born in Genoa, Liguria, to a lower-middle-class family. He was a messenger and briefly attended nautical school before deciding on a career in acting. \n\n=== Document 9: Lipstick (1960 film) ===\nIl rossetto (internationally released as Lipstick) is a 1960 Italian crime-drama film directed by Damiano Damiani. It is the feature film debut of Damiani, after two documentaries and several screenplays. The film\'s plot was loosely inspired by actual events. Pietro Germi reprised, with very slight modifications, the character he played in "Un maledetto imbroglio". \n\n=== Document 10: Commedia all\'italiana ===\nCommedia all\'italiana (i.e. "Comedy in the Italian way"; ] ) or Italian-style comedy is an Italian film genre. It is widely considered to have started with Mario Monicelli\'s "I soliti ignoti" ("Big Deal on Madonna Street") in 1958 and derives its name from the title of Pietro Germi\'s "Divorzio all\'italiana" ("Divorce Italian Style", 1961). \n\n\nProvide your answer based on the information in the supporting documents.'
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
        if frame_module == 'current_script_0':
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
            "iteration": 0,
            "sample_id": "5a88d745554299206df2b378",
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
        "current_script_0", 
        "scripts/current_script_0.py"
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
    question = 'Multi-hop reasoning task:\n\nQuestion: What occupation was shared by David Yates and Pietro Germi?\n\nSupporting Documents:\n=== Document 1: Divorce Italian Style ===\nDivorce Italian Style (Italian: "Divorzio all\'italiana" ) is a 1961 Italian comedy film directed by Pietro Germi. The screenplay was written by Ennio De Concini, Pietro Germi, Alfredo Giannetti, and Agenore Incrocci; based on the novel "Un delitto d\'onore" ("Honour Killing") by Giovanni Arpino. It stars Marcello Mastroianni, Daniela Rocca, Stefania Sandrelli, Lando Buzzanca, and Leopoldo Trieste. The movie won the Academy Award for Best Writing, Story and Screenplay - Written Directly for the Screen; Mastroianni was nominated for Best Actor in a Leading Role (Marcello Mastroianni) and Germi for Best Director. \n\n=== Document 2: Ottavio Alessi ===\nBorn in Cammarata, Province of Agrigento, Alessi entered the film industry in 1940 as an assistant director. In 1945 he started an intense career as a screenwriter, alternating between genre films and art films and collaborating with Pietro Germi, Franco Rossi, Folco Quilici and Luciano Salce, among others. He also directed two films in the 1960s. \n\n=== Document 3: Serafino (film) ===\nSerafino (also known as "Serafino ou L\'amour aux champs" in France) is a 1968 Italian film directed by Pietro Germi. \n\n=== Document 4: Black 13 ===\nBlack 13 is a 1953 British crime drama film directed by Ken Hughes and starring Peter Reynolds, Rona Anderson, Patrick Barr and John Le Mesurier. The film is a remake of the 1948 Italian film "Gioventù perduta" (a.k.a. "Lost Youth") by Pietro Germi. It was made by Vandyke Productions. \n\n=== Document 5: David Yates ===\nDavid Yates (born (1963--)08 1963 ) is an English filmmaker who has directed feature films, short films, and television productions. \n\n=== Document 6: The Testimony (1946 film) ===\nThe Testimony (Italian:Il testimone) is 1946 Italian crime film directed by Pietro Germi and starring Roldano Lupi, Marina Berti and Ernesto Almirante. The film was made at the Cines Studios in Rome. It is one of several films regarded as an antecedent of the later giallo thrillers. \n\n=== Document 7: Cineriz ===\nCineriz was an Italian media company, involved primarily in the production and distribution of films, founded in the early 50s by the businessman Angelo Rizzoli. The company catalogue counts also many movies directed by Federico Fellini, Gillo Pontecorvo, Luchino Visconti, Michelangelo Antonioni, Pier Paolo Pasolini, Pietro Germi, Roberto Rossellini and Vittorio De Sica. \n\n=== Document 8: Pietro Germi ===\nPietro Germi (] ; 14 September 1914 – 5 December 1974) was an Italian actor, screenwriter, and director. Germi was born in Genoa, Liguria, to a lower-middle-class family. He was a messenger and briefly attended nautical school before deciding on a career in acting. \n\n=== Document 9: Lipstick (1960 film) ===\nIl rossetto (internationally released as Lipstick) is a 1960 Italian crime-drama film directed by Damiano Damiani. It is the feature film debut of Damiani, after two documentaries and several screenplays. The film\'s plot was loosely inspired by actual events. Pietro Germi reprised, with very slight modifications, the character he played in "Un maledetto imbroglio". \n\n=== Document 10: Commedia all\'italiana ===\nCommedia all\'italiana (i.e. "Comedy in the Italian way"; ] ) or Italian-style comedy is an Italian film genre. It is widely considered to have started with Mario Monicelli\'s "I soliti ignoti" ("Big Deal on Madonna Street") in 1958 and derives its name from the title of Pietro Germi\'s "Divorzio all\'italiana" ("Divorce Italian Style", 1961). \n\n\nProvide your answer based on the information in the supporting documents.'

    # Call the main function and get the answer
    answer = module.main(question)

    # Log execution completion
    with open(trace_file, 'a', encoding='utf-8') as f:
        end_entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "event": "execution_complete",
            "iteration": 0,
            "sample_id": "5a88d745554299206df2b378",
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
            "iteration": 0,
            "sample_id": "5a88d745554299206df2b378",
            "error": str(e),
            "traceback": traceback.format_exc()
        }
        f.write(json.dumps(error_entry) + "\n")

    print("ERROR_START")
    print(str(e))
    print(traceback.format_exc())
    print("ERROR_END")
