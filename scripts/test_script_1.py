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
trace_file = "archive/trace_iteration_1.jsonl"
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
        "iteration": 1,
        "sample_id": "example_17",
        "question": 'Multi-hop reasoning task:\n\nQuestion: Robert Earl Holding owned an oil company that was originally founded by who?\n\nSupporting Documents:\n=== Document 1: A4 Holding ===\nA4 Holding S.p.A. known as Gruppo A4 Holding (previously as Serenissima Group), is an Italian holding company based in Verona, Veneto region. The company owned "Autostrada Brescia Verona Vicenza Padova" (100%), the operator of Brescia–Padua section of Autostrada A4 and Autostrada A31 (Rovigo via Vicenza to Piovene Rocchette), as well as an equity interests in Autostrada del Brennero, the operator of Autostrada A22 (Modena to Brenner Pass; 4.2327% stake via "Serenissima Partecipazioni" which A4 Holding owned 99.999% stake) and Autostrade Lombarde, the parent company of the operator of Autostrada A35 (Brescia to Milan; 4.90% stake via "Autostrada Brescia–Padova"). \n\n=== Document 2: Skelly Oil ===\nSkelly Oil Company was a medium-sized oil company founded in 1919 by William Grove (Bill) Skelly, Chesley Coleman Herndon and Frederick A. Pielsticker in Tulsa, Oklahoma. J.\xa0Paul Getty acquired control of the company during the 1930s. Skelly Oil became part of Getty Oil Company, Mission Oil Company, Tidewater Oil Company. It became defunct when absorbed by Getty Oil Company in 1974, and the abandoned Skelly brand logo was revived by Nimmons-Joliet Development Corp. in 2012. \n\n=== Document 3: Robert Holding ===\nRobert Earl Holding (November 29, 1926 – April 19, 2013) was an American businessman who owned Sinclair Oil Corporation, the Little America Hotels, the Grand America Hotel, the Westgate Hotel in San Diego, California (directed by Georg Hochfilzer), and two ski resorts, Sun Valley in central Idaho since 1977, and Snowbasin near Ogden, Utah, since 1984. \n\n=== Document 4: Ahvaz Field ===\nThe Ahvaz oil field is an Iranian oil field located in Ahvaz, Khuzestan Province. It was discovered in 1953 and developed by National Iranian Oil Company. It began production in 1954. Ahvaz field is one of the richest oil fields in the world with an estimated proven reserves are around , and production is centered on 750000 oilbbl/d . The field is owned by state-owned National Iranian Oil Company (NIOC) and operated by National Iranian South Oil Company (NISOC). \n\n=== Document 5: Little America, Wyoming ===\nLittle America is a census-designated place (CDP) in Sweetwater County, Wyoming, United States. The population was 68 at the 2010 census. The community got its name from the Little America motel, which was purposefully located in a remote location as a haven, not unlike the base camp the polar explorer Richard E. Byrd set up in the Antarctic in 1928. However, being situated on a coast-to-coast highway and offering travel services, it thrived, launching a chain of travel facilities by the same name. Its developer, Robert Earl Holding, died on April 19, 2013, with a personal net worth of over $3 billion. \n\n=== Document 6: Aghajari oil field ===\nThe Aghajari oil field is an iranian oil field located in Khuzestan Province. It was discovered in 1938 and developed by National Iranian Oil Company. It began production in 1940 and produces oil. The total proven reserves of the Aghajari oil field are around 30 billion barrels (3758×10tonnes), and production is centered on 300000 oilbbl/d . The field is owned by state-owned National Iranian Oil Company (NIOC) and operated by National Iranian South Oil Company (NISOC). \n\n=== Document 7: Carabobo Field ===\nCarabobo is an oil field located in Venezuela\'s Orinoco Belt. As one of the world\'s largest accumulations of recoverable oil, the recent discoveries in the Orinoco Belt have led to Venezuela holding the world\'s largest recoverable reserves in the world, surpassing Saudi Arabia in July 2010. The Carabobo oil field is majority owned by Venezuela\'s national oil company, Petroleos de Venezuela SA (PDVSA). Owning the majority of the Orinoco Belt, and its estimated 1.18 trillion barrels of oil in place, PDVSA is now the fourth largest oil company in the world. The field is well known for its extra Heavy crude oils, having an average specific gravity between 4 and 16 °API. The Orinoco Belt holds 90% of the world\'s extra heavy crude oils, estimated at 256 billion recoverable barrels. While production is in its early development, the Carabobo field is expected to produce 400,000 barrels of oil per day. \n\n=== Document 8: Sinclair Oil Corporation ===\nSinclair Oil Corporation is an American petroleum corporation, founded by Harry F. Sinclair on May 1, 1916, as the Sinclair Oil and Refining Corporation by combining the assets of 11 small petroleum companies. Originally a New York corporation, Sinclair Oil reincorporated in Wyoming in 1976. The corporation\'s logo features the silhouette of a large green dinosaur. \n\n=== Document 9: 101 Ranch Oil Company ===\nFounded in 1908 by oil exploration pioneer E. W. Marland, The 101 Ranch Oil Company was located on the Miller Brothers 101 Ranch and headquartered in Ponca City, Oklahoma. The company’s 1911 oil discovery in North Eastern Oklahoma opened up oil development in a great region from Eastern Oklahoma west to Mervine, Newkirk, Blackwell, Billings and Garber and led to the founding of the Marland Oil Company, later renamed the Continental Oil Company, now known as Conoco. \n\n=== Document 10: Rag Sefid oil field ===\nThe Rag Sefid oil field is an oil field located in Khuzestan Province, approximately 6\xa0km in nearest distance from the Persian Gulf, southwest Iran. It was discovered in 1964 and developed by National Iranian Oil Company and began production in 1966. The total proven reserves of the Rag Sefid oil field are around 14,5 billion barrels, and production is centered on 180000 oilbbl/d . The field is owned by state-owned National Iranian Oil Company (NIOC) and operated by National Iranian South Oil Company (NISOC). \n\n\nProvide your answer based on the information in the supporting documents.'
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
        if frame_module == 'current_script_1':
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
            "iteration": 1,
            "sample_id": "example_17",
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
        "current_script_1", 
        "scripts/current_script_1.py"
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
    question = 'Multi-hop reasoning task:\n\nQuestion: Robert Earl Holding owned an oil company that was originally founded by who?\n\nSupporting Documents:\n=== Document 1: A4 Holding ===\nA4 Holding S.p.A. known as Gruppo A4 Holding (previously as Serenissima Group), is an Italian holding company based in Verona, Veneto region. The company owned "Autostrada Brescia Verona Vicenza Padova" (100%), the operator of Brescia–Padua section of Autostrada A4 and Autostrada A31 (Rovigo via Vicenza to Piovene Rocchette), as well as an equity interests in Autostrada del Brennero, the operator of Autostrada A22 (Modena to Brenner Pass; 4.2327% stake via "Serenissima Partecipazioni" which A4 Holding owned 99.999% stake) and Autostrade Lombarde, the parent company of the operator of Autostrada A35 (Brescia to Milan; 4.90% stake via "Autostrada Brescia–Padova"). \n\n=== Document 2: Skelly Oil ===\nSkelly Oil Company was a medium-sized oil company founded in 1919 by William Grove (Bill) Skelly, Chesley Coleman Herndon and Frederick A. Pielsticker in Tulsa, Oklahoma. J.\xa0Paul Getty acquired control of the company during the 1930s. Skelly Oil became part of Getty Oil Company, Mission Oil Company, Tidewater Oil Company. It became defunct when absorbed by Getty Oil Company in 1974, and the abandoned Skelly brand logo was revived by Nimmons-Joliet Development Corp. in 2012. \n\n=== Document 3: Robert Holding ===\nRobert Earl Holding (November 29, 1926 – April 19, 2013) was an American businessman who owned Sinclair Oil Corporation, the Little America Hotels, the Grand America Hotel, the Westgate Hotel in San Diego, California (directed by Georg Hochfilzer), and two ski resorts, Sun Valley in central Idaho since 1977, and Snowbasin near Ogden, Utah, since 1984. \n\n=== Document 4: Ahvaz Field ===\nThe Ahvaz oil field is an Iranian oil field located in Ahvaz, Khuzestan Province. It was discovered in 1953 and developed by National Iranian Oil Company. It began production in 1954. Ahvaz field is one of the richest oil fields in the world with an estimated proven reserves are around , and production is centered on 750000 oilbbl/d . The field is owned by state-owned National Iranian Oil Company (NIOC) and operated by National Iranian South Oil Company (NISOC). \n\n=== Document 5: Little America, Wyoming ===\nLittle America is a census-designated place (CDP) in Sweetwater County, Wyoming, United States. The population was 68 at the 2010 census. The community got its name from the Little America motel, which was purposefully located in a remote location as a haven, not unlike the base camp the polar explorer Richard E. Byrd set up in the Antarctic in 1928. However, being situated on a coast-to-coast highway and offering travel services, it thrived, launching a chain of travel facilities by the same name. Its developer, Robert Earl Holding, died on April 19, 2013, with a personal net worth of over $3 billion. \n\n=== Document 6: Aghajari oil field ===\nThe Aghajari oil field is an iranian oil field located in Khuzestan Province. It was discovered in 1938 and developed by National Iranian Oil Company. It began production in 1940 and produces oil. The total proven reserves of the Aghajari oil field are around 30 billion barrels (3758×10tonnes), and production is centered on 300000 oilbbl/d . The field is owned by state-owned National Iranian Oil Company (NIOC) and operated by National Iranian South Oil Company (NISOC). \n\n=== Document 7: Carabobo Field ===\nCarabobo is an oil field located in Venezuela\'s Orinoco Belt. As one of the world\'s largest accumulations of recoverable oil, the recent discoveries in the Orinoco Belt have led to Venezuela holding the world\'s largest recoverable reserves in the world, surpassing Saudi Arabia in July 2010. The Carabobo oil field is majority owned by Venezuela\'s national oil company, Petroleos de Venezuela SA (PDVSA). Owning the majority of the Orinoco Belt, and its estimated 1.18 trillion barrels of oil in place, PDVSA is now the fourth largest oil company in the world. The field is well known for its extra Heavy crude oils, having an average specific gravity between 4 and 16 °API. The Orinoco Belt holds 90% of the world\'s extra heavy crude oils, estimated at 256 billion recoverable barrels. While production is in its early development, the Carabobo field is expected to produce 400,000 barrels of oil per day. \n\n=== Document 8: Sinclair Oil Corporation ===\nSinclair Oil Corporation is an American petroleum corporation, founded by Harry F. Sinclair on May 1, 1916, as the Sinclair Oil and Refining Corporation by combining the assets of 11 small petroleum companies. Originally a New York corporation, Sinclair Oil reincorporated in Wyoming in 1976. The corporation\'s logo features the silhouette of a large green dinosaur. \n\n=== Document 9: 101 Ranch Oil Company ===\nFounded in 1908 by oil exploration pioneer E. W. Marland, The 101 Ranch Oil Company was located on the Miller Brothers 101 Ranch and headquartered in Ponca City, Oklahoma. The company’s 1911 oil discovery in North Eastern Oklahoma opened up oil development in a great region from Eastern Oklahoma west to Mervine, Newkirk, Blackwell, Billings and Garber and led to the founding of the Marland Oil Company, later renamed the Continental Oil Company, now known as Conoco. \n\n=== Document 10: Rag Sefid oil field ===\nThe Rag Sefid oil field is an oil field located in Khuzestan Province, approximately 6\xa0km in nearest distance from the Persian Gulf, southwest Iran. It was discovered in 1964 and developed by National Iranian Oil Company and began production in 1966. The total proven reserves of the Rag Sefid oil field are around 14,5 billion barrels, and production is centered on 180000 oilbbl/d . The field is owned by state-owned National Iranian Oil Company (NIOC) and operated by National Iranian South Oil Company (NISOC). \n\n\nProvide your answer based on the information in the supporting documents.'

    # Call the main function and get the answer
    answer = module.main(question)

    # Log execution completion
    with open(trace_file, 'a', encoding='utf-8') as f:
        end_entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "event": "execution_complete",
            "iteration": 1,
            "sample_id": "example_17",
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
            "iteration": 1,
            "sample_id": "example_17",
            "error": str(e),
            "traceback": traceback.format_exc()
        }
        f.write(json.dumps(error_entry) + "\n")

    print("ERROR_START")
    print(str(e))
    print(traceback.format_exc())
    print("ERROR_END")
