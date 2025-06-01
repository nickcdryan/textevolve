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
trace_file = "archive/trace_iteration_4.jsonl"
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
        "iteration": 4,
        "sample_id": "example_28",
        "question": 'Multi-hop reasoning task:\n\nQuestion: What is the middle name of the singer who recorded Would You Like to Take a Walk? with Louis Armstrong in 1951\n\nSupporting Documents:\n=== Document 1: Louis Armstrong Hot Five and Hot Seven Sessions ===\nThe Louis Armstrong Hot Five and Hot Seven Sessions were recorded between 1925 and 1928 by Louis Armstrong with his Hot Five and Hot Seven groups. According to the National Recording Registry, "Louis Armstrong was jazz\'s first great soloist and is among American music\'s most important and influential figures. These sessions, his solos in particular, set a standard musicians still strive to equal in their beauty and innovation." These recordings were added to the National Recording Registry in 2002, the first year of the institution\'s existence. \n\n=== Document 2: Ella Fitzgerald ===\nElla Jane Fitzgerald (April 25, 1917 – June 15, 1996) was an African - American jazz singer often referred to as the First Lady of Song, Queen of Jazz and Lady Ella. She was noted for her purity of tone, impeccable diction, phrasing and intonation, and a "horn-like" improvisational ability, particularly in her scat singing. \n\n=== Document 3: Louis Armstrong New Orleans International Airport ===\nLouis Armstrong New Orleans International Airport (IATA: MSY,\xa0ICAO: KMSY,\xa0FAA LID: MSY) is an international airport in Jefferson Parish, Louisiana, United States. It is owned by the city of New Orleans and is 11 miles west of downtown New Orleans. The airport\'s address is 900 Airline Drive in Kenner, Louisiana. A small portion of Runway 11/29 is in unincorporated St. Charles Parish. Armstrong International is the primary commercial airport for the New Orleans metropolitan area and southeast Louisiana. The airport was formerly known as Moisant Field, and it is also known as Louis Armstrong International Airport and New Orleans International Airport. \n\n=== Document 4: Heebie Jeebies (composition) ===\n"Heebie Jeebies" is a composition written by Boyd Atkins which achieved fame when it was recorded by Louis Armstrong in 1926. The recording on Okeh Records by Louis Armstrong and his Hot Five includes a famous example of scat singing by Armstrong. \n\n=== Document 5: Would You Like to Take a Walk? ===\n"Would You Like to Take a Walk?" is a popular song with music by Harry Warren and lyrics by Mort Dixon and Billy Rose. It appeared in the Broadway show "Sweet and Low" starring James Barton, Fannie Brice and George Jessel. The song was published in 1930 by Remick Music Corporation. The song has become a pop standard, recorded by many artists including Rudy Vallee in 1931, Annette Hanshaw in 1931 , and Bing Crosby. It plays in the 1939 Porky Pig cartoon "Naughty Neighbors" and the 1942 Daffy Duck cartoon "The Daffy Duckaroo". Ella Fitzgerald and Louis Armstrong recorded the song for Decca in 1951, accompanied by the Dave Barbour Orchestra. It was later included on Ella\'s Decca album "Ella and Her Fellas". \n\n=== Document 6: Louis Armstrong Plays W.C. Handy ===\nLouis Armstrong Plays W. C. Handy is a 1954 studio release by Louis Armstrong and His All Stars, described by Allmusic as "Louis Armstrong\'s finest record of the 1950s" and "essential music for all serious jazz collections". Columbia CD released the album on CD in 1986 in a much altered form, with alternative versions in place of many of the original songs, but restored the original with its 1997 re-issue, which also included additional tracks: a brief interview by the producer, George Avakian, with W. C. Handy; a joke told by Louis Armstrong; and several rehearsal versions of the songs. \n\n=== Document 7: Louis Armstrong and His Hot Seven ===\nLouis Armstrong and his Hot Seven was a jazz studio group organized to make a series of recordings for Okeh Records in Chicago, Illinois, in May 1927. Some of the personnel also recorded with Louis Armstrong and His Hot Five, including Johnny Dodds (clarinet), Lil Armstrong (piano), and Johnny St. Cyr (banjo and guitar). These musicians were augmented by Dodds\'s brother, Baby Dodds (drums), Pete Briggs (tuba), and John Thomas (trombone, replacing Armstrong\'s usual trombonist, Kid Ory, who was then touring with King Oliver). Briggs and Thomas were at the time working with Armstrong\'s performing group, the Sunset Stompers. \n\n=== Document 8: Saint Louis Blues (song) ===\n"Saint Louis Blues" is a popular American song composed by W. C. Handy in the blues style and published in September 1914. It remains a fundamental part of jazz musicians\' repertoire. It was also one of the first blues songs to succeed as a pop song. It has been performed by numerous musicians in various styles, including Louis Armstrong, Bessie Smith, Count Basie, Glenn Miller, Guy Lombardo, and the Boston Pops Orchestra. It has been called "the jazzman\'s "Hamlet"." The 1925 version sung by Bessie Smith, with Louis Armstrong on cornet, was inducted into the Grammy Hall of Fame in 1993. The 1929 version by Louis Armstrong & His Orchestra (with Red Allen) was inducted in 2008. \n\n=== Document 9: Potato Head Blues ===\n"Potato Head Blues" is a Louis Armstrong composition regarded as one of his finest recordings. It was made by Louis Armstrong and his Hot Seven for Okeh Records in Chicago, Illinois on May 10, 1927. It was recorded during a remarkably productive week in which Armstrong\'s usual Hot Five was temporarily expanded to seven players by the addition of tuba and drums; over five sessions the group recorded twelve sides. \n\n=== Document 10: Danny Barcelona ===\nDanny Barcelona (July 23, 1929 – April 1, 2007) was a jazz drummer best known for his years with Louis Armstrong\'s All-Stars. He was a Filipino-American born in Waipahu, a community of Honolulu, Hawaii. He was also frequently introduced to audiences by Louis Armstrong as The Little Filipino Boy. Armstrong usually followed up by calling himself "the little Arabian boy". \n\n\nProvide your answer based on the information in the supporting documents.'
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
        if frame_module == 'current_script_4':
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
            "iteration": 4,
            "sample_id": "example_28",
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
        "current_script_4", 
        "scripts/current_script_4.py"
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
    question = 'Multi-hop reasoning task:\n\nQuestion: What is the middle name of the singer who recorded Would You Like to Take a Walk? with Louis Armstrong in 1951\n\nSupporting Documents:\n=== Document 1: Louis Armstrong Hot Five and Hot Seven Sessions ===\nThe Louis Armstrong Hot Five and Hot Seven Sessions were recorded between 1925 and 1928 by Louis Armstrong with his Hot Five and Hot Seven groups. According to the National Recording Registry, "Louis Armstrong was jazz\'s first great soloist and is among American music\'s most important and influential figures. These sessions, his solos in particular, set a standard musicians still strive to equal in their beauty and innovation." These recordings were added to the National Recording Registry in 2002, the first year of the institution\'s existence. \n\n=== Document 2: Ella Fitzgerald ===\nElla Jane Fitzgerald (April 25, 1917 – June 15, 1996) was an African - American jazz singer often referred to as the First Lady of Song, Queen of Jazz and Lady Ella. She was noted for her purity of tone, impeccable diction, phrasing and intonation, and a "horn-like" improvisational ability, particularly in her scat singing. \n\n=== Document 3: Louis Armstrong New Orleans International Airport ===\nLouis Armstrong New Orleans International Airport (IATA: MSY,\xa0ICAO: KMSY,\xa0FAA LID: MSY) is an international airport in Jefferson Parish, Louisiana, United States. It is owned by the city of New Orleans and is 11 miles west of downtown New Orleans. The airport\'s address is 900 Airline Drive in Kenner, Louisiana. A small portion of Runway 11/29 is in unincorporated St. Charles Parish. Armstrong International is the primary commercial airport for the New Orleans metropolitan area and southeast Louisiana. The airport was formerly known as Moisant Field, and it is also known as Louis Armstrong International Airport and New Orleans International Airport. \n\n=== Document 4: Heebie Jeebies (composition) ===\n"Heebie Jeebies" is a composition written by Boyd Atkins which achieved fame when it was recorded by Louis Armstrong in 1926. The recording on Okeh Records by Louis Armstrong and his Hot Five includes a famous example of scat singing by Armstrong. \n\n=== Document 5: Would You Like to Take a Walk? ===\n"Would You Like to Take a Walk?" is a popular song with music by Harry Warren and lyrics by Mort Dixon and Billy Rose. It appeared in the Broadway show "Sweet and Low" starring James Barton, Fannie Brice and George Jessel. The song was published in 1930 by Remick Music Corporation. The song has become a pop standard, recorded by many artists including Rudy Vallee in 1931, Annette Hanshaw in 1931 , and Bing Crosby. It plays in the 1939 Porky Pig cartoon "Naughty Neighbors" and the 1942 Daffy Duck cartoon "The Daffy Duckaroo". Ella Fitzgerald and Louis Armstrong recorded the song for Decca in 1951, accompanied by the Dave Barbour Orchestra. It was later included on Ella\'s Decca album "Ella and Her Fellas". \n\n=== Document 6: Louis Armstrong Plays W.C. Handy ===\nLouis Armstrong Plays W. C. Handy is a 1954 studio release by Louis Armstrong and His All Stars, described by Allmusic as "Louis Armstrong\'s finest record of the 1950s" and "essential music for all serious jazz collections". Columbia CD released the album on CD in 1986 in a much altered form, with alternative versions in place of many of the original songs, but restored the original with its 1997 re-issue, which also included additional tracks: a brief interview by the producer, George Avakian, with W. C. Handy; a joke told by Louis Armstrong; and several rehearsal versions of the songs. \n\n=== Document 7: Louis Armstrong and His Hot Seven ===\nLouis Armstrong and his Hot Seven was a jazz studio group organized to make a series of recordings for Okeh Records in Chicago, Illinois, in May 1927. Some of the personnel also recorded with Louis Armstrong and His Hot Five, including Johnny Dodds (clarinet), Lil Armstrong (piano), and Johnny St. Cyr (banjo and guitar). These musicians were augmented by Dodds\'s brother, Baby Dodds (drums), Pete Briggs (tuba), and John Thomas (trombone, replacing Armstrong\'s usual trombonist, Kid Ory, who was then touring with King Oliver). Briggs and Thomas were at the time working with Armstrong\'s performing group, the Sunset Stompers. \n\n=== Document 8: Saint Louis Blues (song) ===\n"Saint Louis Blues" is a popular American song composed by W. C. Handy in the blues style and published in September 1914. It remains a fundamental part of jazz musicians\' repertoire. It was also one of the first blues songs to succeed as a pop song. It has been performed by numerous musicians in various styles, including Louis Armstrong, Bessie Smith, Count Basie, Glenn Miller, Guy Lombardo, and the Boston Pops Orchestra. It has been called "the jazzman\'s "Hamlet"." The 1925 version sung by Bessie Smith, with Louis Armstrong on cornet, was inducted into the Grammy Hall of Fame in 1993. The 1929 version by Louis Armstrong & His Orchestra (with Red Allen) was inducted in 2008. \n\n=== Document 9: Potato Head Blues ===\n"Potato Head Blues" is a Louis Armstrong composition regarded as one of his finest recordings. It was made by Louis Armstrong and his Hot Seven for Okeh Records in Chicago, Illinois on May 10, 1927. It was recorded during a remarkably productive week in which Armstrong\'s usual Hot Five was temporarily expanded to seven players by the addition of tuba and drums; over five sessions the group recorded twelve sides. \n\n=== Document 10: Danny Barcelona ===\nDanny Barcelona (July 23, 1929 – April 1, 2007) was a jazz drummer best known for his years with Louis Armstrong\'s All-Stars. He was a Filipino-American born in Waipahu, a community of Honolulu, Hawaii. He was also frequently introduced to audiences by Louis Armstrong as The Little Filipino Boy. Armstrong usually followed up by calling himself "the little Arabian boy". \n\n\nProvide your answer based on the information in the supporting documents.'

    # Call the main function and get the answer
    answer = module.main(question)

    # Log execution completion
    with open(trace_file, 'a', encoding='utf-8') as f:
        end_entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "event": "execution_complete",
            "iteration": 4,
            "sample_id": "example_28",
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
            "iteration": 4,
            "sample_id": "example_28",
            "error": str(e),
            "traceback": traceback.format_exc()
        }
        f.write(json.dumps(error_entry) + "\n")

    print("ERROR_START")
    print(str(e))
    print(traceback.format_exc())
    print("ERROR_END")
