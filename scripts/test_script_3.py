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
trace_file = "archive/trace_iteration_3.jsonl"
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
        "iteration": 3,
        "sample_id": "example_25",
        "question": 'Multi-hop reasoning task:\n\nQuestion: What company produced the 1978 movie based on a book written by a radio playwright and children\'s book author born in 1900?\n\nSupporting Documents:\n=== Document 1: An Na ===\nAn Na (born 1972) is a South Korea-born American children\'s book author. Starting her career as a middle school English and History teacher, Na turned to writing novels after taking a young adult literature class while enrolled in an M.F.A. program at Vermont College of Fine Arts. She gained success with her very first novel "A Step From Heaven", published by Front Street Press in 2001, which won the annual Michael L. Printz Award from the American Library Association recognizing the year\'s "best book written for teens, based entirely on its literary merit". It was also a finalist for the National Book Award, Young People\'s Literature, and later found its way onto numerous "best book" lists. Na still makes frequent visits to middle schools to talk about her works and encourages young Asian-American students to become artists and harness their creativity. She cites Frank McCourt\'s "Angela\'s Ashes" and Sandra Cisneros\'s "The House on Mango Street" among the influences on her writing and also admires the work of Madeleine L\'Engle and of her first writing teacher, Jacqueline Woodson. She divides her time between Oakland, California and Warren, Vermont. \n\n=== Document 2: Charles Tazewell ===\nCharles Tazewell (June 2, 1900 – June 26, 1972) was a radio playwright and children\'s book author, whose work has been adapted multiple times for film. \n\n=== Document 3: Hank Zipzer\'s Christmas Catastrophe ===\nHank Zipzer\'s Christmas Catastrophe is a 2016 stand alone British Christmas movie based on the Hank Zipzer series of books by Henry Winkler and Lin Oliver and the TV series airing on CBBC. The film will be airing on CBBC on 12 December 2016. It is written by Joe Williams and is directed by Matt Bloom. The film is produced by Kindle Entertainment in association with Walker Productions and DHX Media with support from Screen Yorkshire’s Yorkshire Content Fund. It is the fourth movie based on a CBBC programme after "", "Shaun the Sheep Movie" and "". It is the second movie based on a CBBC show, which has not been released in cinemas and only shown on TV after "" \n\n=== Document 4: Kraft Suspense Theatre ===\nThe Kraft Suspense Theatre is an American television anthology series that was produced and broadcast from 1963 to 1965 on NBC. Sponsored by Kraft Foods, it was seen three weeks out of every four and was pre-empted for Perry Como\'s "Kraft Music Hall" specials once monthly. Como\'s production company, Roncom Films, also produced "Kraft Suspense Theatre." (The company name, "Roncom Films" stood for "RONnie COMo," Perry\'s son, who was in his early twenties when this series premiered). Writer, editor, critic and radio playwright Anthony Boucher served as consultant on the series. \n\n=== Document 5: The Small One ===\nThe Small One is a 1978 American animated featurette produced by Walt Disney Productions and released theatrically by Buena Vista Distribution on December 16, 1978 with a Christmas 1978 re-issue of "Pinocchio". The story is based on a children\'s book of the same name by Charles Tazewell and was an experiment for the new generation of Disney animators including Don Bluth, Richard Rich, Henry Selick, Gary Goldman and John Pomeroy. \n\n=== Document 6: The Face on the Milk Carton (film) ===\nThe Face on the Milk Carton is a 1995 made for television movie based on the book written by Caroline B. Cooney. The movie stars Kellie Martin as Jennifer Sands/Janie Jessmon, a 16-year-old girl who finds her face on the back of a milk carton and puts the pieces of her past together. \n\n=== Document 7: Pichilemu Blues ===\nPichilemu Blues is a 1993 book written by Chilean politician Esteban Valenzuela. A movie based on the book was also released, starring Peggy Cordero, Ximena Nogueira and Evaristo Acevedo. \n\n=== Document 8: Gábor Nógrádi ===\nGábor Nógrádi (born June 22, 1947, Nyíregyháza) is a Hungarian book author, screenwriter, playwright, essayist, publicist and poet who is best known for his children\'s novels such as the "Pigeon granny" and "The story of" "Pie ("original title PetePite")", a book which won the 2002 Children\'s Book of the Year award, was on the IBBY Honor List (International Board for Young People) and was ranked among the 100 most popular books in Hungary in the 2005 \'Big Book\' competition. \n\n=== Document 9: Randy Romero ===\nRandy Paul Romero (born December 22, 1957 in Erath, Louisiana) is a Hall of Fame jockey in the sport of Thoroughbred horse racing, Born into a family involved with horses, his father Lloyd J. Romero was a Louisiana state trooper who trained American Quarter Horses and later, after a drunk driver crashed into his police car and permanently disabled him, he began training Thoroughbreds for flat racing. The 1978 movie "Casey\'s Shadow" is based on Lloyd Romero and his family. He was elected into the Thoroughbred Racing Hall of Fame May 27, 2010. \n\n=== Document 10: Wilbooks ===\nWilbooks is a children’s book educational publishing company based in West Chester, Pennsylvania. The company was founded by children’s book author Bruce Larkin in 1996. The company publishes fiction, non-fiction, humor, and poetry books geared towards children from Pre-kindergarten to third grade. Wilbooks publishes leveled, educational books with a focus on teaching children how to read. In 2009 Wilbooks (through Bruce Larkin) donated over 500,000 books to schools, teachers, and literacy organizations throughout the United States. \n\n\nProvide your answer based on the information in the supporting documents.'
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
        if frame_module == 'current_script_3':
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
            "iteration": 3,
            "sample_id": "example_25",
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
        "current_script_3", 
        "scripts/current_script_3.py"
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
    question = 'Multi-hop reasoning task:\n\nQuestion: What company produced the 1978 movie based on a book written by a radio playwright and children\'s book author born in 1900?\n\nSupporting Documents:\n=== Document 1: An Na ===\nAn Na (born 1972) is a South Korea-born American children\'s book author. Starting her career as a middle school English and History teacher, Na turned to writing novels after taking a young adult literature class while enrolled in an M.F.A. program at Vermont College of Fine Arts. She gained success with her very first novel "A Step From Heaven", published by Front Street Press in 2001, which won the annual Michael L. Printz Award from the American Library Association recognizing the year\'s "best book written for teens, based entirely on its literary merit". It was also a finalist for the National Book Award, Young People\'s Literature, and later found its way onto numerous "best book" lists. Na still makes frequent visits to middle schools to talk about her works and encourages young Asian-American students to become artists and harness their creativity. She cites Frank McCourt\'s "Angela\'s Ashes" and Sandra Cisneros\'s "The House on Mango Street" among the influences on her writing and also admires the work of Madeleine L\'Engle and of her first writing teacher, Jacqueline Woodson. She divides her time between Oakland, California and Warren, Vermont. \n\n=== Document 2: Charles Tazewell ===\nCharles Tazewell (June 2, 1900 – June 26, 1972) was a radio playwright and children\'s book author, whose work has been adapted multiple times for film. \n\n=== Document 3: Hank Zipzer\'s Christmas Catastrophe ===\nHank Zipzer\'s Christmas Catastrophe is a 2016 stand alone British Christmas movie based on the Hank Zipzer series of books by Henry Winkler and Lin Oliver and the TV series airing on CBBC. The film will be airing on CBBC on 12 December 2016. It is written by Joe Williams and is directed by Matt Bloom. The film is produced by Kindle Entertainment in association with Walker Productions and DHX Media with support from Screen Yorkshire’s Yorkshire Content Fund. It is the fourth movie based on a CBBC programme after "", "Shaun the Sheep Movie" and "". It is the second movie based on a CBBC show, which has not been released in cinemas and only shown on TV after "" \n\n=== Document 4: Kraft Suspense Theatre ===\nThe Kraft Suspense Theatre is an American television anthology series that was produced and broadcast from 1963 to 1965 on NBC. Sponsored by Kraft Foods, it was seen three weeks out of every four and was pre-empted for Perry Como\'s "Kraft Music Hall" specials once monthly. Como\'s production company, Roncom Films, also produced "Kraft Suspense Theatre." (The company name, "Roncom Films" stood for "RONnie COMo," Perry\'s son, who was in his early twenties when this series premiered). Writer, editor, critic and radio playwright Anthony Boucher served as consultant on the series. \n\n=== Document 5: The Small One ===\nThe Small One is a 1978 American animated featurette produced by Walt Disney Productions and released theatrically by Buena Vista Distribution on December 16, 1978 with a Christmas 1978 re-issue of "Pinocchio". The story is based on a children\'s book of the same name by Charles Tazewell and was an experiment for the new generation of Disney animators including Don Bluth, Richard Rich, Henry Selick, Gary Goldman and John Pomeroy. \n\n=== Document 6: The Face on the Milk Carton (film) ===\nThe Face on the Milk Carton is a 1995 made for television movie based on the book written by Caroline B. Cooney. The movie stars Kellie Martin as Jennifer Sands/Janie Jessmon, a 16-year-old girl who finds her face on the back of a milk carton and puts the pieces of her past together. \n\n=== Document 7: Pichilemu Blues ===\nPichilemu Blues is a 1993 book written by Chilean politician Esteban Valenzuela. A movie based on the book was also released, starring Peggy Cordero, Ximena Nogueira and Evaristo Acevedo. \n\n=== Document 8: Gábor Nógrádi ===\nGábor Nógrádi (born June 22, 1947, Nyíregyháza) is a Hungarian book author, screenwriter, playwright, essayist, publicist and poet who is best known for his children\'s novels such as the "Pigeon granny" and "The story of" "Pie ("original title PetePite")", a book which won the 2002 Children\'s Book of the Year award, was on the IBBY Honor List (International Board for Young People) and was ranked among the 100 most popular books in Hungary in the 2005 \'Big Book\' competition. \n\n=== Document 9: Randy Romero ===\nRandy Paul Romero (born December 22, 1957 in Erath, Louisiana) is a Hall of Fame jockey in the sport of Thoroughbred horse racing, Born into a family involved with horses, his father Lloyd J. Romero was a Louisiana state trooper who trained American Quarter Horses and later, after a drunk driver crashed into his police car and permanently disabled him, he began training Thoroughbreds for flat racing. The 1978 movie "Casey\'s Shadow" is based on Lloyd Romero and his family. He was elected into the Thoroughbred Racing Hall of Fame May 27, 2010. \n\n=== Document 10: Wilbooks ===\nWilbooks is a children’s book educational publishing company based in West Chester, Pennsylvania. The company was founded by children’s book author Bruce Larkin in 1996. The company publishes fiction, non-fiction, humor, and poetry books geared towards children from Pre-kindergarten to third grade. Wilbooks publishes leveled, educational books with a focus on teaching children how to read. In 2009 Wilbooks (through Bruce Larkin) donated over 500,000 books to schools, teachers, and literacy organizations throughout the United States. \n\n\nProvide your answer based on the information in the supporting documents.'

    # Call the main function and get the answer
    answer = module.main(question)

    # Log execution completion
    with open(trace_file, 'a', encoding='utf-8') as f:
        end_entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "event": "execution_complete",
            "iteration": 3,
            "sample_id": "example_25",
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
            "iteration": 3,
            "sample_id": "example_25",
            "error": str(e),
            "traceback": traceback.format_exc()
        }
        f.write(json.dumps(error_entry) + "\n")

    print("ERROR_START")
    print(str(e))
    print(traceback.format_exc())
    print("ERROR_END")
