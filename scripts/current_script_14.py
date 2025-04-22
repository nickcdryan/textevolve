import os
import re
import math

def main(question):
    """Schedules meetings by extracting constraints, generating python code to calculate time slots, and verifying if generated code is safe and valid"""
    try:
        # Step 1: Extract meeting information and constraints using LLM
        extraction_result = extract_meeting_info(question)
        if "Error" in extraction_result:
            return extraction_result

        # Step 2: Generate Python code to calculate available time slots
        code_generation_result = generate_python_code(extraction_result)
        if "Error" in code_generation_result:
            return code_generation_result
        
        # Step 3: Verify the generated Python code for safety and validity
        verification_result = verify_python_code(code_generation_result["code"], question)
        if not verification_result["is_safe"]:
            return f"Error: Generated code is unsafe: {verification_result['feedback']}"

        # Step 4: Execute the safe code to get available time slots
        available_slots = execute_python_code(code_generation_result["code"])

        # Step 5: Select the best time slot (using LLM)
        best_time = select_best_time(available_slots, question)

        return best_time

    except Exception as e:
        return f"Error processing the request: {str(e)}"

def extract_meeting_info(question):
    """Extracts meeting details (participants, duration, days, schedules) using LLM"""
    system_instruction = "You are an expert at extracting meeting details from text."
    prompt = f"""
    You are an expert at extracting the key data.
    You MUST respond with a JSON dictionary containing: participants, duration, days, schedules.

    Example 1:
    Question: Schedule a meeting for John and Mary for 30 minutes on Monday. John is busy from 9:00-10:00, Mary is busy from 11:00-12:00.
    Extraction: {{"participants": ["John", "Mary"], "duration": 30, "days": ["Monday"], "schedules": {{"John": ["9:00-10:00"], "Mary": ["11:00-12:00"]}}}}

    Question: {question}
    Extraction:
    """
    return call_llm(prompt, system_instruction)

def generate_python_code(extracted_info):
    """Generates Python code to calculate available time slots based on extracted information"""
    system_instruction = "You are an expert Python code generator for scheduling problems."
    prompt = f"""
    You are an expert at generating Python code. You are provided with extracted data and you need to produce the python code that makes those calculations.
    Generate Python code to calculate available time slots given this data. The code will have access to the extracted data.

    Example:
    Extracted Info: {{"participants": ["John", "Mary"], "duration": 30, "days": ["Monday"], "schedules": {{"John": ["9:00-10:00"], "Mary": ["11:00-12:00"]}}}}
    Code:
    participants = ["John", "Mary"]
    duration = 30
    days = ["Monday"]
    schedules = {{"John": ["9:00-10:00"], "Mary": ["11:00-12:00"]}}

    available_slots = []
    # ... (rest of the Python code to calculate available slots)

    Extracted Info: {extracted_info}
    Code:
    """
    code = call_llm(prompt, system_instruction)
    return {"code": code}

def verify_python_code(code, question):
    """Verifies the generated Python code for safety and validity (e.g., no malicious calls)"""
    system_instruction = "You are a security expert verifying python code."
    prompt = f"""
    You are a security expert that determines if Python code is safe to execute.
    Analyze this Python code and determine if it is safe and will produce the correct calculations:
    - The code must NOT have any harmful functions (e.g., os.system, etc.)
    - The code must ONLY perform calculations related to scheduling and time management
    - The code MUST avoid any external API calls

    Example 1:
    Code: print("hello")
    Is safe: Yes

    Example 2:
    Code: os.system("rm -rf /")
    Is safe: No

    Code: {code}
    Is safe:
    """
    verification_result = call_llm(prompt, system_instruction)
    is_safe = "Yes" in verification_result
    return {"is_safe": is_safe, "feedback": verification_result}

def execute_python_code(code):
    """Executes the generated Python code and returns the available time slots"""
    try:
        # Add extracted_info as a local variable for the generated code
        local_vars = {} #This will be needed to access the "extracted info"
        exec(code, globals(), local_vars)
        #After the code is executed, check if the 'available_slots' is available in the local scope
        available_slots = local_vars.get('available_slots', [])
        return available_slots
    except Exception as e:
        return f"Error executing code: {str(e)}"

def select_best_time(available_slots, question):
    """Selects the best time slot (using LLM) from the available slots"""
    system_instruction = "You are an expert at picking the best time."
    prompt = f"""
    You are an expert meeting scheduler.
    Given available time slots and the original scheduling question, pick the BEST time.

    Example:
    Question: Schedule a meeting for John and Mary for 30 minutes on Monday.
    Available slots: ['Monday 10:00-10:30', 'Monday 14:00-14:30']
    Best time: Here is the proposed time: Monday, 10:00-10:30

    Question: {question}
    Available slots: {available_slots}
    Best time:
    """
    return call_llm(prompt, system_instruction)

def call_llm(prompt, system_instruction=None):
    """Call the Gemini LLM with a prompt and return the response"""
    try:
        from google import genai
        from google.genai import types

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