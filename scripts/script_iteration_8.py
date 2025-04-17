import os
import json
import re
import math

def main(question):
    """
    Schedules meetings by extracting structured data, generating and validating solutions using LLMs with a verification loop.
    This approach emphasizes structured extraction and validation, differing from previous methods.
    """
    try:
        # 1. Extract structured data with examples in the LLM prompt
        extracted_data = extract_structured_data(question)
        if "Error" in extracted_data: return extracted_data
        extracted_data_json = json.loads(extracted_data)

        # 2. Propose and validate solution with verification loop
        proposed_solution = propose_and_validate_solution(question, extracted_data_json)
        if "Error" in proposed_solution: return proposed_solution

        return proposed_solution

    except Exception as e:
        return f"Error in main: {str(e)}"

def extract_structured_data(question):
    """Extracts structured data using LLM with a complete example."""
    system_instruction = "You are a structured data extraction agent."
    prompt = f"""
    Extract the relevant information and represent it in a structured JSON format. 
    Consider participants, duration, days, existing schedules, and preferences.
    Example:
    Input: Schedule a meeting for John and Jane for 30 minutes on Monday. John is busy 1-2pm. Jane prefers to meet before noon.
    Reasoning: Participants: John, Jane. Duration: 30. Days: Monday. John's availability: Not 1-2pm. Jane's preference: Before noon.
    Output: {{"participants": ["John", "Jane"], "duration": 30, "days": ["Monday"], "john_busy": ["1:00-2:00"], "jane_preference": "before noon"}}

    Input: {question}
    """
    try:
        return call_llm(prompt, system_instruction)
    except Exception as e:
        return f"Error extracting structured data: {str(e)}"

def propose_and_validate_solution(question, extracted_data, max_attempts=3):
    """Proposes a solution and validates it in a loop using LLM."""
    for attempt in range(max_attempts):
        system_instruction = "You are an expert meeting scheduler."
        prompt = f"""
        Propose a specific meeting time and validate that time against the following: 
        1. Is the time within work hours (9:00 - 17:00)?
        2. Does the time adhere to existing schedules?
        3. Does the time adhere to any preferences?
        Problem: {question}
        Extracted Data: {extracted_data}

        Example:
        Question: Schedule a meeting for John and Jane for 30 minutes on Monday. John is busy 1-2pm. Jane prefers to meet before noon.
        Extracted Data: {{"participants": ["John", "Jane"], "duration": 30, "days": ["Monday"], "john_busy": ["1:00-2:00"], "jane_preference": "before noon"}}
        Reasoning: Since John is busy from 1-2pm and Jane prefers before noon, a good solution is 9:00-9:30.
        Here is the proposed time: Monday, 9:00 - 9:30

        Now generate a solution for:
        Question: {question}
        Extracted Data: {extracted_data}
        """

        try:
            proposed_time = call_llm(prompt, system_instruction)
        except Exception as e:
            return f"Error proposing solution: {str(e)}"

        #Implement feedback loop here
        system_instruction_verifier = "You are an expert meeting time verifier."
        verification_prompt = f"""
        You are a meeting scheduler. Verify if the proposed time works for everyone.
        Problem: {question}
        Extracted Data: {extracted_data}
        Proposed Time: {proposed_time}
        Example:
        Question: Schedule a meeting for John and Jane for 30 minutes on Monday. John is busy 1-2pm. Jane prefers to meet before noon.
        Extracted Data: {{"participants": ["John", "Jane"], "duration": 30, "days": ["Monday"], "john_busy": ["1:00-2:00"], "jane_preference": "before noon"}}
        Proposed Time: Here is the proposed time: Monday, 1:30 - 2:00
        Output: Invalid - John is busy.

        Question: {question}
        Extracted Data: {extracted_data}
        Proposed Time: {proposed_time}
        """

        try:
            verification = call_llm(verification_prompt, system_instruction_verifier)
        except Exception as e:
            return f"Error verifying solution: {str(e)}"

        if "Invalid" not in verification:
            return proposed_time #Valid

    return "Could not find valid schedule" #After max attempts

def call_llm(prompt, system_instruction=None):
    """Call the Gemini LLM with a prompt and return the response."""
    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

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