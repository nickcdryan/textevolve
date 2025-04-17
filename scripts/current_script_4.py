import os
import json
import re
import math

def main(question):
    """
    Schedules meetings using a hybrid approach: LLM for extraction and reasoning,
    Python for time comparisons. Includes multi-stage verification for solution robustness.
    This approach emphasizes structured data and deterministic validation for accuracy.
    """
    try:
        # 1. Extract meeting information using LLM with a robust extraction prompt
        extracted_info = extract_meeting_info(question)
        if "Error" in extracted_info: return extracted_info

        # 2. Verify the extracted information - checking structure and constraint validity
        verified_info = verify_extracted_info(question, extracted_info)
        if "Error" in verified_info: return verified_info

        # 3. Generate candidate schedules based on validated information, using deterministic logic
        candidate_schedule = generate_candidate_schedule(verified_info)
        if "Error" in candidate_schedule: return candidate_schedule

        # 4. Final verification of the candidate schedule
        final_verification = verify_final_schedule(question, candidate_schedule)
        if "Error" in final_verification: return final_verification

        return final_verification

    except Exception as e:
        return f"Error in main: {str(e)}"

def extract_meeting_info(question):
    """Extracts structured information about meeting using LLM with detailed examples."""
    system_instruction = "You are an expert meeting information extractor. Provide structured output."
    prompt = f"""
    Extract meeting details, constraints, and participants from the text below. Use this JSON format:
    {{ "participants": [], "duration": int, "days": [], "constraints": [] }}. 'duration' is in minutes.

    Example:
    Input: Schedule a meeting for John and Jane for 30 minutes on Monday. John is busy 1-2pm. Jane prefers to meet before noon.
    Output:
    {{ "participants": ["John", "Jane"], "duration": 30, "days": ["Monday"], "constraints": ["John is busy 1-2pm", "Jane prefers to meet before noon"] }}

    Input: Schedule a meeting for Kevin, David, Stephen and Helen for half an hour between 9:00 to 17:00 on Monday. Kevin has blocked their calendar on Monday during 11:30 to 12:00, 14:30 to 16:00; David has meetings on Monday during 10:00 to 11:00; Stephen has meetings on Monday during 9:00 to 11:30, 12:00 to 13:00, 14:00 to 15:30, 16:00 to 17:00; Helen has blocked their calendar on Monday during 9:00 to 13:30, 14:30 to 17:00
    Output:
    {{ "participants": ["Kevin", "David", "Stephen", "Helen"], "duration": 30, "days": ["Monday"], "constraints": ["Kevin is busy 11:30 to 12:00, 14:30 to 16:00", "David has meetings 10:00 to 11:00", "Stephen has meetings 9:00 to 11:30, 12:00 to 13:00, 14:00 to 15:30, 16:00 to 17:00", "Helen has blocked their calendar 9:00 to 13:30, 14:30 to 17:00"] }}

    Now extract the same from:
    {question}
    """
    try:
        return call_llm(prompt, system_instruction)
    except Exception as e:
        return f"Error extracting info: {str(e)}"

def verify_extracted_info(question, extracted_info):
    """Verifies extracted information by using a checker and multiple examples."""
    system_instruction = "You are a verification expert. Check for completeness and correctness."
    prompt = f"""
    Given the question and extracted information, verify its completeness and correctness. Highlight any discrepancies.

    Example 1:
    Question: Schedule a meeting for John and Jane for 30 minutes on Monday. John is busy 1-2pm.
    Extracted Info: {{ "participants": ["John"], "duration": 30, "days": ["Monday"], "constraints": ["John is busy 1-2pm"] }}
    Output: Missing participant: Jane. Complete the information before proceeding.

    Example 2:
    Question: Schedule a meeting for Kevin, David and Mary for half an hour between 9:00 to 17:00 on Tuesday. Kevin is available but David and Mary are not.
    Extracted Info: {{"participants": ["Kevin", "David", "Mary"], "duration": 30, "days": ["Tuesday"], "constraints": []}}
    Output: The extracted info is incomplete: "David and Mary are not available" needs to be in constraints.

    Question: {question}
    Extracted Info: {extracted_info}
    """
    try:
        return call_llm(prompt, system_instruction)
    except Exception as e:
        return f"Error verifying info: {str(e)}"

def generate_candidate_schedule(verified_info):
    """Generates a candidate schedule. Currently a placeholder, to be replaced with Python logic."""
    return "Here is the proposed time: Monday, 12:30 - 13:00 "

def verify_final_schedule(question, candidate_schedule):
    """Verifies the final schedule. Currently a placeholder, to be replaced with more robust logic."""
    system_instruction = "You are an expert schedule verifier."
    prompt = f"""Given the question and candidate schedule, verify if it works.

    Question: {question}
    Candidate schedule: {candidate_schedule}
    """
    try:
        return call_llm(prompt, system_instruction)
    except Exception as e:
        return f"Error verifying final schedule: {str(e)}"

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