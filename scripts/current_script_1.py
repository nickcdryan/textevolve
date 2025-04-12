import os
import json

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

def extract_information_with_examples(problem):
    """Extract key information from the problem statement using embedded examples and chain-of-thought."""
    system_instruction = "You are an information extraction specialist. Extract information in JSON format."
    
    prompt = f"""
    Extract key information from this scheduling problem statement in JSON format.
    
    Example:
    Question: You need to schedule a meeting for John and Jennifer for half an hour between 9:00 to 17:00 on Monday. John is busy 10:00-11:00. Jennifer is busy 14:00-15:00.
    
    Reasoning:
    First, identify the participants: John and Jennifer.
    Second, identify the duration: half an hour.
    Third, identify the time constraints: 9:00 to 17:00 on Monday.
    Fourth, identify John's busy slots: 10:00-11:00.
    Fifth, identify Jennifer's busy slots: 14:00-15:00.
    
    Extracted Information:
    {{
      "participants": ["John", "Jennifer"],
      "duration": "30 minutes",
      "day": "Monday",
      "valid_hours": "9:00-17:00",
      "John": ["10:00-11:00"],
      "Jennifer": ["14:00-15:00"]
    }}
    
    Now, extract information from this new problem:
    {problem}
    """
    
    return call_llm(prompt, system_instruction)

def find_available_time_with_examples(extracted_info):
    """Find an available time slot given the extracted information using examples and chain-of-thought."""
    system_instruction = "You are a scheduling assistant. Find an available time."
    
    prompt = f"""
    Find an available time slot that satisfies the constraints.
    
    Example:
    Extracted Information:
    {{
      "participants": ["John", "Jennifer"],
      "duration": "30 minutes",
      "day": "Monday",
      "valid_hours": "9:00-17:00",
      "John": ["10:00-11:00"],
      "Jennifer": ["14:00-15:00"]
    }}
    
    Reasoning:
    The meeting has to be on Monday between 9:00 and 17:00 and last 30 minutes.
    John is busy from 10:00-11:00.
    Jennifer is busy from 14:00-15:00.
    Therefore, a possible time is 9:00-9:30.
    
    Proposed Time: Monday, 9:00-9:30
    
    Now, find the available time for the following:
    {extracted_info}
    """
    
    return call_llm(prompt, system_instruction)

def verify_solution_with_examples(problem, proposed_solution):
    """Verify if the proposed solution satisfies all constraints using embedded examples and chain-of-thought."""
    system_instruction = "You are a solution verification expert. Check all constraints."
    
    prompt = f"""
    Verify if this proposed solution satisfies all constraints in the problem.
    
    Example:
    Problem: You need to schedule a meeting for John and Jennifer for half an hour between 9:00 to 17:00 on Monday. John is busy 10:00-11:00. Jennifer is busy 14:00-15:00.
    Proposed Solution: Monday, 9:00-9:30
    
    Reasoning:
    1. Duration: 30 minutes -> Correct
    2. Time: 9:00-9:30 is between 9:00 and 17:00 -> Correct
    3. John: 9:00-9:30 does not overlap with 10:00-11:00 -> Correct
    4. Jennifer: 9:00-9:30 does not overlap with 14:00-15:00 -> Correct
    
    Result: Valid
    
    Now, verify the following solution:
    Problem: {problem}
    Proposed Solution: {proposed_solution}
    """
    
    return call_llm(prompt, system_instruction)

def main(question):
    """Main function to schedule meetings."""
    try:
        extracted_info = extract_information_with_examples(question)
        proposed_solution = find_available_time_with_examples(extracted_info)
        verification_result = verify_solution_with_examples(question, proposed_solution)

        # Basic error check - could be improved
        if "Error" in extracted_info or "Error" in proposed_solution or "Error" in verification_result:
            return "Error during analysis."

        return proposed_solution
    except Exception as e:
        print(f"An error occurred: {e}")
        return "Error during analysis."