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

def extract_info_with_cot(problem):
    """Extracts information using chain-of-thought and embedded examples."""
    system_instruction = "You are an expert meeting scheduler. Extract relevant info."
    prompt = f"""
    Extract key information from the problem. Include constraints and participant availability.
    Example:
    Question: Schedule a meeting for A and B for 30 min. A is free all day. B is busy 9-10.
    Let's think step by step:
    Participants: A, B
    Duration: 30 min
    Availability: A: All day, B: Not 9-10
    Output: {{"participants": ["A", "B"], "duration": "30 minutes", "A": "All day", "B": "Not 9-10"}}

    Now, extract info from: {problem}
    """
    return call_llm(prompt, system_instruction)

def find_solution_with_cot(extracted_info):
    """Finds a solution using chain-of-thought, given extracted information."""
    system_instruction = "You are an expert meeting scheduler. Find a valid meeting time."
    prompt = f"""
    Given the following information, find a valid meeting time.
    Example:
    Info: {{"participants": ["A", "B"], "duration": "30 minutes", "A": "All day", "B": "Not 9-10"}}
    Let's think step by step:
    A is free. B is not free 9-10. So 10-10:30 works.
    Output: 10:00 - 10:30

    Now, find a solution given: {extracted_info}
    """
    return call_llm(prompt, system_instruction)

def verify_solution_with_cot(extracted_info, solution):
    """Verifies a solution using chain-of-thought."""
    system_instruction = "You are an expert verifier. Verify the solution."
    prompt = f"""
    Verify if the solution is valid given the information.
    Example:
    Info: {{"participants": ["A", "B"], "duration": "30 minutes", "A": "All day", "B": "Not 9-10"}}
    Solution: 10:00 - 10:30
    Let's think step by step:
    A is free. B is free 10-10:30. So the solution is valid.
    Output: Valid

    Now, verify: Info: {extracted_info}, Solution: {solution}
    """
    return call_llm(prompt, system_instruction)

def main(question):
    """Main function to schedule a meeting."""
    try:
        extracted_info = extract_info_with_cot(question)
        solution = find_solution_with_cot(extracted_info)
        verification = verify_solution_with_cot(extracted_info, solution)

        if "Valid" in verification:
            return f"Here is the proposed time: {solution}"
        else:
            return "No valid solution found."
    except Exception as e:
        return f"Error: {str(e)}"