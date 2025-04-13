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

def extract_information(problem):
    """Extract key information, including participants, duration, and availability."""
    system_instruction = "You are an expert information extractor for scheduling problems."
    prompt = f"""
    Extract the key information from the problem, including participants, duration, valid hours, valid days, availability, and preferences.

    Example:
    Problem: You need to schedule a meeting for John and Mary for 1 hour between 9:00 and 17:00 on Monday or Tuesday. John is free all day. Mary is busy on Monday from 10:00 to 12:00 and on Tuesday from 14:00 to 16:00.
    Extracted Information:
    {{
        "participants": ["John", "Mary"],
        "duration": "1 hour",
        "valid_hours": "9:00-17:00",
        "valid_days": ["Monday", "Tuesday"],
        "availability": {{
            "John": "Free all day",
            "Mary": {{
                "Monday": ["9:00-10:00", "12:00-17:00"],
                "Tuesday": ["9:00-14:00", "16:00-17:00"]
            }}
        }},
        "preferences": {{}}
    }}

    Problem: {problem}
    Extracted Information:
    """
    return call_llm(prompt, system_instruction)

def find_available_time(extracted_info_str):
    """Find an available time slot based on the extracted information."""
    system_instruction = "You are an expert meeting scheduler."
    prompt = f"""
    Given the following extracted information about a scheduling problem, find a suitable time slot.
    Extracted Information:
    {extracted_info_str}

    Example:
    Extracted Information:
    {{
        "participants": ["John", "Mary"],
        "duration": "1 hour",
        "valid_hours": "9:00-17:00",
        "valid_days": ["Monday", "Tuesday"],
        "availability": {{
            "John": "Free all day",
            "Mary": {{
                "Monday": ["9:00-10:00", "12:00-17:00"],
                "Tuesday": ["9:00-14:00", "16:00-17:00"]
            }}
        }},
        "preferences": {{}}
    }}
    Solution: Monday, 12:00 - 13:00

    Solution:
    """
    return call_llm(prompt, system_instruction)

def verify_solution(problem, extracted_info_str, proposed_solution):
    """Verify if the proposed solution satisfies all constraints."""
    system_instruction = "You are a meticulous solution verifier."
    prompt = f"""
    Problem: {problem}
    Extracted Information:
    {extracted_info_str}
    Proposed Solution: {proposed_solution}

    Verify that the proposed solution adheres to all constraints outlined in the problem and extracted information. Check for conflicts in availability, adherence to valid days and hours, and satisfaction of any preferences.

    Example:
    Problem: Schedule John and Mary for 1 hour between 9:00 and 17:00 on Monday or Tuesday. John is free all day. Mary is busy on Monday from 10:00 to 12:00 and on Tuesday from 14:00 to 16:00.
    Extracted Information:
    {{
        "participants": ["John", "Mary"],
        "duration": "1 hour",
        "valid_hours": "9:00-17:00",
        "valid_days": ["Monday", "Tuesday"],
        "availability": {{
            "John": "Free all day",
            "Mary": {{
                "Monday": ["9:00-10:00", "12:00-17:00"],
                "Tuesday": ["9:00-14:00", "16:00-17:00"]
            }}
        }},
        "preferences": {{}}
    }}
    Proposed Solution: Monday, 11:00 - 12:00
    Verification Result: Valid

    Verification Result:
    """
    return call_llm(prompt, system_instruction)

def main(question):
    """Main function to schedule a meeting."""
    try:
        # 1. Extract information
        extracted_info_str = extract_information(question)
        print(f"Extracted Info: {extracted_info_str}")

        # 2. Find an available time
        proposed_solution = find_available_time(extracted_info_str)
        print(f"Proposed Solution: {proposed_solution}")

        # 3. Verify the solution
        verification_result = verify_solution(question, extracted_info_str, proposed_solution)
        print(f"Verification Result: {verification_result}")

        # 4. Return the proposed solution if valid, otherwise return an error message
        if "Valid" in verification_result:
            return proposed_solution
        else:
            return "Error: No valid time found."

    except Exception as e:
        return f"Error: {str(e)}"

# Example usage:
if __name__ == "__main__":
    question = "You need to schedule a meeting for Nicholas, Sara, Helen, Brian, Nancy, Kelly and Judy for half an hour between the work hours of 9:00 to 17:00 on Monday. \n\nHere are the existing schedules for everyone during the day: \nNicholas is busy on Monday during 9:00 to 9:30, 11:00 to 11:30, 12:30 to 13:00, 15:30 to 16:00; \nSara is busy on Monday during 10:00 to 10:30, 11:00 to 11:30; \nHelen is free the entire day.\nBrian is free the entire day.\nNancy has blocked their calendar on Monday during 9:00 to 10:00, 11:00 to 14:00, 15:00 to 17:00; \nKelly is busy on Monday during 10:00 to 11:30, 12:00 to 12:30, 13:30 to 14:00, 14:30 to 15:30, 16:30 to 17:00; \nJudy has blocked their calendar on Monday during 9:00 to 11:30, 12:00 to 12:30, 13:00 to 13:30, 14:30 to 17:00; \n\nFind a time that works for everyone's schedule and constraints."
    answer = main(question)
    print(f"Answer: {answer}")