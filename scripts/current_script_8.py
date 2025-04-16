import os
import re
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

def extract_constraints_with_examples(text):
    """Extract scheduling constraints from the input text using LLM and examples."""
    system_instruction = "You are an expert in extracting scheduling constraints."
    prompt = f"""
    Extract all scheduling constraints from the text.

    Example Input:
    You need to schedule a meeting for Nicholas, Sara, and Helen for half an hour between 9:00 to 17:00 on Monday.
    Nicholas is busy on Monday during 9:00 to 9:30, 11:00 to 11:30, 12:30 to 13:00, 15:30 to 16:00;
    Sara is busy on Monday during 10:00 to 10:30, 11:00 to 11:30;
    Helen is free the entire day.

    Reasoning:
    1. Participants: Identify all participants (Nicholas, Sara, Helen)
    2. Duration: Identify the meeting duration (30 minutes)
    3. Time Range: Identify the possible time range (9:00 to 17:00)
    4. Day: Identify the day (Monday)
    5. Conflicts: Extract the busy times for each participant. Represent "free the entire day" as an empty list. Ensure times are in 24:00 format.

    Extracted Constraints:
    {{
        "participants": ["Nicholas", "Sara", "Helen"],
        "duration": "30 minutes",
        "available_time_range": ["09:00", "17:00"],
        "day": "Monday",
        "conflicts": {{
            "Nicholas": ["09:00-09:30", "11:00-11:30", "12:30-13:00", "15:30-16:00"],
            "Sara": ["10:00-10:30", "11:00-11:30"],
            "Helen": []
        }}
    }}

    Now, extract constraints from this new text:
    {text}
    """
    return call_llm(prompt, system_instruction)

def verify_extracted_constraints(constraints_json, original_text):
    """Verify the extracted constraints using LLM, comparing against original text to ensure correctness."""
    system_instruction = "You are an expert in verifying extracted scheduling constraints against the original text."
    prompt = f"""
    You are given a JSON of extracted scheduling constraints and the original text. Verify the data for correctness, completeness and consistency by comparing it to the original text.

    Example Input:
    Original Text:
    Schedule a meeting for John and Mary for one hour between 9:00 and 17:00 on Tuesday.
    John is busy on Tuesday from 10:00 to 11:00 and 14:00 to 15:00. Mary is free all day.

    Extracted Constraints:
    {{
        "participants": ["John", "Mary"],
        "duration": "1 hour",
        "available_time_range": ["09:00", "17:00"],
        "day": "Tuesday",
        "conflicts": {{
            "John": ["10:00-11:00", "14:00-15:00"],
            "Mary": []
        }}
    }}

    Reasoning:
    1. Participants: Verify if all participants listed in the extracted constraints match those in the original text.
    2. Duration: Ensure the duration is correctly specified and matches the original text.
    3. Time Range: Confirm the available time range is valid and matches the original text.
    4. Day: Verify the day is correctly identified and matches the original text.
    5. Conflicts: Ensure all conflicts are accurately listed for each participant by comparing to their schedules in the original text. Pay close attention to "free the entire day" being represented by an empty list.
    6. Time Format: All times should be in HH:MM format.
    7. Completeness: Confirm that all constraints mentioned in the original text are captured in the extracted constraints.

    Verification Result:
    VALID: The extracted constraints are complete and consistent with the original text.

    Now, verify these constraints using the original text:
    Original Text: {original_text}
    Extracted Constraints: {constraints_json}
    """
    return call_llm(prompt, system_instruction)

def find_available_times_with_examples(constraints_json):
    """Find available meeting times based on extracted constraints using LLM."""
    system_instruction = "You are an expert in finding available meeting times."
    prompt = f"""
    Given these scheduling constraints, find a suitable meeting time.

    Example Input:
    {{
        "participants": ["Nicholas", "Sara", "Helen"],
        "duration": "30 minutes",
        "available_time_range": ["09:00", "17:00"],
        "day": "Monday",
        "conflicts": {{
            "Nicholas": ["09:00-09:30", "11:00-11:30", "12:30-13:00", "15:30-16:00"],
            "Sara": ["10:00-10:30", "11:00-11:30"],
            "Helen": []
        }}
    }}

    Reasoning:
    1. Parse conflicts: Extract the busy time slots for each participant.
    2. Iterate Time: Iterate through possible time slots within the available time range. Granularity is 30 minutes.
    3. Check conflicts: Check if the current time slot conflicts with any participant's schedule.
    4. Find available time: Output a time that doesn't conflict with any participants.

    Available Time:
    Monday, 14:00 - 14:30

    Now, find the available time based on these constraints:
    {constraints_json}
    """
    return call_llm(prompt, system_instruction)

def verify_solution_with_examples(problem, proposed_solution):
    """Verify the proposed meeting time with LLM and example."""
    system_instruction = "You are a critical evaluator verifying meeting schedule solutions."
    prompt = f"""
    Verify if the proposed meeting time satisfies all constraints.

    Example Input:
    Problem: Schedule a meeting for Nicholas, Sara, and Helen for half an hour between 9:00 to 17:00 on Monday.
    Nicholas is busy on Monday during 9:00 to 9:30, 11:00 to 11:30, 12:30 to 13:00, 15:30 to 16:00;
    Sara is busy on Monday during 10:00 to 10:30, 11:00 to 11:30;
    Helen is free the entire day.
    Proposed Solution: Monday, 14:00 - 14:30

    Reasoning:
    1. Parse participants: Identify participants (Nicholas, Sara, Helen).
    2. Check conflicts: Ensure the time slot doesn't conflict with anyone's schedule. Granularity is 30 minutes.
    3. Validate time range: Ensure the time is within the given range.

    Verification Result:
    VALID: The proposed time does not conflict with any participant's schedule and is within the specified time range.

    Now, verify this new solution:
    Problem: {problem}
    Proposed Solution: {proposed_solution}
    """
    return call_llm(prompt, system_instruction)

def main(question):
    """Main function to schedule a meeting."""
    try:
        # Extract scheduling constraints
        constraints_json = extract_constraints_with_examples(question)
        
        # Verify extracted constraints
        verification_result = verify_extracted_constraints(constraints_json, question) # Pass the original question
        if "INVALID" in verification_result:
            return "Could not find a valid meeting time due to constraint extraction error."
        
        # Find available time
        available_time = find_available_times_with_examples(constraints_json)
        
        # Verify solution
        final_verification_result = verify_solution_with_examples(question, available_time)
        
        if "VALID" in final_verification_result:
            return "Here is the proposed time: " + available_time
        else:
            return "Could not find a valid meeting time."
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return "Error occurred while scheduling."

# Example usage:
if __name__ == "__main__":
    question = "You need to schedule a meeting for Nicholas, Sara, Helen, Brian, Nancy, Kelly and Judy for half an hour between the work hours of 9:00 to 17:00 on Monday. \n\nHere are the existing schedules for everyone during the day: \nNicholas is busy on Monday during 9:00 to 9:30, 11:00 to 11:30, 12:30 to 13:00, 15:30 to 16:00; \nSara is busy on Monday during 10:00 to 10:30, 11:00 to 11:30; \nHelen is free the entire day.\nBrian is free the entire day.\nNancy has blocked their calendar on Monday during 9:00 to 10:00, 11:00 to 14:00, 15:00 to 17:00; \nKelly is busy on Monday during 10:00 to 11:30, 12:00 to 12:30, 13:30 to 14:00, 14:30 to 15:30, 16:30 to 17:00; \nJudy has blocked their calendar on Monday during 9:00 to 11:30, 12:00 to 12:30, 13:00 to 13:30, 14:30 to 17:00; \n\nFind a time that works for everyone's schedule and constraints."
    answer = main(question)
    print(answer)