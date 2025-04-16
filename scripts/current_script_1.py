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

def extract_data_and_verify(text):
    """Extract scheduling constraints and verifies if extraction is correct"""
    system_instruction = "You are an information extraction and verification expert."
    prompt = f"""
    Extract scheduling constraints from the text and verify the correctness of the extraction in a chain of thought manner.
    
    Example Input:
    You need to schedule a meeting for Carol and Mark for half an hour between the work hours of 9:00 to 17:00 on Monday.
    Carol has blocked their calendar on Monday during 10:00 to 11:00, 14:30 to 15:00, 15:30 to 17:00;
    Mark has blocked their calendar on Monday during 9:30 to 10:00, 10:30 to 17:00;
    
    Let's think step by step:
    1. Participants: Carol and Mark.
    2. Duration: Half an hour (30 minutes).
    3. Day: Monday.
    4. Time Range: 9:00 to 17:00.
    5. Carol's Conflicts: 10:00-11:00, 14:30-15:00, 15:30-17:00.
    6. Mark's Conflicts: 9:30-10:00, 10:30-17:00.
    
    Extracted data:
    {{
        "participants": ["Carol", "Mark"],
        "duration": "30 minutes",
        "day": "Monday",
        "time_range_start": "9:00",
        "time_range_end": "17:00",
        "conflicts": {{
            "Carol": ["10:00-11:00", "14:30-15:00", "15:30-17:00"],
            "Mark": ["9:30-10:00", "10:30-17:00"]
        }}
    }}
    
    Verification:
    1. Participants are correctly identified.
    2. Duration is accurately extracted as 30 minutes.
    3. Day is correctly identified as Monday.
    4. Time range is accurate.
    5. Carol's conflicts are correctly listed.
    6. Mark's conflicts are correctly listed.
    Result: Extraction is Valid.
    
    Now, extract and verify for this new text:
    {text}
    """
    return call_llm(prompt, system_instruction)

def find_available_time_slots(data):
    """Find available time slots using LLM."""
    system_instruction = "You are an expert at identifying available time slots given constraints."
    prompt = f"""
    Given the data extracted and verified, find available time slots.
    
    Example Input:
    {{
        "participants": ["Carol", "Mark"],
        "duration": "30 minutes",
        "day": "Monday",
        "time_range_start": "9:00",
        "time_range_end": "17:00",
        "conflicts": {{
            "Carol": ["10:00-11:00", "14:30-15:00", "15:30-17:00"],
            "Mark": ["9:30-10:00", "10:30-17:00"]
        }}
    }}
    
    Let's think step by step:
    1. Available Time Range: 9:00 to 17:00 on Monday.
    2. Meeting Duration: 30 minutes.
    3. Carol's Available Times: 9:00-10:00, 11:00-14:30, 15:00-15:30
    4. Mark's Available Times: 9:00-9:30, 10:00-10:30
    5. Combining, the available time slots are: 9:00-9:30, 10:00-10:30.
    
    Proposed Time: Monday, 9:00-9:30.
    
    Now, find the available time based on this data:
    {data}
    """
    return call_llm(prompt, system_instruction)

def main(question):
    """Main function."""
    try:
        extracted_data = extract_data_and_verify(question)
        available_time = find_available_time_slots(extracted_data)
        
        if "Proposed Time:" in available_time:
            proposed_time = available_time.split("Proposed Time:")[1].strip()
            return "Here is the proposed time: " + proposed_time
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