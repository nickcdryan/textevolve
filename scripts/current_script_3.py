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
    """Extract key information from the problem statement using embedded examples. Includes schedule verification agent."""
    system_instruction = "You are an information extraction specialist focusing on identifying key entities, constraints, and verifying participant schedules."
    
    prompt = f"""
    Extract key information from this problem statement. Focus on identifying all entities, relationships, and constraints. Verify extracted schedules using the schedule verification agent.
    
    Example usage:
    
    Question:
    You need to schedule a meeting for John and Jennifer for half an hour between the work hours of 9:00 to 17:00 on either Monday, Tuesday or Wednesday. 
    Here are the existing schedules for everyone during the days: 
    John has no meetings the whole week.
    Jennifer has meetings on Monday during 9:00 to 11:00, 11:30 to 13:00, 13:30 to 14:30, 15:00 to 17:00, Tuesday during 9:00 to 11:30, 12:00 to 17:00, Wednesday during 9:00 to 11:30, 12:00 to 12:30, 13:00 to 14:00, 14:30 to 16:00, 16:30 to 17:00.
    John would like to avoid more meetings on Monday after 14:30. Find a time that works for everyone's schedule and constraints.
    
    Let's think step by step.
    
    The key entities are:
    - John (participant)
    - Jennifer (participant)
    
    The key constraints are:
    - Meeting duration: 30 minutes (half an hour)
    - Valid meeting hours: 9:00-17:00
    - Valid days: Monday, Tuesday, or Wednesday
    - John's availability: All week (no meetings)
    - Jennifer's availability:
      * Monday: Busy 9:00-11:00, 11:30-13:00, 13:30-14:30, 15:00-17:00
      * Tuesday: Busy 9:00-11:30, 12:00-17:00
      * Wednesday: Busy 9:00-11:30, 12:00-12:30, 13:00-14:00, 14:30-16:00, 16:30-17:00
    - Preferences: John prefers to avoid meetings on Monday after 14:30
    
    Schedule Verification Agent: Let's double check Jennifer's availability on Monday. Is she REALLY busy from 9:00-11:00? Yes. Is she REALLY busy from 11:30-13:00? Yes. Is she REALLY busy from 13:30-14:30? Yes. Is she REALLY busy from 15:00-17:00? Yes.
    
    Extracted Information:
    {{
      "participants": ["John", "Jennifer"],
      "duration": "30 minutes",
      "valid_hours": "9:00-17:00",
      "valid_days": ["Monday", "Tuesday", "Wednesday"],
      "availability": {{
        "John": "All times",
        "Jennifer": {{
          "Monday": ["11:00-11:30", "13:00-13:30", "14:30-15:00"],
          "Tuesday": ["11:30-12:00"],
          "Wednesday": ["11:30-12:00", "12:30-13:00", "14:00-14:30", "16:00-16:30"]
        }}
      }},
      "preferences": {{
        "John": "Avoid Monday after 14:30"
      }}
    }}
    
    Now, extract information from this new problem:
    {problem}
    """
    
    return call_llm(prompt, system_instruction)

def find_available_time_with_examples(extracted_info):
    """Find an available meeting time given the extracted information."""
    system_instruction = "You are a meeting scheduling expert. Use the extracted information to find a valid meeting time."
    
    prompt = f"""
    Given the following extracted information, find a suitable meeting time:
    
    Extracted Information:
    {extracted_info}
    
    Example usage:
    
    Extracted Information:
    {{
      "participants": ["John", "Jennifer"],
      "duration": "30 minutes",
      "valid_hours": "9:00-17:00",
      "valid_days": ["Monday", "Tuesday", "Wednesday"],
      "availability": {{
        "John": "All times",
        "Jennifer": {{
          "Monday": ["11:00-11:30", "13:00-13:30", "14:30-15:00"],
          "Tuesday": ["11:30-12:00"],
          "Wednesday": ["11:30-12:00", "12:30-13:00", "14:00-14:30", "16:00-16:30"]
        }}
      }},
      "preferences": {{
        "John": "Avoid Monday after 14:30"
      }}
    }}
    
    Reasoning:
    Let's think step by step.
    1. Consider the valid days: Monday, Tuesday, Wednesday
    2. Consider John's availability: All times
    3. Consider Jennifer's availability:
      * Monday: 11:00-11:30, 13:00-13:30, 14:30-15:00
      * Tuesday: 11:30-12:00
      * Wednesday: 11:30-12:00, 12:30-13:00, 14:00-14:30, 16:00-16:30
    4. Consider the meeting duration: 30 minutes
    5. Consider John's preferences: Avoid Monday after 14:30
    
    Possible meeting times:
    - Tuesday: 11:30-12:00
    
    Final Answer:
    Tuesday, 11:30 - 12:00
    
    Now, find a valid meeting time for the given information:
    """
    return call_llm(prompt, system_instruction)

def verify_solution_with_examples(problem, proposed_solution):
    """Verify if the proposed solution satisfies all constraints using embedded examples."""
    system_instruction = "You are a critical evaluator who verifies if solutions satisfy all constraints. Provide detailed reasoning."
    
    prompt = f"""
    Verify if this proposed solution satisfies all constraints in the problem. Show your reasoning.
    
    Example usage:
    
    Problem:
    You need to schedule a meeting for John and Jennifer for half an hour between the work hours of 9:00 to 17:00 on either Monday, Tuesday or Wednesday. 
    Here are the existing schedules for everyone during the days: 
    John has no meetings the whole week.
    Jennifer has meetings on Monday during 9:00 to 11:00, 11:30 to 13:00, 13:30 to 14:30, 15:00 to 17:00, Tuesday during 9:00 to 11:30, 12:00 to 17:00, Wednesday during 9:00 to 11:30, 12:00 to 12:30, 13:00 to 14:00, 14:30 to 16:00, 16:30 to 17:00.
    John would like to avoid more meetings on Monday after 14:30.
    
    Proposed Solution:
    Here is the proposed time: Wednesday, 13:00 - 13:30
    
    Verification:
    Let me check each constraint step by step:
    1. Duration: The meeting is scheduled for 30 minutes (13:00-13:30) - Satisfied.
    2. Work hours: Meeting time 13:00-13:30 is within 9:00-17:00 - Satisfied.
    3. Valid day: Wednesday is one of the allowed days - Satisfied.
    4. John's availability: John has no meetings all week - Satisfied.
    5. Jennifer's availability on Wednesday:
       - Jennifer is busy 9:00-11:30, 12:00-12:30, 13:00-14:00, 14:30-16:00, 16:30-17:00
       - The proposed time 13:00-13:30 overlaps with Jennifer's busy time 13:00-14:00 - NOT Satisfied.
    6. John's preference: Not applicable (not Monday after 14:30) - Satisfied.
    
    Final Answer: INVALID - The solution conflicts with Jennifer's schedule on Wednesday from 13:00-14:00.
    
    Problem:
    {problem}
    
    Proposed Solution:
    {proposed_solution}
    
    Verification:
    """
    
    return call_llm(prompt, system_instruction)

def main(question):
    """Main function to schedule a meeting."""
    try:
        # 1. Extract information
        extracted_info = extract_information_with_examples(question)
        
        # 2. Find available time
        available_time = find_available_time_with_examples(extracted_info)
        
        # 3. Verify solution
        verification_result = verify_solution_with_examples(question, available_time)
        
        #4. Return verified or invalid result
        if "INVALID" in verification_result:
            return "No valid meeting time found."
        else:
            return available_time
        
    except Exception as e:
        return f"An error occurred: {str(e)}"