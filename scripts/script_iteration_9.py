import os
import json
import re
from datetime import datetime, timedelta

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

def extract_meeting_details(question):
    """Extract meeting details (participants, duration, possible days) using LLM with examples."""
    system_instruction = "You are an expert in extracting meeting details from text."
    prompt = f"""
    Extract the participants, meeting duration (in minutes), and possible days from the following text.
    
    Example:
    Text: You need to schedule a meeting for Kelly and Patricia for half an hour between the work hours of 9:00 to 17:00 on either Monday, Tuesday, Wednesday, Thursday or Friday.
    Output: {{"participants": ["Kelly", "Patricia"], "duration": 30, "possible_days": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]}}
    
    Text: You need to schedule a meeting for Janet and Randy for one hour between the work hours of 9:00 to 17:00 on either Monday, Tuesday or Wednesday. 
    Output: {{"participants": ["Janet", "Randy"], "duration": 60, "possible_days": ["Monday", "Tuesday", "Wednesday"]}}

    Text: {question}
    """
    try:
        response = call_llm(prompt, system_instruction)
        details = json.loads(response)
        return details
    except Exception as e:
        print(f"Error extracting meeting details: {e}")
        return None

def extract_schedules(question, participants):
    """Extract individual schedules for each participant using LLM with examples."""
    system_instruction = "You are an expert in extracting schedules from text, extracting busy times for each person by day."
    prompt = f"""
    For each participant, extract their schedule as a list of time intervals (start time - end time) for each day.
    
    Example:
    Question: Kelly has blocked their calendar on Tuesday during 9:00 to 9:30, Friday during 9:00 to 9:30; Patricia has blocked their calendar on Monday during 9:30 to 16:00, 16:30 to 17:00, Tuesday during 9:00 to 11:00, 12:30 to 16:30, Wednesday during 10:00 to 11:00, 11:30 to 12:00, 12:30 to 14:00, 14:30 to 17:00, Thursday during 9:00 to 10:30, 11:00 to 12:30, 13:30 to 14:30, 15:00 to 15:30, 16:00 to 17:00, Friday during 9:00 to 10:00, 10:30 to 11:30, 12:00 to 14:00, 14:30 to 16:00, 16:30 to 17:00;
    Participants: ['Kelly', 'Patricia']
    Output: 
    {{
        "Kelly": {{
            "Tuesday": ["9:00 - 9:30"],
            "Friday": ["9:00 - 9:30"],
            "Monday": [],
            "Wednesday": [],
            "Thursday": []
        }},
        "Patricia": {{
            "Monday": ["9:30 - 16:00", "16:30 - 17:00"],
            "Tuesday": ["9:00 - 11:00", "12:30 - 16:30"],
            "Wednesday": ["10:00 - 11:00", "11:30 - 12:00", "12:30 - 14:00", "14:30 - 17:00"],
            "Thursday": ["9:00 - 10:30", "11:00 - 12:30", "13:30 - 14:30", "15:00 - 15:30", "16:00 - 17:00"],
            "Friday": ["9:00 - 10:00", "10:30 - 11:30", "12:00 - 14:00", "14:30 - 16:00", "16:30 - 17:00"]
        }}
    }}
    
    Question: {question}
    Participants: {participants}
    """
    
    try:
        response = call_llm(prompt, system_instruction)
        schedules = json.loads(response)
        return schedules
    except Exception as e:
        print(f"Error extracting schedules: {e}")
        return None

def find_available_time(meeting_details, schedules):
    """Find an available time slot that works for all participants."""
    duration = meeting_details["duration"]
    possible_days = meeting_details["possible_days"]
    start_time = datetime.strptime("09:00", "%H:%M").time()
    end_time = datetime.strptime("17:00", "%H:%M").time()
    
    for day in possible_days:
        current_time = start_time
        while current_time <= end_time:
            meeting_end_time = (datetime.combine(datetime.today(), current_time) + timedelta(minutes=duration)).time()
            if meeting_end_time > end_time:
                break
            
            available = True
            for person, schedule in schedules.items():
                busy_times = schedule.get(day, [])
                for busy_time in busy_times:
                    busy_start, busy_end = [datetime.strptime(t.strip(), "%H:%M").time() for t in busy_time.split(" - ")]
                    if current_time < busy_end and meeting_end_time > busy_start:
                        available = False
                        break
                if not available:
                    break
            
            if available:
                return f"Here is the proposed time: {day}, {current_time.strftime('%H:%M')} - {meeting_end_time.strftime('%H:%M')}"
            
            current_time = (datetime.combine(datetime.today(), current_time) + timedelta(minutes=15)).time()
    
    return "No available time slots found."

def verify_solution(question, proposed_solution):
    """Verify the proposed solution against the original question using LLM with example."""
    system_instruction = "You are a solution verifier, ensuring a proposed time slot meets all constraints."
    prompt = f"""
    Given the scheduling question and a proposed solution, verify if the solution satisfies all constraints mentioned in the question.
    
    Example:
    Question: You need to schedule a meeting for Kelly and Patricia for half an hour between the work hours of 9:00 to 17:00 on either Monday, Tuesday, Wednesday, Thursday or Friday. Kelly has blocked their calendar on Tuesday during 9:00 to 9:30, Friday during 9:00 to 9:30; Patricia has blocked their calendar on Monday during 9:30 to 16:00, 16:30 to 17:00, Tuesday during 9:00 to 11:00, 12:30 to 16:30, Wednesday during 10:00 to 11:00, 11:30 to 12:00, 12:30 to 14:00, 14:30 to 17:00, Thursday during 9:00 to 10:30, 11:00 to 12:30, 13:30 to 14:30, 15:00 to 15:30, 16:00 to 17:00, Friday during 9:00 to 10:00, 10:30 to 11:30, 12:00 to 14:00, 14:30 to 16:00, 16:30 to 17:00;
    Proposed Solution: Here is the proposed time: Tuesday, 11:00 - 11:30
    Output: VALID
    
    Question: You need to schedule a meeting for Janet and Randy for one hour between the work hours of 9:00 to 17:00 on either Monday, Tuesday or Wednesday. Randy has blocked their calendar on Monday during 9:00 to 11:00, 14:00 to 16:00, Tuesday during 9:00 to 10:30, 11:30 to 12:00, 14:00 to 14:30, 16:00 to 16:30, Wednesday during 9:00 to 14:00, 15:00 to 17:00;
    Proposed Solution: Here is the proposed time: Monday, 11:00 - 12:00
    Output: VALID

    Question: {question}
    Proposed Solution: {proposed_solution}
    """
    try:
        response = call_llm(prompt, system_instruction)
        if "VALID" in response:
            return True
        else:
            return False
    except Exception as e:
        print(f"Error verifying solution: {e}")
        return False

def main(question):
    """Main function to schedule a meeting."""
    try:
        meeting_details = extract_meeting_details(question)
        if not meeting_details:
            return "Could not extract meeting details."
        
        schedules = extract_schedules(question, meeting_details["participants"])
        if not schedules:
            return "Could not extract schedules."
        
        proposed_solution = find_available_time(meeting_details, schedules)
        
        if verify_solution(question, proposed_solution):
            return proposed_solution
        else:
            return "Proposed solution is invalid based on verification."

    except Exception as e:
        return f"An error occurred: {e}"