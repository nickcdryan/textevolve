import os
import json
import re
import math

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

def extract_meeting_constraints(question):
    """Extract meeting constraints from the question using LLM with example."""
    system_instruction = "You are an information extraction expert specializing in scheduling constraints."
    prompt = f"""
    Extract all the scheduling constraints from the input question.
    
    Example:
    Question: You need to schedule a meeting for Nicholas, Sara, Helen for half an hour between 9:00 to 17:00 on Monday. Nicholas is busy 9:00 to 9:30. Sara is busy 10:00 to 10:30.
    Extracted Constraints:
    {{
      "participants": ["Nicholas", "Sara", "Helen"],
      "duration": "0.5",
      "day": "Monday",
      "start_time": "9:00",
      "end_time": "17:00",
      "Nicholas_busy": ["9:00-9:30"],
      "Sara_busy": ["10:00-10:30"],
      "Helen_busy": []
    }}
    
    Question: {question}
    Extracted Constraints:
    """
    return call_llm(prompt, system_instruction)

def find_available_slots(constraints_json):
    """Find available time slots based on the extracted constraints.  Uses a code based approach for efficiency"""
    try:
        constraints = json.loads(constraints_json)
        
        # Extract relevant information from constraints
        participants = constraints.get("participants", [])
        duration_str = constraints.get("duration", "0.5")
        duration = float(duration_str)  # Convert to float for calculations
        day = constraints.get("day", "Monday")
        start_time_str = constraints.get("start_time", "9:00")
        end_time_str = constraints.get("end_time", "17:00")
        
        # Convert start and end times to minutes from midnight
        start_time = int(start_time_str.split(":")[0]) * 60 + int(start_time_str.split(":")[1]) if ":" in start_time_str else int(start_time_str) * 60
        end_time = int(end_time_str.split(":")[0]) * 60 + int(end_time_str.split(":")[1]) if ":" in end_time_str else int(end_time_str) * 60

        # Collect busy slots for all participants
        all_busy_slots = []
        for participant in participants:
            busy_key = f"{participant}_busy"
            if busy_key in constraints:
                busy_slots = constraints[busy_key]
                for slot in busy_slots:
                    start, end = slot.split("-")
                    start_minutes = int(start.split(":")[0]) * 60 + int(start.split(":")[1]) if ":" in start else int(start) * 60
                    end_minutes = int(end.split(":")[0]) * 60 + int(end.split(":")[1]) if ":" in end else int(end) * 60
                    all_busy_slots.append((start_minutes, end_minutes))
        
        # Combine overlapping busy slots to simplify calculations
        all_busy_slots.sort()
        merged_busy_slots = []
        if all_busy_slots:
            current_start, current_end = all_busy_slots[0]
            for start, end in all_busy_slots[1:]:
                if start <= current_end:
                    current_end = max(current_end, end)
                else:
                    merged_busy_slots.append((current_start, current_end))
                    current_start, current_end = start, end
            merged_busy_slots.append((current_start, current_end))

        # Find available time slots
        available_slots = []
        current_time = start_time
        for busy_start, busy_end in merged_busy_slots:
            if current_time + duration * 60 <= busy_start:
                available_slots.append((current_time, current_time + duration * 60))
            current_time = busy_end
        if current_time + duration * 60 <= end_time:
            available_slots.append((current_time, end_time))

        # Convert the time slots back to HH:MM format
        formatted_slots = []
        for start, end in available_slots:
            start_hour = int(start / 60)
            start_minute = int(start % 60)
            end_hour = int(end / 60)
            end_minute = int(end % 60)
            formatted_slots.append(f"{start_hour:02}:{start_minute:02}-{end_hour:02}:{end_minute:02}")
        
        return formatted_slots

    except json.JSONDecodeError as e:
        return f"Error decoding JSON: {str(e)}"
    except Exception as e:
        return f"Error finding available slots: {str(e)}"

def propose_meeting_time(available_slots):
    """Propose a meeting time from the available slots using LLM."""
    system_instruction = "You are a meeting scheduler who selects the best available time slot."
    prompt = f"""
    Given these available time slots: {available_slots}, propose the best time for a meeting.  Return the time as a string suitable for user output.
    Example:
    Available Slots: ['11:00-11:30', '14:00-14:30']
    Proposed Time: 11:00 - 11:30
    """
    return call_llm(prompt, system_instruction)

def verify_solution(question, proposed_answer):
    """Verify if the proposed solution is valid given the question using LLM with example."""
    system_instruction = "You are a solution verifier who determines if the proposed answer satisfies all constraints."
    prompt = f"""
    Verify if the proposed answer is a valid solution to the question.
    
    Example:
    Question: Schedule a meeting for Nicholas and Sara for half an hour between 9:00 and 17:00. Nicholas is busy from 9:00-9:30. Sara is busy from 10:00-10:30.
    Proposed Answer: 11:00 - 11:30
    Verification: The proposed time 11:00 - 11:30 works for both Nicholas and Sara and satisfies the constraints. VALID
    
    Question: {question}
    Proposed Answer: {proposed_answer}
    Verification:
    """
    return call_llm(prompt, system_instruction)

def main(question):
    """Main function to schedule the meeting."""
    try:
        # 1. Extract meeting constraints
        constraints_json = extract_meeting_constraints(question)
        print(f"Extracted constraints: {constraints_json}")

        # 2. Find available time slots
        available_slots = find_available_slots(constraints_json)
        print(f"Available slots: {available_slots}")
        
        # 3. Propose a meeting time
        proposed_time = propose_meeting_time(available_slots)
        print(f"Proposed time: {proposed_time}")

        # 4. Verify the solution
        verification_result = verify_solution(question, proposed_time)
        print(f"Verification result: {verification_result}")

        return f"Here is the proposed time: {proposed_time} "

    except Exception as e:
        return f"Error: {str(e)}"