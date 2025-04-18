import os
import json
import re
import datetime
from datetime import timedelta

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

def main(question):
    """Schedules meetings using a new approach: LLM-driven extraction and validation, deterministic slot generation, LLM-driven filtering with constraint satisfaction."""
    try:
        # Extract meeting information using LLM with validation loop
        meeting_info = extract_and_validate_info(question)
        if "Error" in meeting_info:
            return "Error extracting or validating meeting information."

        # Generate possible meeting slots deterministically
        possible_slots = generate_meeting_slots(meeting_info)
        if not possible_slots:
            return "No possible meeting slots found."

        # Filter slots based on complex constraints using LLM-driven constraint satisfaction
        filtered_slots = filter_slots_with_llm(meeting_info, possible_slots)
        if not filtered_slots:
            return "No meeting slots available after filtering."

        # Select the best slot (first available)
        best_slot = filtered_slots[0]  # Return the first available slot if any.
        return f"Here is the proposed time: {best_slot['day']}, {best_slot['start']} - {best_slot['end']}"
    except Exception as e:
        return f"An unexpected error occurred: {str(e)}"

def extract_and_validate_info(question, max_attempts=3):
    """Extracts and validates meeting information using iterative refinement."""
    for attempt in range(max_attempts):
        extracted_info = extract_meeting_info(question)
        if "Error" in extracted_info:
            return extracted_info

        validation_result = validate_meeting_info(question, extracted_info)
        if validation_result == "VALID":
            try:
                return json.loads(extracted_info) #parse the extracted info
            except:
                return "Error parsing json info"
        else:
            question = f"RETRY: {validation_result}. Original question: {question}. Previous extraction: {extracted_info}"
    return "Error: Could not extract valid meeting information after multiple attempts."

def extract_meeting_info(question):
    """Extract meeting information using LLM with an embedded example."""
    system_instruction = "You are an expert at extracting information from meeting scheduling requests."
    prompt = f"""
    Extract key information from the scheduling request.

    Example:
    Input: You need to schedule a meeting for John, Jane, and Doe for 30 minutes between 9:00 and 17:00 on Monday. John is busy 10:00-11:00, Jane is busy 13:00-14:00, Doe is free.
    Output: {{"participants": ["John", "Jane", "Doe"], "duration": 30, "days": ["Monday"], "work_hours": ["9:00", "17:00"], "schedules": {{"John": [["10:00", "11:00"]], "Jane": [["13:00", "14:00"]], "Doe": []}}}}

    Input: {question}
    Output:
    """
    return call_llm(prompt, system_instruction)

def validate_meeting_info(question, extracted_info):
    """Validates extracted meeting information using LLM."""
    system_instruction = "You are a meticulous validator."
    prompt = f"""
    Check if the extracted information is consistent with the original question.

    Example:
    Question: Schedule a meeting for A and B for 20 minutes on Tuesday. A is busy 9:00-10:00.
    Extracted: {{"participants": ["A", "B"], "duration": 20, "days": ["Tuesday"], "work_hours": ["9:00", "17:00"], "schedules": {{"A": [["9:00", "10:00"]], "B": []}}}}
    Result: VALID

    Question: {question}
    Extracted: {extracted_info}
    Result:
    """
    return call_llm(prompt, system_instruction).strip() #.strip()

def generate_meeting_slots(meeting_info):
    """Generates possible meeting slots deterministically."""
    slots = []
    start_time_str = meeting_info["work_hours"][0]
    end_time_str = meeting_info["work_hours"][1]
    start_time = datetime.datetime.strptime(start_time_str, "%H:%M").time()
    end_time = datetime.datetime.strptime(end_time_str, "%H:%M").time()
    duration = meeting_info["duration"]
    for day in meeting_info["days"]:
        current_time = datetime.datetime.combine(datetime.date.today(), start_time)
        end_datetime = datetime.datetime.combine(datetime.date.today(), end_time)
        while current_time + timedelta(minutes=duration) <= end_datetime:
            start_str = current_time.strftime("%H:%M")
            end_str = (current_time + timedelta(minutes=duration)).strftime("%H:%M")
            slots.append({"day": day, "start": start_str, "end": end_str})
            current_time += timedelta(minutes=30)
    return slots

def filter_slots_with_llm(meeting_info, possible_slots):
    """Filters meeting slots with LLM using constraint satisfaction."""
    system_instruction = "You are an expert meeting scheduler who understands complex time constraints."
    prompt = f"""
    Given the meeting requirements and possible slots, determine which slots satisfy all constraints.
    Return ONLY the slots that work.

    Example:
    Meeting Info: {{"participants": ["John", "Jane"], "duration": 30, "days": ["Monday"], "work_hours": ["9:00", "17:00"], "schedules": {{"John": [["10:00", "11:00"]], "Jane": []}}}}
    Possible Slots: [{{"day": "Monday", "start": "9:00", "end": "9:30"}}, {{"day": "Monday", "start": "10:30", "end": "11:00"}}]
    Valid Slots: [{{"day": "Monday", "start": "9:00", "end": "9:30"}}]

    Meeting Info: {meeting_info}
    Possible Slots: {possible_slots}
    Valid Slots:
    """
    llm_response = call_llm(prompt, system_instruction)
    try:
      return json.loads(llm_response)
    except:
      return []