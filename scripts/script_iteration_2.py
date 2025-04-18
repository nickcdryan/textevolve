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
    """
    Schedules meetings by first summarizing the schedules into a standard format,
    then generating and filtering slots, and finally selecting the best slot.
    """
    try:
        schedule_summary = summarize_schedules(question)
        if "Error" in schedule_summary:
            return "Error summarizing schedules."

        possible_slots = generate_meeting_slots(schedule_summary)
        if "Error" in possible_slots:
            return "Error generating possible meeting slots."

        filtered_slots = filter_meeting_slots(schedule_summary, possible_slots)
        if "Error" in filtered_slots:
            return "Error filtering meeting slots."

        best_slot = select_best_meeting_slot(schedule_summary, filtered_slots)
        if "Error" in best_slot:
            return "Error selecting the best meeting slot."

        return best_slot
    except Exception as e:
        return f"An unexpected error occurred: {str(e)}"

def summarize_schedules(question):
    """Summarizes the schedules into a structured format using LLM."""
    system_instruction = "You are an expert at summarizing meeting schedules."
    prompt = f"""
    Summarize the following meeting scheduling request into a structured format:

    Example:
    Input: You need to schedule a meeting for John and Jennifer for half an hour between the work hours of 9:00 to 17:00 on either Monday, Tuesday or Wednesday. 
    Here are the existing schedules for everyone during the days: 
    John has no meetings the whole week.
    Jennifer has meetings on Monday during 9:00 to 11:00, 11:30 to 13:00, 13:30 to 14:30, 15:00 to 17:00, Tuesday during 9:00 to 11:30, 12:00 to 17:00, Wednesday during 9:00 to 11:30, 12:00 to 12:30, 13:00 to 14:00, 14:30 to 16:00, 16:30 to 17:00; 
    John would like to avoid more meetings on Monday after 14:30. Tuesday. Wednesday.
    Output:
    {{
      "participants": ["John", "Jennifer"],
      "duration": 30,
      "days": ["Monday", "Tuesday", "Wednesday"],
      "work_hours": ["9:00", "17:00"],
      "schedules": {{
        "John": {{
          "Monday": [],
          "Tuesday": [],
          "Wednesday": []
        }},
        "Jennifer": {{
          "Monday": [["9:00", "11:00"], ["11:30", "13:00"], ["13:30", "14:30"], ["15:00", "17:00"]],
          "Tuesday": [["9:00", "11:30"], ["12:00", "17:00"]],
          "Wednesday": [["9:00", "11:30"], ["12:00", "12:30"], ["13:00", "14:00"], ["14:30", "16:00"], ["16:30", "17:00"]]
        }}
      }},
      "preferences": {{
        "John": {{"Monday": "14:30"}}
      }}
    }}
    Input: {question}
    Output:
    """
    try:
        llm_response = call_llm(prompt, system_instruction)
        return llm_response
    except Exception as e:
        return f"Error summarizing schedules: {str(e)}"

def generate_meeting_slots(schedule_summary_str):
    """Generates possible meeting slots based on the summarized schedule using Python code."""
    try:
        schedule_summary = json.loads(schedule_summary_str)
        duration = schedule_summary["duration"]
        work_hours = schedule_summary["work_hours"]
        days = schedule_summary["days"]

        start_time = datetime.datetime.strptime(work_hours[0], "%H:%M").time()
        end_time = datetime.datetime.strptime(work_hours[1], "%H:%M").time()

        slots = []
        for day in days:
            current_time = datetime.datetime.combine(datetime.date.today(), start_time)
            end_datetime = datetime.datetime.combine(datetime.date.today(), end_time)
            while current_time + timedelta(minutes=duration) <= end_datetime:
                start_str = current_time.strftime("%H:%M")
                end_str = (current_time + timedelta(minutes=duration)).strftime("%H:%M")
                slots.append({"day": day, "start": start_str, "end": end_str})
                current_time += timedelta(minutes=30)  # Increment by 30 minutes

        return json.dumps(slots)
    except Exception as e:
        return f"Error generating meeting slots: {str(e)}"

def filter_meeting_slots(schedule_summary_str, possible_slots_str):
    """Filters out invalid meeting slots based on summarized schedules using LLM reasoning."""
    system_instruction = "You are an expert at filtering meeting slots based on availability and constraints."
    prompt = f"""
    Given the summarized schedule and possible meeting slots, filter the slots to find times that work for everyone, considering their schedules and constraints.

    Example:
    Summarized Schedule:
    {{
      "participants": ["John", "Jennifer"],
      "duration": 30,
      "days": ["Monday"],
      "work_hours": ["09:00", "17:00"],
      "schedules": {{
        "John": {{
          "Monday": []
        }},
        "Jennifer": {{
          "Monday": [["09:00", "11:00"], ["11:30", "13:00"]]
        }}
      }},
      "preferences": {{}}
    }}
    Possible Slots:
    [
      {{"day": "Monday", "start": "11:00", "end": "11:30"}},
      {{"day": "Monday", "start": "13:00", "end": "13:30"}},
      {{"day": "Monday", "start": "15:00", "end": "15:30"}}
    ]
    Filtered Slots:
    [
      {{"day": "Monday", "start": "13:00", "end": "13:30"}},
      {{"day": "Monday", "start": "15:00", "end": "15:30"}}
    ]

    Summarized Schedule: {schedule_summary_str}
    Possible Slots: {possible_slots_str}
    Filtered Slots:
    """
    try:
        llm_response = call_llm(prompt, system_instruction)
        return llm_response
    except Exception as e:
        return f"Error filtering meeting slots: {str(e)}"

def select_best_meeting_slot(schedule_summary_str, filtered_slots_str):
    """Selects the best meeting slot based on preferences using LLM."""
    system_instruction = "You are an expert at selecting the best meeting slot from available options, considering preferences."
    prompt = f"""
    Given the summarized schedule and filtered meeting slots, select the best slot based on stated preferences. If no preferences are stated, return the first available slot.

    Example:
    Summarized Schedule:
    {{
      "participants": ["David", "Ethan", "Bradley", "Natalie"],
      "duration": 30,
      "days": ["Monday"],
      "work_hours": ["09:00", "17:00"],
      "schedules": {{
        "David": {{
          "Monday": [["14:00", "14:30"], ["16:30", "17:00"]]
        }},
        "Ethan": {{
          "Monday": [["13:00", "13:30"], ["14:30", "15:00"]]
        }},
        "Bradley": {{
          "Monday": [["09:30", "10:30"], ["11:00", "12:00"], ["13:30", "14:00"], ["15:30", "17:00"]]
        }},
        "Natalie": {{
          "Monday": [["09:30", "10:00"], ["10:30", "12:00"], ["12:30", "15:30"], ["16:00", "17:00"]]
        }}
      }},
      "preferences": {{
        "Natalie": {{"Monday": "10:30"}}
      }}
    }}
    Filtered Slots:
    [
      {{"day": "Monday", "start": "09:00", "end": "09:30"}},
      {{"day": "Monday", "start": "15:30", "end": "16:00"}}
    ]
    Best Slot: Here is the proposed time: Monday, 09:00 - 09:30

    Summarized Schedule: {schedule_summary_str}
    Filtered Slots: {filtered_slots_str}
    Best Slot:
    """
    try:
        llm_response = call_llm(prompt, system_instruction)
        return "Here is the proposed time: " + llm_response
    except Exception as e:
        return f"Error selecting best slot: {str(e)}"