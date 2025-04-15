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

def extract_schedule_information(text):
    """Extract schedule information for each person using LLM with examples."""
    system_instruction = "You are a scheduling assistant. Extract schedule information."
    prompt = f"""
    Extract the schedule for each person mentioned.
    Example:
    Input:
    Jennifer has blocked their calendar on Monday during 12:30 to 13:00, 14:30 to 15:00, 16:00 to 16:30;
    Christine is busy on Monday during 10:00 to 11:00, 12:00 to 14:30, 16:00 to 16:30
    Output:
    {{
      "Jennifer": {{"Monday": ["12:30-13:00", "14:30-15:00", "16:00-16:30"]}},
      "Christine": {{"Monday": ["10:00-11:00", "12:00-14:30", "16:00-16:30"]}}
    }}
    Now extract information from this text:
    {text}
    """
    return call_llm(prompt, system_instruction)

def find_available_times(schedules, duration, days, start_time="9:00", end_time="17:00"):
    """Find available time slots for a given duration, schedules and days."""
    system_instruction = "You are a scheduling expert. Find available time slots."
    prompt = f"""
    Given the schedules of people, find the available time slots of specified duration on the given days.
    Example:
    Input:
    schedules:
    {{
      "Jennifer": {{"Monday": ["12:30-13:00", "14:30-15:00", "16:00-16:30"]}},
      "Christine": {{"Monday": ["10:00-11:00", "12:00-14:30", "16:00-16:30"]}}
    }}
    duration: 0.5
    days: ["Monday"]
    start_time: "9:00"
    end_time: "17:00"
    Output:
    {{
      "Monday": ["9:00-9:30", "9:30-10:00", "11:00-12:00", "15:00-16:00"]
    }}
    Now find the time slots for this:
    schedules: {schedules}
    duration: {duration}
    days: {days}
    start_time: {start_time}
    end_time: {end_time}
    """
    return call_llm(prompt, system_instruction)

def verify_solution(proposed_time, schedules):
    """Verify that the proposed time works with all the schedules."""
    system_instruction = "You are a solution checker. Verify if the given time works for all the people"
    prompt = f"""
    Verify that the proposed time does not conflict with the schedules of any person.
    Example:
    Input:
    proposed_time: "Monday, 9:00-9:30"
    schedules:
    {{
      "Jennifer": {{"Monday": ["12:30-13:00", "14:30-15:00", "16:00-16:30"]}},
      "Christine": {{"Monday": ["10:00-11:00", "12:00-14:30", "16:00-16:30"]}}
    }}
    Output:
    True
    Now check this:
    proposed_time: {proposed_time}
    schedules: {schedules}
    """
    return call_llm(prompt, system_instruction)

def main(question):
    """Main function to schedule meetings."""
    try:
        # 1. Extract information
        schedule_text = question.split("Here are the existing schedules for everyone during the days: ")[1].split("Find a time")[0]
        schedules_json = extract_schedule_information(schedule_text)
        schedules = json.loads(schedules_json)
        
        # 2. Get Duration
        duration_line = question.split("schedule a meeting for ")[1].split(" for ")[1]
        duration = float(duration_line.split(" ")[0])/2 if "half" in duration_line else float(duration_line.split(" ")[0])

        # 3. Get Days
        days_line = question.split("schedule a meeting for ")[1].split(" on ")[1].split(".")[0]
        days = [d.strip() for d in days_line.split(",")]

        # 4. Find available times
        available_times_json = find_available_times(schedules, duration, days)
        available_times = json.loads(available_times_json)

        # 5. Verify and propose a time
        for day in days:
            if day in available_times:
                times = available_times[day]
                for time in times:
                    proposed_time = f"{day}, {time}"
                    verification_result = verify_solution(proposed_time, schedules)
                    if "True" in verification_result:
                        return f"Here is the proposed time: {proposed_time} "

        return "No suitable time found."

    except Exception as e:
        return f"Error: {str(e)}"