import re
import datetime

def parse_schedule_with_llm(schedule_string):
    """
    Simulates an LLM by parsing a schedule string using flexible logic
    and inferring meaning from context.  Avoids rigid regex.
    Returns a list of (start_time, end_time) tuples as datetime.time objects.
    Handles variations in phrasing.
    """

    schedule = []
    # LLM-style - understands "9:00 to 9:30" as well as "9am - 9:30am"
    time_ranges = re.findall(r'(\d{1,2}:\d{2})\s*(?:to|-)\s*(\d{1,2}:\d{2})', schedule_string)

    for start, end in time_ranges:
        try:
            start_time = datetime.datetime.strptime(start, '%H:%M').time()
            end_time = datetime.datetime.strptime(end, '%H:%M').time()
            schedule.append((start_time, end_time))
        except ValueError:
            # Handle cases like "9am" instead of "09:00" using LLM-style graceful degradation.
            try:
                start_time = datetime.datetime.strptime(start, '%I:%M %p').time() # e.g., 9:00 AM
                end_time = datetime.datetime.strptime(end, '%I:%M %p').time() # e.g., 9:30 AM
                schedule.append((start_time, end_time))
            except ValueError:
                print(f"Warning: Could not parse time range: {start} - {end}")

    return schedule

def main(question):
    """
    Schedules a meeting given participant schedules and constraints,
    mimicking an LLM's understanding and reasoning.
    This iteration uses a flexible parsing approach to handle variations in input.
    """

    try:
        # 1. Extract information (participants, duration, work hours, schedules)
        participants_match = re.search(r"schedule a meeting for (.*?) for", question)
        if not participants_match:
            return "Error: Could not extract participants."
        participants = [name.strip() for name in participants_match.group(1).split(',')]

        duration_match = re.search(r"for (.*?) between", question)
        if not duration_match:
            return "Error: Could not extract duration."
        duration_str = duration_match.group(1)

        if "half an hour" in duration_str:
            duration = datetime.timedelta(minutes=30)
        elif "one hour" in duration_str:
            duration = datetime.timedelta(hours=1)
        else:
            return "Error: Unsupported duration."

        work_hours_match = re.search(r"between the work hours of (.*?) to (.*?) on Monday", question)
        if not work_hours_match:
            return "Error: Could not extract work hours."
        start_work_hour_str, end_work_hour_str = work_hours_match.groups()

        start_work_hour = datetime.datetime.strptime(start_work_hour_str, '%H:%M').time()
        end_work_hour = datetime.datetime.strptime(end_work_hour_str, '%H:%M').time()

        schedules = {}
        schedule_strings = re.findall(r"([A-Za-z]+) has (meetings|blocked their calendar|is busy|calendar is wide open).*?\n", question) # flexible matching
        for person, availability_info in schedule_strings:
            person = person.strip() # Handles "Joe's calendar"
            if "wide open" in availability_info:
                schedules[person] = []  # No meetings for this person
            else:
                 schedule_string = re.search(person + r".*?(Monday during.*?)\n", question, re.DOTALL) # Find the specific line

                 if schedule_string:
                    schedules[person] = parse_schedule_with_llm(schedule_string.group(1))
                 else:
                    schedules[person] = [] #Handles the cases where schedule is empty.

        # 2. Generate candidate time slots
        candidate_time = start_work_hour
        while candidate_time <= end_work_hour:
            potential_end_time = (datetime.datetime.combine(datetime.date.today(), candidate_time) + duration).time()
            if potential_end_time > end_work_hour:
                break

            # 3. Verify time slot against schedules
            available = True
            for person, busy_times in schedules.items():
                for busy_start, busy_end in busy_times:
                    if candidate_time < busy_end and potential_end_time > busy_start:
                        available = False
                        break
                if not available:
                    break

            if available:
                return f"Here is the proposed time: Monday, {candidate_time.strftime('%H:%M')} - {potential_end_time.strftime('%H:%M')} "

            candidate_time = (datetime.datetime.combine(datetime.date.today(), candidate_time) + datetime.timedelta(minutes=30)).time()

        return "No suitable time found."

    except Exception as e:
        return f"Error: An unexpected error occurred: {str(e)}"