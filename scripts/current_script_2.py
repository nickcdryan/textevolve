import re

def parse_schedule(schedule_str):
    """
    Parses a schedule string into a list of time intervals (tuples).
    Handles variations in the schedule format.
    For example: "9:30 to 10:00" becomes (9.5, 10.0)
    """
    try:
        schedule = []
        for time_range in schedule_str.split(','):
            time_range = time_range.strip()  # remove leading/trailing whitespace
            if ' to ' in time_range:
                start_time_str, end_time_str = time_range.split(' to ')
                try:
                    start_hour, start_minute = map(int, start_time_str.split(':'))
                    end_hour, end_minute = map(int, end_time_str.split(':'))
                    start_time = start_hour + start_minute / 60.0
                    end_time = end_hour + end_minute / 60.0
                    schedule.append((start_time, end_time))
                except ValueError:
                    # Handle cases where the time format is unexpected
                    return None
            elif time_range.lower() == 'free the entire day' or time_range.lower() == "calendar is wide open the entire day.":
                # Represent the entire day as free (9:00 to 17:00)
                schedule.append((9.0, 17.0))
            else:
                 # Attempt to parse a single time, if possible (less common)
                try:
                    hour, minute = map(int, time_range.split(':'))
                    time_val = hour + minute / 60.0
                    # Represent a single time as a very short interval (e.g., 0.01 hours)
                    schedule.append((time_val, time_val + 0.01))
                except ValueError:
                    #If unparsable format, return none to indicate failure.
                    return None
        return schedule
    except:
        return None


def is_time_available(time_slot, schedule):
    """
    Checks if a given time slot is available in a given schedule.
    """
    start_time, end_time = time_slot
    for busy_start, busy_end in schedule:
        if start_time < busy_end and end_time > busy_start:
            return False  # Overlap found
    return True


def solve_meeting_schedule(question):
    """
    Solves the meeting scheduling problem using a rule-based inference engine.
    """

    try:
        # 1. Extract information using regular expressions (more robust)
        task_match = re.search(r"TASK: (.*?)Here are the existing schedules", question, re.DOTALL)
        if not task_match:
            return "Error: Could not parse task description."
        task_description = task_match.group(1).strip()

        participants_match = re.findall(r"for (.*?) for", task_description)
        if not participants_match:
            return "Error: Could not parse participant names."
        participants_str = participants_match[0] #.split(', ')
        participants = [name.strip() for name in participants_str.split(',')]


        duration_match = re.search(r"for (.*?)(?:between|on)", task_description)
        if not duration_match:
          return "Error: Could not determine the meeting duration."

        duration_str = duration_match.group(1).strip()
        if "half an hour" in duration_str:
          duration = 0.5
        elif "one hour" in duration_str:
          duration = 1.0
        else:
          duration_val = float(duration_str.replace("hours","").replace("hour","").strip())
          duration = duration_val

        schedule_section_match = re.search(r"Here are the existing schedules for everyone during the day:\s*(.*?)Find a time", question, re.DOTALL)
        if not schedule_section_match:
            return "Error: Could not parse schedule information."
        schedule_section = schedule_section_match.group(1).strip()

        participant_schedules = {}
        for participant in participants:
            schedule_match = re.search(rf"{participant} (?:has meetings on Monday during|is busy on Monday during|has blocked their calendar on Monday during|is free the entire day\.|calendar is wide open the entire day\.) (.*?)(?:\n|$)", schedule_section, re.DOTALL)

            if schedule_match:

                schedule_str = schedule_match.group(1).strip()
                parsed_schedule = parse_schedule(schedule_str)
                if parsed_schedule is None:
                   return f"Error: Could not parse schedule for {participant}"
                participant_schedules[participant] = parsed_schedule
            else:
                participant_schedules[participant] = [] #Assume free



        # 2. Generate candidate time slots
        start_time = 9.0  # 9:00
        end_time = 17.0  # 17:00
        time_slots = []
        current_time = start_time
        while current_time + duration <= end_time:
            time_slots.append((current_time, current_time + duration))
            current_time += 0.5  # Check every 30 minutes


        # 3. Rule-based inference: Check availability for each time slot
        available_slots = []
        for slot in time_slots:
            is_available = True
            for participant, schedule in participant_schedules.items():
                if not is_time_available(slot, schedule):
                    is_available = False
                    break
            if is_available:
                available_slots.append(slot)

        # 4. Prioritize earliest availability (if specified) or return first available
        earliest_preference = "earliest availability" in question.lower()

        if available_slots:
            if earliest_preference:
                best_slot = available_slots[0]
            else:
                best_slot = available_slots[0] #Return the first one
            start_hour = int(best_slot[0])
            start_minute = int((best_slot[0] - start_hour) * 60)
            end_hour = int(best_slot[1])
            end_minute = int((best_slot[1] - end_hour) * 60)

            return f"Here is the proposed time: Monday, {start_hour:02}:{start_minute:02} - {end_hour:02}:{end_minute:02} "
        else:
            return "Unfortunately, there are no available time slots that satisfy the constraints."

    except Exception as e:
        return f"An unexpected error occurred: {str(e)}"


def main(question):
    """
    Main function to solve the meeting scheduling problem.
    """
    return solve_meeting_schedule(question)



# Example usage (for testing):
if __name__ == '__main__':
    example_question = """You are an expert at scheduling meetings. You are given a few constraints on the existing schedule of each participant, the meeting duration, and possibly some preferences on the meeting time. Note there exists a solution that works with existing schedule of every participant. Here are a few example tasks and solutions:

TASK: You need to schedule a meeting for Anthony, Pamela and Zachary for one hour between the work hours of 9:00 to 17:00 on Monday. 

Here are the existing schedules for everyone during the day: 
Anthony has meetings on Monday during 9:30 to 10:00, 12:00 to 13:00, 16:00 to 16:30; 
Pamela is busy on Monday during 9:30 to 10:00, 16:30 to 17:00; 
Zachary has meetings on Monday during 9:00 to 11:30, 12:00 to 12:30, 13:00 to 13:30, 14:30 to 15:00, 16:00 to 17:00; 

Pamela do not want to meet on Monday after 14:30. Find a time that works for everyone's schedule and constraints. 
SOLUTION: """
    answer = main(example_question)
    print(answer)