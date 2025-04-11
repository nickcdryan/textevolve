import re

def solve_meeting_scheduling(question):
    """
    This function schedules a meeting based on the provided question string.

    Approach:
    This iteration introduces a rule-based expert system to solve the scheduling problem.
    It uses regular expressions to extract information and then applies a series of rules,
    prioritized based on constraint importance (hard constraints like existing meetings
    are prioritized over soft constraints like participant preferences).  The system
    attempts to mimic a human expert reasoning through the problem step-by-step.
    This avoids both the purely regex-based and fuzzy logic approaches from previous iterations,
    and the backtracking which proved unstable.

    Error Handling:
    The function includes robust error handling to catch missing or malformed input data,
    preventing the script from crashing. It returns a descriptive error message in such cases.
    """

    try:
        # 1. Information Extraction using Regex
        task_match = re.search(r"TASK:\s*(.*)", question)
        if not task_match:
            return "Error: Could not find TASK description."
        task_description = task_match.group(1)

        participant_names = re.findall(r"(for\s)([\w\s,]+)(for)", question)[0][1].strip().replace(' and', ',').split(',') if re.findall(r"(for\s)([\w\s,]+)(for)", question) else re.findall(r"(meeting\s)([\w\s,]+)(for)", question)[0][1].strip().replace(' and', ',').split(',') if re.findall(r"(meeting\s)([\w\s,]+)(for)", question) else None
        participant_names = [p.strip() for p in participant_names]
        if not participant_names:
            return "Error: Could not identify participants."


        schedule_blocks = {}
        for participant in participant_names:
            schedule_match = re.search(rf"{participant}.*Monday during\s*([\d:to,\s-]+);", question)
            if schedule_match:
                schedule_blocks[participant] = schedule_match.group(1)
            else:
                schedule_blocks[participant] = ""  # Empty schedule if not specified. This handles the "has no meetings" cases.


        preferences = {}
        for participant in participant_names:
            preference_match = re.search(rf"{participant}.*avoid.*Monday after\s*([\d:]+)", question) #preference by avoid specific time
            if preference_match:
                 preferences[participant] = preference_match.group(1)
            else:
                preferences[participant] = None


        duration_match = re.search(r"for\s*(one|half)\s*hour", question)
        duration = 60 if duration_match and duration_match.group(1) == "one" else 30

        # 2. Rule-Based Reasoning Engine
        available_slots = []
        start_time = 9 * 60  # 9:00 AM in minutes
        end_time = 17 * 60  # 5:00 PM in minutes


        #Convert blocks into minutes
        blocked_minutes = {}

        for participant in participant_names:
            blocked_minutes[participant] = []
            if schedule_blocks[participant]:
                time_ranges = schedule_blocks[participant].split(", ")
                for time_range in time_ranges:
                    times = time_range.split(" to ")
                    if len(times) == 2:
                        start = times[0]
                        end = times[1]

                        start_hour, start_minute = map(int, start.split(':'))
                        end_hour, end_minute = map(int, end.split(':'))

                        start_minutes = start_hour * 60 + start_minute
                        end_minutes = end_hour * 60 + end_minute

                        blocked_minutes[participant].extend(range(start_minutes, end_minutes))



        #Find available slots
        for start in range(start_time, end_time - duration + 1, 30): #Iterate 30-minute intervals
            is_available = True
            for participant in participant_names:
                for minute in range(start, start + duration):
                    if minute in blocked_minutes[participant]:
                        is_available = False
                        break
                if not is_available:
                    break

            if is_available:
                available_slots.append(start)



        #Apply preferences
        best_slot = None
        earliest_time = float('inf')
        for slot in available_slots:
            valid_slot = True
            for participant in participant_names:
                if preferences[participant]:
                    hour, minute = map(int, preferences[participant].split(':'))
                    cutoff_time_minutes = hour * 60 + minute
                    if slot > cutoff_time_minutes:
                        valid_slot = False
                        break
            if valid_slot:
                if slot < earliest_time:
                    earliest_time = slot
                    best_slot = slot


        if best_slot is not None:
            start_hour = best_slot // 60
            start_minute = best_slot % 60
            end_hour = (best_slot + duration) // 60
            end_minute = (best_slot + duration) % 60

            start_time_str = f"{start_hour:02}:{start_minute:02}"
            end_time_str = f"{end_hour:02}:{end_minute:02}"

            return f"Here is the proposed time: Monday, {start_time_str} - {end_time_str} "
        else:
            return "Error: No suitable meeting time found."

    except Exception as e:
        return f"Error: An unexpected error occurred: {str(e)}"


#Example usage (for testing)
if __name__ == '__main__':
    example_question = "You are an expert at scheduling meetings. You are given a few constraints on the existing schedule of each participant, the meeting duration, and possibly some preferences on the meeting time. Note there exists a solution that works with existing schedule of every participant. Here are a few example tasks and solutions:\n\nTASK: You need to schedule a meeting for Joyce, Christine and Alexander for half an hour between the work hours of 9:00 to 17:00 on Monday. \n\nHere are the existing schedules for everyone during the day: \nJoyce has meetings on Monday during 11:00 to 11:30, 13:30 to 14:00, 14:30 to 16:30; \nChristinehas no meetings the whole day.\nAlexander has meetings on Monday during 9:00 to 11:00, 12:00 to 12:30, 13:30 to 15:00, 15:30 to 16:00, 16:30 to 17:00; \n\nChristine can not meet on Monday before 12:00. Find a time that works for everyone's schedule and constraints. \nSOLUTION: "
    answer = solve_meeting_scheduling(example_question)
    print(answer)

    example_question_2 = "You are an expert at scheduling meetings. You are given a few constraints on the existing schedule of each participant, the meeting duration, and possibly some preferences on the meeting time. Note there exists a solution that works with existing schedule of every participant. Here are a few example tasks and solutions:\n\nTASK: You need to schedule a meeting for Brian, Billy and Patricia for half an hour between the work hours of 9:00 to 17:00 on Monday. \n\nHere are the existing schedules for everyone during the day: \nBrianhas no meetings the whole day.\nBilly is busy on Monday during 10:00 to 10:30, 11:30 to 12:00, 14:00 to 14:30, 16:30 to 17:00; \nPatricia has blocked their calendar on Monday during 9:00 to 12:30, 13:30 to 14:00, 14:30 to 16:00, 16:30 to 17:00; \n\nBilly would like to avoid more meetings on Monday after 15:30. Find a time that works for everyone's schedule and constraints. \nSOLUTION: "
    answer = solve_meeting_scheduling(example_question_2)
    print(answer)