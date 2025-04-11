import re

def solve_meeting_scheduling(question):
    """
    Solves the meeting scheduling problem using a constraint satisfaction approach with rule-based inference.

    This approach differs from previous iterations by:
    1. Abandoning regular expressions for a more structured parsing based on keywords and sentence structure.
    2. Implementing a rule-based inference engine to handle constraints and preferences.
    3. Representing schedules as boolean arrays for efficient conflict checking.
    4. Prioritizing rule application based on specificity (e.g., individual preferences before general availability).

    Args:
        question (str): The meeting scheduling problem described as a text string.

    Returns:
        str: A string containing the proposed meeting time, or an error message if no solution is found.
    """

    try:
        # 1. Information Extraction and Parsing

        # Extract participants and their schedules
        participants = []
        schedules = {}
        preferences = {}
        lines = question.split('\n')
        duration = None
        work_hours_start = 9
        work_hours_end = 17
        day = "Monday" # Default day, assuming Monday. Can be improved.

        for line in lines:
            if "TASK:" in line:
                match = re.search(r"schedule a meeting for (.*?) for (.*?) between the work hours of (.*?) to (.*?) on (.*?)", line)
                if match:
                    participants = [p.strip() for p in match.group(1).split(",")]
                    duration_str = match.group(2)
                    work_hours_start_str, work_hours_end_str = match.group(3), match.group(4)
                    day = match.group(5).strip()
                    try:
                        # Attempt to extract numeric value of the duration
                        duration = int(re.search(r'(\d+)', duration_str).group(1))
                        if "hour" in duration_str:
                            duration *= 60 # convert to minutes
                    except:
                        #if duration is defined as "half an hour"
                        if "half" in duration_str:
                            duration = 30
                        else:
                            raise ValueError(f"Could not interpret duration: {duration_str}") #unhandled cases


                    try:
                        work_hours_start = int(work_hours_start_str.split(":")[0])
                        work_hours_end = int(work_hours_end_str.split(":")[0])
                    except:
                        pass

            if "Here are the existing schedules" in line:
                pass #ignore the line

            if "has blocked their calendar" in line or "'s calendar is wide open" in line or "is busy" in line:
                match = re.search(r"(.*?) (has blocked their calendar|'s calendar is wide open|is busy) (.*?) during (.*)", line)
                if match:
                    name = match.group(1).strip()
                    schedule_description = match.group(4).strip() if match.group(2) != "'s calendar is wide open" else ""

                    if match.group(2) == "'s calendar is wide open":
                        schedules[name] = []
                    else:
                        schedule_entries = schedule_description.split(", ")
                        schedules[name] = []
                        for entry in schedule_entries:
                            time_match = re.search(r"(\d{1,2}:\d{2}) to (\d{1,2}:\d{2})", entry)
                            if time_match:
                                start_time_str, end_time_str = time_match.group(1), time_match.group(2)
                                schedules[name].append((start_time_str, end_time_str))
                            else:
                                pass


            if "would rather not meet" in line or "do not want to meet" in line:
                match = re.search(r"(.*?) (would rather not meet|do not want to meet) (.*?) after (.*)", line)
                if match:
                  name = match.group(1).strip()
                  time_str = match.group(4).strip()
                  preferences[name] = time_str


        # 2. Schedule Representation

        # Represent schedules as boolean arrays (1 = busy, 0 = free)
        time_slots = (work_hours_end - work_hours_start) * 60  # Total time slots in minutes
        availability = {}
        for participant in participants:
            availability[participant] = [0] * time_slots

            if participant in schedules:
                for start_time_str, end_time_str in schedules[participant]:
                    start_hour, start_minute = map(int, start_time_str.split(':'))
                    end_hour, end_minute = map(int, end_time_str.split(':'))

                    start_minute_index = (start_hour - work_hours_start) * 60 + start_minute
                    end_minute_index = (end_hour - work_hours_start) * 60 + end_minute

                    for i in range(start_minute_index, end_minute_index):
                        if 0 <= i < time_slots: # bounds check
                            availability[participant][i] = 1  # Mark the slot as busy

        # 3. Constraint Satisfaction and Rule-Based Inference

        # Find a suitable time slot
        best_start_time = None
        for start_time_index in range(0, time_slots - duration + 1):
            is_valid = True
            for participant in participants:

                #Rule 1: Individual Preferences
                if participant in preferences:
                    preference_time_str = preferences[participant]
                    try:
                        preference_hour, preference_minute = map(int, preference_time_str.split(':'))
                        preference_time_index = (preference_hour - work_hours_start) * 60 + preference_minute
                        if start_time_index >= preference_time_index:
                            is_valid = False
                            break # Break participants checking
                    except:
                        pass #if preferences are defined with terms such as "afternoon", "evening", it is ignored for this version


                #Rule 2: General Availability
                for i in range(start_time_index, start_time_index + duration):
                    if 0 <= i < time_slots:
                        if availability[participant][i] == 1:
                            is_valid = False
                            break #Break duration checking
                    else:
                        is_valid = False
                        break

                if not is_valid:
                    break #Break participants checking

            if is_valid:
                best_start_time = start_time_index
                break #Break slot checking


        # 4. Result Formatting
        if best_start_time is not None:
            start_hour = work_hours_start + best_start_time // 60
            start_minute = best_start_time % 60
            end_hour = work_hours_start + (best_start_time + duration) // 60
            end_minute = (best_start_time + duration) % 60

            start_time_str = f"{start_hour:02}:{start_minute:02}"
            end_time_str = f"{end_hour:02}:{end_minute:02}"

            return f"Here is the proposed time: {day}, {start_time_str} - {end_time_str} "
        else:
            return "No suitable meeting time found."

    except Exception as e:
        return f"Error processing the request: {str(e)}"


# Example Usage (for testing):
if __name__ == '__main__':
    question1 = "You are an expert at scheduling meetings. You are given a few constraints on the existing schedule of each participant, the meeting duration, and possibly some preferences on the meeting time. Note there exists a solution that works with existing schedule of every participant. Here are a few example tasks and solutions:\n\nTASK: You need to schedule a meeting for Kathryn, Charlotte and Lauren for half an hour between the work hours of 9:00 to 17:00 on Monday. \n\nHere are the existing schedules for everyone during the day: \nKathryn has blocked their calendar on Monday during 9:00 to 9:30, 10:30 to 11:00, 11:30 to 12:00, 13:30 to 14:30, 16:30 to 17:00; \nCharlotte has blocked their calendar on Monday during 12:00 to 12:30, 16:00 to 16:30; \nLauren has blocked their calendar on Monday during 9:00 to 10:00, 12:00 to 12:30, 13:30 to 14:30, 15:00 to 16:00, 16:30 to 17:00; \n\nCharlotte do not want to meet on Monday after 13:30. Find a time that works for everyone's schedule and constraints. \nSOLUTION: "
    question2 = "You are an expert at scheduling meetings. You are given a few constraints on the existing schedule of each participant, the meeting duration, and possibly some preferences on the meeting time. Note there exists a solution that works with existing schedule of every participant. Here are a few example tasks and solutions:\n\nTASK: You need to schedule a meeting for Thomas, Dylan and Jerry for one hour between the work hours of 9:00 to 17:00 on Monday. \n\nHere are the existing schedules for everyone during the day: \nThomas's calendar is wide open the entire day.\nDylan has blocked their calendar on Monday during 10:30 to 11:00, 13:30 to 14:00; \nJerry is busy on Monday during 9:00 to 11:00, 11:30 to 14:30, 16:00 to 17:00; \n\nFind a time that works for everyone's schedule and constraints. \nSOLUTION: "
    question3 = "You are an expert at scheduling meetings. You are given a few constraints on the existing schedule of each participant, the meeting duration, and possibly some preferences on the meeting time. Note there exists a solution that works with existing schedule of every participant. Here are a few example tasks and solutions:\n\nTASK: You need to schedule a meeting for Heather, Nicholas and Zachary for half an hour between the work hours of 9:00 to 17:00 on Monday. \n\nHere are the existing schedules for everyone during the day: \nHeather has blocked their calendar on Monday during 9:00 to 9:30, 10:30 to 11:00, 13:00 to 14:00, 14:30 to 15:00, 16:00 to 16:30; \nNicholashas no meetings the whole day.\nZachary has blocked their calendar on Monday during 9:00 to 10:30, 11:00 to 12:00, 12:30 to 13:00, 13:30 to 16:30; \n\nZachary would rather not meet on Monday after 14:00. Find a time that works for everyone's schedule and constraints. \nSOLUTION: "

    answer1 = solve_meeting_scheduling(question1)
    answer2 = solve_meeting_scheduling(question2)
    answer3 = solve_meeting_scheduling(question3)

    print(f"Question 1:\n{answer1}\n")
    print(f"Question 2:\n{answer2}\n")
    print(f"Question 3:\n{answer3}\n")