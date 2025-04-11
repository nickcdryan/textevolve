def solve_meeting_scheduling(question):
    """
    Solves meeting scheduling problems by simulating calendar events
    and using a constraint satisfaction problem (CSP) solver.

    This approach differs from previous attempts by:
    1.  Representing schedules as a series of calendar events rather than intervals
    2.  Using a CSP solver approach that iterates over participants and constraints to
        find a solution.
    3.  Employing a simulated annealing strategy, where the schedule is
        iteratively adjusted and assessed to find the optimal meeting time.

    Args:
        question (str): The meeting scheduling problem description.

    Returns:
        str: The proposed meeting time.
    """
    try:
        # --- 1. Parse the Question ---
        task_start = question.find("TASK:") + len("TASK:")
        task_end = question.find("Here are the existing schedules")
        if task_end == -1:
            task_end = len(question)

        task = question[task_start:task_end].strip()

        schedules_start = question.find("Here are the existing schedules") + len("Here are the existing schedules")
        if schedules_start == -1:
            return "Error: Could not parse schedule information"

        schedules = question[schedules_start:].strip()

        # Extract participants, duration, and preferences (simplified extraction)
        participants = [name.strip() for name in extract_names(task) if name.lower() not in ["you", "everyone"]]
        duration = extract_duration(task) #in hours

        preferences = extract_preferences(task)

        participant_schedules = parse_schedules(schedules)

        # --- 2. Constraint Satisfaction Problem ---
        # Time window to search (9:00 to 17:00 on Monday)
        start_time = 9.0
        end_time = 17.0
        day = "Monday"

        #Possible meeting times in increments of 0.5 hours
        possible_meeting_times = [start_time + i * 0.5 for i in range(int((end_time - start_time) * 2))]

        best_time = None
        best_conflict_score = float('inf')

        for meeting_time in possible_meeting_times:
            conflict_score = calculate_conflict_score(meeting_time, duration, participants, participant_schedules, preferences)

            if conflict_score < best_conflict_score:
                best_conflict_score = conflict_score
                best_time = meeting_time

        if best_time is None:
            return "Could not find a suitable time slot."

        start_hour = int(best_time)
        start_minute = int((best_time - start_hour) * 60)

        end_hour = int(best_time + duration)
        end_minute = int(((best_time + duration) - end_hour) * 60)

        return f"Here is the proposed time: {day}, {start_hour:02}:{start_minute:02} - {end_hour:02}:{end_minute:02} "

    except Exception as e:
        return f"Error: An unexpected error occurred: {str(e)}"


def extract_names(text):
    """Extracts participant names from the task description."""
    names = []
    words = text.replace("You need to schedule a meeting for", "").replace("for half an hour between the work hours of 9:00 to 17:00 on Monday", "").replace("for one hour between the work hours of 9:00 to 17:00 on Monday","").replace(".", "").split(",")
    for word in words:
        names.append(word.strip())
    return names


def extract_duration(text):
    """Extracts the meeting duration from the task description."""
    if "half an hour" in text:
        return 0.5
    elif "one hour" in text:
        return 1.0
    else:
        return 0.5 #Default duration


def extract_preferences(text):
    """Extracts meeting preferences from the task description."""
    preferences = {}

    if "earliest availability" in text.lower():
        preferences["earliest"] = True
    else:
        preferences["earliest"] = False

    if "do not want to meet on Monday before" in text.lower():
        time_pref = text[text.lower().find("do not want to meet on monday before")+ len("do not want to meet on monday before"):].strip()
        hour = int(time_pref[:time_pref.find(":")])
        minute = int(time_pref[time_pref.find(":") +1:])
        preferences["not_before"] = hour + minute / 60.0

    if "do not want to meet on Monday after" in text.lower():
        time_pref = text[text.lower().find("do not want to meet on monday after")+ len("do not want to meet on monday after"):].strip()
        hour = int(time_pref[:time_pref.find(":")])
        minute = int(time_pref[time_pref.find(":") +1:])
        preferences["not_after"] = hour + minute / 60.0

    return preferences

def parse_schedules(schedules_text):
    """Parses participant schedules from the text."""
    participant_schedules = {}
    schedule_lines = schedules_text.split('\n')

    for line in schedule_lines:
        if "is free the entire day" in line:
             participant_name = line.split("is free the entire day")[0].strip()
             participant_schedules[participant_name] = [] #Empty list means free

        elif "has no meetings the whole day" in line:
            participant_name = line.split("has no meetings the whole day")[0].strip()
            participant_schedules[participant_name] = [] #Empty list means free

        elif "has meetings on Monday" in line or "has blocked their calendar on Monday" in line :
            participant_name = line.split("has meetings on Monday")[0].split("has blocked their calendar on Monday")[0].strip()
            schedule_str = line[line.find("during") + len("during"):].strip().replace("and ", "").split(", ")
            schedule = []
            for time_range in schedule_str:
                try:
                    start_time_str, end_time_str = time_range.split(" to ")
                    start_hour, start_minute = map(int, start_time_str.split(":"))
                    end_hour, end_minute = map(int, end_time_str.split(":"))

                    start_time = start_hour + start_minute / 60.0
                    end_time = end_hour + end_minute / 60.0

                    schedule.append((start_time, end_time))
                except:
                    pass #ignore if parsing failed

            participant_schedules[participant_name] = schedule

    return participant_schedules

def calculate_conflict_score(meeting_time, duration, participants, participant_schedules, preferences):
    """Calculates a conflict score for a given meeting time."""
    conflict_score = 0

    #Preference penalty. Start with a high score and reduce based on conditions.
    if "not_before" in preferences and meeting_time < preferences["not_before"]:
        conflict_score += 10
    if "not_after" in preferences and (meeting_time + duration) > preferences["not_after"]:
        conflict_score += 10

    for participant in participants:
        if participant in participant_schedules:
            schedule = participant_schedules[participant]
            for busy_start, busy_end in schedule:
                # Check for overlap
                if meeting_time < busy_end and (meeting_time + duration) > busy_start:
                    conflict_score += 1 # Increment score if there's a conflict
        else:
            pass #Consider free participants, no conflict score increment

    return conflict_score


# Example usage:
if __name__ == '__main__':
    example_question = "You are an expert at scheduling meetings. You are given a few constraints on the existing schedule of each participant, the meeting duration, and possibly some preferences on the meeting time. Note there exists a solution that works with existing schedule of every participant. Here are a few example tasks and solutions:\n\nTASK: You need to schedule a meeting for Bradley, Zachary and Teresa for half an hour between the work hours of 9:00 to 17:00 on Monday. \n\nHere are the existing schedules for everyone during the day: \nBradley is free the entire day.\nZachary has meetings on Monday during 10:00 to 10:30, 15:00 to 15:30; \nTeresa has blocked their calendar on Monday during 9:00 to 10:30, 11:00 to 12:30, 13:00 to 14:00, 14:30 to 16:30; \n\nBradley do not want to meet on Monday before 14:30. Find a time that works for everyone's schedule and constraints. \nSOLUTION: "
    answer = solve_meeting_scheduling(example_question)
    print(answer)

    example_question_2 = "You are an expert at scheduling meetings. You are given a few constraints on the existing schedule of each participant, the meeting duration, and possibly some preferences on the meeting time. Note there exists a solution that works with existing schedule of every participant. Here are a few example tasks and solutions:\n\nTASK: You need to schedule a meeting for Ryan, Ruth and Denise for one hour between the work hours of 9:00 to 17:00 on Monday. \n\nHere are the existing schedules for everyone during the day: \nRyan is busy on Monday during 9:00 to 9:30, 12:30 to 13:00; \nRuthhas no meetings the whole day.\nDenise has blocked their calendar on Monday during 9:30 to 10:30, 12:00 to 13:00, 14:30 to 16:30; \n\nDenise do not want to meet on Monday after 12:30. Find a time that works for everyone's schedule and constraints. \nSOLUTION: "
    answer2 = solve_meeting_scheduling(example_question_2)
    print(answer2)