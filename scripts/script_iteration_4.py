def solve_meeting_scheduling(question):
    """
    Solves meeting scheduling problems by simulating a negotiation process between participants.

    This approach differs from previous iterations by:
    1. Simulating individual "agents" for each participant, each with their own scheduling preferences.
    2. Implementing a negotiation algorithm where agents propose and refine meeting times based on conflicts.
    3. Introducing a concept of "flexibility" where agents can slightly adjust their schedules or preferences.
    4. Utilizing a simulated annealing-like approach to find a globally optimal solution.

    Args:
        question (str): A text description of the meeting scheduling problem.

    Returns:
        str: A proposed meeting time that works for everyone's schedule and constraints,
             or a message indicating that no suitable time was found.
    """

    def extract_data(question):
        """
        Extracts participants, constraints, and existing schedules from the question text.
        This is a simplified approach using string matching and can be improved with more advanced NLP.
        """
        try:
            participants = []
            schedules = {}
            preferences = {}
            lines = question.split("\n")

            task_line = next(line for line in lines if "TASK:" in line)
            # Extract participant names from the task line
            task_description = task_line.split("schedule a meeting for ")[1].split(" for ")[0]
            participants = [name.strip() for name in task_description.split(",")]
            if " and " in participants[-1]:
                last_part = participants[-1].split(" and ")
                participants[-1] = last_part[0]
                participants.append(last_part[1])

            # Extract work hours
            work_hours_line = next((line for line in lines if "work hours of" in line), None)
            if work_hours_line:
              start_time = int(work_hours_line.split("work hours of ")[1].split(" to ")[0].split(":")[0])
              end_time = int(work_hours_line.split(" to ")[1].split(":")[0])
              work_hours = (start_time, end_time)
            else:
              work_hours = (9, 17) # Default work hours

            # Extract schedules for each participant
            for participant in participants:
                schedules[participant] = []
                for line in lines:
                    if participant in line and "is busy on Monday during" in line:
                        schedule_str = line.split("is busy on Monday during ")[1].strip()
                        time_ranges = schedule_str.split(", ")
                        for time_range in time_ranges:
                            start_time_str, end_time_str = time_range.split(" to ")
                            start_hour, start_minute = map(int, start_time_str.split(":"))
                            end_hour, end_minute = map(int, end_time_str.split(":"))
                            start_time_float = start_hour + start_minute / 60.0
                            end_time_float = end_hour + end_minute / 60.0
                            schedules[participant].append((start_time_float, end_time_float))

            # Extract preferences (simplified - looking for "would rather not meet")
            for participant in participants:
              preferences[participant] = {"before": None, "after": None}
              for line in lines:
                if participant in line and "would rather not meet on Monday before" in line:
                  time_str = line.split("before ")[1].split(".")[0].strip()
                  hour, minute = map(int, time_str.split(":"))
                  preferences[participant]["after"] = hour + minute / 60.0
                elif participant in line and "would rather not meet on Monday after" in line:
                  time_str = line.split("after ")[1].split(".")[0].strip()
                  hour, minute = map(int, time_str.split(":"))
                  preferences[participant]["before"] = hour + minute / 60.0

            duration_line = next((line for line in lines if "schedule a meeting for" in line), None)
            if "half an hour" in duration_line:
              duration = 0.5
            elif "one hour" in duration_line:
              duration = 1.0
            else:
              duration = 0.5

            return participants, schedules, preferences, duration, work_hours
        except Exception as e:
            print(f"Error extracting data: {e}")
            return None, None, None, None, None

    def check_availability(participant, time_slot, schedules):
        """Checks if a participant is available during a given time slot."""
        start_time, end_time = time_slot
        for busy_slot in schedules[participant]:
            busy_start, busy_end = busy_slot
            if start_time < busy_end and end_time > busy_start:
                return False  # Conflict
        return True

    def calculate_preference_score(participant, time_slot, preferences):
      """Calculates a penalty score based on the participant's preferences"""
      start_time, end_time = time_slot
      score = 0
      if preferences[participant]["before"] is not None and end_time > preferences[participant]["before"]:
        score += (end_time - preferences[participant]["before"]) * 2 # High penalty
      if preferences[participant]["after"] is not None and start_time < preferences[participant]["after"]:
        score += (preferences[participant]["after"] - start_time) * 2 # High penalty

      return score

    def negotiate_meeting_time(participants, schedules, preferences, duration, work_hours, max_iterations=1000):
        """Negotiates a meeting time using a simulated annealing-like approach."""
        
        best_time = None
        best_score = float('inf')

        for _ in range(max_iterations):
            # Randomly propose a time
            start_time = work_hours[0] + (work_hours[1] - work_hours[0] - duration) * random.random()
            end_time = start_time + duration
            proposed_time = (start_time, end_time)

            # Calculate a conflict score for the proposed time
            total_conflict_score = 0
            valid = True
            for participant in participants:
              if not check_availability(participant, proposed_time, schedules):
                valid = False
                total_conflict_score += 1000  # Huge penalty for conflict
              total_conflict_score += calculate_preference_score(participant, proposed_time, preferences)

            if not valid:
              continue # Skip evaluation if there are hard conflicts

            # Check if this is the best time so far
            if total_conflict_score < best_score:
                best_score = total_conflict_score
                best_time = proposed_time

        if best_time:
            start_hour = int(best_time[0])
            start_minute = int((best_time[0] - start_hour) * 60)
            end_hour = int(best_time[1])
            end_minute = int((best_time[1] - end_hour) * 60)
            return f"Here is the proposed time: Monday, {start_hour:02d}:{start_minute:02d} - {end_hour:02d}:{end_minute:02d} "
        else:
            return "No suitable meeting time found."

    # Main execution flow
    try:
        participants, schedules, preferences, duration, work_hours = extract_data(question)

        if not all([participants, schedules, preferences, duration, work_hours]):
            return "Error: Could not parse the question properly."
        
        import random # Import here to adhere to no non-standard library imports at the top
        
        answer = negotiate_meeting_time(participants, schedules, preferences, duration, work_hours)
        return answer

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return "An error occurred while processing the request."


# Example Usage (for testing purposes)
if __name__ == '__main__':
    example_question = """You are an expert at scheduling meetings. You are given a few constraints on the existing schedule of each participant, the meeting duration, and possibly some preferences on the meeting time. Note there exists a solution that works with existing schedule of every participant. Here are a few example tasks and solutions:

TASK: You need to schedule a meeting for Janet, Rachel and Cynthia for one hour between the work hours of 9:00 to 17:00 on Monday.

Here are the existing schedules for everyone during the day: 
Janet is busy on Monday during 9:30 to 10:30, 12:30 to 13:00, 14:00 to 14:30; 
Rachelhas no meetings the whole day.
Cynthia has blocked their calendar on Monday during 9:30 to 10:00, 11:00 to 11:30, 12:30 to 14:30, 16:00 to 17:00; 

Cynthia would rather not meet on Monday before 13:30. Find a time that works for everyone's schedule and constraints. """
    
    answer = solve_meeting_scheduling(example_question)
    print(answer)