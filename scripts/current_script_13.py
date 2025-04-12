import datetime

def main(question):
    """
    Schedules a meeting by simulating a round-robin voting process among participants.

    This approach uses LLM-like reasoning by simulating a decision-making process
    where each participant 'votes' on available time slots based on their schedule
    and preferences. The time slot with the most votes wins.

    Args:
        question (str): A string containing the meeting scheduling task details.

    Returns:
        str: A string indicating the proposed meeting time.
    """
    try:
        # Step 1: Extract information using LLM-simulated reasoning
        participants, schedules, duration, work_hours, preferences = extract_info(question)

        # Step 2: Generate potential time slots within work hours
        potential_slots = generate_time_slots(work_hours, duration)

        # Step 3: Simulate voting process - each participant votes for available slots
        votes = {}
        for slot in potential_slots:
            votes[slot] = 0

        for participant in participants:
            for slot in potential_slots:
                if is_slot_available(slot, schedules[participant]):
                    votes[slot] += 1 # Increment vote if the slot is available

            #Applying preferences as well (even though not all examples have them).  Treat as "veto" for exploration.
            if preferences and participant in preferences:
                avoid_before = preferences[participant].get("avoid_before")
                if avoid_before:
                    avoid_time = datetime.datetime.strptime(avoid_before, "%H:%M").time()
                    for slot in potential_slots:
                        start_time = datetime.datetime.strptime(slot.split(', ')[1].split(' - ')[0], "%H:%M").time()

                        if start_time < avoid_time: #Avoid slots starting before preferred time
                            votes[slot] = -999  #effectively vetoing the slot


        # Step 4: Determine the winning time slot (slot with most votes)
        best_slot = None
        max_votes = -1
        for slot, vote_count in votes.items():
            if vote_count > max_votes:
                max_votes = vote_count
                best_slot = slot

        # Step 5: Return the proposed time
        if best_slot:
            return "Here is the proposed time: " + best_slot
        else:
            return "No suitable time found."

    except Exception as e:
        return f"Error: {str(e)}"


def extract_info(question):
    """
    Extracts relevant information from the question string using basic string parsing.

    Simulates LLM information extraction.

    Args:
        question (str): The question string.

    Returns:
        tuple: Participants, their schedules, meeting duration, work hours, and preferences.
    """
    try:
        # Extract participants
        participants_line = question.split("schedule a meeting for ")[1].split(" for ")[0]
        participants = [p.strip() for p in participants_line.split(",")]

        # Extract schedules - assumes a specific structure in the question
        schedules = {}
        schedule_sections = question.split("Here are the existing schedules for everyone during the day:")[1].split("\n")
        for line in schedule_sections:
            if "has blocked their calendar" in line or "is busy on" in line or "is free the entire day" in line:
                name = line.split(" ")[0]
                if "is free the entire day" in line:
                    schedules[name] = []  # Empty schedule means free all day
                    continue
                schedule_str = line.split("Monday during ")[1].replace(";", "").strip()
                busy_times = []
                if schedule_str:
                   for time_range in schedule_str.split(","):
                       busy_times.append(time_range.strip())
                schedules[name] = busy_times

        # Extract duration
        duration_str = question.split(" for ")[1].split(" between ")[0]
        if "half an hour" in duration_str:
            duration = 30
        else:
            duration = int(duration_str.replace(" minutes", "")) #Basic parsing

        # Extract work hours
        work_hours_str = question.split(" between the work hours of ")[1].split(" on Monday")[0]
        start_time_str, end_time_str = work_hours_str.split(" to ")
        start_time = datetime.datetime.strptime(start_time_str, "%H:%M").time()
        end_time = datetime.datetime.strptime(end_time_str, "%H:%M").time()
        work_hours = (start_time, end_time)

        #Extract preferences
        preferences = {}
        if "would rather not meet on Monday before" in question:
            pref_name = question.split("would rather not meet on Monday before")[0].split()[-1]
            pref_time = question.split("would rather not meet on Monday before")[1].split('.')[0].strip()
            preferences[pref_name] = {"avoid_before": pref_time} #Assumes there is only one preference like this per question
        
        return participants, schedules, duration, work_hours, preferences

    except Exception as e:
        raise ValueError(f"Error extracting information: {str(e)}")



def generate_time_slots(work_hours, duration):
    """
    Generates potential meeting time slots.

    Args:
        work_hours (tuple): Start and end times for work hours.
        duration (int): Meeting duration in minutes.

    Returns:
        list: A list of potential time slots as strings.
    """
    start_time = datetime.datetime.combine(datetime.date.today(), work_hours[0])
    end_time = datetime.datetime.combine(datetime.date.today(), work_hours[1])
    slots = []
    current_time = start_time
    while current_time + datetime.timedelta(minutes=duration) <= end_time:
        slot_start = current_time.strftime("%H:%M")
        slot_end = (current_time + datetime.timedelta(minutes=duration)).strftime("%H:%M")
        slots.append(f"Monday, {slot_start} - {slot_end}")
        current_time += datetime.timedelta(minutes=30)  # Check in 30-minute increments
    return slots


def is_slot_available(slot, schedule):
    """
    Checks if a time slot is available based on a given schedule.

    Args:
        slot (str): The time slot to check.
        schedule (list): A list of busy time ranges.

    Returns:
        bool: True if the slot is available, False otherwise.
    """
    try:
        slot_start_str, slot_end_str = slot.split(', ')[1].split(" - ")
        slot_start = datetime.datetime.strptime(slot_start_str, "%H:%M").time()
        slot_end = datetime.datetime.strptime(slot_end_str, "%H:%M").time()

        for busy_time in schedule:
            busy_start_str, busy_end_str = busy_time.split(" to ")
            busy_start = datetime.datetime.strptime(busy_start_str, "%H:%M").time()
            busy_end = datetime.datetime.strptime(busy_end_str, "%H:%M").time()

            if (slot_start < busy_end) and (slot_end > busy_start):
                return False  # Slot conflicts with busy time
        return True  # Slot is available
    except ValueError as e:
        print(f"Error processing schedule: {str(e)}")
        return False


# Example usage (for local testing)
if __name__ == "__main__":
    example_question = """You are an expert at scheduling meetings. You are given a few constraints on the existing schedule of each participant, the meeting duration, and possibly some preferences on the meeting time. Note there exists a solution that works with existing schedule of every participant. Here are a few example tasks and solutions:

TASK: You need to schedule a meeting for Richard, Sarah, Gloria and Kathleen for half an hour between the work hours of 9:00 to 17:00 on Monday. 

Here are the existing schedules for everyone during the day: 
Richard has meetings on Monday during 9:00 to 10:00; 
Sarah has blocked their calendar on Monday during 11:00 to 11:30, 14:00 to 14:30; 
Gloria has blocked their calendar on Monday during 9:00 to 12:30, 13:30 to 14:00, 14:30 to 15:00, 15:30 to 16:00; 
Kathleen has blocked their calendar on Monday during 9:00 to 9:30, 10:30 to 11:00, 12:00 to 12:30, 13:30 to 15:30, 16:00 to 16:30; 

Gloria would rather not meet on Monday before 14:30. Find a time that works for everyone's schedule and constraints. """
    result = main(example_question)
    print(result)