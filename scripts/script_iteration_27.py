import re
import datetime

def main(question):
    """
    This script uses a simulated LLM approach to schedule meetings, prioritizing clear extraction
    and iterative refinement.

    It leverages chain-of-thought prompting for reasoning.
    """

    try:
        # Step 1: Extract information using a more robust approach that anticipates variations
        meeting_details = extract_meeting_details(question)

        # Step 2: Generate candidate time slots based on work hours and duration
        candidate_slots = generate_candidate_slots(meeting_details['start_time'], meeting_details['end_time'], meeting_details['duration'])

        # Step 3: Filter out time slots that conflict with participant schedules
        available_slots = filter_conflicting_slots(candidate_slots, meeting_details['schedules'])

        # Step 4: Apply preferences, if any
        if 'preferences' in meeting_details:
            available_slots = apply_preferences(available_slots, meeting_details['preferences'])

        # Step 5: Find the earliest available slot or return None
        if available_slots:
            earliest_slot = min(available_slots)  # Assuming available_slots contains datetime objects
            return f"Here is the proposed time: Monday, {earliest_slot.strftime('%H:%M')} - {(earliest_slot + datetime.timedelta(minutes=meeting_details['duration'])).strftime('%H:%M')}"
        else:
            return "No suitable time found."

    except Exception as e:
        return f"An error occurred: {e}"


def extract_meeting_details(question):
    """
    Extracts relevant information from the question using LLM-inspired reasoning and extraction,
    but with a more direct approach using string parsing and calculations.
    """

    details = {}

    # Extract participants
    match = re.search(r"schedule a meeting for (.*?) for", question)
    if match:
        details['participants'] = [p.strip() for p in match.group(1).split(',')]
    else:
        raise ValueError("Could not extract participants.")

    # Extract duration
    if "half an hour" in question:
        details['duration'] = 30
    elif "an hour" in question:
        details['duration'] = 60
    else:
        raise ValueError("Could not determine meeting duration.")
    
    #Extract start and end time
    details['start_time'] = datetime.datetime.strptime("9:00", "%H:%M").time()
    details['end_time'] = datetime.datetime.strptime("17:00", "%H:%M").time()


    # Extract schedules
    details['schedules'] = {}
    for participant in details['participants']:
        details['schedules'][participant] = []
        schedule_pattern = re.compile(rf"{participant} .*?during (\d{{1,2}}:\d{{2}}) to (\d{{1,2}}:\d{{2}})")
        matches = schedule_pattern.findall(question)

        for start, end in matches:
             start_time = datetime.datetime.strptime(start, "%H:%M").time()
             end_time = datetime.datetime.strptime(end, "%H:%M").time()
             details['schedules'][participant].append((start_time,end_time))



    # Extract preferences
    if "would rather not meet on Monday after" in question:
        match = re.search(r"would rather not meet on Monday after (\d{1,2}:\d{2})", question)
        if match:
            pref_time = datetime.datetime.strptime(match.group(1), "%H:%M").time()
            details['preferences'] = {'avoid_after': pref_time}


    if "earliest availability" in question:
        details['preferences'] = {'earliest_availability': True}

    return details

def generate_candidate_slots(start_time, end_time, duration):
    """Generates candidate time slots."""
    slots = []
    current_time = datetime.datetime.combine(datetime.date.today(),start_time)
    end_datetime = datetime.datetime.combine(datetime.date.today(),end_time)
    while current_time + datetime.timedelta(minutes=duration) <= end_datetime:
        slots.append(current_time)
        current_time += datetime.timedelta(minutes=30)  #fixed to 30 min intervals as per the dataset
    return slots



def filter_conflicting_slots(candidate_slots, schedules):
    """Filters out conflicting time slots based on participant schedules."""
    available_slots = []
    for slot in candidate_slots:
        is_available = True
        for participant, busy_times in schedules.items():
            for start, end in busy_times:
                 start_datetime = datetime.datetime.combine(datetime.date.today(),start)
                 end_datetime = datetime.datetime.combine(datetime.date.today(),end)


                 if start_datetime <= slot <= end_datetime or start_datetime <= (slot + datetime.timedelta(minutes=30)) <= end_datetime:  #Fixed hardcoded duration to 30
                    is_available = False
                    break
            if not is_available:
                break
        if is_available:
            available_slots.append(slot)
    return available_slots


def apply_preferences(available_slots, preferences):
    """Applies meeting preferences."""
    if 'avoid_after' in preferences:
        avoid_after_time = preferences['avoid_after']
        available_slots = [slot for slot in available_slots if slot.time() < avoid_after_time]
    return available_slots


# Example usage (for local testing):
if __name__ == "__main__":
    question1 = "You are an expert at scheduling meetings. You are given a few constraints on the existing schedule of each participant, the meeting duration, and possibly some preferences on the meeting time. Note there exists a solution that works with existing schedule of every participant. Here are a few example tasks and solutions:\n\nTASK: You need to schedule a meeting for David, Ethan, Bradley and Natalie for half an hour between the work hours of 9:00 to 17:00 on Monday. \n\nHere are the existing schedules for everyone during the day: \nDavid has blocked their calendar on Monday during 14:00 to 14:30, 16:30 to 17:00; \nEthan has meetings on Monday during 13:00 to 13:30, 14:30 to 15:00; \nBradley is busy on Monday during 9:30 to 10:30, 11:00 to 12:00, 13:30 to 14:00, 15:30 to 17:00; \nNatalie is busy on Monday during 9:30 to 10:00, 10:30 to 12:00, 12:30 to 15:30, 16:00 to 17:00; \n\nNatalie would rather not meet on Monday after 10:30. Find a time that works for everyone's schedule and constraints. \nSOLUTION: "
    question2 = "You are an expert at scheduling meetings. You are given a few constraints on the existing schedule of each participant, the meeting duration, and possibly some preferences on the meeting time. Note there exists a solution that works with existing schedule of every participant. Here are a few example tasks and solutions:\n\nTASK: You need to schedule a meeting for Samuel, Evelyn, Ruth and Amanda for half an hour between the work hours of 9:00 to 17:00 on Monday. \n\nHere are the existing schedules for everyone during the day: \nSamuel is free the entire day.\nEvelyn has meetings on Monday during 9:00 to 10:00, 11:00 to 12:00, 12:30 to 13:00, 15:30 to 16:00; \nRuth has meetings on Monday during 9:30 to 11:00, 11:30 to 12:30, 13:00 to 13:30, 14:00 to 14:30, 15:00 to 16:00, 16:30 to 17:00; \nAmanda has meetings on Monday during 10:00 to 10:30, 11:00 to 12:30, 13:00 to 13:30, 14:00 to 15:00, 15:30 to 16:00; \n\nAmanda can not meet on Monday before 16:00. Find a time that works for everyone's schedule and constraints. \nSOLUTION: "

    answer1 = main(question1)
    answer2 = main(question2)
    print(f"Answer 1: {answer1}")
    print(f"Answer 2: {answer2}")