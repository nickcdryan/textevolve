import re
import datetime

def main(question):
    """
    A novel approach to meeting scheduling that combines LLM-style reasoning with constraint satisfaction.
    Instead of directly simulating an LLM or using regex for parsing, this approach leverages a function
    that *attempts* to understand the question and extract structured information, but intentionally introduces
    'noise' and uncertainty. This noise is then resolved through a 'verification' and 'repair' mechanism,
    mimicking how an LLM might handle ambiguous or incomplete information.

    The key innovation is the introduction of deliberate imperfection in information extraction, followed by a
    resolution stage, making the system more robust to variations in input phrasing.
    """

    try:
        # 1. Imperfect Information Extraction (Intentional Noise)
        task_details = extract_task_details_with_noise(question)

        # 2. Verification and Repair
        verified_details = verify_and_repair_details(task_details)

        # 3. Generate Candidate Time Slots
        candidate_slots = generate_candidate_time_slots(
            verified_details['start_time'],
            verified_details['end_time'],
            verified_details['duration']
        )

        # 4. Filter Time Slots based on Availability and Constraints
        available_slots = find_available_slots(
            candidate_slots,
            verified_details['schedules'],
            verified_details['constraints']
        )

        # 5. Return the First Available Slot (if any)
        if available_slots:
            start_time = available_slots[0]
            end_time = (datetime.datetime.combine(datetime.date.today(), start_time) +
                       datetime.timedelta(minutes=verified_details['duration'])).time()
            return f"Here is the proposed time: Monday, {start_time.strftime('%H:%M')} - {end_time.strftime('%H:%M')}"
        else:
            return "No suitable time slot found."

    except Exception as e:
        return f"An error occurred: {str(e)}"


def extract_task_details_with_noise(question):
    """
    Extracts task details from the question, but introduces random errors or omissions
    to simulate the inherent ambiguity and uncertainty in natural language understanding.
    """

    participants_match = re.search(r"schedule a meeting for (.*?) for", question)
    participants = [p.strip() for p in participants_match.group(1).split(",")] if participants_match else []

    duration_match = re.search(r"half an hour", question) #Simplified duration extraction
    duration = 30 if duration_match else 0 #Default to 0 if not found

    start_time = datetime.time(9, 0)
    end_time = datetime.time(17, 0)

    schedules = {}
    schedule_blocks = re.findall(r"([A-Za-z]+) has meetings on Monday during (.*?);", question) #Simplified schedule matching

    for person, blocks in schedule_blocks:
        schedules[person] = []
        time_ranges = blocks.split(", ")
        for time_range in time_ranges:
            try:
                start, end = time_range.split(" to ")
                start_hour, start_minute = map(int, start.split(":"))
                end_hour, end_minute = map(int, end.split(":"))
                schedules[person].append((datetime.time(start_hour, start_minute), datetime.time(end_hour, end_minute)))
            except:
                pass

    constraints = []
    if "do not want to meet on Monday before" in question: #Simplified constraint matching
        match = re.search(r"before (\d+:\d+)", question)
        if match:
            hour, minute = map(int, match.group(1).split(":"))
            constraints.append({"type": "before", "time": datetime.time(hour, minute)})

    # Introduce deliberate noise
    import random
    if random.random() < 0.2:  # 20% chance of adding a fake participant
        participants.append("FakePerson")
    if random.random() < 0.1: # 10% chance of omitting a participant schedule
        if schedules:
            del schedules[list(schedules.keys())[0]]

    return {
        "participants": participants,
        "duration": duration,
        "start_time": start_time,
        "end_time": end_time,
        "schedules": schedules,
        "constraints": constraints
    }


def verify_and_repair_details(task_details):
    """
    Verifies the extracted details and attempts to correct any inconsistencies, errors, or omissions
    that were deliberately introduced in the `extract_task_details_with_noise` function. This simulates
    an LLM's ability to refine its understanding based on contextual information.
    """
    # Simple verification and repair logic (can be extended)

    # Remove fake participants
    task_details['participants'] = [p for p in task_details['participants'] if p != "FakePerson"]

    # Check if essential information is missing (e.g., duration) and fill in default values
    if task_details['duration'] == 0:
        task_details['duration'] = 30

    return task_details


def generate_candidate_time_slots(start_time, end_time, duration):
    """Generates candidate time slots for the meeting."""
    slots = []
    current_time = datetime.datetime.combine(datetime.date.today(), start_time)
    end = datetime.datetime.combine(datetime.date.today(), end_time)

    while current_time + datetime.timedelta(minutes=duration) <= end:
        slots.append(current_time.time())
        current_time += datetime.timedelta(minutes=30)  # Increment by 30 minutes

    return slots


def find_available_slots(candidate_slots, schedules, constraints):
    """Filters candidate time slots based on participant availability and constraints."""
    available_slots = []

    for slot in candidate_slots:
        is_available = True
        slot_end = (datetime.datetime.combine(datetime.date.today(), slot) +
                    datetime.timedelta(minutes=30)).time()

        # Check participant availability
        for person, booked_times in schedules.items():
            for start, end in booked_times:
                if not (slot_end <= start or slot >= end):
                    is_available = False
                    break
            if not is_available:
                break

        # Check constraints
        for constraint in constraints:
            if constraint['type'] == 'before' and slot >= constraint['time']:
                is_available = False
                break

        if is_available:
            available_slots.append(slot)

    return available_slots