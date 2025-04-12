import re
import datetime

def main(question):
    """
    Schedules a meeting based on participant availability and constraints.

    This approach simulates a multi-agent system. One agent extracts information,
    another finds potential time slots, and a third verifies the solution.
    """
    try:
        # Agent 1: Information Extraction (simulated LLM)
        participants, schedules, constraints = extract_info(question)

        # Agent 2: Time Slot Finder (simulated LLM)
        potential_slots = generate_time_slots(schedules, constraints)

        # Agent 3: Solution Verifier (simulated LLM)
        solution = verify_solution(potential_slots, schedules, constraints)

        if solution:
            return f"Here is the proposed time: {solution}"
        else:
            return "No suitable time found."

    except Exception as e:
        return f"Error: {str(e)}"

def extract_info(question):
    """
    Extracts information about participants, their schedules, and constraints
    using simulated LLM reasoning (regex-based parsing).
    """
    try:
        # Extract participants
        match = re.search(r"schedule a meeting for (.*?) for", question)
        if not match:
            raise ValueError("Could not extract participants.")
        participants = [name.strip() for name in match.group(1).split(',')]

        # Extract schedules
        schedules = {}
        for participant in participants:
            schedules[participant] = []
        
        schedule_blocks = re.findall(r"(\w+) has (blocked their calendar|meetings) on Monday during (\d{1,2}:\d{2}) to (\d{1,2}:\d{2})", question)

        for person, _, start_time, end_time in schedule_blocks:
            if person in schedules:
                schedules[person].append((start_time, end_time))

        # Extract constraints
        constraints = {}
        if "would rather not meet on Monday after" in question:
             match = re.search(r"would rather not meet on Monday after (\d{1,2}:\d{2})", question)
             if match:
                constraints['avoid_after'] = match.group(1)
        elif "can not meet on Monday before" in question:
            match = re.search(r"can not meet on Monday before (\d{1,2}:\d{2})", question)
            if match:
                constraints['avoid_before'] = match.group(1)
        
        if "earliest availability" in question:
            constraints['earliest_availability'] = True

        return participants, schedules, constraints

    except Exception as e:
        raise ValueError(f"Error extracting information: {str(e)}")

def generate_time_slots(schedules, constraints):
    """
    Generates potential time slots based on the given schedules and constraints.
    Uses simulated LLM reasoning for time slot generation.
    """
    try:
        start_time = datetime.time(9, 0)
        end_time = datetime.time(17, 0)
        meeting_duration = datetime.timedelta(minutes=30)
        current_time = datetime.datetime.combine(datetime.date.today(), start_time)  # Combine with a date
        end_datetime = datetime.datetime.combine(datetime.date.today(), end_time)

        potential_slots = []
        while current_time + meeting_duration <= end_datetime:  # Use <= for comparison with end_datetime
            potential_slots.append(current_time.time().strftime("%H:%M"))
            current_time += meeting_duration
        
        # Apply preference constraints
        filtered_slots = potential_slots[:] # Create a copy

        if 'avoid_after' in constraints:
            avoid_time = datetime.datetime.strptime(constraints['avoid_after'], "%H:%M").time()
            filtered_slots = [slot for slot in filtered_slots if datetime.datetime.strptime(slot, "%H:%M").time() < avoid_time]
            
        if 'avoid_before' in constraints:
            avoid_time = datetime.datetime.strptime(constraints['avoid_before'], "%H:%M").time()
            filtered_slots = [slot for slot in filtered_slots if datetime.datetime.strptime(slot, "%H:%M").time() >= avoid_time]
        
        return filtered_slots

    except Exception as e:
        raise ValueError(f"Error generating time slots: {str(e)}")

def verify_solution(potential_slots, schedules, constraints):
    """
    Verifies the potential time slots against the schedules and constraints
    using simulated LLM reasoning.
    """
    try:
        available_slots = []
        for slot in potential_slots:
            is_available = True
            slot_start_time = datetime.datetime.strptime(slot, "%H:%M").time()
            slot_end_time = (datetime.datetime.combine(datetime.date.today(), slot_start_time) + datetime.timedelta(minutes=30)).time()

            for participant, blocked_times in schedules.items():
                for blocked_start, blocked_end in blocked_times:
                    blocked_start_time = datetime.datetime.strptime(blocked_start, "%H:%M").time()
                    blocked_end_time = datetime.datetime.strptime(blocked_end, "%H:%M").time()
                    
                    if not (slot_end_time <= blocked_start_time or slot_start_time >= blocked_end_time):
                        is_available = False
                        break

                if not is_available:
                    break

            if is_available:
                available_slots.append(slot)
        
        if 'earliest_availability' in constraints and constraints['earliest_availability']:
            if available_slots:
                return f"Monday, {available_slots[0]} - {(datetime.datetime.combine(datetime.date.today(), datetime.datetime.strptime(available_slots[0], '%H:%M').time()) + datetime.timedelta(minutes=30)).time().strftime('%H:%M')}"
            else:
                return None
        
        if available_slots:
            return f"Monday, {available_slots[0]} - {(datetime.datetime.combine(datetime.date.today(), datetime.datetime.strptime(available_slots[0], '%H:%M').time()) + datetime.timedelta(minutes=30)).time().strftime('%H:%M')}"
        else:
            return None

    except Exception as e:
        raise ValueError(f"Error verifying solution: {str(e)}")

# Example usage:
if __name__ == "__main__":
    question = "You are an expert at scheduling meetings. You are given a few constraints on the existing schedule of each participant, the meeting duration, and possibly some preferences on the meeting time. Note there exists a solution that works with existing schedule of every participant. Here are a few example tasks and solutions:\n\nTASK: You need to schedule a meeting for David, Ethan, Bradley and Natalie for half an hour between the work hours of 9:00 to 17:00 on Monday. \n\nHere are the existing schedules for everyone during the day: \nDavid has blocked their calendar on Monday during 14:00 to 14:30, 16:30 to 17:00; \nEthan has meetings on Monday during 13:00 to 13:30, 14:30 to 15:00; \nBradley is busy on Monday during 9:30 to 10:30, 11:00 to 12:00, 13:30 to 14:00, 15:30 to 17:00; \nNatalie is busy on Monday during 9:30 to 10:00, 10:30 to 12:00, 12:30 to 15:30, 16:00 to 17:00; \n\nNatalie would rather not meet on Monday after 10:30. Find a time that works for everyone's schedule and constraints. "
    answer = main(question)
    print(answer)