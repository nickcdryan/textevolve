import json
import re
from datetime import datetime, timedelta

def main(question):
    """
    Schedules meetings by using a "persona prompting" approach to simulate different meeting scheduling experts,
    each with a unique style and expertise. An LLM is simulated by using if/else statements with a small bit of regex.
    The best solution suggested by the experts is chosen.

    Args:
        question (str): A string describing the meeting scheduling task.

    Returns:
        str: A string containing the proposed meeting time.
    """
    try:
        # Simulate different personas with unique strategies
        persona_solutions = []
        persona_solutions.append(schedule_meeting_persona_1(question))
        persona_solutions.append(schedule_meeting_persona_2(question))
        persona_solutions.append(schedule_meeting_persona_3(question))
        
        # Select the best solution (simplest logic for demonstration, but could use more advanced criteria).
        # In a real LLM scenario we would have it choose the best one.
        best_solution = min(persona_solutions, key=len) # Choose the shortest (simplest) solution as "best"

        return best_solution
    except Exception as e:
        return f"Error: {str(e)}"

def schedule_meeting_persona_1(question):
    """
    Persona 1: The "Direct and Efficient" scheduler.  Focuses on quickly finding the *first* available slot.
    """
    try:
        participants, schedules, start_time, end_time, duration, preferences = extract_info(question)
        available_time = find_available_time_direct(participants, schedules, start_time, end_time, duration, preferences)
        return f"Here is the proposed time (Persona 1): {available_time}"
    except Exception as e:
        return f"Persona 1 could not find a time."

def schedule_meeting_persona_2(question):
    """
    Persona 2: The "Constraint-Focused" scheduler.  Prioritizes satisfying *all* stated preferences.
    """
    try:
        participants, schedules, start_time, end_time, duration, preferences = extract_info(question)
        available_time = find_available_time_constraints(participants, schedules, start_time, end_time, duration, preferences)
        return f"Here is the proposed time (Persona 2): {available_time}"
    except Exception as e:
        return f"Persona 2 could not find a time."

def schedule_meeting_persona_3(question):
    """
    Persona 3: The "Thorough and Comprehensive" scheduler.  Explores *all* possible slots and returns the most optimal (earliest) time.
    """
    try:
        participants, schedules, start_time, end_time, duration, preferences = extract_info(question)
        available_time = find_available_time_thorough(participants, schedules, start_time, end_time, duration, preferences)
        return f"Here is the proposed time (Persona 3): {available_time}"
    except Exception as e:
        return f"Persona 3 could not find a time."

def extract_info(question):
    """
    Simulates information extraction from text.

    Returns:
        tuple: Participants, schedules, start and end times, meeting duration, and preferences.
    """
    participants_match = re.search(r"schedule a meeting for (.*?) for", question)
    participants = [p.strip() for p in participants_match.group(1).split(",")] if participants_match else []

    schedules = {}
    for participant in participants:
        schedule_match = re.search(rf"{participant}.*?during (.*?);", question)
        if schedule_match:
            schedule_str = schedule_match.group(1)
            schedule_str = schedule_str.replace(" and ", ", ")
            times = schedule_str.split(", ")
            schedules[participant] = []
            for time_range in times:
                try:
                  start, end = time_range.split(" to ")
                  schedules[participant].append((start, end))
                except:
                  pass # Handle case of empty meeting times.
    
    start_time = "9:00"
    end_time = "17:00"
    duration = 30  # minutes

    preferences = {}
    for participant in participants:
        preference_match = re.search(rf"{participant}.*?not meet.*?before (.*?)\.", question)
        if preference_match:
            preferences[participant] = preference_match.group(1)

    return participants, schedules, start_time, end_time, duration, preferences

def find_available_time_direct(participants, schedules, start_time, end_time, duration, preferences):
    """
    Finds the *first* available time slot.
    """
    current_time = datetime.strptime(start_time, "%H:%M").time()
    end_time_obj = datetime.strptime(end_time, "%H:%M").time()
    
    while current_time <= end_time_obj:
        current_time_str = current_time.strftime("%H:%M")
        end_meeting_time = (datetime.combine(datetime.today(), current_time) + timedelta(minutes=duration)).time()
        end_meeting_time_str = end_meeting_time.strftime("%H:%M")

        available = True
        for participant in participants:
            if participant in preferences and current_time < datetime.strptime(preferences[participant], "%H:%M").time():
                available = False
                break # Preference not met

            if participant in schedules:
                for busy_start, busy_end in schedules[participant]:
                    busy_start_time = datetime.strptime(busy_start, "%H:%M").time()
                    busy_end_time = datetime.strptime(busy_end, "%H:%M").time()
                    
                    if not (end_meeting_time <= busy_start_time or current_time >= busy_end_time):
                        available = False
                        break
            if not available:
                break

        if available:
            return f"Monday, {current_time_str} - {end_meeting_time_str}"
        
        current_time = (datetime.combine(datetime.today(), current_time) + timedelta(minutes=15)).time()

    return "No suitable time found (Direct)."

def find_available_time_constraints(participants, schedules, start_time, end_time, duration, preferences):
    """
    Prioritizes satisfying *all* stated preferences.
    """
    current_time = datetime.strptime(start_time, "%H:%M").time()
    end_time_obj = datetime.strptime(end_time, "%H:%M").time()
    
    while current_time <= end_time_obj:
        current_time_str = current_time.strftime("%H:%M")
        end_meeting_time = (datetime.combine(datetime.today(), current_time) + timedelta(minutes=duration)).time()
        end_meeting_time_str = end_meeting_time.strftime("%H:%M")

        available = True
        
        # Check preferences FIRST.
        for participant in participants:
            if participant in preferences and current_time < datetime.strptime(preferences[participant], "%H:%M").time():
                available = False # preference not met
                break
        
        if available:
            for participant in participants:
                if participant in schedules:
                    for busy_start, busy_end in schedules[participant]:
                        busy_start_time = datetime.strptime(busy_start, "%H:%M").time()
                        busy_end_time = datetime.strptime(busy_end, "%H:%M").time()
                        
                        if not (end_meeting_time <= busy_start_time or current_time >= busy_end_time):
                            available = False
                            break
                if not available:
                    break

        if available:
            return f"Monday, {current_time_str} - {end_meeting_time_str}"
        
        current_time = (datetime.combine(datetime.today(), current_time) + timedelta(minutes=15)).time()

    return "No suitable time found (Constraints)."

def find_available_time_thorough(participants, schedules, start_time, end_time, duration, preferences):
    """
    Explores *all* possible slots and returns the most optimal (earliest) time.
    """
    available_times = []
    current_time = datetime.strptime(start_time, "%H:%M").time()
    end_time_obj = datetime.strptime(end_time, "%H:%M").time()
    
    while current_time <= end_time_obj:
        current_time_str = current_time.strftime("%H:%M")
        end_meeting_time = (datetime.combine(datetime.today(), current_time) + timedelta(minutes=duration)).time()
        end_meeting_time_str = end_meeting_time.strftime("%H:%M")

        available = True
        for participant in participants:
            if participant in preferences and current_time < datetime.strptime(preferences[participant], "%H:%M").time():
                available = False
                break # Preference not met
            
            if participant in schedules:
                for busy_start, busy_end in schedules[participant]:
                    busy_start_time = datetime.strptime(busy_start, "%H:%M").time()
                    busy_end_time = datetime.strptime(busy_end, "%H:%M").time()
                    
                    if not (end_meeting_time <= busy_start_time or current_time >= busy_end_time):
                        available = False
                        break
            if not available:
                break

        if available:
            available_times.append(f"Monday, {current_time_str} - {end_meeting_time_str}")
        
        current_time = (datetime.combine(datetime.today(), current_time) + timedelta(minutes=15)).time()

    if available_times:
        return min(available_times) # Returns EARLIEST available time
    return "No suitable time found (Thorough)."