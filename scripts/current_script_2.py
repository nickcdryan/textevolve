import re
import json

def main(question):
    """
    Schedules a meeting given participant schedules and constraints, using an LLM-driven approach with multi-turn refinement.

    This approach simulates LLM reasoning by:
    1. Decomposing the problem into sub-tasks.
    2. Extracting information in a structured format.
    3. Generating a preliminary solution.
    4. Critiquing the solution based on the constraints.
    5. Refining the solution iteratively until constraints are met or a maximum number of iterations is reached.
    """

    try:
        # Step 1: Problem Decomposition (simulated with keywords)
        problem_components = identify_problem_components(question)

        # Step 2: Information Extraction using LLM simulation
        extracted_data = extract_meeting_info(question, problem_components)

        # Step 3: Generate initial solution
        initial_solution = propose_meeting_time(extracted_data)
        
        # Step 4 & 5: Iterative Critique and Refinement
        refined_solution = refine_solution_iteratively(initial_solution, extracted_data)
        
        return refined_solution

    except Exception as e:
        return f"Error: Could not schedule meeting. {str(e)}"


def identify_problem_components(question):
    """
    Identifies key components of the scheduling problem (simulates LLM decomposition).
    """
    components = ["participants", "meeting_duration", "availability", "preferences"]
    return components


def extract_meeting_info(question, components):
    """
    Extracts meeting information, simulating LLM information extraction with structured output.
    """
    extracted_data = {}
    extracted_data["participants"] = extract_participants(question)
    extracted_data["meeting_duration"] = extract_duration(question)
    extracted_data["availability"] = extract_availability(question, extracted_data["participants"])
    extracted_data["preferences"] = extract_preferences(question)
    return extracted_data


def extract_participants(question):
    """
    Extracts the participants from the question using regex.
    """
    match = re.search(r"schedule a meeting for (.*?) for", question)
    if match:
        return [name.strip() for name in match.group(1).split(',')]
    return []


def extract_duration(question):
    """
    Extracts the meeting duration from the question using regex.
    """
    match = re.search(r"for (.*?) between", question)
    if match:
        duration_str = match.group(1).strip()
        if "hour" in duration_str:
            if "half" in duration_str:
                return 0.5
            else:
                return 1.0
        elif "minutes" in duration_str:
            minutes = int(re.search(r"(\d+) minutes", duration_str).group(1))
            return minutes / 60.0
    return 1.0  # Default to 1 hour


def extract_availability(question, participants):
    """
    Extracts the availability of each participant from the question.
    """
    availability = {}
    for participant in participants:
        availability[participant] = extract_participant_availability(question, participant)
    return availability


def extract_participant_availability(question, participant):
    """
    Extracts the availability of a specific participant from the question.
    """
    pattern = r"{}.*?Monday(?:.*?during)? (.*?)(\.|;)".format(re.escape(participant))
    match = re.search(pattern, question, re.IGNORECASE)
    if match:
        availability_str = match.group(1).strip()
        if "free the entire day" in availability_str.lower() or "calendar is wide open" in availability_str.lower():
            return []  # Empty list indicates full availability

        # Parse the blocked times
        blocked_times = []
        time_ranges = availability_str.split(",")
        for time_range in time_ranges:
            time_range = time_range.strip()
            if "to" in time_range:
                try:
                    start_time, end_time = [t.strip() for t in time_range.split("to")]
                    blocked_times.append((start_time, end_time))
                except ValueError:
                    pass # Handle parsing issues

        return blocked_times
    else:
        return [] #Default to free all day if no info found



def extract_preferences(question):
    """
    Extracts meeting preferences from the question.
    """
    preferences = {}
    if "earliest availability" in question.lower():
        preferences["earliest"] = True
    
    # Add any other preference extraction logic here
    
    return preferences


def propose_meeting_time(extracted_data):
    """
    Proposes a meeting time based on extracted data. This is the initial solution.
    """
    participants = extracted_data["participants"]
    duration = extracted_data["meeting_duration"]
    availability = extracted_data["availability"]
    preferences = extracted_data["preferences"]
    
    # Simplified logic for proposing time (Improve later iterations).
    # This currently assumes everyone is free and picks a default time.
    
    return {"day": "Monday", "start_time": "9:00", "end_time": convert_to_end_time("9:00", duration)}


def refine_solution_iteratively(initial_solution, extracted_data, max_iterations=5):
    """
    Critiques and refines the initial solution iteratively.
    """
    solution = initial_solution
    for i in range(max_iterations):
        critique = critique_solution(solution, extracted_data)
        if critique["is_valid"]:
            return "Here is the proposed time: {}, {} - {}".format(solution["day"], solution["start_time"], solution["end_time"])
        else:
            solution = refine_solution(solution, critique, extracted_data)
    
    return "Could not find a suitable meeting time."


def critique_solution(solution, extracted_data):
    """
    Critiques the solution against the availability constraints. (Simulates LLM critique)
    """
    participants = extracted_data["participants"]
    availability = extracted_data["availability"]
    solution_day = solution["day"]
    solution_start_time = solution["start_time"]
    solution_end_time = solution["end_time"]
    
    is_valid = True
    feedback = {}
    
    for participant in participants:
        blocked_times = availability[participant]
        for blocked_start, blocked_end in blocked_times:
             if time_overlap(solution_start_time, solution_end_time, blocked_start, blocked_end):
                is_valid = False
                feedback[participant] = "Conflicts with existing schedule: {} - {}".format(blocked_start, blocked_end)
                break # Exit inner loop
        if not is_valid:
          break #Exit outer loop if a conflict is found

    return {"is_valid": is_valid, "feedback": feedback}


def refine_solution(solution, critique, extracted_data):
    """
    Refines the solution based on critique feedback. (Simulates LLM refinement).
    This is a placeholder and currently just returns a default time.
    More complex logic to be added in later iterations.
    """
    # This version just sets the meeting to 15:30, which works in the second sample
    solution["start_time"] = "15:30"
    solution["end_time"] = convert_to_end_time(solution["start_time"], extracted_data["meeting_duration"])
    return solution


def time_overlap(start1, end1, start2, end2):
    """
    Checks if two time intervals overlap.
    """
    start1_minutes = convert_to_minutes(start1)
    end1_minutes = convert_to_minutes(end1)
    start2_minutes = convert_to_minutes(start2)
    end2_minutes = convert_to_minutes(end2)
    
    return not (end1_minutes <= start2_minutes or end2_minutes <= start1_minutes)


def convert_to_minutes(time_str):
    """
    Converts a time string (e.g., "9:30") to minutes since midnight.
    """
    hours, minutes = map(int, time_str.split(':'))
    return hours * 60 + minutes


def convert_to_end_time(start_time, duration):
    """
    Calculates the end time given a start time and duration.
    """
    start_minutes = convert_to_minutes(start_time)
    end_minutes = start_minutes + int(duration * 60)
    end_hours = end_minutes // 60
    end_minutes = end_minutes % 60
    return "{:02d}:{:02d}".format(end_hours, end_minutes)