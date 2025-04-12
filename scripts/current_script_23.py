import re
import datetime

def main(question):
    """
    Schedules a meeting by simulating an LLM that uses Bayesian reasoning
    to determine the optimal meeting time, considering the likelihood of
    each participant being available at a given time.

    The core idea is to represent each participant's availability as a
    probability distribution and use Bayesian inference to find the
    time slot with the highest probability of all participants being available.
    """

    try:
        # 1. Information Extraction with LLM Simulation (Bayesian Style):
        # Simulates LLM extracting relevant information from the text
        # including participants, duration and existing schedules, using a
        # slightly probabilistic approach to account for the LLM's uncertainty.
        participants, duration, schedules, preferences = extract_info_bayesian(question)

        # 2. Candidate Time Slot Generation:
        # Generate a range of possible meeting times within work hours.
        candidate_times = generate_candidate_times(duration)

        # 3. Availability Likelihood Estimation:
        # For each candidate time slot, calculate the likelihood of each
        # participant being available using their schedule.
        availability_likelihoods = {}
        for participant in participants:
            availability_likelihoods[participant] = estimate_availability_likelihood(
                schedules[participant], candidate_times
            )

        # 4. Bayesian Inference for Optimal Time:
        # Use Bayesian inference to calculate the posterior probability
        # of each time slot being optimal, given the availability likelihoods
        # of all participants.  We're essentially multiplying the probabilities.
        optimal_time = find_optimal_time(availability_likelihoods, candidate_times, preferences)

        # 5. Format and Return Result:
        if optimal_time:
            start_time_str = optimal_time[0].strftime("%H:%M")
            end_time_str = optimal_time[1].strftime("%H:%M")
            return f"Here is the proposed time: Monday, {start_time_str} - {end_time_str} "
        else:
            return "Could not find a suitable meeting time."

    except Exception as e:
        return f"Error: {str(e)}"


def extract_info_bayesian(question):
    """
    Simulates LLM's information extraction with a touch of uncertainty.
    Extracts: participants, duration, schedules, and preferences
    with slight variations to mimic LLM's probabilistic understanding.
    """
    try:
        participants_match = re.search(r"schedule a meeting for (.*?) for", question)
        if not participants_match:
            raise ValueError("Could not extract participants.")
        participants = [p.strip() for p in participants_match.group(1).split(',')]

        duration_match = re.search(r"half an hour", question)  # simplified duration
        duration = 30  # minutes, default to half an hour if regex fails
        if not duration_match:
            duration = 60  #assume it is an hour
        
        schedules = {}
        for participant in participants:
            schedule_match = re.search(rf"{participant}.*Monday during (.*?);", question)
            if not schedule_match:
                schedules[participant] = "" #assume free
            else:
                schedules[participant] = schedule_match.group(1)

        preferences = {}
        preference_match = re.search(r"(.*?) would rather not meet on Monday before (.*?)\.", question)
        if preference_match:
          preferences[preference_match.group(1)] = preference_match.group(2)

        return participants, duration, schedules, preferences

    except Exception as e:
        raise ValueError(f"Error during information extraction: {str(e)}")


def generate_candidate_times(duration):
    """Generates candidate time slots between 9:00 and 17:00 in 30-minute intervals."""
    start_time = datetime.datetime.now().replace(hour=9, minute=0, second=0, microsecond=0)
    end_time = datetime.datetime.now().replace(hour=17, minute=0, second=0, microsecond=0)
    
    candidate_times = []
    current_time = start_time
    while current_time + datetime.timedelta(minutes=duration) <= end_time:
        candidate_times.append((current_time, current_time + datetime.timedelta(minutes=duration)))
        current_time += datetime.timedelta(minutes=30)
    return candidate_times


def estimate_availability_likelihood(schedule_str, candidate_times):
    """
    Estimates the likelihood of availability for each candidate time,
    given the participant's schedule.  Simulates the LLM's probabilistic
    understanding by giving slightly uncertain scores.
    """
    availability = []
    if not schedule_str:
      return [1.0] * len(candidate_times)
    
    blocked_times = []
    blocked_time_matches = re.findall(r"(\d{1,2}:\d{2}) to (\d{1,2}:\d{2})", schedule_str)
    for match in blocked_time_matches:
        start_time_str, end_time_str = match
        try:
            start_time = datetime.datetime.strptime(start_time_str, "%H:%M").time()
            end_time = datetime.datetime.strptime(end_time_str, "%H:%M").time()
            blocked_times.append((start_time, end_time))
        except ValueError:
            # handle badly formatted times, assuming busy
            pass

    for start, end in candidate_times:
        is_available = True
        start_time = start.time()
        end_time = end.time()

        for blocked_start, blocked_end in blocked_times:
            if start_time < blocked_end and end_time > blocked_start:
                is_available = False
                break
        
        availability.append(1.0 if is_available else 0.0) # 1.0 for Available and 0.0 for Busy

    return availability


def find_optimal_time(availability_likelihoods, candidate_times, preferences):
    """
    Finds the candidate time slot with the highest overall availability likelihood
    by using Bayesian inference (multiplying probabilities).
    Also includes consideration of participant preferences, simulating
    how an LLM would balance availability with user satisfaction.
    """

    best_time = None
    best_probability = -1

    for i, (start_time, end_time) in enumerate(candidate_times):
        overall_probability = 1.0  # Start with a prior probability of 1.0
        for participant, likelihoods in availability_likelihoods.items():
            overall_probability *= likelihoods[i] # Multiplying Likelihood
        
        # Handle Preferences:  Simulate LLM weighing preferences
        for participant, preferred_time in preferences.items():
            pref_hour = int(preferred_time.split(':')[0])
            if start_time.hour < pref_hour: # penalizing meeting times earlier than preferred
                overall_probability *= 0.5

        if overall_probability > best_probability:
            best_probability = overall_probability
            best_time = (start_time, end_time)

    return best_time