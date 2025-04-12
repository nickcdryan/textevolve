import re

def main(question):
    """
    This script adopts a novel LLM-free approach leveraging time interval arithmetic 
    and constraint satisfaction to schedule meetings.  It avoids regex and 
    simulated LLM calls, focusing instead on direct parsing and calculation.

    1.  Parses the question to extract participants, duration, work hours, 
        and individual schedules using string manipulation instead of regex.
    2.  Represents time slots as intervals.
    3.  Calculates the intersection of free time intervals for all participants.
    4.  Searches for a meeting slot within the intersection that satisfies 
        duration and other constraints.
    5.  If no suitable slot is found, returns an error message.

    Error handling is included to gracefully handle unexpected input formats.
    """

    try:
        # 1. Extract Information (Direct Parsing)
        participants_str = question.split("for ")[1].split(" for")[0] # e.g., "John, Andrea and Lisa"
        participants = [p.strip() for p in participants_str.replace(" and ", ", ").split(",")]
        
        duration_str = question.split(" for ")[1].split(" between")[0]
        duration_hours = 0
        duration_minutes = 0
        if "hour" in duration_str:
            duration_hours = int(duration_str.split(" hour")[0])
        if "half" in duration_str:
             duration_minutes = 30
        
        work_hours_str = question.split("between the work hours of ")[1].split(" on ")[0] #e.g "9:00 to 17:00"
        work_start_str, work_end_str = work_hours_str.split(" to ")
        work_start = time_to_minutes(work_start_str)
        work_end = time_to_minutes(work_end_str)

        schedule_start_index = question.find("Here are the existing schedules") + len("Here are the existing schedules for everyone during the day: \n")
        schedules_str = question[schedule_start_index:]
        schedules = {}
        for person in participants:
            schedule_start = schedules_str.find(person)
            if schedule_start == -1:
                schedules[person] = [] # Free all day
                continue

            person_schedule_str = schedules_str[schedules_str.find(person):]
            person_schedule_end = person_schedule_str.find("\n")
            if person_schedule_end == -1:
                 person_schedule_end = len(person_schedule_str)

            person_schedule_str = person_schedule_str[:person_schedule_end]
            busy_times = []
            if "free the entire day" not in person_schedule_str:
                time_ranges = person_schedule_str.split("during ")[1].split(", ")
                for time_range in time_ranges:
                    start_time_str, end_time_str = time_range.split(" to ")
                    start_time = time_to_minutes(start_time_str)
                    end_time = time_to_minutes(end_time_str)
                    busy_times.append((start_time, end_time))
            
            schedules[person] = busy_times
                
        constraints_str = question.split("\n")[-2]
        if "can not meet on Monday after" in constraints_str:
            name = constraints_str.split("can not meet on Monday after ")[0]
            time_limit_str = constraints_str.split("can not meet on Monday after ")[1]
            time_limit = time_to_minutes(time_limit_str)
        else:
            time_limit = work_end
            

        # 2. Represent Time as Intervals and Find Common Free Time
        available_slots = []
        for start_time in range(work_start, time_limit + 1):
            end_time = start_time + duration_hours * 60 + duration_minutes
            if end_time > time_limit:
                continue
            
            valid_slot = True
            for person, busy_times in schedules.items():
                for busy_start, busy_end in busy_times:
                    if intervals_overlap((start_time, end_time), (busy_start, busy_end)):
                        valid_slot = False
                        break
                if not valid_slot:
                    break
            
            if valid_slot:
                available_slots.append((start_time, end_time))

        # 3. Find Earliest Available Time
        if available_slots:
            earliest_start, earliest_end = available_slots[0]
            earliest_start_str = minutes_to_time(earliest_start)
            earliest_end_str = minutes_to_time(earliest_end)

            return f"Here is the proposed time: Monday, {earliest_start_str} - {earliest_end_str} "
        else:
            return "No suitable time found."
            
    except Exception as e:
        return f"Error: Could not find a valid meeting time. {str(e)}"

def time_to_minutes(time_str):
    """Converts a time string (e.g., "9:00") to minutes since midnight."""
    hours, minutes = map(int, time_str.split(':'))
    return hours * 60 + minutes

def minutes_to_time(minutes):
    """Converts minutes since midnight to a time string (e.g., "9:00")."""
    hours = minutes // 60
    minutes = minutes % 60
    return "{:02d}:{:02d}".format(hours, minutes)

def intervals_overlap(interval1, interval2):
    """Checks if two time intervals overlap."""
    start1, end1 = interval1
    start2, end2 = interval2
    return start1 < end2 and start2 < end1