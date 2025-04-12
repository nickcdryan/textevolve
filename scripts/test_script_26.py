
import sys
import traceback

# Add the scripts directory to the path
sys.path.append("scripts")

try:
    # Import the script as a module
    from current_script_26 import main

    # Execute the main function with the question
    question = "You are an expert at scheduling meetings. You are given a few constraints on the existing schedule of each participant, the meeting duration, and possibly some preferences on the meeting time. Note there exists a solution that works with existing schedule of every participant. Here are a few example tasks and solutions:\n\nTASK: You need to schedule a meeting for Donna, Lisa, Philip and Albert for half an hour between the work hours of 9:00 to 17:00 on Monday. \n\nHere are the existing schedules for everyone during the day: \nDonnahas no meetings the whole day.\nLisahas no meetings the whole day.\nPhilip has blocked their calendar on Monday during 9:00 to 10:30, 11:00 to 12:00, 12:30 to 14:00, 14:30 to 15:00, 15:30 to 17:00; \nAlbert has meetings on Monday during 9:30 to 10:00, 11:00 to 12:30, 13:30 to 14:30, 15:00 to 15:30, 16:00 to 16:30; \n\nFind a time that works for everyone's schedule and constraints. \nSOLUTION: "
    answer = main(question)

    # Print the answer for capture
    print("ANSWER_START")
    print(answer)
    print("ANSWER_END")

except Exception as e:
    print("ERROR_START")
    print(str(e))
    print(traceback.format_exc())
    print("ERROR_END")
