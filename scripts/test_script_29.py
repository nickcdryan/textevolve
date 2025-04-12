
import sys
import traceback

# Add the scripts directory to the path
sys.path.append("scripts")

try:
    # Import the script as a module
    from current_script_29 import main

    # Execute the main function with the question
    question = "You are an expert at scheduling meetings. You are given a few constraints on the existing schedule of each participant, the meeting duration, and possibly some preferences on the meeting time. Note there exists a solution that works with existing schedule of every participant. Here are a few example tasks and solutions:\n\nTASK: You need to schedule a meeting for Elizabeth, Carol, Christian and Isabella for half an hour between the work hours of 9:00 to 17:00 on Monday. \n\nHere are the existing schedules for everyone during the day: \nElizabeth has meetings on Monday during 11:30 to 12:00, 13:30 to 14:00, 15:00 to 15:30, 16:30 to 17:00; \nCarol is free the entire day.\nChristian has meetings on Monday during 9:00 to 9:30, 10:00 to 10:30, 11:00 to 11:30, 12:30 to 13:30, 15:30 to 17:00; \nIsabella has blocked their calendar on Monday during 9:30 to 11:30, 12:30 to 13:30, 15:00 to 15:30, 16:00 to 16:30; \n\nCarol do not want to meet on Monday before 12:30. Find a time that works for everyone's schedule and constraints. \nSOLUTION: "
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
