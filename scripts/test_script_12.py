
import sys
import traceback

# Add the scripts directory to the path
sys.path.append("scripts")

try:
    # Import the script as a module
    from current_script_12 import main

    # Execute the main function with the question
    question = "You are an expert at scheduling meetings. You are given a few constraints on the existing schedule of each participant, the meeting duration, and possibly some preferences on the meeting time. Note there exists a solution that works with existing schedule of every participant. Here are a few example tasks and solutions:\n\nTASK: You need to schedule a meeting for Jacqueline, Adam, Gerald and Wayne for half an hour between the work hours of 9:00 to 17:00 on Monday. \n\nHere are the existing schedules for everyone during the day: \nJacqueline has meetings on Monday during 10:00 to 10:30, 11:00 to 11:30, 12:30 to 13:00, 14:00 to 15:00, 16:00 to 16:30; \nAdamhas no meetings the whole day.\nGerald has blocked their calendar on Monday during 9:00 to 11:00, 11:30 to 12:00, 14:00 to 15:00, 15:30 to 17:00; \nWayne is busy on Monday during 9:00 to 10:00, 10:30 to 11:30, 13:00 to 15:00, 15:30 to 16:00, 16:30 to 17:00; \n\nYou would like to schedule the meeting at their earlist availability.\nFind a time that works for everyone's schedule and constraints. \nSOLUTION: "
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
