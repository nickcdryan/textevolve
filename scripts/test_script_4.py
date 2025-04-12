
import sys
import traceback

# Add the scripts directory to the path
sys.path.append("scripts")

try:
    # Import the script as a module
    from current_script_4 import main

    # Execute the main function with the question
    question = "You are an expert at scheduling meetings. You are given a few constraints on the existing schedule of each participant, the meeting duration, and possibly some preferences on the meeting time. Note there exists a solution that works with existing schedule of every participant. Here are a few example tasks and solutions:\n\nTASK: You need to schedule a meeting for Virginia, Charles and Megan for half an hour between the work hours of 9:00 to 17:00 on Monday. \n\nHere are the existing schedules for everyone during the day: \nVirginia has meetings on Monday during 10:00 to 12:00; \nCharles has blocked their calendar on Monday during 12:00 to 12:30, 13:00 to 13:30; \nMegan is busy on Monday during 9:00 to 12:00, 13:30 to 16:00, 16:30 to 17:00; \n\nCharles would like to avoid more meetings on Monday before 14:30. Find a time that works for everyone's schedule and constraints. \nSOLUTION: "
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
