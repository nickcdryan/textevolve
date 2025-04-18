
import sys
import traceback
import os

# Add the scripts directory to the path
sys.path.append("scripts")

# Ensure the Gemini API key is available to the script
os.environ["GEMINI_API_KEY"] = "AIzaSyD_DWppm-TR9CN7xTTVmrW5ngTax7xsLDA"

try:
    # Import the script as a module
    from current_script_11 import main

    # Execute the main function with the question
    question = "You are an expert at scheduling meetings. You are given a few constraints on the existing schedule of each participant, the meeting duration, and possibly some preferences on the meeting time. Note there exists a solution that works with existing schedule of every participant. Here are a few example tasks and solutions:\n\nTASK: You need to schedule a meeting for Donna, Albert, Jeremy, Grace, Matthew, Jean and Dylan for half an hour between the work hours of 9:00 to 17:00 on Monday. \n\nHere are the existing schedules for everyone during the day: \nDonna is busy on Monday during 10:00 to 10:30, 11:00 to 12:30, 13:00 to 13:30, 14:30 to 15:00; \nAlbert is busy on Monday during 10:00 to 10:30, 11:00 to 11:30, 13:30 to 15:00, 16:00 to 16:30; \nJeremyhas no meetings the whole day.\nGrace has blocked their calendar on Monday during 13:00 to 13:30, 16:30 to 17:00; \nMatthew has blocked their calendar on Monday during 10:00 to 10:30, 11:00 to 11:30, 12:00 to 15:00, 15:30 to 16:30; \nJean is busy on Monday during 11:00 to 13:30, 14:30 to 15:30, 16:30 to 17:00; \nDylan has meetings on Monday during 9:00 to 9:30, 10:00 to 10:30, 11:30 to 12:00, 12:30 to 13:00, 13:30 to 14:00, 15:00 to 15:30, 16:00 to 17:00; \n\nDonna would like to avoid more meetings on Monday before 10:00. Find a time that works for everyone's schedule and constraints. \nSOLUTION: "
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
