
import sys
import traceback
import os

# Add the scripts directory to the path
sys.path.append("scripts")

# Ensure the Gemini API key is available to the script
os.environ["GEMINI_API_KEY"] = "AIzaSyD_DWppm-TR9CN7xTTVmrW5ngTax7xsLDA"

try:
    # Import the script as a module
    from current_script_5 import main

    # Execute the main function with the question
    question = "You are an expert at scheduling meetings. You are given a few constraints on the existing schedule of each participant, the meeting duration, and possibly some preferences on the meeting time. Note there exists a solution that works with existing schedule of every participant. Here are a few example tasks and solutions:\n\nTASK: You need to schedule a meeting for Danielle and Amanda for half an hour between the work hours of 9:00 to 17:00 on either Monday, Tuesday, Wednesday, Thursday or Friday. \n\nHere are the existing schedules for everyone during the days: \nDanielle has blocked their calendar on Monday during 9:30 to 10:00, 11:00 to 12:00, 13:00 to 13:30, Tuesday during 10:30 to 11:00, 15:30 to 16:00, Wednesday during 9:30 to 10:00, 12:30 to 13:00, Thursday during 11:30 to 12:00, Friday during 9:30 to 10:00, 12:00 to 12:30, 15:30 to 16:00; \nAmanda has blocked their calendar on Monday during 9:30 to 10:30, 11:00 to 12:00, 12:30 to 15:00, 15:30 to 17:00, Tuesday during 9:00 to 10:30, 11:00 to 17:00, Wednesday during 9:00 to 9:30, 10:30 to 12:30, 13:00 to 16:00, Thursday during 9:00 to 10:30, 11:00 to 12:30, 13:30 to 14:00, 16:00 to 17:00, Friday during 10:00 to 10:30, 12:00 to 13:00, 14:30 to 15:30, 16:00 to 17:00; \n\nDanielle do not want to meet on Thursday. The group would like to meet at their earlist availability.\nFind a time that works for everyone's schedule and constraints. \nSOLUTION: "
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
