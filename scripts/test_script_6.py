
import sys
import traceback
import os
import json

# Add the scripts directory to the path
sys.path.append("scripts")

# Ensure the Gemini API key is available to the script
os.environ["GEMINI_API_KEY"] = "AIzaSyD_DWppm-TR9CN7xTTVmrW5ngTax7xsLDA"

try:
    # Import the script as a module
    from current_script_6 import main

    # Execute the main function with the question string
    question = 'Grid Transformation Task\n\nTraining Examples:\n[{"input":[[0,9,9,1,9,9,9],[0,0,9,1,9,9,0],[9,0,9,1,9,9,0],[0,0,0,1,9,0,0],[0,9,9,1,9,9,9]],"output":[[0,0,0],[0,0,0],[0,0,0],[0,8,8],[0,0,0]]},{"input":[[0,0,0,1,9,0,0],[9,0,9,1,9,9,9],[0,9,9,1,9,9,9],[0,0,0,1,9,9,9],[0,9,9,1,9,9,9]],"output":[[0,8,8],[0,0,0],[0,0,0],[0,0,0],[0,0,0]]},{"input":[[9,0,0,1,9,0,9],[9,0,0,1,0,9,0],[9,0,0,1,9,0,0],[0,9,9,1,0,9,9],[0,0,9,1,0,9,0]],"output":[[0,8,0],[0,0,8],[0,8,8],[8,0,0],[8,0,0]]},{"input":[[0,9,9,1,9,0,9],[9,0,0,1,9,0,0],[9,9,9,1,9,9,9],[0,9,0,1,0,0,0],[9,0,0,1,9,0,0]],"output":[[0,0,0],[0,8,8],[0,0,0],[8,0,8],[0,8,8]]},{"input":[[0,9,9,1,9,0,9],[9,0,9,1,9,9,9],[9,9,9,1,0,0,9],[9,0,0,1,9,0,0],[9,9,9,1,0,0,9]],"output":[[0,0,0],[0,0,0],[0,0,0],[0,8,8],[0,0,0]]}]\n\nTest Input:\n[[9,9,0,1,0,9,0],[0,9,9,1,0,0,0],[9,9,0,1,0,9,0],[9,9,9,1,9,0,9],[0,9,9,1,0,9,9]]\n\nTransform the test input according to the pattern shown in the training examples.'

    # Call the main function and get the answer
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
