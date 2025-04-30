
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
    from current_script_16 import main

    # Execute the main function with the question string
    question = 'Grid Transformation Task\n\n=== TRAINING EXAMPLES ===\n\nExample 1:\nInput Grid:\n[\n  [5, 5, 0]\n  [5, 0, 5]\n  [0, 5, 0]\n]\n\nOutput Grid:\n[\n  [1]\n]\nExample 2:\nInput Grid:\n[\n  [8, 0, 8]\n  [0, 8, 0]\n  [8, 0, 8]\n]\n\nOutput Grid:\n[\n  [2]\n]\nExample 3:\nInput Grid:\n[\n  [5, 0, 5]\n  [0, 5, 0]\n  [5, 0, 5]\n]\n\nOutput Grid:\n[\n  [2]\n]\nExample 4:\nInput Grid:\n[\n  [0, 1, 1]\n  [0, 1, 1]\n  [1, 0, 0]\n]\n\nOutput Grid:\n[\n  [3]\n]\nExample 5:\nInput Grid:\n[\n  [0, 8, 8]\n  [0, 8, 8]\n  [8, 0, 0]\n]\n\nOutput Grid:\n[\n  [3]\n]\nExample 6:\nInput Grid:\n[\n  [4, 4, 0]\n  [4, 0, 4]\n  [0, 4, 0]\n]\n\nOutput Grid:\n[\n  [1]\n]\nExample 7:\nInput Grid:\n[\n  [0, 5, 0]\n  [5, 5, 5]\n  [0, 5, 0]\n]\n\nOutput Grid:\n[\n  [6]\n]\n\n=== TEST INPUT ===\n[\n  [0, 8, 0]\n  [8, 8, 8]\n  [0, 8, 0]\n]\n\nTransform the test input according to the pattern shown in the training examples.'

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
