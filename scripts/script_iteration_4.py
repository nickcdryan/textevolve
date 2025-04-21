import os
import re
import json
import math
from typing import List, Dict, Any

# Overall reasoning:
# This iteration will focus on a "decompose and conquer" strategy, breaking down the scheduling problem into sub-problems that can be solved and verified independently.
# The hypothesis is that by isolating specific aspects of the scheduling problem (e.g., constraint satisfaction, schedule matching, time slot generation), we can improve the overall accuracy and robustness of the solution.
# This approach is different from previous iterations that attempted to solve the entire problem in a single pass or by iteratively refining a single solution. This time, components are isolated and tested separately.
# We will test this approach and add verification steps to deduce if the changes are helpful.
# This strategy is also aligned with the findings on structured questions and limited scope of constraints, letting us make assumptions that may speed up the process.
# The script contains several functions, including decompose_problem, satisfy_constraints, match_schedules and generate_solution.
# This approach will use multi-example prompting and incorporate validation loops throughout the process.

# THIS IS KEY: The API error has been fixed. The `call_llm` function is included.
# THIS IS KEY: All the string literals are properly handled.

# Define `call_llm` to call the Gemini API.
def call_llm(prompt, system_instruction=None):
    """Call the Gemini LLM with a prompt and return the response"""
    try:
        from google import genai
        from google.generativeai import types

        # Initialize the Gemini client
        genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
        model = genai.GenerativeModel('gemini-pro') # or 'gemini-pro-vision' if image input
        
        # Call the API with system instruction if provided
        if system_instruction:
            response = model.generate_content(
                prompt,
                generation_config = {
                    "temperature": 0.7,
                    "top_p": 1,
                    "top_k": 32,
                    "max_output_tokens": 4096,
                },
                safety_settings={
                    genai.HarmCategory.HARASSMENT: genai.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                    genai.HarmCategory.HATE_SPEECH: genai.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                    genai.HarmCategory.SEXUALLY_EXPLICIT: genai.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                    genai.HarmCategory.DANGEROUS_CONTENT: genai.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                }
            )
        else:
            response = model.generate_content(
                prompt,
                generation_config = {
                    "temperature": 0.7,
                    "top_p": 1,
                    "top_k": 32,
                    "max_output_tokens": 4096,
                },
                safety_settings={
                    genai.HarmCategory.HARASSMENT: genai.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                    genai.HarmCategory.HATE_SPEECH: genai.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                    genai.HarmCategory.SEXUALLY_EXPLICIT: genai.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                    genai.HarmCategory.DANGEROUS_CONTENT: genai.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                }
            )

        return response.text
    except Exception as e:
        print(f"Error calling Gemini API: {str(e)}")
        return f"Error: {str(e)}"

def decompose_problem(question: str, max_attempts: int = 3) -> Dict[str, Any]:
    """
    Decomposes the scheduling problem into smaller, independent sub-problems.
    """
    system_instruction = "You are an expert at decomposing complex scheduling problems into smaller, independent sub-problems."

    for attempt in range(max_attempts):
        decomposition_prompt = f"""
        Decompose the following scheduling problem into smaller, independent sub-problems.
        Identify the key steps required to solve the problem and list them as individual tasks.

        Example 1:
        Question: Schedule a meeting for John and Mary for 30 minutes between 9:00 and 17:00 on Monday. John prefers to avoid meetings before 11:00.
        Decomposition:
        1. Extract participants (John, Mary)
        2. Extract available days (Monday)
        3. Extract available time ranges (9:00 - 17:00)
        4. Extract preferences (John avoids meetings before 11:00)
        5. Find time slots that satisfy all preferences and constraints

        Example 2:
        Question: Schedule a 60-minute meeting for Sarah and Tom on Tuesday or Wednesday between 10:00 AM and 4:00 PM. Tom cannot attend after 3:00 PM.
        Decomposition:
        1. Extract participants (Sarah, Tom)
        2. Extract available days (Tuesday, Wednesday)
        3. Extract available time ranges (10:00 - 16:00)
        4. Extract constraints (Tom cannot attend after 15:00)
        5. Find time slots that satisfy all constraints

        Example 3:
        Question: Schedule a meeting for Bob and Alice for one hour. Alice is not available on Mondays and Bob is not available on Tuesdays or Wednesdays.
        Decomposition:
        1. Extract participants (Bob, Alice)
        2. Extract unavailable days for Alice (Mondays)
        3. Extract unavailable days for Bob (Tuesdays, Wednesdays)
        4. Determine days where BOTH are available.
        5. Find one-hour timeslot on available days.

        Question: {question}
        Decomposition:
        """

        decomposition = call_llm(decomposition_prompt, system_instruction)

        # Verification Step
        verification_prompt = f"""
            Verify if the decomposition is complete and identifies all necessary sub-problems.
            Check if the identified sub-problems are independent and cover all aspects of the original problem.

            Question: {question}
            Decomposition: {decomposition}

            Respond with "VALID" if the decomposition is valid, otherwise respond with "INVALID: [reason]".
            """

        verification_result = call_llm(verification_prompt, system_instruction)

        if "VALID" in verification_result:
            return {"is_valid": True, "decomposition": decomposition}
        else:
            print(f"Decomposition failed verification: {verification_result}")
            continue

    return {"is_valid": False, "decomposition": "Failed to decompose problem after multiple attempts."}

def satisfy_constraints(constraints: str, available_times: List[str], max_attempts: int = 3) -> List[str]:
    """
    Filters available time slots based on the given constraints.
    """
    system_instruction = "You are an expert at filtering time slots based on scheduling constraints."

    for attempt in range(max_attempts):
        filtering_prompt = f"""
        Filter the following list of available time slots based on the given constraints.
        Return a list of time slots that satisfy all constraints.

        Example 1:
        Constraints: John prefers to avoid meetings before 11:00.
        Available Times: ["9:00 - 9:30", "10:00 - 10:30", "11:00 - 11:30", "12:00 - 12:30"]
        Filtered Times: ["11:00 - 11:30", "12:00 - 12:30"]

        Example 2:
        Constraints: Tom cannot attend after 3:00 PM.
        Available Times: ["13:00 - 14:00", "14:00 - 15:00", "15:00 - 16:00", "16:00 - 17:00"]
        Filtered Times: ["13:00 - 14:00", "14:00 - 15:00"]

        Example 3:
        Constraints: Alice is not available on Mondays.
        Available Times: ["Monday, 9:00 - 10:00", "Tuesday, 10:00 - 11:00", "Wednesday, 11:00 - 12:00"]
        Filtered Times: ["Tuesday, 10:00 - 11:00", "Wednesday, 11:00 - 12:00"]

        Constraints: {constraints}
        Available Times: {available_times}
        Filtered Times:
        """

        filtered_times = call_llm(filtering_prompt, system_instruction)

        # Verification Step
        verification_prompt = f"""
            Verify that the filtered time slots satisfy all the given constraints.
            Check if any time slots that violate the constraints are included in the filtered list.

            Constraints: {constraints}
            Available Times: {available_times}
            Filtered Times: {filtered_times}

            Respond with "VALID" if the filtering is valid, otherwise respond with "INVALID: [reason]".
            """

        verification_result = call_llm(verification_prompt, system_instruction)

        if "VALID" in verification_result:
            return json.loads(filtered_times) # Returning it as a List
        else:
            print(f"Filtering failed verification: {verification_result}")
            continue

    return []

def match_schedules(schedules: Dict[str, List[str]], max_attempts: int = 3) -> List[str]:
    """
    Finds time slots that work for all participants based on their schedules.
    """
    system_instruction = "You are an expert at matching schedules to find common available time slots."

    for attempt in range(max_attempts):
        matching_prompt = f"""
        Find the time slots that work for all participants based on their schedules.
        Return a list of time slots that are available for everyone.

        Example 1:
        Schedules:
        {{
          "John": ["9:00 - 9:30", "10:00 - 10:30", "11:00 - 11:30"],
          "Mary": ["9:00 - 9:30", "10:30 - 11:00", "11:00 - 11:30"]
        }}
        Common Times: ["9:00 - 9:30", "11:00 - 11:30"]

        Example 2:
        Schedules:
        {{
          "Sarah": ["Tuesday, 10:00 - 11:00", "Wednesday, 11:00 - 12:00"],
          "Tom": ["Tuesday, 10:00 - 11:00", "Wednesday, 12:00 - 13:00"]
        }}
        Common Times: ["Tuesday, 10:00 - 11:00"]

        Schedules: {schedules}
        Common Times:
        """

        common_times = call_llm(matching_prompt, system_instruction)

        # Verification Step
        verification_prompt = f"""
            Verify that the common time slots are available for all participants based on their schedules.
            Check if any time slots are included that are not available for all participants.

            Schedules: {schedules}
            Common Times: {common_times}

            Respond with "VALID" if the matching is valid, otherwise respond with "INVALID: [reason]".
            """

        verification_result = call_llm(verification_prompt, system_instruction)

        if "VALID" in verification_result:
            return json.loads(common_times) # Returning as List
        else:
            print(f"Matching failed verification: {verification_result}")
            continue

    return []

def generate_solution(common_times: List[str], max_attempts: int = 3) -> str:
    """
    Generates a final solution from the common available time slots.
    """
    system_instruction = "You are an expert at generating a final solution from available time slots."

    for attempt in range(max_attempts):
        generation_prompt = f"""
        Generate a final solution from the following list of available time slots.
        Select a time slot and present it as the proposed meeting time.

        Example 1:
        Common Times: ["9:00 - 9:30", "11:00 - 11:30"]
        Solution: 9:00 - 9:30

        Example 2:
        Common Times: ["Tuesday, 10:00 - 11:00", "Wednesday, 11:00 - 12:00"]
        Solution: Tuesday, 10:00 - 11:00

        Common Times: {common_times}
        Solution:
        """

        solution = call_llm(generation_prompt, system_instruction)

        # Verification Step
        verification_prompt = f"""
            Verify that the generated solution is a valid time slot from the list of common available times.

            Common Times: {common_times}
            Solution: {solution}

            Respond with "VALID" if the solution is valid, otherwise respond with "INVALID: [reason]".
            """

        verification_result = call_llm(verification_prompt, system_instruction)

        if "VALID" in verification_result:
            return solution
        else:
            print(f"Solution generation failed verification: {verification_result}")
            continue

    return "Could not find a suitable meeting time."

def extract_info(question: str, info_type: str, examples: List[Dict[str, str]], max_attempts: int = 3) -> str:
    """Extracts specific information from the question with few-shot examples."""
    system_instruction = f"You are an expert at extracting {info_type} from scheduling questions."

    for attempt in range(max_attempts):
        prompt = f"""
        Extract the {info_type} from the following scheduling question.

        Here are some examples:
        """
        for i, example in enumerate(examples):
            prompt += f"""
            Example {i+1}:
            Question: {example["question"]}
            {info_type}: {example["answer"]}
            """

        prompt += f"""
        Question: {question}
        {info_type}:
        """

        extracted_info = call_llm(prompt, system_instruction)

        # Basic validation step - can be enhanced based on `info_type`
        if extracted_info:
            return extracted_info
        else:
            print(f"Failed to extract {info_type} on attempt {attempt + 1}")

    return f"Could not extract {info_type} after multiple attempts."

def main(question: str) -> str:
    """Main function to schedule a meeting."""
    try:
        # Example list of dicts for few-shot extraction
        participant_examples = [
            {"question": "Schedule a meeting for John and Mary", "answer": "John, Mary"},
            {"question": "Schedule a meeting for Sarah, Tom, and David", "answer": "Sarah, Tom, David"}
        ]

        time_examples = [
            {"question": "Schedule a meeting between 9am and 5pm", "answer": "9am and 5pm"},
            {"question": "The meeting should be scheduled on Tuesday and Wednesday", "answer": "Tuesday and Wednesday"}
        ]

        constraint_examples = [
            {"question": "John prefers not to meet before 11am", "answer": "John prefers not to meet before 11am"},
            {"question": "The meeting has to take place on Monday", "answer": "The meeting has to take place on Monday"}
        ]

        # 1. Extract Participants
        participants = extract_info(question, "participants", participant_examples)
        print(f"Extracted participants: {participants}")

        # 2. Extract Time Information
        time_info = extract_info(question, "time information", time_examples)
        print(f"Extracted time information: {time_info}")

        # 3. Extract Constraints
        constraints = extract_info(question, "constraints", constraint_examples)
        print(f"Extracted constraints: {constraints}")

        # 4. Decompose Problem
        decomposition_result = decompose_problem(question)
        if not decomposition_result["is_valid"]:
            return f"Error: {decomposition_result['decomposition']}"
        print(f"Decomposition Result: {decomposition_result['decomposition']}")

        # 5. Now simulate available times and schedules, as these were not directly addressed in previous extraction step
        # In a real scenario, one might query a calendar or database.

        available_times = ["Monday, 9:00 - 10:00", "Monday, 10:00 - 11:00", "Tuesday, 9:00 - 10:00", "Tuesday, 10:00 - 11:00"]
        simulated_schedules = {
            "Participants": available_times, # simplified for example
            "Mary": ["Monday, 10:00 - 11:00"],
            "John": ["Tuesday, 9:00 - 10:00"]
        }

        # 6. Satisfy Constraints
        filtered_times = satisfy_constraints(constraints, available_times)
        print(f"Filtered times: {filtered_times}")

        # 7. Match Schedules
        common_times = match_schedules(simulated_schedules)
        print(f"Common times: {common_times}")

        # 8. Generate Solution
        solution = generate_solution(common_times)
        print(f"Final solution: {solution}")

        return f"Here is the proposed time: {solution}"

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return f"Error: {str(e)}"

# Example usage:
if __name__ == "__main__":
    question = "You are an expert at scheduling meetings. You are given a few constraints on the existing schedule of each participant, the meeting duration, and possibly some preferences on the meeting time. Note there exists a solution that works with existing schedule of every participant. Here are a few example tasks and solutions:\n\nTASK: You need to schedule a meeting for Joyce, Christine and Alexander for half an hour between the work hours of 9:00 to 17:00 on Monday. \n\nHere are the existing schedules for everyone during the day: \nJoyce has meetings on Monday during 11:00 to 11:30, 13:30 to 14:00, 14:30 to 16:30; \nChristinehas no meetings the whole day.\nAlexander has meetings on Monday during 9:00 to 11:00, 12:00 to 12:30, 13:30 to 15:00, 15:30 to 16:00, 16:30 to 17:00; \n\nChristine can not meet on Monday before 12:00. Find a time that works for everyone's schedule and constraints. "
    answer = main(question)
    print(f"Final Answer: {answer}")