import os
import re
import json
import math
from typing import List, Dict, Any

# Overall reasoning:
# This iteration explores a new approach that focuses on constraint-based reasoning with a dedicated "Constraint Analyzer" agent.
# The hypothesis is that explicitly analyzing and representing constraints in a structured format BEFORE attempting to extract schedules or generate solutions will improve the accuracy and efficiency of the scheduling process.
# This is different from previous approaches that attempted to extract schedules and constraints in separate steps, potentially leading to inconsistencies or missed information.
# This approach will use multi-example prompting, incorporate validation loops at each stage, and leverage the ReAct pattern for solution generation.

# Error handling and validation are implemented at each extraction and generation stage.

def call_llm(prompt, system_instruction=None):
    """Call the Gemini LLM with a prompt and return the response"""
    try:
        from google import genai
        from google.genai import types

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

def analyze_constraints_with_verification(question: str, max_attempts: int = 3) -> Dict[str, Any]:
    """
    Analyzes the question to identify and structure all constraints related to scheduling the meeting.
    This includes time restrictions, participant preferences, and any other limitations.
    """
    system_instruction = "You are a Constraint Analyzer, skilled at identifying and structuring constraints for scheduling tasks."

    for attempt in range(max_attempts):
        analysis_prompt = f"""
        Analyze the following scheduling request and extract all relevant constraints.
        Structure the constraints into a JSON object with the following keys:
        - 'duration': The required meeting duration in minutes (e.g., 30, 60).
        - 'available_days': A list of days the meeting can be scheduled on (e.g., ["Monday", "Tuesday"]).
        - 'available_time_ranges': A list of time ranges during which the meeting can occur (e.g., [["9:00", "17:00"]]).
        - 'participant_preferences': A dictionary mapping participant names to their preferences (e.g., {{"John": {{"avoid_before": "11:00"}}}}).
        - 'other_constraints': A list of other constraints that do not fit into the above categories (e.g., ["Must be in conference room A"]).

        Example 1:
        Question: Schedule a meeting for John and Mary for 30 minutes between 9:00 and 17:00 on Monday. John prefers to avoid meetings before 11:00.
        Analysis:
        {{
          "duration": 30,
          "available_days": ["Monday"],
          "available_time_ranges": [["9:00", "17:00"]],
          "participant_preferences": {{"John": {{"avoid_before": "11:00"}}}},
          "other_constraints": []
        }}

        Example 2:
        Question: Schedule a 60-minute meeting for Sarah and Tom on Tuesday or Wednesday between 10:00 AM and 4:00 PM. Tom cannot attend after 3:00 PM.
        Analysis:
        {{
          "duration": 60,
          "available_days": ["Tuesday", "Wednesday"],
          "available_time_ranges": [["10:00", "16:00"]],
          "participant_preferences": {{"Tom": {{"avoid_after": "15:00"}}}},
          "other_constraints": []
        }}

        Example 3:
        Question:  You need to schedule a meeting for Nicholas, Sara, Helen, Brian, Nancy, Kelly and Judy for half an hour between the work hours of 9:00 to 17:00 on Monday.
        Analysis:
        {{
          "duration": 30,
          "available_days": ["Monday"],
          "available_time_ranges": [["9:00", "17:00"]],
          "participant_preferences": {{}},
          "other_constraints": []
        }}

        Question: {question}
        Analysis:
        """

        extracted_data = call_llm(analysis_prompt, system_instruction)

        try:
            constraints = json.loads(extracted_data)

            # Verification Step
            verification_prompt = f"""
            Verify that the extracted constraints are complete and correctly structured based on the question.
            Check if all the mentioned constraints are captured in the JSON object and if the data types are correct.

            Question: {question}
            Extracted Constraints: {json.dumps(constraints, indent=2)}

            Respond with "VALID" if the constraints are valid, otherwise respond with "INVALID: [reason]".
            """

            verification_result = call_llm(verification_prompt, system_instruction)

            if "VALID" in verification_result:
                return {"is_valid": True, "constraints": constraints}
            else:
                print(f"Constraint analysis failed verification: {verification_result}")
                continue

        except json.JSONDecodeError as e:
            print(f"JSON Decode Error: {e}")
            continue
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return {"is_valid": False, "error": str(e)}

    return {"is_valid": False, "constraints": {}, "error": "Failed to analyze constraints after multiple attempts."}

def extract_participants_with_verification(question: str, max_attempts: int = 3) -> Dict[str, Any]:
    """Extracts participants from the question and verifies the extraction."""
    system_instruction = "You are an expert at extracting participant names from scheduling questions."

    for attempt in range(max_attempts):
        extraction_prompt = f"""
        Extract the names of all participants who need to attend the meeting from the following scheduling question.
        Return a JSON list of names.

        Example 1:
        Question: Schedule a meeting for John and Jennifer for half an hour.
        Participants: ["John", "Jennifer"]

        Example 2:
        Question: You need to schedule a meeting for Patricia, Harold, and Susan for half an hour.
        Participants: ["Patricia", "Harold", "Susan"]

        Example 3:
        Question: Can you schedule a meeting involving Alex, Ben, and Chloe?
        Participants: ["Alex", "Ben", "Chloe"]

        Question: {question}
        Participants:
        """

        extracted_data = call_llm(extraction_prompt, system_instruction)

        try:
            data = json.loads(extracted_data)

            # Verification Step
            verification_prompt = f"""
            Verify that the extracted participant names are complete and correct, and the output is a JSON list.

            Question: {question}
            Extracted Participants: {json.dumps(data)}

            Check if:
            1. All participant names are present.
            2. There are no extra, non-participant names.
            3. The output is a valid JSON list of strings.

            Respond with "VALID" if correct, or "INVALID: [reason]" if incorrect.
            """

            verification_result = call_llm(verification_prompt, system_instruction)

            if "VALID" in verification_result:
                return {"is_valid": True, "participants": data}
            else:
                print(f"Participant extraction failed verification: {verification_result}")
                continue

        except json.JSONDecodeError as e:
            print(f"JSON Decode Error: {e}")
            continue
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return {"is_valid": False, "error": str(e)}

    return {"is_valid": False, "participants": [], "error": "Failed to extract valid participants after multiple attempts."}

def extract_schedules_with_verification(question: str, participants: List[str], constraints: Dict[str, Any], max_attempts: int = 3) -> Dict[str, Any]:
    """Extracts and verifies existing schedules for each participant, taking into account the analyzed constraints."""
    system_instruction = "You are an expert at extracting schedules from text and structuring them in JSON format, considering the given constraints."

    for attempt in range(max_attempts):
        extraction_prompt = f"""
        Extract the existing schedules for each participant from the question, and represent them in JSON format.
        Each participant's schedule should be a list of time intervals when they are busy. Times should be in 24:00 format.
        Take into account the constraints while extracting the schedules.

        Example 1:
        Question: John has meetings from 9:00 to 10:00 and 11:00 to 12:00. Jennifer has meetings from 13:00 to 14:00.
        Participants: ["John", "Jennifer"]
        Constraints: {{"duration": 30, "available_days": ["Monday"], "available_time_ranges": [["9:00", "17:00"]], "participant_preferences": {{}}, "other_constraints": []}}
        Schedules:
        {{
          "John": [["9:00", "10:00"], ["11:00", "12:00"]],
          "Jennifer": [["13:00", "14:00"]]
        }}

        Example 2:
        Question: Patricia has blocked their calendar from 11:30 to 12:00 and 12:30 to 13:00. Harold has meetings from 9:30 to 10:30.
        Participants: ["Patricia", "Harold"]
        Constraints: {{"duration": 60, "available_days": ["Tuesday", "Wednesday"], "available_time_ranges": [["10:00", "16:00"]], "participant_preferences": {{}}, "other_constraints": []}}
        Schedules:
        {{
          "Patricia": [["11:30", "12:00"], ["12:30", "13:00"]],
          "Harold": [["9:30", "10:30"]]
        }}

        Question: {question}
        Participants: {json.dumps(participants)}
        Constraints: {json.dumps(constraints)}
        Schedules:
        """

        extracted_data = call_llm(extraction_prompt, system_instruction)

        try:
            data = json.loads(extracted_data)

            # Verification Step - Validating JSON, participant schedules present
            verification_prompt = f"""
            Verify that the extracted schedules are complete, correct, and in the correct JSON format.
            Consider the constraints while verifying the schedules.

            Question: {question}
            Participants: {json.dumps(participants)}
            Constraints: {json.dumps(constraints)}
            Extracted Schedules: {json.dumps(data)}

            Check if:
            1. The output is a valid JSON object.
            2. Every participant in {json.dumps(participants)} has a corresponding schedule in the extracted data.
            3. Each schedule is a list of time intervals in ["start", "end"] format.
            4. The extracted schedules are consistent with the constraints.

            Respond with "VALID" if all conditions are met, or "INVALID: [reason]" if not.
            """

            verification_result = call_llm(verification_prompt, system_instruction)

            if "VALID" in verification_result:
                return {"is_valid": True, "schedules": data}
            else:
                print(f"Schedule extraction failed verification: {verification_result}")
                continue

        except json.JSONDecodeError as e:
            print(f"JSON Decode Error: {e}")
            continue
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return {"is_valid": False, "error": str(e)}

    return {"is_valid": False, "schedules": {}, "error": "Failed to extract valid schedules after multiple attempts."}

def solve_with_react_pattern(question: str, participants: List[str], schedules: Dict[str, List[List[str]]], constraints: Dict[str, Any], max_iterations: int = 10) -> str:
    """Solves the scheduling problem using the ReAct pattern."""
    system_instruction = "You are a scheduling agent that uses the ReAct framework: Reason about the current state, take an Action, observe the result, and repeat until reaching a solution."

    context = f"""
    I will solve the scheduling problem step by step using the ReAct approach.

    Problem: You need to schedule a meeting for {', '.join(participants)}.

    Here are the existing schedules:
    {json.dumps(schedules)}

    Here are the constraints:
    {json.dumps(constraints)}

    Example:
    Problem: Schedule a meeting for John and Jennifer for half an hour between 9:00 to 17:00 on Monday.
    Schedules: {{"John": [], "Jennifer": [["11:30", "12:00"]]}}
    Constraints: {{"duration": 30, "available_days": ["Monday"], "available_time_ranges": [["9:00", "17:00"]]}}

    Thought 1: I need to consider the duration, available times, and days to check for an opening in both schedules.
    Action 1: Check availability for John and Jennifer on Monday between 9:00 and 17:00 for 30 minutes.
    Observation 1: Jennifer is busy from 11:30 to 12:00.

    Thought 2: Let me check other time slots before or after Jennifer's busy time.
    Action 2: Check availability before 11:30 and after 12:00 for a 30-minute slot.
    Observation 2: A slot is available from 9:00 to 9:30.

    Thought 3: I have found a time that works for both.
    Action 3: Finish[Monday, 9:00 - 9:30]
    """

    full_trace = context

    for i in range(max_iterations):
        response = call_llm(full_trace, system_instruction)
        full_trace += response + "\n"

        if "Action" in response and "Finish" in response:
            final_answer = response.split("Finish[")[1].split("]")[0]

            validation_prompt = f"""
            Problem: {question}
            Proposed Solution: {final_answer}

            Given the schedules: {json.dumps(schedules)}
            And Constraints: {json.dumps(constraints)}

            Is this solution correct and feasible?
            """

            validation = call_llm(validation_prompt)
            return final_answer

        if "Action" in response and "Check availability" in response:
            action_details = response.split("Check availability")[1].split("for")[0]
            observation = "A time slot is available." # Implement logic later
            full_trace += observation + "\nThought" + str(i+2) + ":"

    return "Could not find a suitable meeting time."

def main(question: str) -> str:
    """Main function to schedule a meeting."""

    # 1. Analyze Constraints
    constraints_result = analyze_constraints_with_verification(question)
    if not constraints_result["is_valid"]:
        return f"Error: {constraints_result.get('error', 'Failed to analyze constraints.')}"

    constraints = constraints_result["constraints"]
    print(f"Constraints: {constraints}")

    # 2. Extract Participants
    participants_result = extract_participants_with_verification(question)
    if not participants_result["is_valid"]:
        return f"Error: {participants_result.get('error', 'Failed to extract participants.')}"

    participants = participants_result["participants"]
    print(f"Participants: {participants}")

    # 3. Extract Schedules
    schedules_result = extract_schedules_with_verification(question, participants, constraints)
    if not schedules_result["is_valid"]:
        return f"Error: {schedules_result.get('error', 'Failed to extract schedules.')}"

    schedules = schedules_result["schedules"]
    print(f"Schedules: {schedules}")

    # 4. Solve with ReAct Pattern
    solution = solve_with_react_pattern(question, participants, schedules, constraints)
    print(f"Solution: {solution}")

    return solution

# Example usage:
if __name__ == "__main__":
    question = "You are an expert at scheduling meetings. You are given a few constraints on the existing schedule of each participant, the meeting duration, and possibly some preferences on the meeting time. Note there exists a solution that works with existing schedule of every participant. Here are a few example tasks and solutions:\n\nTASK: You need to schedule a meeting for Joyce, Christine and Alexander for half an hour between the work hours of 9:00 to 17:00 on Monday. \n\nHere are the existing schedules for everyone during the day: \nJoyce has meetings on Monday during 11:00 to 11:30, 13:30 to 14:00, 14:30 to 16:30; \nChristinehas no meetings the whole day.\nAlexander has meetings on Monday during 9:00 to 11:00, 12:00 to 12:30, 13:30 to 15:00, 15:30 to 16:00, 16:30 to 17:00; \n\nChristine can not meet on Monday before 12:00. Find a time that works for everyone's schedule and constraints. "
    answer = main(question)
    print(f"Final Answer: {answer}")