import os
import re
import json
import math

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

def extract_participants_with_verification(question, max_attempts=3):
    """Extract participants from the question with verification steps."""
    system_instruction = "You are an expert at extracting participant names from scheduling questions."

    for attempt in range(max_attempts):
        # Improved prompt with multi-example few-shot learning
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

def extract_schedules_with_verification(question, participants, max_attempts=3):
    """Extract and verify existing schedules for each participant."""
    system_instruction = "You are an expert at extracting schedules from text and structuring them in JSON format."

    for attempt in range(max_attempts):
        # Enhanced prompt with multiple examples, and focusing on structured output
        extraction_prompt = f"""
        Extract the existing schedules for each participant from the question, and represent them in JSON format.
        Each participant's schedule should be a list of time intervals when they are busy. Times should be in 24:00 format.

        Example 1:
        Question: John has meetings from 9:00 to 10:00 and 11:00 to 12:00. Jennifer has meetings from 13:00 to 14:00.
        Participants: ["John", "Jennifer"]
        Schedules:
        {{
          "John": [["9:00", "10:00"], ["11:00", "12:00"]],
          "Jennifer": [["13:00", "14:00"]]
        }}

        Example 2:
        Question: Patricia has blocked their calendar from 11:30 to 12:00 and 12:30 to 13:00. Harold has meetings from 9:30 to 10:30.
        Participants: ["Patricia", "Harold"]
        Schedules:
        {{
          "Patricia": [["11:30", "12:00"], ["12:30", "13:00"]],
          "Harold": [["9:30", "10:30"]]
        }}

        Question: {question}
        Participants: {json.dumps(participants)}
        Schedules:
        """

        extracted_data = call_llm(extraction_prompt, system_instruction)

        try:
            data = json.loads(extracted_data)

            # Verification Step - Validating JSON, participant schedules present
            verification_prompt = f"""
            Verify that the extracted schedules are complete, correct, and in the correct JSON format.

            Question: {question}
            Participants: {json.dumps(participants)}
            Extracted Schedules: {json.dumps(data)}

            Check if:
            1. The output is a valid JSON object.
            2. Every participant in {json.dumps(participants)} has a corresponding schedule in the extracted data.
            3. Each schedule is a list of time intervals in ["start", "end"] format.

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

def extract_constraints_with_verification(question, participants, max_attempts=3):
    """Extract and verify constraints from the scheduling question."""
    system_instruction = "You are an expert at extracting constraints from scheduling questions."

    for attempt in range(max_attempts):
        # Prompt enhanced with multi-example prompt and detailed formatting instructions
        extraction_prompt = f"""
        Extract the constraints on the meeting time from the question. Constraints may include preferred days, times, or durations.
        Represent the constraints as a JSON object.

        Example 1:
        Question: Schedule a meeting for John and Jennifer for half an hour between 9:00 to 17:00 on Monday. John would like to avoid meetings after 14:00.
        Participants: ["John", "Jennifer"]
        Constraints:
        {{
          "duration": "30 minutes",
          "available_time": [["9:00", "17:00"]],
          "days": ["Monday"],
          "preferences": {{"John": {{"avoid_after": "14:00"}}}}
        }}

        Example 2:
        Question: Schedule a meeting for Patricia and Harold for an hour between 10:00 and 16:00 on Tuesday or Wednesday. Harold would rather not meet before 11:00.
        Participants: ["Patricia", "Harold"]
        Constraints:
        {{
          "duration": "60 minutes",
          "available_time": [["10:00", "16:00"]],
          "days": ["Tuesday", "Wednesday"],
          "preferences": {{"Harold": {{"avoid_before": "11:00"}}}}
        }}

        Question: {question}
        Participants: {json.dumps(participants)}
        Constraints:
        """

        extracted_data = call_llm(extraction_prompt, system_instruction)

        try:
            data = json.loads(extracted_data)

            # Verification Step - JSON validity and semantic correctness
            verification_prompt = f"""
            Verify that the extracted constraints are complete, correct, and in the correct JSON format.

            Question: {question}
            Participants: {json.dumps(participants)}
            Extracted Constraints: {json.dumps(data)}

            Check if:
            1. The output is a valid JSON object.
            2. All constraints mentioned in the question are captured in the JSON.
            3. Time values are in the correct ["start", "end"] format if present.
            4. Duration is specified if available.

            Respond with "VALID" if all conditions are met, or "INVALID: [reason]" if not.
            """

            verification_result = call_llm(verification_prompt, system_instruction)

            if "VALID" in verification_result:
                return {"is_valid": True, "constraints": data}
            else:
                print(f"Constraint extraction failed verification: {verification_result}")
                continue

        except json.JSONDecodeError as e:
            print(f"JSON Decode Error: {e}")
            continue
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return {"is_valid": False, "error": str(e)}

    return {"is_valid": False, "constraints": {}, "error": "Failed to extract valid constraints after multiple attempts."}

def solve_with_react_pattern(question, participants, schedules, constraints, max_iterations=10):
    """Solve the scheduling problem using the ReAct pattern."""
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
    Constraints: {{"duration": "30 minutes", "available_time": [["9:00", "17:00"]], "days": ["Monday"]}}

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

def generate_solution_plan(question, participants, schedules, constraints):
    """Generate a solution plan for scheduling a meeting."""
    system_instruction = "You are an AI assistant that creates solution plans for scheduling problems."

    prompt = f"""
    Create a step-by-step solution plan for scheduling a meeting.

    Problem: {question}
    Participants: {json.dumps(participants)}
    Schedules: {json.dumps(schedules)}
    Constraints: {json.dumps(constraints)}

    Example 1:
    Problem: Schedule a meeting for John and Jennifer.
    Plan:
    1. Identify the available time slots for John.
    2. Identify the available time slots for Jennifer.
    3. Find the overlapping time slots between John and Jennifer.
    4. Select a time slot that meets the constraints.

    Example 2:
    Problem: Schedule a meeting for Patricia, Harold, and Susan.
    Plan:
    1. Extract available time intervals from Patricia's schedule.
    2. Extract available time intervals from Harold's schedule.
    3. Extract available time intervals from Susan's schedule.
    4. Find the common available intervals.
    5. Incorporate any given preferences on meeting times or days.
    """

    return call_llm(prompt, system_instruction)

def main(question):
    """Main function to schedule a meeting."""

    # 1. Extract Participants
    participants_result = extract_participants_with_verification(question)
    if not participants_result["is_valid"]:
        return f"Error: {participants_result.get('error', 'Failed to extract participants.')}"

    participants = participants_result["participants"]
    print(f"Participants: {participants}")

    # 2. Extract Schedules
    schedules_result = extract_schedules_with_verification(question, participants)
    if not schedules_result["is_valid"]:
        return f"Error: {schedules_result.get('error', 'Failed to extract schedules.')}"

    schedules = schedules_result["schedules"]
    print(f"Schedules: {schedules}")

    # 3. Extract Constraints
    constraints_result = extract_constraints_with_verification(question, participants)
    if not constraints_result["is_valid"]:
        return f"Error: {constraints_result.get('error', 'Failed to extract constraints.')}"

    constraints = constraints_result["constraints"]
    print(f"Constraints: {constraints}")

    # 4. Solve with ReAct Pattern
    solution = solve_with_react_pattern(question, participants, schedules, constraints)
    print(f"Solution: {solution}")

    return solution