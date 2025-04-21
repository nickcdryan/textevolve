import os
import re
import json
import math
from typing import List, Dict, Any

# Overall reasoning:
# This iteration explores a new approach that focuses on a hybrid approach:
# 1) A rule-based system extracts preliminary data (participants and initial time possibilities)
# 2) An LLM refines these possibilities based on complex constraints

# The hypothesis is that using a rule-based system for preliminary data extraction will reduce the load on the LLM,
# allowing it to focus on complex constraint reasoning. This hybrid approach should improve efficiency and accuracy.
# We will test this approach and add verification steps to deduce if the changes are helpful.

# Unlike previous iterations, this iteration will not call LLM to extract the participants

# The script contains several functions including extract_participants_rule_based, refine_schedule_with_llm and main.
# This approach will use multi-example prompting and incorporate validation loops at the refinement stage.

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

def extract_participants_rule_based(question: str) -> List[str]:
    """
    Extracts participants from the question using a rule-based approach (regex).
    """
    try:
        match = re.search(r"schedule a meeting for (.*?) for", question)
        if match:
            participants = [name.strip() for name in match.group(1).split(',')]
            return participants
        else:
            return []
    except Exception as e:
        print(f"Error extracting participants: {str(e)}")
        return []

def refine_schedule_with_llm(question: str, participants: List[str], max_attempts: int = 3) -> str:
    """
    Refines the schedule using LLM, considering complex constraints.
    """
    system_instruction = "You are an expert at refining meeting schedules based on complex constraints."

    for attempt in range(max_attempts):
        refinement_prompt = f"""
        Refine the meeting schedule based on the following information.
        Participants: {participants}
        Question: {question}

        Consider all constraints and provide the best possible meeting schedule.

        Example 1:
        Participants: ["John", "Jennifer"]
        Question: Schedule a meeting for John and Jennifer for half an hour between 9:00 to 17:00 on Monday. John would like to avoid meetings after 14:00.
        Refined Schedule: Monday, 9:00 - 9:30

        Example 2:
        Participants: ["Patricia", "Harold"]
        Question: Schedule a meeting for Patricia and Harold for an hour between 10:00 and 16:00 on Tuesday or Wednesday. Harold would rather not meet before 11:00.
        Refined Schedule: Tuesday, 11:00 - 12:00

        Example 3:
        Participants: ["Nicholas", "Sara", "Helen", "Brian", "Nancy", "Kelly", "Judy"]
        Question: You need to schedule a meeting for Nicholas, Sara, Helen, Brian, Nancy, Kelly and Judy for half an hour between the work hours of 9:00 to 17:00 on Monday.
        Refined Schedule: Monday, 14:00 - 14:30

        Refined Schedule:
        """

        refined_schedule = call_llm(refinement_prompt, system_instruction)

        # Verification Step
        verification_prompt = f"""
            Verify if the refined schedule is feasible and satisfies all constraints.
            Participants: {participants}
            Question: {question}
            Refined Schedule: {refined_schedule}

            Respond with "VALID" if the schedule is valid, or "INVALID: [reason]" if not.
            """

        verification_result = call_llm(verification_prompt, system_instruction)

        if "VALID" in verification_result:
            return refined_schedule
        else:
            print(f"Schedule refinement failed verification: {verification_result}")
            continue

    return "Could not find a suitable meeting time."

def main(question: str) -> str:
    """Main function to schedule a meeting."""
    try:
        # 1. Extract Participants (Rule-based)
        participants = extract_participants_rule_based(question)
        if not participants:
            return "Error: Could not extract participants."

        # 2. Refine Schedule with LLM
        refined_schedule = refine_schedule_with_llm(question, participants)
        if not refined_schedule:
            return "Error: Could not refine schedule."

        return f"Here is the proposed time: {refined_schedule}"

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return f"Error: {str(e)}"

# Example usage:
if __name__ == "__main__":
    question = "You are an expert at scheduling meetings. You are given a few constraints on the existing schedule of each participant, the meeting duration, and possibly some preferences on the meeting time. Note there exists a solution that works with existing schedule of every participant. Here are a few example tasks and solutions:\n\nTASK: You need to schedule a meeting for Joyce, Christine and Alexander for half an hour between the work hours of 9:00 to 17:00 on Monday. \n\nHere are the existing schedules for everyone during the day: \nJoyce has meetings on Monday during 11:00 to 11:30, 13:30 to 14:00, 14:30 to 16:30; \nChristinehas no meetings the whole day.\nAlexander has meetings on Monday during 9:00 to 11:00, 12:00 to 12:30, 13:30 to 15:00, 15:30 to 16:00, 16:30 to 17:00; \n\nChristine can not meet on Monday before 12:00. Find a time that works for everyone's schedule and constraints. "
    answer = main(question)
    print(f"Final Answer: {answer}")