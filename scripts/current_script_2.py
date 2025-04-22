import os
import re
import math

def main(question):
    """Main function to schedule meetings using a new LLM-driven approach."""
    try:
        # Step 1: Analyze the question and extract relevant information using LLM
        analyzed_info = analyze_question(question)

        # Step 2: Generate potential meeting times using LLM
        potential_times = generate_potential_times(analyzed_info)

        # Step 3: Validate each potential time against constraints
        validated_times = validate_potential_times(potential_times, analyzed_info, question)

        # Step 4: Select the best time from validated times
        best_time = select_best_time(validated_times)

        return best_time

    except Exception as e:
        return f"Error processing the request: {str(e)}"

def analyze_question(question):
    """Analyzes the question using LLM to extract key information."""
    system_instruction = "You are an expert at analyzing meeting scheduling questions."
    prompt = f"""
    Analyze the following question to extract key information such as participants, duration, days, and schedules.

    Example 1:
    Question: Schedule a meeting for John and Mary for 30 minutes on Monday. John is busy from 9-10, Mary is busy from 11-12.
    Analysis: Participants: John, Mary; Duration: 30 minutes; Days: Monday; John's schedule: 9:00-10:00; Mary's schedule: 11:00-12:00.

    Example 2:
    Question: Schedule a meeting for Alice, Bob, and Charlie for 1 hour on Tuesday and Wednesday. Alice is busy from 14:00-15:00 on Tuesday, Bob is busy from 10:00-11:00 on Wednesday. Charlie is free.
    Analysis: Participants: Alice, Bob, Charlie; Duration: 1 hour; Days: Tuesday, Wednesday; Alice's schedule (Tuesday): 14:00-15:00; Bob's schedule (Wednesday): 10:00-11:00; Charlie is free.

    Question: {question}
    Analysis:
    """
    return call_llm(prompt, system_instruction)

def generate_potential_times(analyzed_info):
    """Generates potential meeting times using LLM based on the analyzed information."""
    system_instruction = "You are an expert at generating potential meeting times."
    prompt = f"""
    Based on the analyzed information, generate 3 potential meeting times.

    Example 1:
    Analyzed Information: Participants: John, Mary; Duration: 30 minutes; Days: Monday; John's schedule: 9:00-10:00; Mary's schedule: 11:00-12:00.
    Potential Times: Monday 10:00-10:30, Monday 14:00-14:30, Monday 15:00-15:30

    Example 2:
    Analyzed Information: Participants: Alice, Bob, Charlie; Duration: 1 hour; Days: Tuesday, Wednesday; Alice's schedule (Tuesday): 14:00-15:00; Bob's schedule (Wednesday): 10:00-11:00; Charlie is free.
    Potential Times: Tuesday 10:00-11:00, Wednesday 11:00-12:00, Wednesday 14:00-15:00

    Analyzed Information: {analyzed_info}
    Potential Times:
    """
    return call_llm(prompt, system_instruction)

def validate_potential_times(potential_times, analyzed_info, question):
    """Validates each potential meeting time against the constraints using LLM."""
    system_instruction = "You are an expert at validating potential meeting times against constraints."
    prompt = f"""
    For each potential meeting time, determine if it works for all participants based on their schedules extracted from the original question.
    Respond as a list of tuples, where each tuple contains the potential time and a boolean indicating its validity (True if valid, False otherwise).
    Format the response as plain text.

    Example 1:
    Question: Schedule a meeting for John and Mary for 30 minutes on Monday. John is busy from 9-10, Mary is busy from 11-12.
    Potential Times: Monday 10:00-10:30, Monday 14:00-14:30, Monday 15:00-15:30
    Validation: [("Monday 10:00-10:30", True), ("Monday 14:00-14:30", True), ("Monday 15:00-15:30", True)]

    Example 2:
    Question: Schedule a meeting for Alice, Bob, and Charlie for 1 hour on Tuesday and Wednesday. Alice is busy from 14:00-15:00 on Tuesday, Bob is busy from 10:00-11:00 on Wednesday. Charlie is free.
    Potential Times: Tuesday 10:00-11:00, Wednesday 11:00-12:00, Wednesday 14:00-15:00
    Validation: [("Tuesday 10:00-11:00", True), ("Wednesday 11:00-12:00", False), ("Wednesday 14:00-15:00", True)]

    Question: {question}
    Analyzed Information: {analyzed_info}
    Potential Times: {potential_times}
    Validation:
    """
    return call_llm(prompt, system_instruction)

def select_best_time(validated_times):
    """Selects the best meeting time from the validated times."""
    try:
        # Use regex to parse the LLM response
        times = re.findall(r'\("([^"]*)", (True|False)\)', validated_times)
        
        # Convert string booleans to actual booleans
        times = [(time, valid == "True") for time, valid in times]

        # Filter out invalid times
        valid_times = [time for time, valid in times if valid]

        if valid_times:
            # Select the first valid time
            return f"Here is the proposed time: {valid_times[0]}"
        else:
            return "No suitable meeting time found."
    except Exception as e:
        print(f"Error parsing validated times: {e}")
        return "Error: Could not determine a valid meeting time."
    

def call_llm(prompt, system_instruction=None):
    """Call the Gemini LLM with a prompt and return the response"""
    try:
        from google import genai
        from google.genai import types
        import os

        # Initialize the Gemini client
        client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

        # Call the API with system instruction if provided
        if system_instruction:
            response = client.models.generate_content(
                model="gemini-2.0-flash",
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction
                ),
                contents=prompt
            )
        else:
            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=prompt
            )

        return response.text
    except Exception as e:
        print(f"Error calling Gemini API: {str(e)}")
        return f"Error: {str(e)}"