import os

def call_llm(prompt, system_instruction=None):
    """Call the Gemini LLM with a prompt and return the response"""
    try:
        from google import genai
        from google.genai import types

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

def extract_data_with_llm(problem):
    """Extract structured data (JSON) representing participants' schedules."""
    system_instruction = "You are a JSON data extractor. Your task is to extract information about peoples' schedules in a JSON format. You must find the participants, the days they're being scheduled for, the start and end work hours, and their availability."
    prompt = f"Extract the schedule and constraints in JSON format: {problem}"
    return call_llm(prompt, system_instruction)

def generate_candidate_times_with_llm(schedule_json, duration):
    """Generate candidate meeting times given a schedule."""
    system_instruction = "You are an expert in generating candidate meeting times based on availability. You should adhere to the constraints and give all possible times."
    prompt = f"Based on this schedule data: {schedule_json}, and duration {duration}, generate a list of potential meeting times."
    return call_llm(prompt, system_instruction)

def verify_solution_with_llm(schedule_json, proposed_solution):
    """Verify that the proposed solution is valid given a schedule."""
    system_instruction = "You are a meticulous solution verifier. Your task is to check if a proposed meeting time violates any constraints in the schedule. Return VALID or INVALID with a detailed explanation."
    prompt = f"Given the schedule: {schedule_json}, is the proposed solution: {proposed_solution} VALID or INVALID? Explain."
    return call_llm(prompt, system_instruction)

def main(question):
    """Main function to schedule a meeting."""
    try:
        # 1. Extract structured data using LLM
        schedule_json = extract_data_with_llm(question)

        # 2. Determine meeting duration - simplified for this example
        duration = "half an hour"

        # 3. Generate candidate meeting times using LLM
        candidate_times = generate_candidate_times_with_llm(schedule_json, duration)

        # 4. Verify each candidate time using LLM
        candidate_times_list = candidate_times.split("\n") # Assuming list formatting

        for time in candidate_times_list:
            verification_result = verify_solution_with_llm(schedule_json, time)
            if "VALID" in verification_result:
                return f"Here is the proposed time: {time}"

        return "No valid meeting time found."

    except Exception as e:
        return f"Error: {str(e)}"