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

def extract_info_with_llm(problem):
    """Extract relevant information using LLM."""
    system_instruction = "You are an expert information extractor."
    prompt = f"Extract participants, schedules, duration, and time constraints from: {problem}"
    return call_llm(prompt, system_instruction)

def generate_candidate_times_with_llm(info):
    """Generate candidate meeting times using LLM."""
    system_instruction = "You are a meeting time generator."
    prompt = f"Generate 3 candidate meeting times based on the following information: {info}"
    return call_llm(prompt, system_instruction)

def validate_time_with_llm(problem, candidate_time):
    """Validate a candidate time against the problem constraints using LLM."""
    system_instruction = "You are a meeting scheduler validator."
    prompt = f"Determine if '{candidate_time}' is a valid meeting time for the following problem: {problem}. Explain your reasoning."
    return call_llm(prompt, system_instruction)

def main(question):
    """Main function to schedule a meeting."""

    # 1. Extract key information
    info = extract_info_with_llm(question)
    if "Error" in info:
        return "Error extracting information."

    # 2. Generate candidate times
    candidate_times_str = generate_candidate_times_with_llm(info)
    if "Error" in candidate_times_str:
        return "Error generating candidate times."

    candidate_times = candidate_times_str.split('\n')

    # 3. Validate candidate times and return the first valid one
    for candidate_time in candidate_times:
        validation_result = validate_time_with_llm(question, candidate_time)
        if "valid" in validation_result.lower():
            return f"Here is the proposed time: {candidate_time}"

    return "No valid meeting time found."