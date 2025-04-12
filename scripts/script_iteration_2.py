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

def extract_info_llm(question):
    """LLM to extract participants, duration, day, and work hours."""
    system_instruction = "Extract meeting details. Specify participants, duration, day, and work hours."
    prompt = f"Extract all relevant information from: {question}"
    return call_llm(prompt, system_instruction)

def generate_candidate_times_llm(info):
    """LLM to generate candidate meeting times."""
    system_instruction = "Generate a list of 5 possible meeting times based on the extracted info. Provide the times as a list."
    prompt = f"Generate meeting times based on this info: {info}"
    return call_llm(prompt, system_instruction)

def validate_and_select_time_llm(question, candidate_times):
    """LLM to validate candidate times and select the best one."""
    system_instruction = "You are a meeting scheduler. Review the candidate times and select the BEST time that satisfies all the constraints in the question. Return ONLY the selected time in the format 'Day, Start Time - End Time'."
    prompt = f"Question: {question}\nCandidate Times: {candidate_times}\nSelect the best meeting time."
    return call_llm(prompt, system_instruction)

def main(question):
    """Main function to schedule a meeting."""
    try:
        # 1. Extract information
        extracted_info = extract_info_llm(question)

        # 2. Generate candidate times
        candidate_times = generate_candidate_times_llm(extracted_info)

        # 3. Validate and select the best time
        best_time = validate_and_select_time_llm(question, candidate_times)

        return f"Here is the proposed time: {best_time}"
    except Exception as e:
        return f"Error: {str(e)}"