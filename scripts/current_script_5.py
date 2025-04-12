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

def extract_info_llm(problem):
    """Extract info with LLM. Handles errors."""
    system_instruction = "You are a meticulous information extractor that identifies all people involved, their schedules, meeting duration and desired days."
    prompt = f"Extract all relevant information (participants, schedules, duration, days) from the text below:\n{problem}"
    return call_llm(prompt, system_instruction)

def generate_candidate_times_llm(extracted_info):
    """Generate candidate meeting times with LLM, considering preferences. Handles errors."""
    system_instruction = "You are an expert meeting scheduler. Generate possible meeting times based on the extracted information, taking into account the preferences and generating 3 distinct candidate times."
    prompt = f"Generate 3 candidate meeting times based on the following information:\n{extracted_info}\nBe very concise and only include meeting times. Format as: 'Day, Start Time - End Time'."
    return call_llm(prompt, system_instruction)

def validate_and_select_time_llm(problem, candidate_times):
    """Validate and select the best meeting time, acting as final decider. Handles errors."""
    system_instruction = "You are the final decider. You must meticulously validate each candidate meeting time against the original problem, and select the BEST one. State the selected meeting time. Ensure that the output has the prefix 'Here is the proposed time:' and the format of the final answer MUST be: [DAY], [START_TIME] - [END_TIME]"
    prompt = f"Original problem:\n{problem}\n\nCandidate meeting times:\n{candidate_times}\n\n Carefully analyze the constraints from the original problem, and only propose a time that follows ALL constraints. If none of the times work, generate a valid time. Ensure that the output has the prefix 'Here is the proposed time:' and the format of the final answer MUST be: [DAY], [START_TIME] - [END_TIME]"
    return call_llm(prompt, system_instruction)

def main(question):
    """Main function to schedule a meeting using LLM calls."""
    try:
        # 1. Extract info
        extracted_info = extract_info_llm(question)

        # 2. Generate candidate times
        candidate_times = generate_candidate_times_llm(extracted_info)

        # 3. Validate and select best time
        final_answer = validate_and_select_time_llm(question, candidate_times)
        if "Here is the proposed time:" not in final_answer:
            #Force final answer to be in format "Here is the proposed time:"
            final_answer = validate_and_select_time_llm(question, candidate_times)

        return final_answer
    except Exception as e:
        return f"Error: {str(e)}"