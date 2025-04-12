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

def meeting_scheduler(question):
    """Schedules a meeting based on the given question."""

    # Agent 1: Problem Decomposer
    system_instruction_decomposer = "You are an expert at understanding complex meeting scheduling requests and decomposing them into smaller parts."
    prompt_decomposer = f"Decompose the following scheduling request into participants, available days, time constraints, duration, and preferences: {question}. Return the information as a single string."
    decomposed_info = call_llm(prompt_decomposer, system_instruction_decomposer)

    # Agent 2: Candidate Time Generator
    system_instruction_generator = "You are a creative scheduler that generates possible meeting times given the constraints."
    prompt_generator = f"Given the following information: {decomposed_info}, suggest three possible meeting times. Return these in the following format: Time 1: [Day], [Start Time] - [End Time]; Time 2: ...; Time 3: ..."
    candidate_times = call_llm(prompt_generator, system_instruction_generator)

    # Agent 3: Solution Validator and Selector
    system_instruction_validator = "You are a strict validator and decision maker. Select the *single* best meeting time from a list of candidates."
    prompt_validator = f"Here are the candidate meeting times: {candidate_times}. Considering the original request: {question}, select the *single* best time that meets all constraints and preferences. Explicitly state your reasoning. Then, format your answer as: Here is the proposed time: [Day], [Start Time] - [End Time]."
    final_answer = call_llm(prompt_validator, system_instruction_validator)

    # Return the final answer
    return final_answer

def main(question):
    """Main function to execute the meeting scheduler."""
    try:
        answer = meeting_scheduler(question)
        return answer
    except Exception as e:
        return f"An error occurred: {str(e)}"

# Example usage
if __name__ == '__main__':
    example_question = "You need to schedule a meeting for Carol and Mark for half an hour between the work hours of 9:00 to 17:00 on Monday. Carol has blocked their calendar on Monday during 10:00 to 11:00, 14:30 to 15:00, 15:30 to 17:00; Mark has blocked their calendar on Monday during 9:30 to 10:00, 10:30 to 17:00;"
    result = main(example_question)
    print(result)