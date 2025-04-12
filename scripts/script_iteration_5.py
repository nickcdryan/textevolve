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
    """Extract participants, duration, and time constraints"""
    system_instruction = "You are an expert information extractor for scheduling meetings. Extract the participants, meeting duration, and time constraints."
    prompt = f"Extract information from the following problem: {problem}"
    return call_llm(prompt, system_instruction)

def propose_solution_llm(extracted_info):
    """Propose a meeting time based on extracted information."""
    system_instruction = "You are an expert meeting scheduler. Given the extracted information, propose a valid meeting time."
    prompt = f"Propose a meeting time based on this information: {extracted_info}"
    return call_llm(prompt, system_instruction)

def critique_solution_llm(problem, proposed_solution):
    """Critique the proposed solution and suggest improvements."""
    system_instruction = "You are a solution critic. Critique the proposed solution and point out any potential issues or improvements."
    prompt = f"Critique this proposed solution: {proposed_solution} for the problem: {problem}"
    return call_llm(prompt, system_instruction)

def main(question):
    """Main function to schedule meetings."""
    try:
        extracted_info = extract_info_llm(question)
        if "Error" in extracted_info:
            return "Could not extract information."

        proposed_solution = propose_solution_llm(extracted_info)
        if "Error" in proposed_solution:
            return "Could not propose a solution."

        critique = critique_solution_llm(question, proposed_solution)
        if "Error" in critique:
            return "Could not critique the solution."

        return proposed_solution  # Return the proposed solution directly

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return f"Error: {str(e)}"