import os

def call_llm(prompt, system_instruction=None):
    """Call the Gemini LLM with a prompt and return the response"""
    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

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
    """Extract relevant information like participants, duration, and constraints."""
    system_instruction = "You are an information extraction expert. Extract key details from the scheduling problem."
    prompt = f"Extract participants, duration, possible days, and individual constraints from this text: {problem}"
    return call_llm(prompt, system_instruction)

def suggest_times_with_llm(extracted_info):
    """Suggest candidate meeting times based on extracted information."""
    system_instruction = "You are a scheduling assistant. Suggest a few possible meeting times."
    prompt = f"Based on this information, suggest 3 possible meeting times: {extracted_info}"
    return call_llm(prompt, system_instruction)

def validate_time_with_llm(problem, proposed_time):
    """Validate if a proposed time works for all participants given the constraints."""
    system_instruction = "You are a meticulous meeting validator. Ensure the proposed time works for everyone."
    prompt = f"Given the original problem: {problem}, does this proposed time work: {proposed_time}? Explain your reasoning step by step and explicitly state YES or NO at the end."
    return call_llm(prompt, system_instruction)

def main(question):
    """Main function to schedule a meeting."""
    try:
        # 1. Extract information
        extracted_info = extract_info_with_llm(question)

        # 2. Suggest candidate times
        suggested_times = suggest_times_with_llm(extracted_info)

        # 3. Validate times (loop through suggestions, pick the first valid one)
        possible_times = suggested_times.split('\n') # Split into times
        
        for proposed_time in possible_times:
            validation_result = validate_time_with_llm(question, proposed_time)
            if "YES" in validation_result:  # Look for a 'YES' confirmation
                return f"Here is the proposed time: {proposed_time}"
            
        return "No suitable time found based on the proposed times."
    except Exception as e:
        return f"An error occurred: {str(e)}"