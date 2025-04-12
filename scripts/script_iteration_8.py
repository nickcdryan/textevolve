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

def extract_info_llm(problem):
    """Extract key information using LLM"""
    system_instruction = "You are an expert at extracting relevant scheduling information."
    prompt = f"Extract the participants, duration, working hours, and existing schedules from: {problem}"
    return call_llm(prompt, system_instruction)

def suggest_time_llm(info):
    """Suggest a meeting time based on extracted info using LLM"""
    system_instruction = "You are an expert meeting scheduler."
    prompt = f"Based on this information, suggest a possible meeting time: {info}"
    return call_llm(prompt, system_instruction)

def verify_time_llm(problem, suggested_time):
    """Verify if the suggested time is valid using LLM"""
    system_instruction = "You are a meticulous meeting time verifier."
    prompt = f"Given the problem: {problem} and suggested time: {suggested_time}, is the suggested time valid? Explain why or why not."
    return call_llm(prompt, system_instruction)

def main(question):
    """Main function to schedule a meeting"""
    try:
        extracted_info = extract_info_llm(question)
        suggested_time = suggest_time_llm(extracted_info)
        verification_result = verify_time_llm(question, suggested_time)

        if "valid" in verification_result.lower():
            return f"Here is the proposed time: {suggested_time}"
        else:
            return "Could not find a valid time."

    except Exception as e:
        print(f"An error occurred: {e}")
        return "Error processing the request."