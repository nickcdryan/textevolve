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

def extract_info_agent(problem):
    """Extract participants, duration, and work hours using LLM."""
    system_instruction = "You are an information extraction expert."
    prompt = f"Extract participants, meeting duration, and work hours from: {problem}"
    return call_llm(prompt, system_instruction)

def schedule_solution_agent(info):
    """Propose a solution based on extracted info using LLM."""
    system_instruction = "You are a scheduling expert."
    prompt = f"Based on the following information, propose a meeting time: {info}"
    return call_llm(prompt, system_instruction)

def critique_solution_agent(problem, proposed_solution):
    """Critique the proposed solution against the problem constraints using LLM."""
    system_instruction = "You are a solution critique expert."
    prompt = f"Critique this proposed solution: {proposed_solution} given the problem: {problem}"
    return call_llm(prompt, system_instruction)

def main(question):
    """Main function to schedule a meeting."""
    try:
        # Extract information
        info = extract_info_agent(question)

        # Propose a solution
        proposed_solution = schedule_solution_agent(info)

        # Critique the solution
        critique = critique_solution_agent(question, proposed_solution)

        # Return the solution (can refine based on critique in future iterations)
        return proposed_solution
    except Exception as e:
        return f"Error: {str(e)}"