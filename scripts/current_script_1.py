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

def problem_decomposer(problem):
    """Decompose the problem into sub-problems"""
    system_instruction = "You are an expert problem decomposer."
    prompt = f"Decompose the following scheduling problem into smaller parts: {problem}"
    return call_llm(prompt, system_instruction)

def constraint_extractor(problem):
    """Extract constraints from the problem description"""
    system_instruction = "You are an expert at extracting scheduling constraints."
    prompt = f"Extract all explicit and implicit scheduling constraints from: {problem}"
    return call_llm(prompt, system_instruction)

def solution_generator(constraints):
    """Generate a potential solution given the constraints"""
    system_instruction = "You are an expert scheduling assistant."
    prompt = f"Generate a possible meeting time given these constraints: {constraints}"
    return call_llm(prompt, system_instruction)

def solution_verifier(problem, solution):
    """Verify if a solution is valid given the problem description"""
    system_instruction = "You are a critical solution verifier."
    prompt = f"Verify if the proposed solution '{solution}' is valid for the problem: {problem}"
    return call_llm(prompt, system_instruction)

def main(question):
    """Main function to solve the scheduling problem"""
    try:
        decomposed_problem = problem_decomposer(question)
        constraints = constraint_extractor(decomposed_problem)
        proposed_solution = solution_generator(constraints)
        verification_result = solution_verifier(question, proposed_solution)

        if "valid" in verification_result.lower():
            return f"Here is the proposed time: {proposed_solution}"
        else:
            return "Could not find a valid solution."
    except Exception as e:
        return f"An error occurred: {str(e)}"