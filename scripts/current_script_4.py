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

def problem_clarifier_with_llm(problem):
    """Clarify ambiguous parts of the problem for better understanding"""
    system_instruction = "You are an expert problem clarifier. Identify any ambiguities in the problem statement."
    prompt = f"Identify ambiguities and ask clarifying questions about this scheduling problem: {problem}"
    return call_llm(prompt, system_instruction)

def constraint_extractor_with_llm(problem):
    """Extract scheduling constraints from the problem statement using LLM."""
    system_instruction = "You are an information extractor specialized in identifying scheduling constraints."
    prompt = f"Extract all scheduling constraints from this text: {problem}"
    return call_llm(prompt, system_instruction)

def solution_generator_with_llm(problem, constraints):
    """Generate a potential solution based on problem and constraints."""
    system_instruction = "You are a scheduling expert who generates solutions based on constraints."
    prompt = f"Generate a possible meeting time that satisfies these constraints: {constraints} for this problem: {problem}"
    return call_llm(prompt, system_instruction)

def solution_verifier_with_llm(problem, proposed_solution):
    """Verify if the proposed solution meets all constraints using LLM."""
    system_instruction = "You are a critical evaluator who verifies if solutions satisfy all given constraints."
    prompt = f"Verify if this proposed solution satisfies all constraints: {proposed_solution} for problem: {problem}"
    return call_llm(prompt, system_instruction)

def main(question):
    """Main function to solve the scheduling problem."""
    try:
        clarified_problem = problem_clarifier_with_llm(question)
        constraints = constraint_extractor_with_llm(clarified_problem)
        solution = solution_generator_with_llm(clarified_problem, constraints)
        verification = solution_verifier_with_llm(clarified_problem, solution)

        if "satisfies" in verification.lower():
            return f"Here is the proposed time: {solution}"
        else:
            return "No valid solution found."
    except Exception as e:
        return f"An error occurred: {str(e)}"