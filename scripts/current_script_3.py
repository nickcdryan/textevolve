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

def problem_decomposer_agent(problem):
    """Decompose the problem into smaller parts"""
    system_instruction = "You are an expert problem decomposer for scheduling tasks."
    prompt = f"Break down the following problem into steps to find a meeting time: {problem}"
    return call_llm(prompt, system_instruction)

def constraint_extractor_agent(problem):
    """Extract constraints from the problem description"""
    system_instruction = "You are an expert constraint extractor."
    prompt = f"List all explicit and implicit constraints for this scheduling problem: {problem}"
    return call_llm(prompt, system_instruction)

def solution_generator_agent(problem, constraints):
    """Generate a solution based on the constraints"""
    system_instruction = "You are an expert scheduler who generates solutions based on constraints."
    prompt = f"Given the problem: {problem} and constraints: {constraints}, propose a meeting time."
    return call_llm(prompt, system_instruction)

def solution_verifier_agent(problem, proposed_solution):
    """Verify if the generated solution is valid"""
    system_instruction = "You are an expert at verifying if a scheduling solution is valid."
    prompt = f"Determine if this solution: {proposed_solution} is valid for the problem: {problem}. Explain your reasoning."
    return call_llm(prompt, system_instruction)

def main(question):
    """Main function to schedule meetings"""
    try:
        # Decompose the problem
        decomposition = problem_decomposer_agent(question)
        print(f"Decomposition: {decomposition}")

        # Extract constraints
        constraints = constraint_extractor_agent(question)
        print(f"Constraints: {constraints}")

        # Generate solution
        solution = solution_generator_agent(question, constraints)
        print(f"Proposed solution: {solution}")

        # Verify solution
        verification = solution_verifier_agent(question, solution)
        print(f"Verification: {verification}")

        return solution
    except Exception as e:
        return f"Error processing the request: {str(e)}"