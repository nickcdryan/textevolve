import os
import json

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

def problem_decomposer_llm(problem):
    system_instruction = "You are an expert at breaking down scheduling problems into manageable parts. Identify participants, constraints, and objectives."
    prompt = f"Decompose this problem: {problem}"
    return call_llm(prompt, system_instruction)

def constraint_extractor_llm(problem_decomposition):
    system_instruction = "You are an information extraction specialist focusing on scheduling conflicts. Extract all time conflicts for each participant."
    prompt = f"Based on this problem decomposition, extract all time conflicts: {problem_decomposition}"
    return call_llm(prompt, system_instruction)

def solution_generator_llm(constraints, problem):
    system_instruction = "You are a scheduling assistant that can propose solutions based on extracted constraints. Suggest a meeting time that satisfies all conditions."
    prompt = f"Considering these constraints: {constraints}, generate a suitable meeting time for the problem: {problem}"
    return call_llm(prompt, system_instruction)

def solution_verifier_llm(solution, constraints, problem):
    system_instruction = "You are a meticulous verifier. Check if a given meeting time satisfies ALL scheduling constraints, and identify any violations."
    prompt = f"Verify if this solution: {solution} is valid given these constraints: {constraints} for problem: {problem}. Indicate any violations."
    return call_llm(prompt, system_instruction)

def main(question):
    try:
        decomposition = problem_decomposer_llm(question)
        constraints = constraint_extractor_llm(decomposition)
        proposed_solution = solution_generator_llm(constraints, question)
        verification_result = solution_verifier_llm(proposed_solution, constraints, question)

        if "no violations" in verification_result.lower():
            return f"Here is the proposed time: {proposed_solution}"
        else:
            return f"Error: Initial solution invalid. Verification result: {verification_result}"

    except Exception as e:
        return f"An error occurred: {str(e)}"