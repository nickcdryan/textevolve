import os
import re
import json
import math # for react
from google import genai
from google.genai import types

def call_llm(prompt, system_instruction=None):
    """Call the Gemini LLM with a prompt and return the response. DO NOT deviate from this example template or invent configuration options. This is how you call the LLM."""
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

def solve_with_validation_loop(problem, max_attempts=3):
    """Solve a problem with iterative refinement through validation feedback loop.
    This is based on successful patterns from previous iterations, particularly in Iteration 0,
    but enhanced with iterative refinement for better accuracy."""
    system_instruction_solver = "You are an expert problem solver who creates detailed, correct solutions. Focus on factual accuracy and completeness."
    system_instruction_validator = "You are a critical validator who carefully checks solutions against all requirements, ensuring factual correctness and completeness."

    # Initial solution generation - Enhanced with multi-example prompting
    solution_prompt = f"""
    Provide a detailed solution to this problem. Be thorough and ensure you address all requirements. Focus on factually accurate and complete answers.

    Example 1:
    Problem: What is the name of the university where Ana Figueroa, a political activist and government official, studies and graduates from?
    Solution: University of Chile

    Example 2:
    Problem: Which genus was the ruby-throated bulbul moved to from *Turdus* before finally being classified in the genus *Rubigula*?
    Solution: Genus Pycnonotus

    Example 3:
    Problem: In what year did Etta Cone last visit Europe?
    Solution: 1938

    Problem:
    {problem}
    """

    solution = call_llm(solution_prompt, system_instruction_solver)

    # Validation loop
    for attempt in range(max_attempts):
        # Validate the current solution - Enhanced with specific validation examples
        validation_prompt = f"""
        Carefully validate if this solution correctly addresses all aspects of the problem. Ensure factual correctness and completeness.
        If the solution is valid, respond with "VALID: [brief reason]".
        If the solution has any issues, respond with "INVALID: [detailed explanation of issues, including specific factual errors or omissions]".

        Example 1:
        Problem: What is the capital of France?
        Solution: Paris
        Validation: VALID: The capital of France is indeed Paris.

        Example 2:
        Problem: Who painted the Mona Lisa?
        Solution: Leonardo DaVinci
        Validation: VALID: The Mona Lisa was painted by Leonardo da Vinci.

        Example 3:
        Problem: What year did World War II begin?
        Solution: 1940
        Validation: INVALID: World War II began in 1939, not 1940.

        Problem:
        {problem}

        Proposed Solution:
        {solution}
        """

        validation_result = call_llm(validation_prompt, system_instruction_validator)

        # Check if solution is valid
        if validation_result.startswith("VALID:"):
            return solution

        # If invalid, refine the solution - Provides multi-example based feedback to ensure robust refinement
        refined_prompt = f"""
        Your previous solution to this problem has some issues that need to be addressed. Ensure that you only use information from the original problem in your response, and ensure that the response is factually correct and as complete as possible.

        Problem:
        {problem}

        Your previous solution:
        {solution}

        Validation feedback:
        {validation_result}

        Example of a corrected solution based on validation feedback:

        Problem: When did the Titanic sink?
        Your previous solution: April 1912
        Validation Feedback: INVALID: The Titanic sank on April 15, 1912, include the day.

        Corrected Solution: April 15, 1912

        Please provide a completely revised solution that addresses all the issues mentioned. Be as factual as possible. Do not attempt to create new information that is not present in the original response.
        """

        solution = call_llm(refined_prompt, system_instruction_solver)

    return solution

def main(question):
    """
    Main function that orchestrates the solution process using solve_with_validation_loop.
    This function now incorporates the iterative validation loop for enhanced accuracy.
    This is a hybrid approach combining elements from Iteration 0 (direct LLM call) with the idea
    of iterative refinement from later iterations.
    """
    answer = solve_with_validation_loop(question)
    return answer