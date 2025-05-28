import os
from google import genai
from google.genai import types

def call_llm(prompt, system_instruction=None):
    """Call the Gemini LLM with a prompt and return the response. DO NOT deviate from this example template or invent configuration options. This is how you call the LLM."""
    try:
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

def main(question):
    """
    This script implements a "Reflexion-Enhanced Solution Synthesis" approach.
    Hypothesis: Adding a reflection step, where the LLM critiques its OWN initial solution, will lead to improved accuracy.
    This directly addresses the need for better precision by explicitly focusing on self-critique.
    """

    # === Step 1: Initial Solution Generation ===
    def generate_initial_solution(question):
        """Generates an initial solution to the problem."""
        system_instruction = "You are an expert problem solver. Generate a detailed initial solution."
        prompt = f"""
        Provide a detailed initial solution to the following problem:

        Example:
        Problem: What is the area of a square with side length 5?
        Solution: The area of a square is side * side. So, the area is 5 * 5 = 25.
        Answer: 25

        Problem: {question}
        Solution:
        """
        try:
            solution = call_llm(prompt, system_instruction)
            print(f"Initial Solution: {solution}")
            return solution
        except Exception as e:
            print(f"Error generating initial solution: {e}")
            return "Error: Could not generate initial solution."

    # === Step 2: Reflexion and Critique ===
    def reflect_and_critique(question, solution):
        """Reflects on the initial solution and identifies potential issues."""
        system_instruction = "You are a critical self-evaluator. Analyze the solution and identify any potential errors, inconsistencies, or areas for improvement."
        prompt = f"""
        Analyze the following solution to the problem and identify any potential errors or inconsistencies.

        Example:
        Problem: What is the area of a circle with radius 5?
        Solution: The area of a circle is radius * radius. So, the area is 5 * 5 = 25.
        Critique: The solution incorrectly states the formula for the area of a circle. It should be pi * radius^2.

        Problem: {question}
        Solution: {solution}
        Critique:
        """
        try:
            critique = call_llm(prompt, system_instruction)
            print(f"Critique: {critique}")
            return critique
        except Exception as e:
            print(f"Error generating critique: {e}")
            return "Error: Could not generate critique."

    # === Step 3: Refined Solution Synthesis ===
    def synthesize_refined_solution(question, initial_solution, critique):
        """Synthesizes a refined solution based on the initial solution and the critique."""
        system_instruction = "You are an expert problem solver. Use the critique to generate a refined solution."
        prompt = f"""
        Based on the critique, generate a refined solution to the problem.

        Example:
        Problem: What is the area of a circle with radius 5?
        Initial Solution: The area of a circle is radius * radius. So, the area is 5 * 5 = 25.
        Critique: The solution incorrectly states the formula for the area of a circle. It should be pi * radius^2.
        Refined Solution: The area of a circle is pi * radius^2. So, the area is pi * 5^2 = 25pi.
        Answer: 25pi

        Problem: {question}
        Initial Solution: {initial_solution}
        Critique: {critique}
        Refined Solution:
        """
        try:
            refined_solution = call_llm(prompt, system_instruction)
            print(f"Refined Solution: {refined_solution}")
            return refined_solution
        except Exception as e:
            print(f"Error generating refined solution: {e}")
            return "Error: Could not generate refined solution."
    
    # === Step 4: Verification of the Refined Solution ===
    def verify_refined_solution(question, refined_solution):
        """Verifies if the refined solution correctly addresses the problem."""
        system_instruction = "You are a solution validator. Check the solution for correctness and completeness."
        prompt = f"""
        Validate the refined solution for correctness and completeness.

        Example 1:
        Question: What is 2 + 2?
        Solution: 4
        Verdict: Correct.

        Example 2:
        Question: What is the capital of France?
        Solution: London
        Verdict: Incorrect.

        Question: {question}
        Solution: {refined_solution}
        Verdict:
        """
        try:
            validation = call_llm(prompt, system_instruction)
            print(f"Validation: {validation}")
            return validation
        except Exception as e:
            print(f"Error validating solution: {e}")
            return "Error: Could not validate the solution."

    # Call the functions in sequence
    initial_solution = generate_initial_solution(question)
    critique = reflect_and_critique(question, initial_solution)
    refined_solution = synthesize_refined_solution(question, initial_solution, critique)
    validation_result = verify_refined_solution(question, refined_solution)

    return f"Initial Solution: {initial_solution}\nCritique: {critique}\nRefined Solution: {refined_solution}\nValidation: {validation_result}"