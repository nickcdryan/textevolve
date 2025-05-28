import os

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

def main(question):
    """
    This script combines the "Decompose-Solve-Verify" approach (Iteration 3) and
    "Reflexion-Enhanced Solution Synthesis" (Iteration 6) to create a robust hybrid.
    It first decomposes the problem, then generates an initial solution, reflects on it,
    synthesizes a refined solution, and finally verifies the refined solution for coherency.
    This addresses the need for both detailed solutions and strong validation.
    The goal is to create a more robust solution than any individual approach by layering different strengths.
    """

    # === Step 1: Decompose the problem (from Iteration 3) ===
    def decompose_problem(question):
        """Breaks down the problem into smaller steps. (From Iteration 3)"""
        system_instruction = "You are an expert at decomposing complex math problems into smaller, solvable steps."
        prompt = f"""
        Decompose the following math problem into smaller, manageable steps.

        Example 1:
        Problem: What is the area of a square with side length 10, and what is the area if the side length is increased by 50%?
        Decomposition:
        1. Calculate the area of the square with side length 10.
        2. Calculate the new side length after increasing it by 50%.
        3. Calculate the area of the square with the new side length.

        Example 2:
        Problem: A train travels at 60 mph for 2.5 hours. How far does it go and how much time is spent going the first half of the distance if the train travels at a constant velocity?
        Decomposition:
        1. Calculate the total distance traveled.
        2. Divide the total distance by 2.
        3. Calculate the time spent for the first half.

        Example 3:
        Problem: Find the value of x in the equation 2x + 5 = 11.
        Decomposition:
        1. Subtract 5 from both sides of the equation.
        2. Divide both sides of the equation by 2.
        3. State the value of x.

        Problem: {question}
        Decomposition:
        """
        return call_llm(prompt, system_instruction)

    # === Step 2: Generate Initial Solution (Adapted from Iteration 6) ===
    def generate_initial_solution(question, decomposition):
        """Generates an initial solution to the problem based on decomposition."""
        system_instruction = "You are an expert problem solver. Generate a detailed initial solution based on these steps."
        prompt = f"""
        Provide a detailed initial solution to the following problem, following the steps outlined in the decomposition:

        Example:
        Problem: What is the area of a square with side length 5?
        Decomposition:
        1. Calculate the area of the square.
        Solution: The area of a square is side * side. So, the area is 5 * 5 = 25.
        Answer: 25

        Example 2:
        Problem: Solve for x: 2x + 3 = 7
        Decomposition:
        1. Subtract 3 from both sides of the equation.
        2. Divide both sides of the equation by 2.
        Solution:
        1. Subtract 3 from both sides: 2x = 4.
        2. Divide both sides by 2: x = 2.
        Answer: 2

        Problem: {question}
        Decomposition: {decomposition}
        Solution:
        """
        try:
            solution = call_llm(prompt, system_instruction)
            print(f"Initial Solution: {solution}")
            return solution
        except Exception as e:
            print(f"Error generating initial solution: {e}")
            return "Error: Could not generate initial solution."

    # === Step 3: Reflexion and Critique (From Iteration 6) ===
    def reflect_and_critique(question, solution):
        """Reflects on the initial solution and identifies potential issues. (From Iteration 6)"""
        system_instruction = "You are a critical self-evaluator. Analyze the solution and identify any potential errors, inconsistencies, or areas for improvement, focusing on arithmetic accuracy and logical steps."
        prompt = f"""
        Analyze the following solution to the problem and identify any potential errors, inconsistencies, or areas for improvement. Pay special attention to arithmetic and logical correctness.

        Example:
        Problem: What is the area of a circle with radius 5?
        Solution: The area of a circle is radius * radius. So, the area is 5 * 5 = 25.
        Critique: The solution incorrectly states the formula for the area of a circle. It should be pi * radius^2. There appears to be an arithmetic error: it doesn't include pi in the final answer.

        Example 2:
        Problem: Solve for x: 2x + 3 = 7
        Solution: 1. Subtract 3 from both sides: 2x = 4. 2. Divide both sides by 2: x = 3. Answer: x = 3
        Critique: There's an arithmetic error. 4/2 is 2, not 3.

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

    # === Step 4: Refined Solution Synthesis (Adapted from Iteration 6) ===
    def synthesize_refined_solution(question, initial_solution, critique):
        """Synthesizes a refined solution based on the initial solution and the critique. (Adapted from Iteration 6)"""
        system_instruction = "You are an expert problem solver. Use the critique to generate a refined solution, ensuring arithmetic accuracy and clear, logical steps."
        prompt = f"""
        Based on the critique, generate a refined solution to the problem. Ensure all steps are arithmetically sound and logically clear.

        Example:
        Problem: What is the area of a circle with radius 5?
        Initial Solution: The area of a circle is radius * radius. So, the area is 5 * 5 = 25.
        Critique: The solution incorrectly states the formula for the area of a circle. It should be pi * radius^2. There appears to be an arithmetic error: it doesn't include pi in the final answer.
        Refined Solution: The area of a circle is pi * radius^2. So, the area is pi * 5^2 = 25pi.
        Answer: 25pi

        Example 2:
        Problem: Solve for x: 2x + 3 = 7
        Initial Solution: 1. Subtract 3 from both sides: 2x = 4. 2. Divide both sides by 2: x = 3. Answer: x = 3
        Critique: There's an arithmetic error. 4/2 is 2, not 3.
        Refined Solution: 1. Subtract 3 from both sides: 2x = 4. 2. Divide both sides by 2: x = 2. Answer: 2

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
    
    # === Step 5: Verification of the Refined Solution with Coherency Check (Combining Iteration 6 & 3) ===
    def verify_refined_solution(question, refined_solution):
        """Verifies if the refined solution correctly addresses the problem and is coherent with the question."""
        system_instruction = "You are a solution validator. Check the solution for correctness, completeness, and coherency with the original question. Focus on finding any logical fallacies or arithmetic errors."
        prompt = f"""
        Validate the refined solution for correctness, completeness, and coherency. Check for logical fallacies and arithmetic errors.

        Example 1:
        Question: What is 2 + 2?
        Solution: 4
        Verdict: Correct and Coherent.

        Example 2:
        Question: What is the capital of France?
        Solution: London
        Verdict: Incorrect and Incoherent.

        Example 3:
        Question: What is the area of a square with side length 5?
        Solution: The area of a square is side * side = 5*5=25 therefore the perimeter is 4*5 = 20.
        Verdict: Incorrect. Solved for both area and perimeter when only area was required. Not Coherent.

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
    decomposition = decompose_problem(question)
    initial_solution = generate_initial_solution(question, decomposition)
    critique = reflect_and_critique(question, initial_solution)
    refined_solution = synthesize_refined_solution(question, initial_solution, critique)
    validation_result = verify_refined_solution(question, refined_solution)

    return f"Decomposition: {decomposition}\nInitial Solution: {initial_solution}\nCritique: {critique}\nRefined Solution: {refined_solution}\nValidation: {validation_result}"