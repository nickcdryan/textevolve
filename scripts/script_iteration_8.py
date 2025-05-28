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
    This script implements a hybrid approach combining "Decompose-Solve-Verify" (from iteration 3)
    with "Reflexion-Enhanced Solution Synthesis" (from iteration 6) for robust problem-solving.
    It also incorporates Knowledge Retrieval (from iteration 4) for enhanced information gathering.

    The hypothesis is that combining decomposition, reflection, and knowledge retrieval will lead to more accurate and reliable solutions.
    """

    # === Step 1: Knowledge Retrieval (from iteration 4) ===
    def retrieve_knowledge(question):
        """Retrieves relevant knowledge based on the question."""
        system_instruction = "You are a knowledge retrieval expert. Identify key concepts and retrieve relevant information to solve the given problem."
        prompt = f"""
        Identify the mathematical concepts required to solve the following problem and retrieve relevant formulas or theorems.

        Example 1:
        Problem: What is the area of a circle with a radius of 5?
        Concepts: Area of a circle
        Retrieved Information: The area of a circle is given by the formula A = πr^2, where r is the radius.

        Example 2:
        Problem: Solve for x: 2x + 3 = 7
        Concepts: Solving linear equations
        Retrieved Information: To solve a linear equation, isolate the variable by performing inverse operations on both sides.

        Example 3:
        Problem: What are the prime factors of 36?
        Concepts: Prime factorization
        Retrieved Information: Prime factors are the prime numbers that divide evenly into a given number. To find them, repeatedly divide by prime numbers until you reach 1.

        Problem: {question}
        Concepts and Retrieved Information:
        """
        try:
            knowledge = call_llm(prompt, system_instruction)
            print(f"Retrieved Knowledge: {knowledge}")
            return knowledge
        except Exception as e:
            print(f"Error retrieving knowledge: {e}")
            return "Error: Could not retrieve knowledge."

    # === Step 2: Decompose the problem into smaller, manageable steps (from iteration 3) ===
    def decompose_problem(question, knowledge):
        """Breaks down the problem into smaller steps, incorporating retrieved knowledge."""
        system_instruction = "You are an expert at decomposing complex math problems into smaller, solvable steps. Use the provided knowledge to guide your decomposition."
        prompt = f"""
        Decompose the following math problem into smaller, manageable steps. Consider the retrieved knowledge to inform your decomposition.

        Example 1:
        Problem: What is the area of a square with side length 10, and what is the area if the side length is increased by 50%?
        Retrieved Knowledge: Area of a square = side * side; Percentage increase = (new value - old value) / old value * 100
        Decomposition:
        1. Calculate the area of the square with side length 10 using the formula: side * side.
        2. Calculate the new side length after increasing it by 50%.
        3. Calculate the area of the square with the new side length using the formula: side * side.

        Example 2:
        Problem: A train travels at 60 mph for 2.5 hours. How far does it go and how much time is spent going the first half of the distance if the train travels at a constant velocity?
        Retrieved Knowledge: Distance = speed * time
        Decomposition:
        1. Calculate the total distance traveled using the formula: speed * time.
        2. Divide the total distance by 2 to find the first half distance.
        3. Calculate the time spent for the first half using the formula: time = distance / speed.

        Example 3:
        Problem: Solve for x: 2x + 3 = 7
        Retrieved Knowledge: To solve for x, isolate the variable on one side of the equation.
        Decomposition:
        1. Subtract 3 from both sides of the equation.
        2. Divide both sides of the equation by 2.

        Problem: {question}
        Retrieved Knowledge: {knowledge}
        Decomposition:
        """
        return call_llm(prompt, system_instruction)

    # === Step 3: Generate an Initial Solution (from iteration 6, modified) ===
    def generate_initial_solution(question, decomposition):
        """Generates an initial solution to the problem based on the decomposition."""
        system_instruction = "You are an expert problem solver. Generate a detailed initial solution based on the provided decomposition."
        prompt = f"""
        Provide a detailed initial solution to the following problem, following the steps outlined in the decomposition.

        Example:
        Problem: What is the area of a square with side length 5?
        Decomposition: 1. Calculate the area of the square using the formula: side * side.
        Solution: The area of a square is side * side. So, the area is 5 * 5 = 25.
        Answer: 25

        Example:
        Problem: Solve for x: 2x + 3 = 7
        Decomposition: 1. Subtract 3 from both sides of the equation. 2. Divide both sides of the equation by 2.
        Solution: Subtracting 3 from both sides gives 2x = 4. Dividing both sides by 2 gives x = 2.
        Answer: x = 2

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

    # === Step 4: Reflect and Critique (from iteration 6) ===
    def reflect_and_critique(question, solution):
        """Reflects on the initial solution and identifies potential issues."""
        system_instruction = "You are a critical self-evaluator. Analyze the solution and identify any potential errors, inconsistencies, or areas for improvement. Focus especially on arithmetic errors."
        prompt = f"""
        Analyze the following solution to the problem and identify any potential errors or inconsistencies. Be extremely thorough, especially looking for arithmetic mistakes.

        Example:
        Problem: What is the area of a circle with radius 5?
        Solution: The area of a circle is radius * radius. So, the area is 5 * 5 = 25.
        Critique: The solution incorrectly states the formula for the area of a circle. It should be pi * radius^2. The arithmetic is correct, but the underlying formula is wrong.

        Example:
        Problem: Solve for x: 2x + 3 = 7
        Solution: Subtract 3 from both sides gives 2x = 10. Dividing both sides by 2 gives x = 5.
        Critique: There is an arithmetic error. Subtracting 3 from 7 should result in 4, not 10. The solution should be x=2.

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

    # === Step 5: Synthesize Refined Solution (from iteration 6, modified) ===
    def synthesize_refined_solution(question, initial_solution, critique):
        """Synthesizes a refined solution based on the initial solution and the critique."""
        system_instruction = "You are an expert problem solver. Use the critique to generate a refined solution, correcting any identified errors."
        prompt = f"""
        Based on the critique, generate a refined solution to the problem. Incorporate all feedback to ensure correctness.

        Example:
        Problem: What is the area of a circle with radius 5?
        Initial Solution: The area of a circle is radius * radius. So, the area is 5 * 5 = 25.
        Critique: The solution incorrectly states the formula for the area of a circle. It should be pi * radius^2.
        Refined Solution: The area of a circle is pi * radius^2. So, the area is pi * 5^2 = 25pi.
        Answer: 25pi

        Example:
        Problem: Solve for x: 2x + 3 = 7
        Initial Solution: Subtract 3 from both sides gives 2x = 10. Dividing both sides by 2 gives x = 5.
        Critique: There is an arithmetic error. Subtracting 3 from 7 should result in 4, not 10.
        Refined Solution: Subtracting 3 from both sides gives 2x = 4. Dividing both sides by 2 gives x = 2.
        Answer: x = 2

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

    # === Step 6: Verify Refined Solution (from iteration 3, modified) ===
    def verify_refined_solution(question, refined_solution):
        """Verifies if the refined solution correctly addresses the problem. Focus on arithmetic and logical consistency."""
        system_instruction = "You are a solution validator. Check the solution for correctness and completeness. Double-check all calculations and logical steps."
        prompt = f"""
        Validate the refined solution for correctness and completeness. Be extremely thorough, focusing on both arithmetic and logical consistency. Provide a detailed assessment.

        Example 1:
        Question: What is 2 + 2?
        Solution: 4
        Verdict: Correct. The solution is correct and complete.

        Example 2:
        Question: What is the capital of France?
        Solution: London
        Verdict: Incorrect. The solution is incorrect. The capital of France is Paris.

        Example 3:
        Question: What is the area of a square with side length 5?
        Solution: The area of a square is side * side. So, the area is 5 * 5 = 25.
        Verdict: Correct.

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

    # Orchestrate the process
    try:
        knowledge = retrieve_knowledge(question)
        decomposition = decompose_problem(question, knowledge)
        initial_solution = generate_initial_solution(question, decomposition)
        critique = reflect_and_critique(question, initial_solution)
        refined_solution = synthesize_refined_solution(question, initial_solution, critique)
        validation_result = verify_refined_solution(question, refined_solution)

        return f"Knowledge: {knowledge}\nDecomposition: {decomposition}\nInitial Solution: {initial_solution}\nCritique: {critique}\nRefined Solution: {refined_solution}\nValidation: {validation_result}"

    except Exception as e:
        return f"Overall Error: {str(e)}"