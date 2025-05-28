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
    This script implements a 'Decompose-Solve-Verify' approach with multi-example prompting for each step.
    Hypothesis: Explicit examples in prompts improve accuracy and robustness by guiding the LLM. The solution is checked with another prompt to make sure the response is coherent.
    """

    # Step 1: Decompose the problem into smaller, manageable steps
    def decompose_problem(question):
        """Breaks down the problem into smaller steps."""
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

        Problem: {question}
        Decomposition:
        """
        return call_llm(prompt, system_instruction)

    # Step 2: Solve each sub-problem independently
    def solve_sub_problems(decomposition):
        """Solves each sub-problem from the decomposition."""
        system_instruction = "You are an expert at solving math sub-problems."
        prompt = f"""
        Solve the following sub-problems.

        Example:
        Sub-problems:
        1. Calculate the area of the square with side length 10.
        2. Calculate the new side length after increasing it by 50%.
        3. Calculate the area of the square with the new side length.
        Solutions:
        1. 100
        2. 15
        3. 225

         Sub-problems: {decomposition}
        Solutions:
        """
        return call_llm(prompt, system_instruction)

    # Step 3: Synthesize the solutions into a final answer
    def synthesize_solutions(question, sub_problems, solutions):
        """Synthesizes the solutions to the sub-problems into a final answer."""
        system_instruction = "You are an expert at synthesizing solutions to math problems."
        prompt = f"""
        Synthesize the following solutions into a final answer for the given question.

        Example:
        Question: What is the area of a square with side length 10, and what is the area if the side length is increased by 50%?
        Sub-problems:
        1. Calculate the area of the square with side length 10.
        2. Calculate the new side length after increasing it by 50%.
        3. Calculate the area of the square with the new side length.
        Solutions:
        1. 100
        2. 15
        3. 225
        Final Answer: The area of the square with side length 10 is 100. If the side length is increased by 50%, the new area is 225.

        Question: {question}
        Sub-problems: {sub_problems}
        Solutions: {solutions}
        Final Answer:
        """
        return call_llm(prompt, system_instruction)

    #Step 4: Check for response coherency
    def check_coherency(question, solution):
        """Verifies if the solution is coherent."""
        system_instruction = "You are an expert solution coherency verifier."
        prompt = f"""
        Is this response coherent with the question?

        Example 1:
        Question: What is the capital of France?
        Solution: The capital of France is Paris.
        Coherent: True

        Example 2:
        Question: What is the capital of France?
        Solution: I like apples.
        Coherent: False

        Question: {question}
        Solution: {solution}
        Coherent:
        """
        return call_llm(prompt, system_instruction)
    try:
        # Call the decomposition function
        decomposition = decompose_problem(question)
        print(f"Decomposition: {decomposition}")

        # Call the solve sub-problems function
        solutions = solve_sub_problems(decomposition)
        print(f"Solutions: {solutions}")

        # Call the synthesize solutions function
        final_answer = synthesize_solutions(question, decomposition, solutions)
        print(f"Final Answer: {final_answer}")

        #Call the coherency checker
        is_coherent = check_coherency(question, final_answer)
        print(f"Coherency: {is_coherent}")
        if "True" in is_coherent:
            return final_answer
        else:
            return f"Response not coherent. Answer: {final_answer}"
    except Exception as e:
        print(f"Error: {e}")
        return f"Error: {e}"