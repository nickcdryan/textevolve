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
    This script employs a "Knowledge Retrieval and Solution Synthesis" approach.
    Hypothesis: Combining targeted knowledge retrieval with a final synthesis agent can improve problem-solving accuracy. We will also add a verification call to validate the "Targeted Knowledge Retrieval" stage to understand its success in the pipeline.
    """

    # === Step 1: Targeted Knowledge Retrieval ===
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

    def verify_knowledge_retrieval(question, knowledge):
        """Verifies if the retrieved knowledge is relevant and complete."""
        system_instruction = "You are an expert at verifying the relevance of the knowledge."
        prompt = f"""
        Verify if the retrieved knowledge is sufficient and necessary to solve the given problem.

        Example 1:
        Problem: What is the area of a circle with a radius of 5?
        Retrieved Knowledge: The area of a circle is given by the formula A = πr^2, where r is the radius.
        Verification: The retrieved knowledge is both necessary and sufficient to solve the problem.

        Example 2:
        Problem: Solve for x: 2x + 3 = 7
        Retrieved Knowledge: To solve a linear equation, isolate the variable by performing inverse operations on both sides.
        Verification: The retrieved knowledge is necessary and sufficient to solve the problem.

        Problem: {question}
        Retrieved Knowledge: {knowledge}
        Verification:
        """

        try:
            verification = call_llm(prompt, system_instruction)
            print(f"Knowledge Verification: {verification}")  # Print validation result
            return verification
        except Exception as e:
            print(f"Error validating solution: {e}")
            return "Error: Could not validate the retrieved knowledge."

    # === Step 2: Solution Synthesis ===
    def synthesize_solution(question, knowledge):
        """Synthesizes a solution based on the retrieved knowledge."""
        system_instruction = "You are a solution synthesis expert. Use the retrieved knowledge to generate a step-by-step solution to the problem."
        prompt = f"""
        Synthesize a step-by-step solution to the problem using the retrieved knowledge.

        Example 1:
        Problem: What is the area of a circle with a radius of 5?
        Retrieved Knowledge: The area of a circle is given by the formula A = πr^2, where r is the radius.
        Solution:
        1. Identify the formula: A = πr^2
        2. Substitute the radius: A = π(5)^2
        3. Calculate the area: A = 25π
        Answer: The area of the circle is 25π.

        Example 2:
        Problem: Solve for x: 2x + 3 = 7
        Retrieved Knowledge: To solve a linear equation, isolate the variable by performing inverse operations on both sides.
        Solution:
        1. Subtract 3 from both sides: 2x = 4
        2. Divide both sides by 2: x = 2
        Answer: x = 2

        Problem: {question}
        Retrieved Knowledge: {knowledge}
        Solution:
        """
        try:
            solution = call_llm(prompt, system_instruction)
            print(f"Synthesized Solution: {solution}")
            return solution
        except Exception as e:
            print(f"Error synthesizing solution: {e}")
            return "Error: Could not synthesize the solution."

    # Call the knowledge retrieval function
    knowledge = retrieve_knowledge(question)
    knowledge_verification = verify_knowledge_retrieval(question, knowledge)

    # Call the solution synthesis function
    solution = synthesize_solution(question, knowledge)

    return f"Knowledge: {knowledge}\nKnowledge Verification: {knowledge_verification}\nSolution: {solution}"