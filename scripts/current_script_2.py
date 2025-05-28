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
    """Main function to answer math questions using a two-agent approach:
    1. Problem Analyzer: Understands and formats the problem.
    2. Solution Generator: Generates a solution based on the formatted problem.
    This tests the hypothesis that specialized agents improve performance.
    """

    # === Agent 1: Problem Analyzer ===
    def analyze_problem(question):
        """Analyzes the problem and extracts key information."""
        system_instruction = "You are a problem analyzer. Your task is to understand the question and format it for a solution generator."
        prompt = f"""
        Analyze the math question and extract relevant information in a structured format.

        Example 1:
        Question: What is the area of a circle with a radius of 5?
        Analysis:
        {{
          "problem_type": "geometry",
          "topic": "circle",
          "known": {{"radius": 5}},
          "unknown": "area",
          "formula": "area = pi * radius^2"
        }}

        Example 2:
        Question: Solve for x: 2x + 3 = 7
        Analysis:
        {{
          "problem_type": "algebra",
          "topic": "equation solving",
          "known": {{"equation": "2x + 3 = 7"}},
          "unknown": "x",
          "steps": ["subtract 3 from both sides", "divide both sides by 2"]
        }}

        Question: {question}
        Analysis:
        """
        try:
            analysis = call_llm(prompt, system_instruction)
            # Print statement to understand the analysis
            print(f"Problem Analysis: {analysis}")
            return analysis
        except Exception as e:
            print(f"Error analyzing problem: {e}")
            return "Error: Could not analyze the problem."

    # === Agent 2: Solution Generator ===
    def generate_solution(analysis):
        """Generates a solution based on the analyzed problem."""
        system_instruction = "You are a solution generator. Use the problem analysis to generate a step-by-step solution."
        prompt = f"""
        Generate a step-by-step solution based on the problem analysis.

        Example 1:
        Analysis:
        {{
          "problem_type": "geometry",
          "topic": "circle",
          "known": {{"radius": 5}},
          "unknown": "area",
          "formula": "area = pi * radius^2"
        }}
        Solution:
        1. Identify the formula: area = pi * radius^2
        2. Substitute the radius: area = pi * 5^2
        3. Calculate: area = 25 * pi
        Answer: 25 * pi

        Example 2:
        Analysis:
        {{
          "problem_type": "algebra",
          "topic": "equation solving",
          "known": {{"equation": "2x + 3 = 7"}},
          "unknown": "x",
          "steps": ["subtract 3 from both sides", "divide both sides by 2"]
        }}
        Solution:
        1. Subtract 3 from both sides: 2x = 4
        2. Divide both sides by 2: x = 2
        Answer: 2

        Analysis: {analysis}
        Solution:
        """
        try:
            solution = call_llm(prompt, system_instruction)
            # Print statement to understand the solution
            print(f"Generated Solution: {solution}")
            return solution
        except Exception as e:
            print(f"Error generating solution: {e}")
            return "Error: Could not generate the solution."

    # === Validation Step ===
    def validate_solution(question, solution):
        """Validates the generated solution against the original question."""
        system_instruction = "You are a solution validator. Check the solution for correctness, completeness, and relevance."
        prompt = f"""
        Validate the generated solution against the original question. Provide a short verdict.

        Example 1:
        Question: What is 2 + 2?
        Solution: 4
        Verdict: Correct.

        Example 2:
        Question: What is the capital of France?
        Solution: London
        Verdict: Incorrect.

        Question: {question}
        Solution: {solution}
        Verdict:
        """
        try:
            validation = call_llm(prompt, system_instruction)
            print(f"Validation: {validation}")  # Print validation result
            return validation
        except Exception as e:
            print(f"Error validating solution: {e}")
            return "Error: Could not validate the solution."
    # Call the problem analyzer
    analysis = analyze_problem(question)
    # Call the solution generator with the analysis
    solution = generate_solution(analysis)
    # Validate the solution
    validation_result = validate_solution(question, solution)

    return f"Analysis: {analysis}\nSolution: {solution}\nValidation: {validation_result}"

# Example usage
if __name__ == "__main__":
    question = "Let $n$ be a natural number with exactly 2 positive prime divisors.  If $n^2$ has 27 divisors, how many does $n$ have?"
    answer = main(question)
    print(f"Question: {question}")
    print(f"Answer: {answer}")