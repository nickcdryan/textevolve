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

def decompose_problem_with_llm(problem):
    """Break down a problem into key components using LLM reasoning"""
    system_instruction = "You are a problem analyzer. Break down scheduling problems into key components."
    prompt = f"Analyze this scheduling problem and extract key parts: {problem}"
    return call_llm(prompt, system_instruction)

def extract_constraints_with_llm(problem):
    """Extract scheduling constraints from the problem statement"""
    system_instruction = "You are an information extractor. Find all scheduling constraints."
    prompt = f"Identify scheduling constraints in this text: {problem}"
    response = call_llm(prompt, system_instruction)
    return response

def suggest_solution_with_llm(constraints):
    """Suggest a solution given the constraints"""
    system_instruction = "You are a scheduling expert. Suggest a meeting time based on constraints."
    prompt = f"Suggest a meeting time that satisfies these constraints: {constraints}"
    return call_llm(prompt, system_instruction)

def verify_solution_with_llm(problem, proposed_solution):
    """Verify if the proposed solution meets all constraints"""
    system_instruction = "You are a solution checker. Verify the solution satisfies all constraints."
    prompt = f"Verify this solution against constraints: Solution: {proposed_solution}, Problem: {problem}"
    return call_llm(prompt, system_instruction)

def main(question):
    """Main function to schedule a meeting"""
    try:
        decomposition = decompose_problem_with_llm(question)
        constraints = extract_constraints_with_llm(question)
        proposed_solution = suggest_solution_with_llm(constraints)
        verification = verify_solution_with_llm(question, proposed_solution)

        return f"Proposed solution: {proposed_solution}. Verification: {verification}"
    except Exception as e:
        return f"Error processing the request: {str(e)}"

# Example usage (replace with your actual question)
if __name__ == "__main__":
    example_question = "You need to schedule a meeting for Kathryn, Charlotte and Lauren for half an hour between 9:00 and 17:00 on Monday. Kathryn has blocked their calendar on Monday during 9:00 to 9:30, 10:30 to 11:00, 11:30 to 12:00, 13:30 to 14:30, 16:30 to 17:00; Charlotte has blocked their calendar on Monday during 12:00 to 12:30, 16:00 to 16:30; Lauren has blocked their calendar on Monday during 9:00 to 10:00, 12:00 to 12:30, 13:30 to 14:30, 15:00 to 16:00, 16:30 to 17:00; Charlotte do not want to meet on Monday after 13:30."
    answer = main(example_question)
    print(answer)