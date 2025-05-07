import google.generativeai as genai
import os

# Set up the Gemini API key
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")  # Retrieve from environment variable
genai.configure(api_key=GOOGLE_API_KEY)

# Select the Gemini model
model = genai.GenerativeModel('gemini-pro')

def call_llm(prompt, system_instruction=None):
    """Calls the LLM with the given prompt and optional system instruction."""
    try:
        if system_instruction:
            response = model.generate_content([system_instruction, prompt])
        else:
            response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"Error calling LLM: {e}")
        return None

def main(question):
    """Main function to answer questions using LLM-driven agentic approach."""

    # Step 1: Analyze the question type
    analysis_prompt = f"""
    Analyze this question and determine what type of reasoning is required. Give the reasoning type in one word.
    Example:
    Question: What is the capital of France?
    Answer: Factual

    Question: {question}
    Answer: """
    reasoning_type = call_llm(analysis_prompt)

    if reasoning_type is None:
        return "Could not determine the type of reasoning required."

    # Step 2: Generate a step-by-step reasoning plan
    plan_prompt = f"""
    Based on the question and identified reasoning type, create a step-by-step reasoning plan to answer the question correctly.
    Example:
    Question: What is 5 + 3 * 2? Reasoning Type: Numerical
    Plan:
    1. Multiply 3 by 2.
    2. Add 5 to the result.
    3. State the final answer.

    Question: {question} Reasoning Type: {reasoning_type}
    Plan: """

    reasoning_plan = call_llm(plan_prompt)

    if reasoning_plan is None:
        return "Could not generate a reasoning plan."

    # Step 3: Execute the reasoning plan using the LLM

    execution_prompt = f"""
    Execute the reasoning plan step-by-step and provide the final answer.
    Example:
    Question: What is the capital of France? Reasoning Type: Factual.
    Plan:
    1. Recall that Paris is the capital of France.
    2. State the final answer.
    Answer: Paris is the capital of France.

    Question: {question} Reasoning Type: {reasoning_type}
    Plan: {reasoning_plan}
    Answer: """

    final_answer = call_llm(execution_prompt)

    if final_answer is None:
        return "Could not execute the reasoning plan and generate the final answer."

    return final_answer

# Example usage
if __name__ == "__main__":
    question = "If a train leaves Chicago at 7am traveling 60mph, and another leaves New York at 8am traveling 80mph, when will they meet?"
    answer = main(question)
    print(f"Question: {question}")
    print(f"Answer: {answer}")