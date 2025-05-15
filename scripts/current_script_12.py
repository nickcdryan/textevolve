import google.generativeai as genai
import os

# Set up the Gemini API (replace with your actual API key)
genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
model = genai.GenerativeModel('gemini-pro')


def call_llm(prompt, system_instruction=None):
    """
    Calls the LLM with the given prompt and optional system instruction.
    Includes basic error handling.
    """
    try:
        if system_instruction:
            response = model.generate_content(
                contents=prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.2),
                safety_settings={
                    genai.types.HarmCategory.HARM_CATEGORY_HARASSMENT: genai.types.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                    genai.types.HarmCategory.HARM_CATEGORY_HATE_SPEECH: genai.types.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                    genai.types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: genai.types.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                    genai.types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: genai.types.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                },
            )
        else:
            response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"Error calling LLM: {e}")
        return "Error: Could not generate response."


def main(question):
    """
    Main function that takes a question string as input and returns the answer string.
    Uses multiple LLM calls for different reasoning steps.
    """

    # Step 1: Analyze the question type using LLM
    analysis_prompt = f"""
    Analyze the following question and determine its type (e.g., math, reasoning, general knowledge).
    Example:
    Question: What is the capital of France?
    Type: General Knowledge

    Question: {question}
    Type:
    """
    question_type = call_llm(analysis_prompt)

    # Step 2: Generate a plan to answer the question using LLM
    plan_prompt = f"""
    Given that the question type is {question_type}, generate a step-by-step plan to answer the question.
    Example:
    Question Type: General Knowledge
    Question: What is the capital of France?
    Plan: 1. Search for the capital of France. 2. Return the answer.

    Question Type: {question_type}
    Question: {question}
    Plan:
    """
    plan = call_llm(plan_prompt)

    # Step 3: Execute the plan using LLM
    execution_prompt = f"""
    Execute the following plan to answer the question:
    Example:
    Question: What is the capital of France?
    Plan: 1. Search for the capital of France. 2. Return the answer.
    Answer: Paris

    Question: {question}
    Plan: {plan}
    Answer:
    """
    answer = call_llm(execution_prompt)

    return answer


# Example usage
if __name__ == "__main__":
    question = "What is the boiling point of water in Celsius?"
    answer = main(question)
    print(f"Question: {question}")
    print(f"Answer: {answer}")