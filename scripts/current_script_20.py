import google.generativeai as genai
import os

# Replace with your actual Gemini API key
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

# Select the Gemini model
model = genai.GenerativeModel('gemini-pro')

def call_llm(prompt, system_instruction=None, max_output_tokens=200):
    """
    Calls the LLM with a prompt and optional system instruction.

    Includes example usage within the docstring for clarity.
    """
    try:
        if system_instruction:
            response = model.generate_content(
                contents=prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=max_output_tokens
                ),
                safety_settings={
                  genai.types.HarmCategory.HARASSMENT: genai.types.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                  genai.types.HarmCategory.HATE_SPEECH: genai.types.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                  genai.types.HarmCategory.SEXUALLY_EXPLICIT: genai.types.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                  genai.types.HarmCategory.DANGEROUS_CONTENT: genai.types.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                }
            )
        else:
            response = model.generate_content(
                contents=prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=max_output_tokens
                ),
                safety_settings={
                  genai.types.HarmCategory.HARASSMENT: genai.types.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                  genai.types.HarmCategory.HATE_SPEECH: genai.types.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                  genai.types.HarmCategory.SEXUALLY_EXPLICIT: genai.types.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                  genai.types.HarmCategory.DANGEROUS_CONTENT: genai.types.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                }
            )
        return response.text
    except Exception as e:
        return f"Error calling LLM: {str(e)}"

def main(question):
    """
    Main function to answer a question using LLM-driven reasoning.

    Example:
    question = "What is the capital of France?"
    answer = main(question)
    print(answer)
    """

    # Step 1: Clarify the question (LLM call 1)
    clarification_prompt = f"""
    Given the question: "{question}", clarify any ambiguities or assumptions.

    Example:
    Question: "What is the best way to lose weight?"
    Clarification: "This question assumes a healthy individual.  The best way to lose weight depends on individual circumstances but typically involves a balanced diet and exercise."

    Question: "{question}"
    Clarification:
    """
    clarified_question = call_llm(clarification_prompt)

    # Step 2: Generate a step-by-step reasoning plan (LLM call 2)
    reasoning_prompt = f"""
    Create a step-by-step reasoning plan to answer the question: "{clarified_question}".

    Example:
    Question: "What is the population of the US?"
    Plan:
    1. Search online for the current population of the US.
    2. Summarize the findings from multiple sources.
    3. Provide the most up-to-date estimate.

    Question: "{clarified_question}"
    Plan:
    """
    reasoning_plan = call_llm(reasoning_prompt)

    # Step 3: Execute the reasoning plan and generate an answer (LLM call 3)
    execution_prompt = f"""
    Execute the following reasoning plan to answer the question: "{clarified_question}".

    Reasoning Plan:
    {reasoning_plan}

    Provide a detailed answer based on the plan.

    Example:
    Question: What is the capital of France?
    Reasoning Plan: Search the web. Respond with the capital
    Answer: Paris is the capital of France.

    Question: "{clarified_question}"
    Reasoning Plan: {reasoning_plan}
    Answer:
    """
    answer = call_llm(execution_prompt)

    return answer

if __name__ == "__main__":
    question = "What is the meaning of life?"
    answer = main(question)
    print(answer)