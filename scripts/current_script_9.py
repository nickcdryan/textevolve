import os
import re
import math

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

def main(question, max_attempts=3):
    """Solve factual questions using a new approach: Question Transformation & Focused Verification."""

    # Hypothesis: Transform the question into multiple simpler questions each focused on a specific aspect. Then, focus the verification process specifically on validating the transformations and intermediate answers. This approach isolates potential failure points and allows for targeted refinement.

    # Step 1: Question Transformation - Transform the original question into multiple simpler questions (with examples)
    transformation_prompt = f"""
    Transform the original question into a set of simpler questions that, when answered individually, will collectively answer the original question. Focus each question on a specific aspect of the original.

    Example 1:
    Original Question: What is the capital of the country where the Great Barrier Reef is located, and what is the population of that capital?
    Transformed Questions:
    1.  What country is the Great Barrier Reef located in?
    2.  What is the capital of [country from question 1]?
    3.  What is the population of [capital from question 2]?

    Example 2:
    Original Question: In what year was Jamini Roy awarded the Padma Bhushan, and what was his primary artistic style?
    Transformed Questions:
    1. In what year was Jamini Roy awarded the Padma Bhushan?
    2. What was Jamini Roy's primary artistic style?

    Original Question: {question}
    Transformed Questions:
    """
    transformed_questions = call_llm(transformation_prompt, system_instruction="You are an expert question transformer.").split("\n")
    print(f"Transformed Questions: {transformed_questions}")

    # Step 2: Answer the transformed questions (with error handling).
    answers = []
    for q in transformed_questions:
        if not q.strip():
            continue  # Skip empty questions
        search_query = call_llm(f"Generate a search query for: {q}", "You are a search query generator.")
        search_results = call_llm(f"Simulated search results for: {search_query}.", "You are a search engine.")
        answer = call_llm(f"Answer the question '{q}' using the search results: {search_results}", "You are an answer extraction expert.")
        answers.append(answer)
    print(f"Answers: {answers}")
    # Step 3: Synthesize the answers into a final answer (with examples).
    synthesis_prompt = f"""
    Synthesize the individual answers into a coherent final answer to the original question.
    Example 1:
    Original Question: What is the capital of the country where the Great Barrier Reef is located, and what is the population of that capital?
    Individual Answers:
    1. Australia
    2. Canberra
    3. 450,000
    Final Answer: The capital of Australia, where the Great Barrier Reef is located, is Canberra, which has a population of approximately 450,000.

    Original Question: {question}
    Individual Answers:
    {answers}
    Final Answer:
    """
    final_answer = call_llm(synthesis_prompt, system_instruction="You are an expert answer synthesizer.")

    # Step 4: Focused Validation
    validation_prompt = f"""
    Validate the transformation and the synthesized answer.
    First, confirm that the TRANSFORMED QUESTIONS, if answered completely and accurately, would address the original question completely. Then, validate the final answer based on the transformed questions and answers.
    Original Question: {question}
    Transformed Questions: {transformed_questions}
    Final Answer: {final_answer}

    Example 1:
    Original Question: What is the capital of the country where the Great Barrier Reef is located, and what is the population of that capital?
    Transformed Questions:
    1.  What country is the Great Barrier Reef located in?
    2.  What is the capital of [country from question 1]?
    3.  What is the population of [capital from question 2]?
    Final Answer: The capital of Australia, where the Great Barrier Reef is located, is Canberra, which has a population of approximately 450,000.
    Validation: VALID. The transformed questions cover all aspects of the original. The final answer correctly synthesizes the answers to those questions.

    """
    validation_result = call_llm(validation_prompt, "You are an expert validator.")

    if "VALID" in validation_result:
        return final_answer
    else:
        return "Could not be validated."