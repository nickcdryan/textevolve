import os
import re
import math

def main(question):
    """
    Solve the question using a novel LLM-driven approach that focuses on iterative refinement
    of the answer based on feedback from multiple verifiers with different expertise.
    """
    try:
        # Step 1: Initial answer generation
        initial_answer = generate_initial_answer(question)
        if "Error" in initial_answer:
            return "Error generating initial answer"

        # Step 2: Iterative refinement with multiple verifiers
        refined_answer = refine_answer_with_multiple_verifiers(question, initial_answer)
        if "Error" in refined_answer:
            return "Error refining answer"

        return refined_answer

    except Exception as e:
        return f"General Error: {str(e)}"

def generate_initial_answer(question):
    """Generates an initial answer to the question."""
    system_instruction = "You are an expert question answering system."
    prompt = f"""
    Generate an initial answer to the following question. Be as accurate and complete as possible.

    Example:
    Question: Who caught the final touchdown of the game?
    Answer: Jarrett Boykin

    Question: {question}
    Answer:
    """
    return call_llm(prompt, system_instruction)

def refine_answer_with_multiple_verifiers(question, initial_answer, max_iterations=3):
    """Refines the answer iteratively using feedback from multiple verifiers."""
    answer = initial_answer
    for i in range(max_iterations):
        # Step 1: Get feedback from the factual verifier
        factual_feedback = get_factual_feedback(question, answer)
        if "Error" in factual_feedback:
            return "Error getting factual feedback"

        # Step 2: Get feedback from the arithmetic verifier (if applicable)
        arithmetic_feedback = get_arithmetic_feedback(question, answer)
        if "Error" in arithmetic_feedback:
            return "Error getting arithmetic feedback"

        # Step 3: Combine feedback and refine the answer
        combined_feedback = f"Factual Feedback: {factual_feedback}\nArithmetic Feedback: {arithmetic_feedback}"
        refined_answer = refine_answer(question, answer, combined_feedback)
        if "Error" in refined_answer:
            return "Error refining answer"

        answer = refined_answer  # Update the answer for the next iteration
    return answer

def get_factual_feedback(question, answer):
    """Gets feedback on the factual accuracy of the answer."""
    system_instruction = "You are an expert at verifying the factual accuracy of answers to questions."
    prompt = f"""
    Verify the factual accuracy of the following answer to the question. Provide specific feedback on any errors or omissions. If the answer is factually correct, say "Factually correct.".

    Example:
    Question: Who caught the final touchdown of the game?
    Answer: Jarrett Boykin
    Feedback: Factually correct.

    Question: {question}
    Answer: {answer}
    Feedback:
    """
    return call_llm(prompt, system_instruction)

def get_arithmetic_feedback(question, answer):
    """Gets feedback on the arithmetic accuracy of the answer (if applicable)."""
    system_instruction = "You are an expert at verifying the arithmetic accuracy of answers to questions."
    prompt = f"""
    Verify the arithmetic accuracy of the following answer to the question. If the question requires a calculation and the answer is arithmetically incorrect, provide the correct calculation. If the question does not require a calculation or the answer is arithmetically correct, say "No arithmetic required or arithmetically correct.".

    Example:
    Question: How many running backs ran for a touchdown?
    Answer: 2
    Feedback: No arithmetic required or arithmetically correct.

    Question: How many yards did Chris Johnson's first touchdown and Jason Hanson's first field goal combine for?
    Answer: 50
    Feedback: Arithmetically incorrect. 6 + 53 = 59. The correct answer is 59.

    Question: {question}
    Answer: {answer}
    Feedback:
    """
    return call_llm(prompt, system_instruction)

def refine_answer(question, answer, feedback):
    """Refines the answer based on the provided feedback."""
    system_instruction = "You are an expert at refining answers to questions based on feedback."
    prompt = f"""
    Refine the following answer to the question based on the provided feedback.

    Example:
    Question: Who caught the final touchdown of the game?
    Answer: Jarrett Boykin
    Feedback: Factually correct.
    Refined Answer: Jarrett Boykin

    Question: How many yards did Chris Johnson's first touchdown and Jason Hanson's first field goal combine for?
    Answer: 50
    Feedback: Arithmetically incorrect. 6 + 53 = 59. The correct answer is 59.
    Refined Answer: 59

    Question: {question}
    Answer: {answer}
    Feedback: {feedback}
    Refined Answer:
    """
    return call_llm(prompt, system_instruction)

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