import os
import re
import math # for react
from google import genai
from google.genai import types

# This script introduces a new approach: LLM-Guided Recursive Decomposition & Verification (LLM-RDRV)
# Hypothesis: By recursively decomposing complex questions into simpler sub-questions and verifying each intermediate answer,
# we can improve accuracy and handle complex queries more effectively.

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

def decompose_question(question):
    """Decomposes a complex question into simpler, answerable sub-questions."""
    system_instruction = "You are an expert at breaking down complex questions into simpler sub-questions."
    prompt = f"""
    Decompose the following complex question into simpler, independent sub-questions that can be answered individually.

    Example 1:
    Complex Question: What is the capital of the country where the Great Barrier Reef is located, and what is the population of that capital?
    Sub-Questions:
    1. What country is the Great Barrier Reef located in?
    2. What is the capital of Australia?
    3. What is the population of Canberra?

    Example 2:
    Complex Question: In which month and year was Satyanarayan Gangaram Pitroda appointed as advisor to the Indian Prime Minister, and what was his rank?
    Sub-Questions:
    1. In which month and year was Satyanarayan Gangaram Pitroda appointed as advisor to the Indian Prime Minister?
    2. What was Satyanarayan Gangaram Pitroda's rank as advisor?

    Question: {question}
    Sub-Questions:
    """
    return call_llm(prompt, system_instruction)

def answer_sub_question(sub_question):
    """Answers a single sub-question using a direct LLM call."""
    system_instruction = "You are an expert at answering questions directly."
    prompt = f"""
    Answer the following question concisely and accurately.

    Example 1:
    Question: What is the capital of France?
    Answer: Paris

    Example 2:
    Question: In what year did World War II begin?
    Answer: 1939

    Question: {sub_question}
    Answer:
    """
    return call_llm(prompt, system_instruction)

def verify_answer(question, answer):
    """Verifies the answer against the original question to ensure relevance and accuracy."""
    system_instruction = "You are a critical validator who checks if an answer is factually correct and relevant to the question."
    prompt = f"""
    Verify if the following answer accurately and completely answers the question. Respond with VALID or INVALID, followed by a brief explanation.

    Example 1:
    Question: What is the capital of France?
    Answer: Paris
    Verification: VALID: Paris is indeed the capital of France.

    Example 2:
    Question: In what year did World War II begin?
    Answer: 1940
    Verification: INVALID: World War II began in 1939.

    Question: {question}
    Answer: {answer}
    Verification:
    """
    return call_llm(prompt, system_instruction)

def synthesize_answers(original_question, sub_questions_answers):
    """Synthesizes the answers to the sub-questions into a coherent answer to the original question."""
    system_instruction = "You are an expert at synthesizing information to answer complex questions."
    prompt = f"""
    Synthesize the following answers to sub-questions into a coherent and complete answer to the original question.

    Example 1:
    Original Question: What is the capital of the country where the Great Barrier Reef is located, and what is the population of that capital?
    Sub-Questions and Answers:
    1. What country is the Great Barrier Reef located in? Answer: Australia
    2. What is the capital of Australia? Answer: Canberra
    3. What is the population of Canberra? Answer: 431,500
    Synthesized Answer: The capital of Australia, where the Great Barrier Reef is located, is Canberra, and its population is 431,500.

   Example 2:
    Original Question: In which month and year was Satyanarayan Gangaram Pitroda appointed as advisor to the Indian Prime Minister, and what was his rank?
    Sub-Questions and Answers:
    1. In which month and year was Satyanarayan Gangaram Pitroda appointed as advisor to the Indian Prime Minister? Answer: October 2009
    2. What was Satyanarayan Gangaram Pitroda's rank as advisor? Answer: Cabinet Minister
    Synthesized Answer: Satyanarayan Gangaram Pitroda was appointed as advisor to the Indian Prime Minister in October 2009 with the rank of Cabinet Minister.

    Original Question: {original_question}
    Sub-Questions and Answers:
    {sub_questions_answers}
    Synthesized Answer:
    """
    return call_llm(prompt, system_instruction)

def main(question):
    """Main function to orchestrate the LLM-Guided Recursive Decomposition & Verification process."""
    # Step 1: Decompose the question
    sub_questions = decompose_question(question)
    print(f"Sub-questions: {sub_questions}")

    # Step 2: Answer each sub-question
    sub_questions_list = sub_questions.split("\n")
    sub_questions_answers = []

    for i, sub_question in enumerate(sub_questions_list):
        if sub_question.strip(): # Skip empty lines
            answer = answer_sub_question(sub_question)
            verification = verify_answer(sub_question, answer)
            print(f"Verification result: {verification}")
            if "INVALID" not in verification:
                sub_questions_answers.append(f"{i+1}. {sub_question} Answer: {answer}")
            else:
                return "Could not find the answer."

    # Step 3: Synthesize the answers
    synthesized_answer = synthesize_answers(question, "\n".join(sub_questions_answers))
    print(f"Synthesized answer: {synthesized_answer}")

    return synthesized_answer