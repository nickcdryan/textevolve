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
    """Solve factual questions using a new approach: Multi-faceted Question Decomposition and Answer Aggregation with Confidence Scoring."""

    # Hypothesis: Decomposing the question into *multiple* different types of sub-questions, extracting potential answers for each, assigning confidence scores, and then aggregating the high-confidence answers will improve accuracy. This tests whether diverse question perspectives lead to better information retrieval and synthesis. This is a NEW approach.

    # New approach: Question Decomposition into multiple different types of questions with confidence scoring and answer aggregation

    # Step 1: Multi-Faceted Question Decomposition (with examples) - Generating sub-questions of different kinds.
    decomposition_prompt = f"""
    Decompose the original question into 3 different types of sub-questions: simpler factual questions, hypothetical questions, and clarifying questions.

    Example:
    Original Question: What is the capital of Australia and its population?
    Sub-questions:
    1. Factual: What is the capital of Australia?
    2. Hypothetical: What if Australia didn't have a designated capital, what would be its most likely candidate?
    3. Clarifying: What type of government system does Australia have that determines how the capital is chosen?

    Original Question: {question}
    Sub-questions:
    """
    sub_questions = call_llm(decomposition_prompt, system_instruction="You are an expert at breaking down questions into different facets.").split("\n")
    print(f"Sub-questions: {sub_questions}")

    # Step 2: Fact Extraction for Each Sub-Question (with examples)
    extracted_answers = {}
    for sub_question in sub_questions:
        extraction_prompt = f"""
        Extract a concise answer to the sub-question and assign a confidence score (1-10).

        Example:
        Sub-question: What is the capital of Australia?
        Answer: Canberra (Confidence: 9)

        Sub-question: {sub_question}
        Answer:
        """
        answer = call_llm(extraction_prompt, system_instruction="You are an expert at extracting concise answers with confidence.").strip()
        extracted_answers[sub_question] = answer
    print(f"Extracted Answers: {extracted_answers}")

    # Step 3: Answer Aggregation and Confidence-Based Synthesis (with examples)
    aggregation_prompt = f"""
    Synthesize a final answer to the original question by aggregating the answers to the sub-questions, weighting them by their confidence scores. Only use answers with confidence above 7.

    Example:
    Original Question: What is the capital of Australia and its population?
    Sub-questions:
    1. What is the capital of Australia? Answer: Canberra (Confidence: 9)
    2. What if Australia didn't have a designated capital? Answer: Sydney (Confidence: 4)
    Final Answer: Canberra

    Original Question: {question}
    Sub-questions and Answers: {extracted_answers}
    Final Answer:
    """
    final_answer = call_llm(aggregation_prompt, system_instruction="You are an expert at synthesizing information with confidence.").strip()
    print(f"Final Answer: {final_answer}")

    # Step 4: Verification (with examples)
    verification_prompt = f"""
    Verify that the synthesized answer correctly and completely answers the original question.

    Example:
    Original Question: What is the capital of Australia and its population?
    Answer: Canberra
    Verification: The answer is correct.

    Original Question: {question}
    Answer: {final_answer}
    Verification:
    """
    validation = call_llm(verification_prompt, system_instruction="You are a validator of answer accuracy.").strip()

    if "correct" in validation.lower():
        return final_answer
    else:
        return "Could not be validated."