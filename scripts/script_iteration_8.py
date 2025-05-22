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
    """Solve factual questions using a new approach: Iterative Question Decomposition and Targeted Information Extraction with Confidence Scoring."""

    # Hypothesis: Iteratively decomposing the question into smaller, more manageable parts, then extracting information targeted to each part *with explicit confidence scores* will improve accuracy. By focusing extraction on smaller components, we reduce the complexity of each extraction step.

    # Step 1: Initial question decomposition (with examples)
    decomposition_prompt = f"""
    Decompose the question into smaller, independent sub-questions that, when answered, will collectively answer the original question.

    Example 1:
    Question: What is the capital of Australia and what is its population?
    Sub-questions:
    1. What is the capital of Australia?
    2. What is the population of Canberra?

    Example 2:
    Question: In what year was Jamini Roy awarded the Padma Bhushan, and what was his primary artistic style?
    Sub-questions:
    1. In what year was Jamini Roy awarded the Padma Bhushan?
    2. What was Jamini Roy's primary artistic style?

    Question: {question}
    Sub-questions:
    """
    sub_questions = call_llm(decomposition_prompt, system_instruction="You are an expert question decomposer.").split("\n")
    print (f"Sub-questions: {sub_questions}")

    # Step 2: Iteratively extract targeted information for EACH sub-question AND assign confidence score
    answers_with_confidence = []
    for sub_question in sub_questions:
        extraction_prompt = f"""
        Extract a concise answer to the following sub-question, AND provide a confidence score (1-10) for the accuracy of your answer.

        Example:
        Sub-question: What is the capital of Australia?
        Answer: Canberra (Confidence: 9)

        Sub-question: {sub_question}
        Answer:
        """
        extracted_answer_raw = call_llm(extraction_prompt, system_instruction="You are an expert at concise answer extraction.").strip()

        try:
            extracted_answer = extracted_answer_raw.split('(Confidence:')[0].strip()
            confidence = int(extracted_answer_raw.split('(Confidence:')[1].replace(')','').strip())
        except:
            extracted_answer = extracted_answer_raw
            confidence = 5 #low confidence score to force validation to work

        answers_with_confidence.append({"sub_question": sub_question, "answer": extracted_answer, "confidence": confidence})
    print (f"Answers with confidence: {answers_with_confidence}")

    # Step 3: Synthesize final answer, taking into account confidence scores (with example)
    synthesis_prompt = f"""
    Synthesize the answers to the sub-questions into a single, coherent answer to the original question. Consider the confidence scores of each sub-answer. If any sub-answer has low confidence (<7), indicate uncertainty.

    Example:
    Question: What is the capital of Australia and what is its population?
    Sub-questions:
    1. What is the capital of Australia? Answer: Canberra (Confidence: 9)
    2. What is the population of Canberra? Answer: 450,000 (Confidence: 6)
    Final Answer: The capital of Australia is Canberra. The population is approximately 450,000, but this number is uncertain.

    Question: {question}
    Sub-questions and answers:
    {answers_with_confidence}
    Final Answer:
    """
    final_answer = call_llm(synthesis_prompt, system_instruction="You are an expert at synthesizing information.").strip()

    # Step 4: Validation of final answer (with example)
    validation_prompt = f"""
    Validate that the following extracted and synthesized answer correctly answers the original question.

    Example:
    Question: What is the capital of Australia and what is its population?
    Answer: The capital of Australia is Canberra. The population is approximately 450,000, but this number is uncertain.
    Validation: Correct; Canberra is the capital, and the population estimate reflects the lower confidence score. VALID.

    Question: {question}
    Answer: {final_answer}
    Validation:
    """

    validation_result = call_llm(validation_prompt, system_instruction="You are an expert answer validator.")

    if "VALID" in validation_result:
        return final_answer
    else:
        return "Could not be validated."