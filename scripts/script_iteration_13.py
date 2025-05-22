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
    """Solve factual questions using a new approach: Question Decomposition + Focused Fact Extraction and Ranking + Direct Answer Synthesis"""

    # Hypothesis: Decomposing the question into sub-questions, extracting *multiple* candidate facts with relevance ranking, and directly synthesizing the answer from top-ranked facts will improve accuracy. This contrasts with previous single-extraction approaches and emphasizes gathering multiple pieces of evidence before synthesizing. The goal is to test this hypothesis.
    # New approach from previous: This approach will change and make multiple extraction runs to get fact candidates and rank those candidates.

    # Step 1: Question Decomposition (with examples)
    decomposition_prompt = f"""
    Break down the original question into simpler, more specific sub-questions.

    Example 1:
    Original Question: What is the capital of Australia and its population?
    Sub-questions:
    1. What is the capital of Australia?
    2. What is the population of Canberra?

    Example 2:
    Original Question: In what year did Jamini Roy receive the Padma Bhushan, and what art style did he use?
    Sub-questions:
    1. In what year did Jamini Roy receive the Padma Bhushan?
    2. What art style did Jamini Roy use?

    Original Question: {question}
    Sub-questions:
    """
    sub_questions = call_llm(decomposition_prompt, system_instruction="You are an expert at breaking down questions.").split("\n")
    print(f"Sub-questions: {sub_questions}")

    # Step 2: Fact Extraction and Ranking (with examples) - Extract multiple candidate facts for each sub-question, and rank them by relevance.
    candidate_facts = {}
    for sub_question in sub_questions:
        extraction_prompt = f"""
        Extract 3 candidate facts for the following sub-question, and rank their relevance (1-10). Be concise.

        Example:
        Sub-question: What is the capital of Australia?
        1. Canberra is the capital (Relevance: 10)
        2. Sydney is a city in Australia (Relevance: 2)
        3. Australia is an island nation. (Relevance: 1)

        Sub-question: {sub_question}
        Candidate Facts:
        """
        facts = call_llm(extraction_prompt, system_instruction="You are an expert in extracting facts and their relevance.")
        candidate_facts[sub_question] = facts
    print(f"Extracted Candidate Facts: {candidate_facts}")

    # Step 3: Direct Answer Synthesis (with examples). Synthesize the facts directly into an answer.
    synthesis_prompt = f"""
    Synthesize a direct answer to the original question from the candidate facts.

    Example:
    Original Question: What is the capital of Australia and its population?
    Candidate Facts:
    Sub-question: What is the capital of Australia?
    1. Canberra is the capital (Relevance: 10)
    2. Sydney is a city in Australia (Relevance: 2)
    Sub-question: What is the population of Canberra?
    1. Canberra has a population of 450,000 (Relevance: 10)
    2. Australia has a population of 25 million (Relevance: 3)
    Answer: Canberra has a population of 450,000

    Original Question: {question}
    Candidate Facts: {candidate_facts}
    Answer:
    """
    answer = call_llm(synthesis_prompt, system_instruction="You are an expert at synthesizing accurate answers.")
    print(f"Synthesized Answer: {answer}")

    # Step 4: Verification (with example). Validating the answer directly.
    verification_prompt = f"""
    Verify that the extracted answer accurately answers the original question.

    Example:
    Original Question: What is the capital of Australia and its population?
    Answer: Canberra has a population of 450,000
    Verification: The answer is correct.

    Original Question: {question}
    Answer: {answer}
    Verification:
    """
    validation = call_llm(verification_prompt, system_instruction="You are a validator of answer accuracy.")

    if "correct" in validation.lower():
        return answer
    else:
        return "Could not be validated."