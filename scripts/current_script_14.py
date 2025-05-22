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
    """Solve factual questions using a new approach: Contextualized Question Answering with Iterative Context Expansion and Answer Ranking."""

    # Hypothesis: By actively expanding the context of the question in multiple iterations and ranking potential answers based on both the expanded context and the initial question, we can improve accuracy. This approach moves away from simple fact extraction and aims for a deeper understanding of the question's context. This is a fundamentally new approach.

    # Step 1: Initial Contextualization (with examples)
    contextualization_prompt = f"""
    Provide a contextualized version of the original question by adding background information and related details.

    Example 1:
    Original Question: What is the capital of Australia?
    Contextualized Question: What is the capital city of Australia, a country in the southern hemisphere known for its unique wildlife?

    Example 2:
    Original Question: Who is known as the first rock star of the Middle East?
    Contextualized Question: Who is the musician widely recognized as the first rock and roll star of the Middle East?

    Original Question: {question}
    Contextualized Question:
    """
    contextualized_question = call_llm(contextualization_prompt, system_instruction="You are an expert at providing context to questions.").strip()
    print(f"Contextualized Question: {contextualized_question}")

    # Step 2: Iterative Context Expansion and Answer Extraction (with examples)
    extracted_answers = []
    current_question = contextualized_question

    for i in range(2): # Iterate twice to expand the context
        expansion_extraction_prompt = f"""
        Expand the current question with additional context, and then extract a *potential* answer from it. Rank the answer for its likelihood to answer the question (1-10). Be concise.

        Example:
        Current Question: What is the capital city of Australia, a country in the southern hemisphere known for its unique wildlife?
        Expanded Question: What is the capital city of Australia, which is Canberra, a planned city also known for being the home to Parliament House?
        Potential Answer: Canberra (Relevance: 9)

        Current Question: {current_question}
        Expanded Question:
        """
        expanded_text = call_llm(expansion_extraction_prompt, system_instruction="You are an expert at expanding questions with context and extracting answers.").strip()
        print(f"Expanded Text: {expanded_text}")

        # Simple parsing to extract answer and expanded question (error prone JSON is avoided)
        potential_answer_match = re.search(r"Potential Answer:\s*(.*?)\s*\(", expanded_text)
        potential_answer = potential_answer_match.group(1) if potential_answer_match else "No answer"

        current_question = expanded_text.split("Potential Answer:")[0].replace("Expanded Question:", "").strip()

        extracted_answers.append(potential_answer)
        print(f"Extracted answers after iteration {i+1}: {extracted_answers}")

    # Step 3: Answer Ranking (with examples)
    answer_ranking_prompt = f"""
    Rank the following potential answers for their likelihood to answer the *original* question, considering the expanded context from each iteration (1-10, 10 is best). Also give a short reasoning about the score.

    Original Question: {question}
    Potential Answers:
    {extracted_answers}

    Example:
    Original Question: What is the capital of Australia?
    Potential Answers:
    ['Canberra', 'Sydney']
    Ranking:
    1. Canberra (10) - Canberra is the capital.
    2. Sydney (2) - Sydney is only a major city.

    Ranking:
    """
    ranked_answers = call_llm(answer_ranking_prompt, system_instruction="You are an expert at ranking answers.").strip()
    print(f"Ranked Answers: {ranked_answers}")

    # Basic parsing for the best answer
    try:
        best_answer = ranked_answers.split("1. ")[1].split(" (")[0].strip()
    except:
        best_answer = "No answer found."

    # Step 4: Validation (with example)
    validation_prompt = f"""
    Validate that the selected answer correctly and completely answers the original question.

    Example:
    Question: What is the capital of Australia?
    Answer: Canberra
    Validation: VALID - Canberra is the capital.

    Question: {question}
    Answer: {best_answer}
    Validation:
    """
    validation_result = call_llm(validation_prompt, system_instruction="You are a strict validator.").strip()

    if "VALID" in validation_result:
        return best_answer
    else:
        return "Could not be validated."