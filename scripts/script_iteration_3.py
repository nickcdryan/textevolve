import os
import re
import math
from google import genai
from google.genai import types

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

def main(question):
    """
    EXPLORATION: This script uses a question-decomposition and answer-synthesis approach
    to enhance reasoning and accuracy. This is a new approach compared to previous iterations.
    This approach focuses on breaking the original question into a series of sub-questions
    which allow the LLM to address specific elements. Finally, after the LLM has answered each
    sub-question, it synthesizes the sub-answers into a final answer. This is fundamentally different
    than the three-stage approach of the past few successful iterations.
    """

    # Step 1: Decompose the original question into sub-questions
    decomposition_prompt = f"""
    Decompose the original question into a series of simpler, more specific sub-questions that, when answered,
    will provide all the information needed to answer the original question.

    Example 1:
    Original Question: How many yards longer was the longest touchdown pass than the longest field goal?
    Sub-questions:
    1. What was the length of the longest touchdown pass?
    2. What was the length of the longest field goal?
    3. What is the difference between the length of the longest touchdown pass and the length of the longest field goal?

    Example 2:
    Original Question: Which player kicked the only field goal of the game?
    Sub-questions:
    1. Was there a field goal in the game?
    2. Which player kicked the field goal?

    Original Question: {question}
    Sub-questions:
    """

    try:
        sub_questions_str = call_llm(decomposition_prompt, "You are an expert at question decomposition.")
        sub_questions = [q.strip() for q in sub_questions_str.split('\n') if q.strip()]  # Splits and cleans
    except Exception as e:
        print(f"Error decomposing question: {e}")
        return "Error decomposing question."

    # Step 2: Answer each sub-question
    sub_answers = []
    for i, sub_question in enumerate(sub_questions):
        answer_prompt = f"""
        Answer the following sub-question concisely.

        Sub-question: {sub_question}

        Answer:
        """
        try:
            answer = call_llm(answer_prompt, "You are an expert at answering questions concisely.")
            sub_answers.append(answer)
        except Exception as e:
            print(f"Error answering sub-question {i+1}: {e}")
            sub_answers.append(f"Error answering sub-question {i+1}.")

    # Step 3: Synthesize the sub-answers into a final answer
    synthesis_prompt = f"""
    Synthesize the following sub-answers into a final, comprehensive answer to the original question.

    Original Question: {question}
    Sub-questions and Answers:
    {chr(10).join([f"{i+1}. {q}: {a}" for i, (q, a) in enumerate(zip(sub_questions, sub_answers))])}

    Final Answer:
    """

    try:
        final_answer = call_llm(synthesis_prompt, "You are an expert at synthesizing information.")
        return final_answer
    except Exception as e:
        print(f"Error synthesizing final answer: {e}")
        return "Error synthesizing final answer."