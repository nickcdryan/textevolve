import os
import re
import math

# Hypothesis: Implement a "Chain of Reasoning with Knowledge Base Augmentation and Iterative Validation" approach.
# This approach focuses on combining Chain of Thought reasoning with augmenting knowledge from a database and iterative validation to identify and correct errors.
# We will use explicit chain of reasoning steps with a dedicated "reasoning validator" agent, then augment with external data, then validate the final answer.

def call_llm(prompt, system_instruction=None):
    """Call the Gemini LLM with a prompt and return the response."""
    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

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

def initial_reasoning(question):
    """Generate initial reasoning steps for the question."""
    system_instruction = "You are a reasoning expert. Generate initial reasoning steps for the question."
    prompt = f"""
    Generate initial reasoning steps to answer the question.

    Example:
    Question: What was Pankaj Mithal's position just before being appointed as a judge of the Supreme Court of India?
    Reasoning Steps:
    1. Identify the person: Pankaj Mithal
    2. Identify the target position: position just before Supreme Court judge appointment
    3. Search for information about Pankaj Mithal's career before the Supreme Court appointment

    Question: {question}
    Reasoning Steps:
    """
    return call_llm(prompt, system_instruction)

def validate_reasoning(question, reasoning_steps):
    """Validate the reasoning steps."""
    system_instruction = "You are a strict reasoning validator. Check if the steps are complete and logical."
    prompt = f"""
    Validate the reasoning steps for answering the question.

    Example:
    Question: What was Pankaj Mithal's position just before being appointed as a judge of the Supreme Court of India?
    Reasoning Steps:
    1. Identify the person: Pankaj Mithal
    2. Identify the target position: position just before Supreme Court judge appointment
    3. Search for information about Pankaj Mithal's career before the Supreme Court appointment
    Validation: VALID - The reasoning steps are complete and logical.

    Question: {question}
    Reasoning Steps: {reasoning_steps}
    Validation:
    """
    return call_llm(prompt, system_instruction)

def augment_knowledge(question, reasoning_steps):
    """Augment knowledge from a database based on the reasoning steps."""
    system_instruction = "You are a knowledge augmenter. Find data related to the question and reasoning steps."
    prompt = f"""
    Augment knowledge from available sources based on the reasoning steps.

    Example:
    Question: What was Pankaj Mithal's position just before being appointed as a judge of the Supreme Court of India?
    Reasoning Steps:
    1. Identify the person: Pankaj Mithal
    2. Identify the target position: position just before Supreme Court judge appointment
    3. Search for information about Pankaj Mithal's career before the Supreme Court appointment
    Augmented Knowledge: Pankaj Mithal was Chief Justice of Rajasthan High Court before his Supreme Court appointment.

    Question: {question}
    Reasoning Steps: {reasoning_steps}
    Augmented Knowledge:
    """
    return call_llm(prompt, system_instruction)

def extract_answer(question, augmented_knowledge):
    """Extract the answer from the augmented knowledge."""
    system_instruction = "You are an expert answer extractor. Extract the answer from the augmented knowledge."
    prompt = f"""
    Extract the concise answer from the augmented knowledge.

    Example:
    Question: What was Pankaj Mithal's position just before being appointed as a judge of the Supreme Court of India?
    Augmented Knowledge: Pankaj Mithal was Chief Justice of Rajasthan High Court before his Supreme Court appointment.
    Answer: Chief Justice Rajasthan High Court

    Question: {question}
    Augmented Knowledge: {augmented_knowledge}
    Answer:
    """
    return call_llm(prompt, system_instruction)

def validate_answer(question, answer):
    """Validate the extracted answer."""
    system_instruction = "You are a fact validator. Ensure the answer is correct and complete."
    prompt = f"""
    Validate if the answer accurately and completely answers the question.

    Example:
    Question: What was Pankaj Mithal's position just before being appointed as a judge of the Supreme Court of India?
    Answer: Chief Justice Rajasthan High Court
    Validation: VALID - Pankaj Mithal was indeed the Chief Justice of Rajasthan High Court.

    Question: {question}
    Answer: {answer}
    Validation:
    """
    return call_llm(prompt, system_instruction)

def main(question):
    """Solve questions using the Chain of Reasoning with Knowledge Base Augmentation and Iterative Validation."""
    try:
        # 1. Initial Reasoning
        reasoning_steps = initial_reasoning(question)
        print(f"Reasoning steps: {reasoning_steps}")

        # 2. Validate Reasoning
        validation_result = validate_reasoning(question, reasoning_steps)
        if "VALID" not in validation_result:
            return "Could not validate reasoning."

        # 3. Augment Knowledge
        augmented_knowledge = augment_knowledge(question, reasoning_steps)
        print(f"Augmented Knowledge: {augmented_knowledge}")

        # 4. Extract Answer
        answer = extract_answer(question, augmented_knowledge)
        print(f"Extracted Answer: {answer}")

        # 5. Validate Answer
        final_validation = validate_answer(question, answer)
        if "VALID" not in final_validation:
            return "Could not be validated."

        return answer

    except Exception as e:
        return f"Error: {str(e)}"