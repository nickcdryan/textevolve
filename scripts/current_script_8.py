import os
import re
import math # for react
from google import genai
from google.genai import types

# This script implements a new approach: LLM-Guided Iterative Context Expansion & Focused Summarization (LLM-ICE-FS)
# Hypothesis: By iteratively expanding the context around key entities and then focusing summarization on the most relevant parts,
# we can improve accuracy in question answering by capturing nuanced relationships and reducing hallucination.

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

def extract_key_entities(question):
    """Extract key entities from the question for context expansion."""
    system_instruction = "You are an expert at identifying key entities in a question."
    prompt = f"""
    Identify the key entities (people, places, organizations, events, etc.) in the following question.

    Example 1:
    Question: What is the full name of the younger daughter of Mehbooba Mufti, a politician from Kashmir?
    Entities: Mehbooba Mufti, daughter

    Example 2:
    Question: In what patch did the item Mechanical Glove change to only apply its damage buff to melee weapons instead of all weapon types in Terraria?
    Entities: Mechanical Glove, Terraria

    Question: {question}
    Entities:
    """
    return call_llm(prompt, system_instruction)

def expand_context(question, entities, iteration):
    """Expand the context around the key entities by retrieving related information."""
    system_instruction = "You are an expert at gathering information about specific entities."
    prompt = f"""
    Gather relevant information about the following entities to answer the question.

    Example 1:
    Question: What is the full name of the younger daughter of Mehbooba Mufti, a politician from Kashmir?
    Entities: Mehbooba Mufti, daughter
    Information: Iltija Mufti is the younger daughter of Mehbooba Mufti.

    Example 2:
    Question: In what patch did the item Mechanical Glove change to only apply its damage buff to melee weapons instead of all weapon types in Terraria?
    Entities: Mechanical Glove, Terraria
    Information: The Mechanical Glove's damage buff was changed in patch 1.2.3.

    Question: {question}
    Entities: {entities}
    Information:
    """
    return call_llm(prompt, system_instruction)

def summarize_context(question, context):
    """Summarize the expanded context, focusing on the answer to the question."""
    system_instruction = "You are an expert at summarizing text to answer a specific question."
    prompt = f"""
    Summarize the following information to answer the question, providing only the key facts.

    Example 1:
    Question: What is the full name of the younger daughter of Mehbooba Mufti, a politician from Kashmir?
    Information: Iltija Mufti is the younger daughter of Mehbooba Mufti. Other information about Mehbooba Mufti that's not relevant.
    Summary: Iltija Mufti

    Example 2:
    Question: In what patch did the item Mechanical Glove change to only apply its damage buff to melee weapons instead of all weapon types in Terraria?
    Information: The Mechanical Glove's damage buff was changed in patch 1.2.3. Irrelevant information about the game.
    Summary: 1.2.3

    Question: {question}
    Information: {context}
    Summary:
    """
    return call_llm(prompt, system_instruction)

def verify_answer(question, answer):
    """Verify the answer against the original question to ensure accuracy."""
    system_instruction = "You are a critical validator who checks if the answer is factually correct and relevant."
    prompt = f"""
    Verify if the following answer is accurate and completely answers the question. Respond with VALID or INVALID, followed by a brief explanation.

    Example 1:
    Question: What is the capital of France?
    Answer: Paris
    Verification: VALID: Paris is the capital of France.

    Example 2:
    Question: In what year did World War II begin?
    Answer: 1940
    Verification: INVALID: World War II began in 1939.

    Question: {question}
    Answer: {answer}
    Verification:
    """
    return call_llm(prompt, system_instruction)

def main(question):
    """Main function to orchestrate the LLM-Guided Iterative Context Expansion & Focused Summarization process."""
    # Step 1: Extract key entities
    entities = extract_key_entities(question)
    print(f"Entities: {entities}")

    # Step 2: Iteratively expand context
    context = ""
    for i in range(2):  # Iterate twice for deeper context
        context = expand_context(question, entities, i)
        print(f"Context (Iteration {i+1}): {context}")

    # Step 3: Summarize context to answer the question
    answer = summarize_context(question, context)
    print(f"Initial Answer: {answer}")

    # Step 4: Verify the answer
    verification = verify_answer(question, answer)
    print(f"Verification result: {verification}")

    if "INVALID" not in verification:
        return answer
    else:
        return "Could not find the answer."