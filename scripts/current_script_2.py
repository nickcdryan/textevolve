import os
import re
import math

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

def main(question, max_attempts=3):
    """Solve factual questions using a multi-stage reasoning approach with knowledge graph integration."""

    # Hypothesis: Integrating a simple knowledge graph lookup (simulated) into the reasoning process will improve accuracy.
    # The knowledge graph provides structured knowledge that can be leveraged.

    # Step 1: Identify key entities and relationships from the question (with examples)
    entity_extraction_prompt = f"""
    Extract key entities and relationships from the following question.

    Example 1:
    Question: What is the capital of the country where the Great Barrier Reef is located?
    Entities: Great Barrier Reef, Australia
    Relationship: located in

    Example 2:
    Question: Before the New 52, who murdered the supervillain Monsieur Mallah?
    Entities: Monsieur Mallah, New 52
    Relationship: murdered by

    Question: {question}
    Entities and Relationships:
    """
    entities_relationships = call_llm(entity_extraction_prompt, "You are an expert at extracting entities and relationships.")

    # Step 2: Lookup relevant information in a simulated knowledge graph (with examples)
    knowledge_graph_lookup_prompt = f"""
    Given entities and relationships, look up relevant information in a knowledge graph.

    Example 1:
    Entities: Great Barrier Reef, Australia
    Relationship: located in
    Knowledge Graph Result: The Great Barrier Reef is located in Australia. Australia's capital is Canberra.

    Example 2:
    Entities: Monsieur Mallah, New 52
    Relationship: murdered by
    Knowledge Graph Result: Monsieur Mallah was murdered by Gorilla Grodd before the New 52.

    Entities and Relationships: {entities_relationships}
    Knowledge Graph Result:
    """
    knowledge_graph_results = call_llm(knowledge_graph_lookup_prompt, "You are an expert at knowledge graph lookups.")

    # Step 3: Synthesize information from the knowledge graph and answer the question (with examples)
    answer_synthesis_prompt = f"""
    Synthesize information from the knowledge graph to answer the question.

    Example 1:
    Question: What is the capital of the country where the Great Barrier Reef is located?
    Knowledge Graph Result: The Great Barrier Reef is located in Australia. Australia's capital is Canberra.
    Answer: Canberra

    Example 2:
    Question: Before the New 52, who murdered the supervillain Monsieur Mallah?
    Knowledge Graph Result: Monsieur Mallah was murdered by Gorilla Grodd before the New 52.
    Answer: Gorilla Grodd

    Question: {question}
    Knowledge Graph Result: {knowledge_graph_results}
    Answer:
    """
    final_answer = call_llm(answer_synthesis_prompt, "You are an expert at synthesizing information and answering questions.")

    # Step 4: Validate the answer against the question
    validation_prompt = f"""
    Validate if the answer is correct and completely answers the question.

    Question: {question}
    Proposed Answer: {final_answer}

    Is the answer correct? Respond 'Correct' or 'Incorrect'.
    """
    validation_result = call_llm(validation_prompt, "You are an expert validator.")

    if "Correct" in validation_result:
        return final_answer
    else:
        return "Could not be validated."