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
    EXPLORATION: This script uses an **Iterative Question Decomposition and Contextual Fact Verification** approach.

    HYPOTHESIS: By first decomposing complex questions into simpler sub-questions and then performing targeted fact verification on each sub-question within the original context,
    we can improve accuracy and reduce the risk of hallucination. This focuses validation on *individual components*, not the entire answer.

    This approach is DIFFERENT from previous iterations by:
    1. Explicitly decomposing questions into sub-questions and answering each individually *within the passage's context.*
    2. Performing targeted fact verification for *each sub-question* using the original passage.
    3. Synthesizing sub-question answers to generate the final answer.
    """

    # Step 1: Question Decomposition (with embedded examples)
    decomposition_prompt = f"""
    Decompose the original question into simpler, self-contained sub-questions that can be answered independently.

    Example 1:
    Original Question: What is the capital of the country where the Great Barrier Reef is located, and what is the population of that capital?
    Sub-questions:
    1.  Which country is the Great Barrier Reef located in?
    2.  What is the capital of Australia?
    3.  What is the population of Canberra?

    Example 2:
    Original Question: Which player kicked the only field goal of the game, and how long was it?
    Sub-questions:
    1.  Which player kicked the only field goal of the game?
    2.  How long was the field goal?

    Original Question: {question}
    Sub-questions:
    """
    try:
        sub_questions = call_llm(decomposition_prompt, "You are an expert question decomposer.").strip().split("\n")
        sub_questions = [q.split(". ")[-1].strip() for q in sub_questions if q.strip()] # Extract sub-questions cleanly
    except Exception as e:
        print(f"Error decomposing question: {e}")
        return "Error decomposing question."

    # Step 2: Answer Sub-questions and Contextual Fact Verification (with embedded examples)
    answers = []
    for sub_question in sub_questions:
        # Answer sub-question (with fact verification within the passage context)
        answer_verification_prompt = f"""
        Answer the following sub-question based on the provided text, and verify your answer's accuracy using the same text.

        Example 1:
        Sub-question: Which player kicked the only field goal of the game?
        Answer: Josh Scobee
        Verification: The passage states, "In the fourth quarter, the Jaguars drew closer as kicker Josh Scobee nailed a 47-yard field goal." This confirms Josh Scobee kicked the field goal.

        Example 2:
        Sub-question: What is the capital of Australia?
        Answer: Canberra
        Verification: General geographical knowledge confirms that Canberra is the capital of Australia.

        Sub-question: {sub_question}
        Answer:
        """
        try:
            answer = call_llm(answer_verification_prompt, "You are an expert at answering questions and verifying facts.").strip()
            answers.append(answer)
        except Exception as e:
            print(f"Error answering sub-question: {e}")
            answers.append(f"Error: {e}")

    # Step 3: Synthesize Sub-question Answers into Final Answer (with embedded examples)
    synthesis_prompt = f"""
    Synthesize the following answers to sub-questions into a comprehensive and coherent final answer to the original question.

    Example 1:
    Original Question: What is the capital of the country where the Great Barrier Reef is located, and what is the population of that capital?
    Sub-question Answers:
    1. The Great Barrier Reef is located in Australia.
    2. The capital of Australia is Canberra.
    3. The population of Canberra is approximately 431,500.
    Final Answer: The capital of the country where the Great Barrier Reef is located (Australia) is Canberra, and its population is approximately 431,500.

    Example 2:
    Original Question: Which player kicked the only field goal of the game, and how long was it?
    Sub-question Answers:
    1. Josh Scobee kicked the only field goal of the game.
    2. The field goal was 47 yards long.
    Final Answer: Josh Scobee kicked the only field goal of the game, and it was 47 yards long.

    Original Question: {question}
    Sub-question Answers:
    {", ".join(answers)}
    Final Answer:
    """
    try:
        final_answer = call_llm(synthesis_prompt, "You are an expert at synthesizing information.")
        return final_answer.strip()
    except Exception as e:
        print(f"Error synthesizing final answer: {e}")
        return "Error synthesizing final answer."