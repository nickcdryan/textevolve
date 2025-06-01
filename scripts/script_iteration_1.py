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
    Leveraging multi-stage analysis with targeted information extraction and verification:

    Hypothesis: By breaking the problem into stages with distinct LLM calls focused on specific aspects (extraction, reasoning, and synthesis), combined with verification steps at each stage, we can improve accuracy and conciseness. This approach tests whether a more structured reasoning process reduces errors compared to a direct, single-step solution. This exploration focuses on improved information retrieval and answer extraction.
    """

    # Stage 1: Keyword Identification and Passage Simplification
    keywords_prompt = f"""
    Identify the key entities and concepts in the question and passage. Use these keywords to simplify the passage,
    focusing on the most relevant sentences.

    Example 1:
    Question: Which player kicked the only field goal of the game?
    Passage: Game SummaryComing off their Thanksgiving road win over the Falcons, the Colts went home for a Week 13 AFC South rematch with the Jacksonville Jaguars.  In the first quarter, Indianapolis scored first with QB Peyton Manning completing a 5-yard TD pass to TE Dallas Clark, along with a 48-yard TD pass to WR Reggie Wayne.  In the second quarter, the Jaguars got on the board with RB Maurice Jones-Drew getting a 2-yard TD run. Afterwards, the Colts replied with Manning and Clark hooking up with each other again on a 14-yard TD pass. In the third quarter, Jacksonville tried to come back as QB David Garrard completed a 2-yard TD pass to TE Mercedes Lewis for the only score of the period. In the fourth quarter, the Jaguars drew closer as kicker Josh Scobee nailed a 47-yard field goal. However, the Colts responded with Manning completing a 1-yard TD pass to RB Luke Lawton. Afterwards, Jacksonville tried to come back as Garrard completed a 17-yard TD pass to WR Dennis Northcutt (along with getting the 2-point conversion run). Indianapolis' defense managed to seal the deal. With their season-sweep over the Jaguars, the Colts improved to 10-2. During the game, the Colts gave Garrard his first interception of the year, courtesy of Safety Antoine Bethea.

    Keywords: player, field goal
    Simplified Passage: In the fourth quarter, the Jaguars drew closer as kicker Josh Scobee nailed a 47-yard field goal.

    Example 2:
    Question: Which star has a smaller mass, Nu Phoenicis or Gliese 915?
    Passage: Nu Phoenicis is a yellow-white main sequence star of spectral type F9V and magnitude 4.96. Lying some 49 light years distant, it is around 1.2 times as massive as our sun, and likely to be surrounded by a disk of dust. It is the closest star in the constellation that is visible with the unaided eye. Gliese 915 is a white dwarf only 26 light years away. It is of magnitude 13.05, too faint to be seen with the naked eye. White dwarfs are extremely dense stars compacted into a volume the size of the Earth. With around 85% of the mass of the Sun, Gliese 915 has a surface gravity of 108.39 ± 0.01 (2.45 · 108) centimetre·second−2, or approximately 250,000 of Earths gravity.

    Keywords: star, mass, Nu Phoenicis, Gliese 915
    Simplified Passage: Nu Phoenicis is around 1.2 times as massive as our sun. Gliese 915 has around 85% of the mass of the Sun.

    Question: {question}
    Keywords:
    Simplified Passage:
    """

    try:
        keywords_and_simplified = call_llm(keywords_prompt, "You are an expert at simplifying passages by extracting keywords and relevant sentences.")
        simplified_passage = keywords_and_simplified.split("Simplified Passage:")[1].strip()
    except:
        simplified_passage = question # If keyword extraction fails, use the original question

    # Stage 2: Information Extraction and Focused Reasoning
    extraction_prompt = f"""
    Based on the simplified passage, answer the question concisely. Provide ONLY the answer entity (name, value, etc.) without extra words or context.

    Example 1:
    Question: Which player kicked the only field goal of the game?
    Passage: In the fourth quarter, the Jaguars drew closer as kicker Josh Scobee nailed a 47-yard field goal.
    Answer: Josh Scobee

    Example 2:
    Question: Which star has a smaller mass, Nu Phoenicis or Gliese 915?
    Passage: Nu Phoenicis is around 1.2 times as massive as our sun. Gliese 915 has around 85% of the mass of the Sun.
    Answer: Gliese 915

    Question: {question}
    Passage: {simplified_passage}
    Answer:
    """

    try:
        answer = call_llm(extraction_prompt, "You are an information extraction expert. Give the single most relevant entity as the answer.")
        answer = answer.strip()
    except:
        answer = "Could not extract answer."

    # Stage 3: Verification
    verification_prompt = f"""
    Verify that the extracted answer directly and concisely answers the question based on the original passage.
    If the answer is incorrect or incomplete, provide the correct answer without any extra information.

    Example 1:
    Question: Which player kicked the only field goal of the game?
    Extracted Answer: Josh Scobee
    Correct: True

    Example 2:
    Question: Which star has a smaller mass, Nu Phoenicis or Gliese 915?
    Extracted Answer: Gliese 915
    Correct: True

    Question: {question}
    Extracted Answer: {answer}
    Correct:
    """
    try:
        correctness = call_llm(verification_prompt, "You are a precise answer checker.").strip()
        if "False" in correctness:
            answer = call_llm(f"Based on the question {question} and the original passage, what is the CORRECT answer without any extra information or verbosity?", "You are a precise information retriever.")
    except:
        pass

    return answer