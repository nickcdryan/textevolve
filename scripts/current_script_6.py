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
    Hybrid approach: Combines elements from Iteration 2 (multi-stage with simplification, extraction, verification),
    Iteration 4 (fact verification and self-correction), and Iteration 0 (direct approach).
    This approach leverages the strengths of each while addressing their weaknesses.
    """

    # --- 1. Initial Answer Generation with Contextualized Prompt (from Iteration 4 & 0) ---
    initial_prompt = f"""
    Provide a concise answer to the question based on the provided text. Focus on extracting the most relevant entity or value.

    Example 1:
    Question: Which player kicked the only field goal of the game?
    Answer: Josh Scobee

    Example 2:
    Question: Which star has a smaller mass, Nu Phoenicis or Gliese 915?
    Answer: Gliese 915

    Example 3:
    Question: How many yards longer was the longest touchdown pass than the longest field goal?
    Answer: 32

    Question: {question}
    Answer:
    """
    try:
        initial_answer = call_llm(initial_prompt, "You are a precise information retriever. Focus on brevity.")
        initial_answer = initial_answer.strip()
    except Exception as e:
        print(f"Error generating initial answer: {e}")
        return "Error generating initial answer."

    # --- 2. Fact Verification and Self-Correction with Specific Examples (from Iteration 4) ---
    verification_prompt = f"""
    Analyze the answer for factual correctness based on the original question. If the answer is incorrect, provide a corrected answer using the available information.
    If the answer is already concise and accurate, just repeat it.

    Example 1:
    Question: Which player kicked the only field goal of the game?
    Proposed Answer: Tom Brady
    Corrected Answer: Josh Scobee

    Example 2:
    Question: Which star has a smaller mass, Nu Phoenicis or Gliese 915?
    Proposed Answer: Gliese 915
    Corrected Answer: Gliese 915

    Example 3:
    Question: How many yards longer was the longest touchdown pass than the longest field goal?
    Proposed Answer: 30
    Corrected Answer: 32

    Question: {question}
    Proposed Answer: {initial_answer}
    Corrected Answer:
    """
    try:
        verification_response = call_llm(verification_prompt, "You are a fact-checker and self-correction expert. If the answer is correct as is, repeat the exact answer given to you.")
        corrected_answer = verification_response.strip()
    except Exception as e:
        print(f"Error during fact verification: {e}")
        corrected_answer = initial_answer # Fallback in case verification fails

    # --- 3. Keyword Identification and Passage Simplification (from Iteration 2, enhanced with examples) ---
    keywords_prompt = f"""
    Identify the key entities and concepts in the question. Use these keywords to find and extract the most relevant sentence from the original text.

    Example 1:
    Question: Which player kicked the only field goal of the game?
    Passage: Game SummaryComing off their Thanksgiving road win over the Falcons, the Colts went home for a Week 13 AFC South rematch with the Jacksonville Jaguars.  In the first quarter, Indianapolis scored first with QB Peyton Manning completing a 5-yard TD pass to TE Dallas Clark, along with a 48-yard TD pass to WR Reggie Wayne.  In the second quarter, the Jaguars got on the board with RB Maurice Jones-Drew getting a 2-yard TD run. Afterwards, the Colts replied with Manning and Clark hooking up with each other again on a 14-yard TD pass. In the third quarter, Jacksonville tried to come back as QB David Garrard completed a 2-yard TD pass to TE Mercedes Lewis for the only score of the period. In the fourth quarter, the Jaguars drew closer as kicker Josh Scobee nailed a 47-yard field goal. However, the Colts responded with Manning completing a 1-yard TD pass to RB Luke Lawton. Afterwards, Jacksonville tried to come back as Garrard completed a 17-yard TD pass to WR Dennis Northcutt (along with getting the 2-point conversion run). Indianapolis' defense managed to seal the deal. With their season-sweep over the Jaguars, the Colts improved to 10-2. During the game, the Colts gave Garrard his first interception of the year, courtesy of Safety Antoine Bethea.

    Keywords: player, field goal
    Relevant Sentence: In the fourth quarter, the Jaguars drew closer as kicker Josh Scobee nailed a 47-yard field goal.

    Example 2:
    Question: Which star has a smaller mass, Nu Phoenicis or Gliese 915?
    Passage: Nu Phoenicis is a yellow-white main sequence star of spectral type F9V and magnitude 4.96. Lying some 49 light years distant, it is around 1.2 times as massive as our sun, and likely to be surrounded by a disk of dust. It is the closest star in the constellation that is visible with the unaided eye. Gliese 915 is a white dwarf only 26 light years away. It is of magnitude 13.05, too faint to be seen with the naked eye. White dwarfs are extremely dense stars compacted into a volume the size of the Earth. With around 85% of the mass of the Sun, Gliese 915 has a surface gravity of 108.39 ± 0.01 (2.45 · 108) centimetre·second−2, or approximately 250,000 of Earths gravity.

    Keywords: star, mass, Nu Phoenicis, Gliese 915
    Relevant Sentence: Gliese 915 has around 85% of the mass of the Sun.

    Example 3:
    Question: How many yards longer was the longest touchdown pass than the longest field goal?
    Passage: Hoping to rebound from their first loss of the season, the Broncos returned home for an AFC West divisional rematch with the Kansas City Chiefs. After Peyton Manning became the NFL's all-time leader in regular season passing yardage, the game turned sour for the Broncos. Following a Manning interception, the Chiefs capitalized, with a 4-yard touchdown run by running back Charcandrick West. The Broncos' offense went three-and-out on their next two possessions, and the Chiefs increased their lead to 10-0, with a 48-yard field goal by placekicker Cairo Santos. The Chiefs increased their lead to 19-0 at halftime, with three more field goals by Santos &#8212 from 49, 34 and 33 yards out. By halftime, Manning had thrown three interceptions and the Broncos' offense had earned only one first down. The Broncos went three-and-out on their first possession of the second half, and a 50-yarder field goal by Santos increased the Chiefs' lead to 22-0. After Manning threw his fourth interception of the game on the Broncos' next possession, he was pulled and replaced by backup quarterback Brock Osweiler for the remainder of the game. Osweiler drove the Broncos' into the red zone early in the fourth quarter, but was intercepted by Chiefs' safety Eric Berry. Two plays later, the Chiefs increased their lead to 29-0, when quarterback Alex Smith connected with West on an 80-yard touchdown pass. The Broncos' finally got on the scoreboard with 5:31 remaining in the game, with running back Ronnie Hillman rushing for a 1-yard touchdown (two-point conversion attempt unsuccessful), followed by a 7-yard touchdown pass from Osweiler to wide receiver Andre Caldwell, but the Chiefs' lead was too much for the Broncos to overcome. Peyton Manning finished the day with the first 0.0 passer rating of his career.

    Keywords: yards, longest touchdown pass, longest field goal
    Relevant Sentence: The Chiefs increased their lead to 29-0, when quarterback Alex Smith connected with West on an 80-yard touchdown pass. The Chiefs increased their lead to 10-0, with a 48-yard field goal by placekicker Cairo Santos.

    Question: {question}
    Passage: {corrected_answer}
    Keywords:
    Relevant Sentence:
    """

    try:
        keywords_and_sentence = call_llm(keywords_prompt, "You are an expert at extracting keywords and relevant sentences.")
        relevant_sentence = keywords_and_sentence.split("Relevant Sentence:")[1].strip()
    except:
        relevant_sentence = corrected_answer  # If keyword extraction fails, use the corrected answer

    # --- 4. Final Validation with Relevant Sentence (from Iteration 2 and 4, combined) ---
    final_validation_prompt = f"""
    Validate that the {corrected_answer} accurately answers the {question} based on the {relevant_sentence}. If it isn't, provide the correct answer.

    Example 1:
    Question: Which player kicked the only field goal of the game?
    Corrected Answer: Josh Scobee
    Relevant Sentence: In the fourth quarter, the Jaguars drew closer as kicker Josh Scobee nailed a 47-yard field goal.
    Final Answer: Josh Scobee

    Example 2:
    Question: Which star has a smaller mass, Nu Phoenicis or Gliese 915?
    Corrected Answer: Gliese 915
    Relevant Sentence: Gliese 915 has around 85% of the mass of the Sun.
    Final Answer: Gliese 915

    Example 3:
    Question: How many yards longer was the longest touchdown pass than the longest field goal?
    Corrected Answer: 32
    Relevant Sentence: The Chiefs increased their lead to 29-0, when quarterback Alex Smith connected with West on an 80-yard touchdown pass. The Chiefs increased their lead to 10-0, with a 48-yard field goal by placekicker Cairo Santos.
    Final Answer: 32

    Question: {question}
    Corrected Answer: {corrected_answer}
    Relevant Sentence: {relevant_sentence}
    Final Answer:
    """

    try:
        final_answer = call_llm(final_validation_prompt, "You are a final validator. Ensure conciseness and accuracy.").strip()
        return final_answer
    except Exception as e:
        print(f"Error during final validation: {e}")
        return corrected_answer # As a final safety