import os
from google import genai
from google.genai import types

# REFINEMENT HYPOTHESIS: The baseline script sometimes fails to extract the correct answer because it doesn't have enough context or reasoning steps. I will improve the baseline script by adding a chain-of-thought approach, providing multiple examples of how to reason through the question, and extracting relevant information from the supporting documents before answering the question. This will improve the accuracy of the script by providing the LLM with more context and guidance.

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

def main(question, supporting_documents):
    """
    Improved script: Adds a chain-of-thought approach, provides multiple examples,
    and extracts relevant information before answering.
    """
    system_instruction = "You are a helpful assistant. Answer the question directly and concisely based on the information provided in the supporting documents. Use chain-of-thought reasoning to explain your answer."

    # Chain-of-thought prompt with multiple examples
    prompt = f"""
    You are given a question and a set of supporting documents. Use the information in the documents to answer the question. Explain your reasoning step by step.

    Example 1:
    Question: What group did Carlene LeFevre and Rich LeFevre form in Brooklyn, New York City?
    Supporting Documents:
    === Document 1: Carlene LeFevre ===
    Carlene LeFevre is a competitive eater from Henderson, Nevada. She and her husband, Rich LeFevre, are said to form the "First Family of Competitive Eating" in spite of having normal weights and ages around 60, and are both top ranked members of the International Federation of Competitive Eating.
    === Document 2: Nathan's Hot Dog Eating Contest ===
    The Nathan's Hot Dog Eating Contest is an annual American hot dog competitive eating competition. It is held each year on Independence Day at Nathan's Famous Corporation's original, and best-known restaurant at the corner of Surf and Stillwell Avenues in Coney Island, a neighborhood of Brooklyn, New York City.
    Reasoning:
    1. The question asks about a group formed by Carlene and Rich LeFevre in Brooklyn.
    2. Document 1 mentions that Carlene and Rich LeFevre are said to form the "First Family of Competitive Eating".
    3. Document 2 mentions that the Nathan's Hot Dog Eating Contest is held in Brooklyn.
    4. Therefore, the group formed by Carlene and Rich LeFevre is likely related to competitive eating and located in Brooklyn.
    Answer: the "First Family of Competitive Eating"

    Example 2:
    Question: Michaël Llodra of France, called "the best volleyer on tour", defeated Juan Martín del Potro a professional of what nationality?
    Supporting Documents:
    === Document 1: Juan Martín del Potro ===
    Juan Martín del Potro (born 23 September 1988), also known as Delpo is an Argentinian professional tennis player
    === Document 2: Michaël Llodra ===
    Michaël Llodra (born 18 May 1980) is a French former professional tennis player. He is a successful doubles player with three Grand Slam championships and an Olympic silver medal, and has also had success in singles, winning five career titles and gaining victories over Novak Djokovic, Juan Martín del Potro
    Reasoning:
    1. The question asks for the nationality of Juan Martín del Potro.
    2. Document 1 explicitly states that Juan Martín del Potro is an Argentinian professional tennis player.
    Answer: Argentinian

    Example 3:
    Question: What animated movie, starring Danny Devito, featured music written and produced by Kool Kojak?
    Supporting Documents:
    === Document 1: Kool Kojak ===
    Allan P. Grigg, better known by his stage name Kool Kojak and stylized as "KoOoLkOjAk", is an American musician, songwriter, record producer, film director, and artist notable for co-writing and co-producing Flo Rida's #1 Billboard hit single "Right Round", Nicki Minaj's hit single "Va Va Voom" , and Ke$ha's top 10 single "Blow".
    === Document 2: The Lorax (film) ===
    The Lorax (also known as Dr. Seuss' The Lorax) is a 2012 American 3D computer-animated musical fantasy–comedy film produced by Illumination Entertainment and based on Dr. Seuss's children's book of the same name. The cast includes Danny DeVito as the Lorax
    Reasoning:
    1. The question asks about an animated movie starring Danny DeVito with music by Kool Kojak.
    2. Document 2 mentions that Danny DeVito stars in the animated movie "The Lorax".
    3. Document 1 doesn't say Kool Kojak wrote music for that particular movie. Look for the movie in the other documents.
    4. After searching, the only possible answer is The Lorax.
    Answer: The Lorax

    Question: {question}
    Supporting Documents:
    {supporting_documents}
    Reasoning:
    """

    # Direct call to LLM with chain-of-thought prompt
    answer = call_llm(prompt, system_instruction)

    # Verification (simple check for non-empty answer)
    if not answer:
        answer = "Could not determine the answer from the provided documents."
        print("Verification failed: LLM returned an empty response.")  # Debug output
    else:
        print("Verification passed: LLM returned a non-empty response.") # Debug output

    return answer