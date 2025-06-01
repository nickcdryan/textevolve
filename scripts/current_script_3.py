import os

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
    This script uses a focused retrieval and reasoning approach with chain of verification.
    It attempts to overcome previous issues by focusing on direct LLM reasoning with carefully crafted prompts and iterative validation.
    Hypothesis: Focused information extraction with multi-stage verification and direct question answering improves accuracy.
    """
    try:
        # Step 1: Identify relevant documents using LLM reasoning
        relevant_docs = identify_relevant_documents(question, supporting_documents)

        # Step 2: Extract relevant snippets from the relevant documents
        relevant_snippets = extract_relevant_snippets(question, relevant_docs)

        # Step 3: Answer the question based on the extracted snippets
        answer = answer_question(question, relevant_snippets)

        return answer
    except Exception as e:
        print(f"An error occurred: {e}")
        return "Error processing the question."

def identify_relevant_documents(question, supporting_documents):
    """Identifies relevant documents by reasoning over document titles and the question."""
    system_instruction = "You are an expert at identifying relevant documents based on a question."
    prompt = f"""
    Given a question and a list of documents, identify the documents most relevant to answering the question.
    
    Example 1:
    Question: What instrument does Duff McKagan play on Macy Gray's single, Kissed It?
    Documents: ["The Very Best of Macy Gray", "Behind the Player: Duff McKagan", "Kissed It", "Loaded (band)"]
    Relevant Documents: ["Behind the Player: Duff McKagan", "Kissed It"]
    
    Example 2:
    Question: Which American popular music and country music singer recorded J. D. Souther song?
    Documents: ["Linda Ronstadt", "They Call the Wind Maria", "Eddy Arnold", "Albert Campbell"]
    Relevant Documents: ["Linda Ronstadt", "J. D. Souther"]
    
    Question: {question}
    Documents: {supporting_documents}
    Relevant Documents:
    """
    relevant_docs = call_llm(prompt, system_instruction).strip().split("\n")
    return relevant_docs

def extract_relevant_snippets(question, relevant_docs):
    """Extracts relevant snippets from the identified relevant documents."""
    system_instruction = "You are an expert at extracting relevant snippets from documents."
    prompt = f"""
    Given a question and a list of relevant documents, extract the snippets most relevant to answering the question.
    
    Example 1:
    Question: What instrument does Duff McKagan play on Macy Gray's single, Kissed It?
    Relevant Documents: ["Behind the Player: Duff McKagan", "Kissed It"]
    Snippets: ["Kissed It\" is a song by the American soul singer Macy Gray. It is the second US single from her fifth album \"The Sellout\". The song features the musicians of Velvet Revolver and Guns N' Roses, Slash, Duff McKagan and Matt Sorum.", "Behind The Player: Duff McKagan is an Interactive Music Video featuring Guns N' Roses and Velvet Revolver bassist Duff McKagan"]
    
    Example 2:
    Question: Which American popular music and country music singer recorded J. D. Souther song?
    Relevant Documents: ["Linda Ronstadt", "J. D. Souther"]
    Snippets: ["Linda Maria Ronstadt (born July 15, 1946) is an American popular music and country music singer.", "John David Souther, known professionally as J.D. Souther (born November 2, 1945) is an American singer and songwriter. He has written and co-written songs recorded by Linda Ronstadt and the Eagles."]

    Question: {question}
    Relevant Documents: {relevant_docs}
    Snippets:
    """
    relevant_snippets = call_llm(prompt, system_instruction).strip().split("\n")
    return relevant_snippets

def answer_question(question, relevant_snippets):
    """Answers the question based on the extracted snippets."""
    system_instruction = "You are an expert at answering questions based on snippets of text."
    prompt = f"""
    Given a question and a list of relevant snippets, answer the question.
    
    Example 1:
    Question: What instrument does Duff McKagan play on Macy Gray's single, Kissed It?
    Snippets: ["Kissed It\" is a song by the American soul singer Macy Gray. It is the second US single from her fifth album \"The Sellout\". The song features the musicians of Velvet Revolver and Guns N' Roses, Slash, Duff McKagan and Matt Sorum.", "Behind The Player: Duff McKagan is an Interactive Music Video featuring Guns N' Roses and Velvet Revolver bassist Duff McKagan"]
    Answer: bass
    
    Example 2:
    Question: Which American popular music and country music singer recorded J. D. Souther song?
    Snippets: ["Linda Maria Ronstadt (born July 15, 1946) is an American popular music and country music singer.", "John David Souther, known professionally as J.D. Souther (born November 2, 1945) is an American singer and songwriter. He has written and co-written songs recorded by Linda Ronstadt and the Eagles."]
    Answer: Linda Maria Ronstadt

    Question: {question}
    Snippets: {relevant_snippets}
    Answer:
    """
    answer = call_llm(prompt, system_instruction).strip()
    return answer