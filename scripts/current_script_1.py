import os

def call_llm(prompt, system_instruction=None):
    """Call the Gemini LLM with a prompt and return the response"""
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
    This script attempts to address multi-hop reasoning by:
    1. Summarizing each document independently to reduce information overload, then verifying the summarization.
    2. Reasoning across the summaries to find the answer.
    """

    # Hypothesis: Summarizing documents before reasoning will improve accuracy. This explores a document reduction strategy.
    # Addressing previous errors: Information Overload, Inability to Connect Disparate Facts
    # Verification goal: Verify the document summarization is both concise and accurate.
    
    summaries = []
    for i, doc in enumerate(supporting_documents):
        summary_result = summarize_document_with_verification(doc, i, question)
        if summary_result.get("is_valid"):
            summaries.append(summary_result["summary"])
        else:
            print(f"Error summarizing document {i}: {summary_result.get('validation_feedback')}")
            return f"Error summarizing document {i}"
    
    # Now, reason across the summaries to answer the question
    answer = reason_across_summaries(question, summaries)
    return answer

def summarize_document_with_verification(document, doc_id, question, max_attempts=3):
    """Summarizes a document and verifies the summary."""
    system_instruction = "You are an expert summarizer who creates concise, accurate summaries."
    
    #Attempt summarization, then verify
    for attempt in range(max_attempts):
        summary_prompt = f"""
        Summarize this document, focusing on information relevant to this question: {question}.
        Be concise and retain all critical information related to the question.
        
        Document: {document}
        
        Example 1:
        Document: The capital of Australia is Canberra. Canberra is located in the Australian Capital Territory.
        Summary: The capital of Australia is Canberra, located in the Australian Capital Territory.
        
        Example 2:
        Document:  Tommy's Honour is a 2016 historical drama film depicting the lives and careers of, and the complex relationship between, the pioneering Scottish golfing champions Old Tom Morris and his son Young Tom Morris.
        Summary: Tommy's Honour is a 2016 film about Scottish golfers Old Tom Morris and his son.

        Summary:
        """
        
        summary = call_llm(summary_prompt, system_instruction)
        
        # Verify the summary - does it retain relevant information?
        verification_prompt = f"""
        Verify that this summary of document {doc_id} retains all information relevant to the question: {question}.
        If not, explain what is missing.
        
        Document: {document}
        Summary: {summary}
        
        Respond with "VALID" if the summary is valid, or "INVALID: [reason]" if not.

        Example 1:
        Document: The Prime Minister of the UK is Rishi Sunak, who assumed office in 2022.
        Summary: Rishi Sunak is the UK Prime Minister.
        Verification: VALID
        
        Example 2:
        Document: Tommy's Honour is a film about Old Tom Morris and his son. Jack Lowden starred in it.
        Summary: Tommy's Honour is a film about Old Tom Morris.
        Verification: INVALID: The summary is missing the fact that Jack Lowden starred in it.
        """
        
        verification_result = call_llm(verification_prompt, system_instruction)
        
        if "VALID" in verification_result:
            return {"is_valid": True, "summary": summary}
        else:
            print(f"Summary verification failed for doc {doc_id}, attempt {attempt+1}: {verification_result}")
            if attempt < max_attempts-1:
                continue
            else:
                return {"is_valid": False, "summary": summary, "validation_feedback": verification_result}
    return {"is_valid": False, "summary": "", "validation_feedback": "Failed to generate a valid summary after multiple attempts."}

def reason_across_summaries(question, summaries):
    """Reasons across the summaries to answer the question."""
    system_instruction = "You are an expert at answering questions based on summaries of documents."
    
    reasoning_prompt = f"""
    Based on these summaries, answer the question: {question}. Synthesize information from multiple summaries if needed.
    
    Summaries:
    {summaries}

    Example 1:
    Question: What is the capital of Australia?
    Summaries: ['The capital of Australia is Canberra.']
    Answer: Canberra

    Example 2:
    Question: Tommy's Honour was a film that starred who?
    Summaries: ["Tommy's Honour is a 2016 film about Scottish golfers Old Tom Morris and his son.", "Jack Lowden starred in War & Peace"]
    Answer: Jack Lowden

    Answer:
    """
    
    answer = call_llm(reasoning_prompt, system_instruction)
    return answer

# Example usage (replace with actual data and document splitting)
if __name__ == "__main__":
    question = "Tommy's Honour was a drama film that included the actor who found success with what 2016 BBC miniseries?"
    supporting_documents = [
        "Tommy's Honour is a 2016 historical drama film depicting the lives and careers of, and the complex relationship between, the pioneering Scottish golfing champions Old Tom Morris and his son Young Tom Morris. The film is directed by Jason Connery, and the father and son are portrayed by Peter Mullan and Jack Lowden. The film won Best Feature Film at the 2016 British Academy Scotland Awards.",
        "Jack Andrew Lowden (born 2 June 1990) is a Scottish stage, television, and film actor. Following a highly successful and award-winning four-year stage career, his first major international onscreen success was in the 2016 BBC miniseries War & Peace, which led to starring roles in feature films."
    ]

    answer = main(question, supporting_documents)
    print(f"Answer: {answer}")