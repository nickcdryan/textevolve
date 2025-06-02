import os
import re
import math
import json

# HYPOTHESIS: Improve the answer validation by adding examples that explicitly demonstrate the criteria for a VALID response,
# emphasizing conciseness and directness. This will address the weakness of the model providing verbose answers.

import openai
from openai import OpenAI

def call_llm(prompt, system_instruction=None):


    # Set your API key (keep this safe and secure)
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    # Call the chat completion endpoint
    response = client.responses.create(
        model="gpt-4o-mini-2024-07-18",
        input=[
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": prompt}
        ]
    )

    # Print the response content
    print(response.output[0].content[0].text)
    return response.output[0].content[0].text

def main(question):
    """
    Solve problems using a dynamic retrieval-augmented generation approach with adaptive prompt engineering.
    The system analyzes the question, designs a retrieval prompt, retrieves relevant documents, and constructs a final answer prompt.
    """

    # HYPOTHESIS: Explicitly engineering the retrieval prompt based on question characteristics and
    # dynamically selecting examples for prompt augmentation will enhance retrieval accuracy and, therefore, answer quality.
    # We will test this by analyzing the question characteristics, designing retrieval prompts based on those characteristics,
    # and then using the retrieved content to build the final answer. A validation step will then evaluate answer quality based on the extracted entities and relationships.
    
    # Step 1: Analyze the question and determine key characteristics.
    question_analysis_prompt = f"""
    Analyze the following question and identify:
    1. The type of question (e.g., Who, What, When, Where, How).
    2. Key entities mentioned in the question.
    3. Any specific constraints or requirements.
    4. The primary domain/topic of the question.

    Example 1:
    Question: What animated movie, starring Danny Devito, featured music written and produced by Kool Kojak?
    Analysis:
    {{
      "question_type": "What",
      "entities": ["Danny Devito", "Kool Kojak", "animated movie"],
      "constraints": ["featured music written and produced by Kool Kojak", "starring Danny Devito"],
      "domain": "Movies"
    }}

    Example 2:
    Question: Out of the actors who have played the role of Luc Deveraux in the Universal Soldier franchise, which actor has also starred in the movies Holby City, Doctor Strange, the Bourne Ultimatum and Zero Dark Thirty?
    Analysis:
    {{
      "question_type": "Which",
      "entities": ["Luc Deveraux", "Universal Soldier", "Holby City", "Doctor Strange", "The Bourne Ultimatum", "Zero Dark Thirty"],
      "constraints": ["actors who have played Luc Deveraux", "also starred in Holby City, Doctor Strange, The Bourne Ultimatum and Zero Dark Thirty"],
      "domain": "Movies"
    }}

    Question: {question}
    Analysis:
    """
    question_analysis = call_llm(question_analysis_prompt, "You are an expert question analyzer.")
    
    try:
        analysis_data = json.loads(question_analysis)
        question_type = analysis_data.get("question_type", "Unknown")
        entities = analysis_data.get("entities", [])
        constraints = analysis_data.get("constraints", [])
        domain = analysis_data.get("domain", "General")
    except Exception as e:
        question_type = "Unknown"
        entities = []
        constraints = []
        domain = "General"
    
    # Step 2: Design a retrieval prompt based on question characteristics.
    retrieval_prompt_design_prompt = f"""
    Based on the question type, entities, constraints, and domain, design an effective retrieval prompt
    to find relevant documents to answer the question.
    Question Type: {question_type}
    Entities: {entities}
    Constraints: {constraints}
    Domain: {domain}

    Example 1:
    Question Type: What
    Entities: ["Danny Devito", "Kool Kojak", "animated movie"]
    Constraints: ["featured music written and produced by Kool Kojak", "starring Danny Devito"]
    Domain: Movies
    Retrieval Prompt: "animated movie starring Danny DeVito music Kool Kojak"

    Example 2:
    Question Type: Which
    Entities: ["Luc Deveraux", "Universal Soldier", "Holby City", "Doctor Strange", "The Bourne Ultimatum", "Zero Dark Thirty"]
    Constraints: ["actors who have played Luc Deveraux", "also starred in Holby City, Doctor Strange, The Bourne Ultimatum and Zero Dark Thirty"]
    Domain: Movies
    Retrieval Prompt: "actor Luc Deveraux Universal Soldier Holby City Doctor Strange The Bourne Ultimatum Zero Dark Thirty"

    Now, design the retrieval prompt for the question: {question}
    """
    retrieval_prompt = call_llm(retrieval_prompt_design_prompt, "You are an expert retrieval prompt designer.")
    
    # Step 3: Perform retrieval (simulated here by using the existing supporting documents).
    retrieved_documents = retrieval_prompt # In a real system, this would call a document retrieval system, but we just use the prompt for the task
    
    # Step 4: Construct final answer prompt.
    final_answer_prompt = f"""
    Using the following retrieved documents, answer the question: {question}
    Retrieved Documents: {retrieved_documents}

    Example 1:
    Question: What animated movie, starring Danny Devito, featured music written and produced by Kool Kojak?
    Retrieved Documents: [Relevant documents about The Lorax, Danny DeVito, and Kool Kojak]
    Answer: The Lorax

    Example 2:
    Question: Out of the actors who have played the role of Luc Deveraux in the Universal Soldier franchise, which actor has also starred in the movies Holby City, Doctor Strange, the Bourne Ultimatum and Zero Dark Thirty?
    Retrieved Documents: [Relevant documents about Luc Deveraux, Universal Soldier, Holby City, Doctor Strange, The Bourne Ultimatum and Zero Dark Thirty, Scott Adkins, Jean Claude Van Damme]
    Answer: Scott Adkins

    Now, answer the question: {question}
    """

    answer = call_llm(final_answer_prompt, "You are an expert at answering questions using provided documents.")
    
    # Step 5: Validate the answer
    validation_prompt = f"""
    Problem: {question}
    Proposed Answer: {answer}
    Analyze the problem, propose a solution yourself, then check if the proposed answer is correct, comprehensive, and logically consistent with the initial ask. The answer should be CONCISE and DIRECT, providing ONLY the information requested, nothing more.

    Example 1:
    Problem: What is the capital of France?
    Proposed Answer: The capital of France is Paris.
    Evaluation: VALID

    Example 2:
    Problem: What is the capital of France?
    Proposed Answer: France is a country in Europe. The capital of France is Paris, which is a major city.
    Evaluation: INVALID: The answer is too verbose. It should only state the capital.

    Example 3:
    Problem: Who directed the movie E.T.?
    Proposed Answer: Steven Spielberg
    Evaluation: VALID

    Example 4:
    Problem: Who directed the movie E.T.?
    Proposed Answer: The movie E.T. was directed by Steven Spielberg.
    Evaluation: INVALID: The answer is too verbose. It should only state the director's name.

    Respond only with the following:
        - VALID: If the provided answer satisfies the criteria
        - INVALID: If the provided answer doesn't satisfy the criteria. Indicate why not by providing a feedback description
    """

    validation_result = call_llm(validation_prompt, "You are an expert answer validator.")

    if "VALID" in validation_result:
        concise_answer = call_llm(f"The following answer has been verified as correct. Your job is to extract or consolidate the part of the answer that directly and concisely answers the stated question. Please do not attempt to change the answer, you are only here to make possibly long answers short, concise, and directly in response to the question. The answer should only be at most a few words, typically a name, date, place, etc. Think step by step to make sure you give a direct answer to the question. ONLY return the answer. Here is the ANSWER: {answer}. And here is the QUESTION: {question}", "You are an expert answer conciser.")
        return concise_answer
    else:
        return "Could not determine the final answer"