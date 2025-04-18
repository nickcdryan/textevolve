import google.generativeai as genai
import os

# Replace with your actual Gemini API key
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

model = genai.GenerativeModel('gemini-pro')


def call_llm(prompt, model=model):
    """
    Calls the Gemini LLM with the given prompt and returns the response.
    Handles potential errors during the API call.
    """
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"Error calling LLM: {e}")
        return None


def extract_answer_from_solution(solution):
    """
    Extracts the final answer from a detailed solution string, handling potential errors.
    """
    try:
        # LLM call to extract answer
        prompt = f"""
        Extract the final answer from the following solution:

        Solution:
        {solution}

        Example:
        Solution: The cost of apples is $3.60 and the cost of oranges is $1.60 for a total of $5.20. The change from $10 is $4.80.
        Answer: $4.80

        Now extract the answer from:
        {solution}
        """

        answer = call_llm(prompt)
        return answer
    except Exception as e:
        print(f"Error extracting answer: {e}")
        return "Could not extract answer."


def solve_question(question):
    """
    Solves a question by breaking it down into smaller reasoning steps using LLM calls.
    """
    try:
        # Step 1: Understand the question
        understanding_prompt = f"""
        Understand the question and identify the key information needed to answer it.

        Question:
        {question}

        Example:
        Question: John has 5 apples and buys 3 more. How many does he have now?
        Key information: John initially has 5 apples. John buys 3 more apples.
        The question asks for the total number of apples.
        """
        understanding = call_llm(understanding_prompt)

        # Step 2: Devise a plan
        plan_prompt = f"""
        Devise a plan to answer the question, given the following understanding:

        Understanding:
        {understanding}

        Question:
        {question}

        Example:
        Understanding: John initially has 5 apples. John buys 3 more apples. The question asks for the total number of apples.
        Plan: Add the initial number of apples (5) to the number of apples John buys (3) to find the total number of apples.
        """
        plan = call_llm(plan_prompt)

        # Step 3: Execute the plan
        execution_prompt = f"""
        Execute the plan and provide the final answer.

        Plan:
        {plan}

        Question:
        {question}

        Example:
        Plan: Add the initial number of apples (5) to the number of apples John buys (3) to find the total number of apples.
        Answer: 5 + 3 = 8. John has 8 apples.
        """
        execution = call_llm(execution_prompt)

        # Step 4: Extract the final answer from the solution
        final_answer = extract_answer_from_solution(execution)
        return final_answer

    except Exception as e:
        print(f"An error occurred: {e}")
        return "Could not solve the question."


def main(question):
    """
    Main function to solve the question using LLM.
    """
    return solve_question(question)


if __name__ == "__main__":
    example_question = "What is the capital of France?"
    answer = main(example_question)
    print(f"Question: {example_question}\nAnswer: {answer}")