import google.generativeai as genai
import os

# Replace with your actual Gemini API key or use environment variable
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

def call_llm(prompt, model_name="gemini-1.5-flash-002"):
    """Calls the LLM with error handling."""
    try:
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error calling LLM: {str(e)}"

def main(question):
    """Main function to answer questions using LLM reasoning."""

    # Step 1: Break down the question into sub-questions
    breakdown_prompt = f"""
    Example:
    Question: What were the main causes of World War II?
    Breakdown:
    1. What were the political tensions in Europe before World War II?
    2. What were the economic factors contributing to the war?
    3. What were the key events that led to the outbreak of the war?

    Question: {question}
    Breakdown:
    """
    sub_questions = call_llm(breakdown_prompt)

    # Step 2: Answer each sub-question using LLM
    answers = []
    for sub_q in sub_questions.split("\n"):
        if not sub_q.strip():
            continue
        answer_prompt = f"""
        Example:
        Question: What were the political tensions in Europe before World War II?
        Answer: The Treaty of Versailles imposed harsh terms on Germany, leading to resentment and political instability.

        Question: {sub_q}
        Answer:
        """
        answer = call_llm(answer_prompt)
        answers.append(answer)

    # Step 3: Synthesize the answers into a final response
    synthesis_prompt = f"""
    Example:
    Sub-questions:
    1. What were the political tensions in Europe before World War II?
    2. What were the economic factors contributing to the war?
    3. What were the key events that led to the outbreak of the war?
    Answers:
    1. The Treaty of Versailles imposed harsh terms on Germany...
    2. The Great Depression created economic hardship...
    3. The invasion of Poland by Germany triggered declarations of war...
    Synthesis: World War II was caused by a combination of political tensions, economic factors, and aggressive actions...

    Sub-questions: {sub_questions}
    Answers: {answers}
    Synthesis:
    """
    final_answer = call_llm(synthesis_prompt)

    return final_answer

if __name__ == "__main__":
    question = "Explain the process of photosynthesis."
    answer = main(question)
    print(f"Question: {question}")
    print(f"Answer: {answer}")