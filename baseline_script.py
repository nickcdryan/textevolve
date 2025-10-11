from system_tools import call_llm

def main(question):
    """
    Baseline script - Direct LLM inference with no special techniques.
    
    This script serves as a baseline for comparison against more sophisticated scripts.
    It simply passes the question directly to the inference model without any:
    - Parsing or preprocessing
    - Multi-step reasoning
    - Verification loops
    - Deterministic rules
    - Special prompting strategies
    
    Use this to establish baseline performance before applying any techniques.
    """
    # Direct call to LLM with minimal instruction
    system_instruction = "You are a helpful assistant. Answer the question accurately."
    
    answer = call_llm(question, system_instruction=system_instruction)
    
    return answer
