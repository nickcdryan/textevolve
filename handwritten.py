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
    system_instruction = "You are a helpful assistant. Answer the question accurately. You are an expert at identifying and applying grid transformation patterns. Focus on identifying high-level transformation meta-patterns before applying specific rules. It might involve rotation, replication, reflection, movement of different parts, repetition, filtering, number counting or majority rule, number addition or averaging, object extraction or alignment, object bounding boxes, object replacement, symmetry completion, border creation, hole filling, masking and overlay, conditional transformations, etc."
    
    answer = call_llm(question, system_instruction=system_instruction)
    
    return answer


