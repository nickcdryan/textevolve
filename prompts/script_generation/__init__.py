from .strategies import get_explore_instructions
from .strategies import get_refine_instructions
from .strategies import get_exploit_instructions

from .prompting_guides import (
    multi_example_prompting_guide,
    llm_reasoning_prompting_guide,
    validation_prompting_guide,
    meta_programming_prompting_guide,
    code_execution_prompting_guide,
)

from .llm_patterns import(
    as_example_code,
    extract_information_with_examples,
    verify_solution_with_examples,
    solve_with_validation_loop,
    best_of_n,
    solve_with_react_pattern,
    
    chain_of_thought_reasoning,
    verification_with_feedback,
    multi_perspective_analysis,
    self_consistency_approach,
    pattern_identification,
    wait_injection,
    solve_with_meta_programming,
    self_modifying_solver,
    debate_approach,
    adaptive_chain_solver,
    dynamic_memory_pattern,
    test_time_training,
    combination_example,
)
