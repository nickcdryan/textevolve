# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Environment

This project uses `uv` for dependency management. Use `uv run` to execute Python scripts:

```bash
# Install dependencies
uv sync

# Run main system
uv run python run_script.py --dataset hendrycks_math/math_test.jsonl --loader math --iterations 5

# Validate results
uv run python validate_script.py --script scripts/script_iteration_4.py --dataset hendrycks_math/math_test.jsonl --loader math --start 100 --end 199

# Reset system (wipes memory)
uv run python reset_system.py

# Verify setup
uv run python verify_setup.py
```

## Core Architecture

TextEvolve is an LLM-driven system that iteratively improves solutions to datasets through exploration, exploitation, and refinement strategies.

### Key Components

- **AgentSystem** (`agent_system.py`): Main orchestrator that runs iterations and manages the learning loop
- **DatasetLoader** (`dataset_loader.py`): Universal interface for loading various dataset formats with standard field mapping
- **SystemTools** (`system_tools.py`): Injectable functions for generated scripts (LLM calls, database access, file operations, code execution)
- **Evaluators** (`evaluators.py`): Different evaluation methods (LLM, F1, exact match, custom like TicketWorld)
- **Sandbox** (`sandbox.py`): Docker-based code execution environment for safety
- **Prompts** (`prompts/`): Modular prompt system for different aspects of the learning process

### Generated Scripts

The system generates Python scripts in `scripts/` that contain advanced agentic patterns. These scripts:
- Use functions from `system_tools.py` (injected at runtime)
- Are executed in Docker sandbox for safety
- Implement patterns like ReAct, chain-of-thought, verification loops
- Are progressively tested and refined based on performance

### Dataset Support

Built-in loaders: `arc`, `jsonl`, `json`, `math`, `gpqa`, `hotpotqa`, `simpleqa`, `natural_plan`, `custom`

All datasets are normalized to standard fields: `question`, `answer`, `id`

## Key Files to Understand

- `run_script.py`: Main entry point for running the system
- `agent_system.py:50-100`: Core iteration logic and strategy selection
- `dataset_loader.py:16-50`: Base loader interface and field normalization
- `system_tools.py:10-30`: Core system functions that get injected into generated scripts
- `prompts/script_generation/strategies.py`: Strategy-specific prompt generation
- `evaluators.py`: Different evaluation methods for measuring script performance

## Testing

- `test_sandbox.py`: Tests Docker sandbox functionality
- `test_mcp_script.py`: Tests MCP (Model Context Protocol) integration
- No formal test suite - system validates through iterative performance measurement

## Environment Variables

- `GEMINI_API_KEY`: Required for LLM calls (uses Gemini 2.0 Flash)

## Data Flow

1. Dataset loaded and normalized via `DatasetLoader`
2. `AgentSystem` runs iterations with explore/exploit/refine strategies
3. Generated scripts executed in Docker sandbox
4. Performance measured via evaluators
5. Results stored in `archive/` and `learnings.txt` for future iterations
6. Best scripts saved in `scripts/` directory

## Important Patterns

- All generated scripts must use the system_tools functions (call_llm, execute_code, etc.)
- Scripts are validated through AST parsing before execution
- Progressive testing balances speed vs accuracy measurement
- Memory system tracks learnings across iterations in structured format