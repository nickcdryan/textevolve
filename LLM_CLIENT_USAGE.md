# LLM Client System - Usage Guide

## Overview

The codebase now has a modular LLM client architecture that cleanly separates:
- **Orchestrator LLM**: Used for meta-level operations (generating feedback, learnings, scripts)
- **Inference LLM**: Used for script execution and problem-solving

## Quick Start

### Using CLI Arguments (Recommended)

```bash
# Set both orchestrator and inference models
python run_script.py \
  --dataset datasets/simpleqa/simpleqa_train_1k.jsonl \
  --loader simpleqa \
  --iterations 5 \
  --orchestrator-llm gemini-2.5-flash \
  --inference-llm gemini-2.0-flash

# Validate a script with specific models
python validate_script.py \
  --script scripts/script_iteration_14.py \
  --dataset datasets/simpleqa/simpleqa_train_1k.jsonl \
  --loader simpleqa \
  --start 0 \
  --end 99 \
  --orchestrator-llm gemini-1.5-pro \
  --inference-llm gemini-2.0-flash
```

### Using Environment Variables

```bash
# Set via environment variables
export ORCHESTRATOR_LLM_MODEL="gemini-2.5-flash"
export INFERENCE_LLM_MODEL="gemini-2.0-flash"
export GEMINI_API_KEY="your_api_key"

python run_script.py --dataset datasets/simpleqa/simpleqa_train_1k.jsonl --loader simpleqa
```

### Priority Order

The system resolves LLM models in this order:
1. CLI arguments (`--orchestrator-llm`, `--inference-llm`)
2. Environment variables (`ORCHESTRATOR_LLM_MODEL`, `INFERENCE_LLM_MODEL`)
3. Default values (orchestrator: `gemini-2.5-flash`, inference: `gemini-2.0-flash`)

## Architecture

### Core Components

1. **`llm_client.py`** - Abstract base class and implementations
   - `LLMClient` (ABC)
   - `GeminiClient` (implemented)
   - `ClaudeClient` (placeholder)
   - `OpenAIClient` (placeholder)
   - `LLMClientFactory` (creates clients)

2. **Orchestrator LLM** - Used in:
   - `agent_system.py` - Main agent system
   - `system_improver.py` - System improvement
   - `script_flow_graph.py` - Flow analysis

3. **Inference LLM** - Used in:
   - `system_tools.py` - Injected into generated scripts
   - All generated scripts via `call_llm()` function

### How It Works

```python
# In agent_system.py (orchestrator)
self.orchestrator_llm = LLMClientFactory.create_orchestrator_client(
    model=orchestrator_llm_model
)
response = self.orchestrator_llm.generate(
    prompt=prompt,
    system_instruction=system_instruction
)

# In system_tools.py (inference)
_inference_llm_client = LLMClientFactory.create_inference_client()
def call_llm(prompt, system_instruction=None):
    client = _get_inference_client()
    return client.generate(prompt, system_instruction)
```

## Available Models

### Gemini Models
- `gemini-2.5-flash` - Fast, high-quality (default orchestrator)
- `gemini-2.0-flash` - Fast (default inference)
- `gemini-1.5-pro` - More powerful
- `gemini-1.5-flash` - Balanced

### Future Support
- Claude: `claude-3-5-sonnet-20241022`, etc.
- OpenAI: `gpt-4`, `gpt-4-turbo`, etc.

## Adding New Providers

To add a new LLM provider (e.g., Claude):

1. Implement the `LLMClient` ABC in `llm_client.py`:

```python
class ClaudeClient(LLMClient):
    def __init__(self, model: str = "claude-3-5-sonnet-20241022", **config):
        super().__init__(model, **config)
        # Initialize Claude client
        
    def generate(self, prompt: str, system_instruction: Optional[str] = None, **kwargs) -> str:
        # Implement Claude API call
        pass
        
    def get_model_name(self) -> str:
        return self.model
```

2. Register in `LLMClientFactory.PROVIDERS`:

```python
PROVIDERS = {
    "gemini": GeminiClient,
    "claude": ClaudeClient,  # Add here
    "openai": OpenAIClient,
}
```

3. Use it:

```bash
export ORCHESTRATOR_LLM_PROVIDER="claude"
export ORCHESTRATOR_LLM_MODEL="claude-3-5-sonnet-20241022"
export ANTHROPIC_API_KEY="your_key"
```

## Benefits

✅ **Clean Separation**: Orchestrator vs inference LLMs clearly separated
✅ **Easy Switching**: Change models via CLI without code changes
✅ **Consistent Interface**: All LLM calls use the same pattern
✅ **Extensible**: Adding new providers is straightforward
✅ **Testable**: Can mock the client for testing
✅ **Configuration Flexibility**: CLI args, env vars, or defaults

## Examples

### Use a more powerful orchestrator for better script generation:
```bash
python run_script.py \
  --orchestrator-llm gemini-1.5-pro \
  --inference-llm gemini-2.0-flash \
  --dataset datasets/medmcqa/medmcqa_train_500.jsonl \
  --loader medmcqa
```

### Use faster models for quick experiments:
```bash
python run_script.py \
  --orchestrator-llm gemini-2.0-flash \
  --inference-llm gemini-1.5-flash \
  --dataset datasets/simpleqa/simpleqa_train_1k.jsonl \
  --loader simpleqa
```

### Same models for both:
```bash
python run_script.py \
  --orchestrator-llm gemini-2.5-flash \
  --inference-llm gemini-2.5-flash \
  --dataset datasets/ticketworld/ticketworld.json \
  --loader ticketworld
```

## Files Modified

- ✅ `llm_client.py` (new)
- ✅ `agent_system.py`
- ✅ `system_improver.py`
- ✅ `system_tools.py`
- ✅ `script_flow_graph.py`
- ✅ `run_script.py`
- ✅ `validate_script.py`
- ✅ `prompts/script_generation/llm_api_call_library.py`
- ✅ `system_prompt.md`

## Troubleshooting

### Error: "GEMINI_API_KEY not found"
Make sure you've set your API key:
```bash
export GEMINI_API_KEY="your_api_key_here"
```

### Error: "Unknown provider"
Check that the provider name is correct (gemini, claude, openai).

### Different model not being used
Check the priority order - CLI args override env vars, which override defaults.

## Notes

- Generated scripts don't need to know about the client abstraction
- The `call_llm()` function signature remains unchanged for backward compatibility
- Model configuration happens at the script runner level, not in individual scripts

