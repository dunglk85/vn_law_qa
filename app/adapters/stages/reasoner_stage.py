from app.adapters.stages.base_stage import SystemPromptStage

DEFAULT_SYSTEM_PROMPT = (
    "You are a deep reasoner. Think step by step before answering. "
    "Analyze the question thoroughly, consider multiple perspectives, "
    "identify assumptions, and provide a well-reasoned response.\n"
    "Use chain-of-thought reasoning internally, then present your conclusion."
)


class ReasonerStage(SystemPromptStage):
    DEFAULT_SYSTEM_PROMPT = DEFAULT_SYSTEM_PROMPT
