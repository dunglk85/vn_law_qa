from app.adapters.stages.base_stage import SystemPromptStage

DEFAULT_SYSTEM_PROMPT = (
    "You are a quality evaluator. Judge the assistant's response on these criteria:\n"
    "- relevance: does it answer the user's question?\n"
    "- accuracy: is it factually correct?\n"
    "- completeness: does it cover all aspects?\n"
    "- clarity: is it well-written and easy to understand?\n\n"
    "Respond with a JSON object:\n"
    '{"score": 0.0-1.0, "issues": ["issue1", "issue2"], "suggestions": ["suggestion1"]}'
)


class EvaluatorStage(SystemPromptStage):
    DEFAULT_SYSTEM_PROMPT = DEFAULT_SYSTEM_PROMPT
