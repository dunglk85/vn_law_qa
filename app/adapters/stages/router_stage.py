from app.adapters.stages.base_stage import SystemPromptStage

DEFAULT_SYSTEM_PROMPT = (
    "You are a query router. Classify the user's question into exactly one category.\n"
    "Respond with ONLY the category label, nothing else.\n\n"
    "Categories: [general, technical, policy, legal, code]"
)


class RouterStage(SystemPromptStage):
    DEFAULT_SYSTEM_PROMPT = DEFAULT_SYSTEM_PROMPT
