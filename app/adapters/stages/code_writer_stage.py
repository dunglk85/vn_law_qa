from app.adapters.stages.base_stage import SystemPromptStage

DEFAULT_SYSTEM_PROMPT = (
    "You are a code generation specialist. Write clean, production-quality code "
    "in response to the user's request. Include type hints, handle errors, "
    "and add a brief usage example if applicable.\n"
    "Respond with the code in a language-appropriate code block."
)


class CodeWriterStage(SystemPromptStage):
    DEFAULT_SYSTEM_PROMPT = DEFAULT_SYSTEM_PROMPT
