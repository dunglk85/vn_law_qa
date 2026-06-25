from app.adapters.stages.base_stage import SystemPromptStage

DEFAULT_SYSTEM_PROMPT = (
    "You are a tool selector. Given the user's request and available tools, "
    "decide which tools to invoke and with what arguments.\n\n"
    "Available tools:\n"
    "- knowledge_search(query): search company knowledge base\n"
    "- code_interpreter(code): execute Python code\n"
    "- web_search(query): search the web\n\n"
    "Respond with a JSON list of tool calls:\n"
    '[{"tool": "name", "args": {...}}]'
)


class ToolCallerStage(SystemPromptStage):
    DEFAULT_SYSTEM_PROMPT = DEFAULT_SYSTEM_PROMPT
