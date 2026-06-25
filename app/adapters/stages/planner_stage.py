from app.adapters.stages.base_stage import SystemPromptStage

DEFAULT_SYSTEM_PROMPT = (
    "You are a task planner. Decompose the user's request into a sequence of "
    "concrete, executable sub-tasks. Return a numbered list of tasks.\n"
    "Format:\n"
    "1. task description\n"
    "2. task description\n"
    "..."
)


class PlannerStage(SystemPromptStage):
    DEFAULT_SYSTEM_PROMPT = DEFAULT_SYSTEM_PROMPT
