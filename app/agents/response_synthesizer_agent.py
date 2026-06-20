"""ResponseSynthesizerAgent — generates grounded Vietnamese legal responses."""
import logging
from typing import Optional

from langchain_openai import ChatOpenAI

from shared import Citation, format_citations

logger = logging.getLogger(__name__)


class ResponseSynthesizerAgent:
    def __init__(self, llm: Optional[ChatOpenAI] = None):
        self.llm = llm or ChatOpenAI(model="gpt-4o", temperature=0)

    async def synthesize(self, query: str, citations: list[Citation]) -> dict:
        prompt = (
            "Bạn là trợ lý tư vấn pháp luật Việt Nam. "
            "Trả lời câu hỏi dựa trên các điều luật đã xác minh sau.\n\n"
            f"Câu hỏi: {query}\n\n"
            f"Điều luật:\n{format_citations(citations)}\n\n"
            "Yêu cầu: trả lời tiếng Việt, trích dẫn điều luật, "
            "thêm khuyến nghị nếu cần, lưu ý tham khảo luật sư.\n\nTrả lời:"
        )
        response = await self.llm.ainvoke(prompt)
        return {
            "response": response.content,
            "citations": [c.model_dump() for c in citations],
            "metadata": {
                "citation_count": len(citations),
                "response_length": len(response.content),
            },
        }
