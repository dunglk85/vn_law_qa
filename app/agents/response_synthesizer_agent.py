"""ResponseSynthesizerAgent — generates grounded Vietnamese legal responses."""
import logging
from dataclasses import asdict
from typing import Optional

from langchain_core.language_models import BaseChatModel

from app.core.models import Citation, format_citations, llm_ainvoke

logger = logging.getLogger(__name__)


class ResponseSynthesizerAgent:
    def __init__(self, llm: Optional[BaseChatModel] = None):
        if llm is None:
            raise ValueError("ResponseSynthesizerAgent requires a chat model")
        self.llm = llm

    async def synthesize(self, query: str, citations: list[Citation]) -> dict:
        if not citations:
            return {
                "response": "Không tìm thấy điều luật liên quan để trả lời câu hỏi này.",
                "citations": [],
                "metadata": {"citation_count": 0, "response_length": 0},
            }
        prompt = (
            "Bạn là trợ lý tư vấn pháp luật Việt Nam. "
            "Trả lời câu hỏi dựa trên các điều luật đã xác minh sau.\n\n"
            f"Câu hỏi: {query}\n\n"
            f"Điều luật:\n{format_citations(citations)}\n\n"
            "Yêu cầu: trả lời tiếng Việt, trích dẫn điều luật, "
            "thêm khuyến nghị nếu cần, lưu ý tham khảo luật sư.\n\nTrả lời:"
        )
        try:
            response = await llm_ainvoke(self.llm, prompt)
            return {
                "response": response.content,
                "citations": [asdict(c) for c in citations],
                "metadata": {
                    "citation_count": len(citations),
                    "response_length": len(response.content),
                },
            }
        except Exception as exc:
            logger.error("Synthesis LLM call failed: %s", exc)
            return {
                "response": "Không thể tạo câu trả lời do lỗi hệ thống. Vui lòng thử lại.",
                "citations": [],
                "metadata": {"citation_count": 0, "response_length": 0, "error": str(exc)},
            }
