"""Unit tests for SupervisorAgent routing and quality gate"""
from unittest.mock import MagicMock

import pytest

from app.agents.supervisor_agent import SupervisorAgent


class TestSupervisorRouting:
    @pytest.fixture
    def supervisor(self):
        return SupervisorAgent(
            research_agent=MagicMock(),
            citation_agent=MagicMock(),
            synthesis_agent=MagicMock(),
            llm=MagicMock(),
        )

    def test_route_after_research_no_error(self, supervisor):
        state = {"error": None}
        result = supervisor.route_after_research(state)
        assert result == "citation_check"

    def test_route_after_research_with_error(self, supervisor):
        state = {"error": "Research failed"}
        result = supervisor.route_after_research(state)
        assert result == "error"

    def test_route_after_citation_check_has_citations(self, supervisor):
        state = {
            "error": None,
            "verified_citations": [{"id": "1"}],
            "retry_count": 0,
        }
        result = supervisor.route_after_citation_check(state)
        assert result == "synthesis"

    def test_route_after_citation_check_no_citations_retry(self, supervisor):
        state = {
            "error": None,
            "verified_citations": [],
            "retry_count": 0,
        }
        result = supervisor.route_after_citation_check(state)
        assert result == "retry_research"

    def test_route_after_citation_check_no_citations_max_retries(self, supervisor):
        state = {
            "error": None,
            "verified_citations": [],
            "retry_count": 2,
        }
        result = supervisor.route_after_citation_check(state)
        assert result == "synthesis"

    def test_route_after_citation_check_error_retry(self, supervisor):
        state = {
            "error": "Citation check failed",
            "verified_citations": [],
            "retry_count": 0,
        }
        result = supervisor.route_after_citation_check(state)
        assert result == "error"

    def test_route_after_citation_check_error_max_retries(self, supervisor):
        state = {
            "error": "Citation check failed",
            "verified_citations": [],
            "retry_count": 2,
        }
        result = supervisor.route_after_citation_check(state)
        assert result == "synthesis"

    def test_route_after_validation_good_quality(self, supervisor):
        state = {
            "error": None,
            "quality_score": 0.85,
            "retry_count": 0,
        }
        result = supervisor.route_after_validation(state)
        assert result == "complete"

    def test_route_after_validation_poor_quality_retry(self, supervisor):
        state = {
            "error": None,
            "quality_score": 0.5,
            "retry_count": 0,
        }
        result = supervisor.route_after_validation(state)
        assert result == "retry_synthesis"

    def test_route_after_validation_poor_quality_max_retries(self, supervisor):
        state = {
            "error": None,
            "quality_score": 0.5,
            "retry_count": 2,
        }
        result = supervisor.route_after_validation(state)
        assert result == "complete"

    def test_route_after_validation_with_error(self, supervisor):
        state = {
            "error": "Validation failed",
            "quality_score": 0.9,
            "retry_count": 0,
        }
        result = supervisor.route_after_validation(state)
        assert result == "error"

    def test_route_after_validation_no_score(self, supervisor):
        state = {
            "error": None,
            "quality_score": None,
            "retry_count": 0,
        }
        result = supervisor.route_after_validation(state)
        assert result == "retry_synthesis"


class TestSupervisorHeuristicScore:
    @pytest.fixture
    def supervisor(self):
        return SupervisorAgent(
            research_agent=MagicMock(),
            citation_agent=MagicMock(),
            synthesis_agent=MagicMock(),
            llm=MagicMock(),
        )

    def test_heuristic_score_no_response(self, supervisor):
        score = supervisor._heuristic_score("", [])
        assert score == 0.0

    def test_heuristic_score_response_only(self, supervisor):
        score = supervisor._heuristic_score("This is a response", [])
        assert score == 0.4

    def test_heuristic_score_response_with_citations(self, supervisor):
        citations = [MagicMock(article_id="1", content="test")]
        score = supervisor._heuristic_score("This is a response", citations)
        assert score == 0.7

    def test_heuristic_score_response_with_article_reference(self, supervisor):
        citations = [MagicMock(article_id="1", content="test")]
        score = supervisor._heuristic_score("Response mentions 1", citations)
        assert score == 0.9

    def test_heuristic_score_long_response(self, supervisor):
        citations = [MagicMock(article_id="1", content="test")]
        score = supervisor._heuristic_score("A" * 150 + " 1", citations)
        assert score == 1.0
