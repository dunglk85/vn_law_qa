"""Unit tests for token_tracker.py"""
from app.core.token_tracker import TokenTracker, get_tracker, reset_tracker, set_tracker


class TestTokenTracker:
    def test_init_defaults(self):
        tracker = TokenTracker()
        assert tracker.prompt_tokens == 0
        assert tracker.completion_tokens == 0
        assert tracker.total_tokens == 0
        assert tracker.llm_call_count == 0

    def test_record_single_call(self):
        tracker = TokenTracker()
        tracker.record(prompt_tokens=100, completion_tokens=50, call_name="test_call")

        assert tracker.prompt_tokens == 100
        assert tracker.completion_tokens == 50
        assert tracker.total_tokens == 150
        assert tracker.llm_call_count == 1

    def test_record_multiple_calls(self):
        tracker = TokenTracker()
        tracker.record(100, 50, "call1")
        tracker.record(200, 100, "call2")
        tracker.record(50, 25, "call3")

        assert tracker.prompt_tokens == 350
        assert tracker.completion_tokens == 175
        assert tracker.total_tokens == 525
        assert tracker.llm_call_count == 3

    def test_to_dict(self):
        tracker = TokenTracker()
        tracker.record(100, 50, "call1")
        tracker.record(200, 100, "call2")

        result = tracker.to_dict()
        assert result == {
            "prompt_tokens": 300,
            "completion_tokens": 150,
            "total_tokens": 450,
            "llm_call_count": 2,
        }


class TestContextVar:
    def test_get_tracker_returns_none_by_default(self):
        set_tracker(None)
        assert get_tracker() is None

    def test_set_and_get_tracker(self):
        tracker = TokenTracker()
        set_tracker(tracker)
        try:
            assert get_tracker() is tracker
        finally:
            set_tracker(None)

    def test_reset_tracker(self):
        old_tracker = TokenTracker()
        old_tracker.record(100, 50)
        set_tracker(old_tracker)

        new_tracker = reset_tracker()
        try:
            assert new_tracker is not old_tracker
            assert new_tracker.prompt_tokens == 0
            assert get_tracker() is new_tracker
        finally:
            set_tracker(None)
