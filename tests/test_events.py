import pytest

from clipper.core.errors import Cancelled, ClipperError
from clipper.core.events import (
    CancelToken,
    Message,
    Reporter,
    StageFinished,
    StageProgress,
    StageStarted,
)


def collecting_reporter() -> tuple[list, Reporter]:
    events: list = []
    return events, Reporter(events.append)


def test_stage_lifecycle():
    events, reporter = collecting_reporter()
    with reporter.stage("download", "Загрузка", total=10, unit="bytes") as stage:
        stage.update(10)
        stage.result = "готово"
    assert events[0] == StageStarted("download", "Загрузка", 10, "bytes")
    assert events[1] == StageProgress("download", 10, 10, "")
    assert isinstance(events[2], StageFinished)
    assert events[2].ok and events[2].message == "готово"


def test_stage_failure_is_reported_and_reraised():
    events, reporter = collecting_reporter()
    with pytest.raises(ClipperError), reporter.stage("x", "X"):
        raise ClipperError("сломалось", hint="почините")
    assert events[-1].ok is False
    assert events[-1].message == "сломалось"


def test_cancel_stops_stage():
    events, reporter = collecting_reporter()
    with pytest.raises(Cancelled), reporter.stage("x", "X", total=5) as stage:
        stage.update(1)
        reporter.cancel.cancel()
        stage.update(2)
    assert events[-1].ok is False
    assert events[-1].message == "отменено"


def test_cancelled_passes_through_except_exception():
    token = CancelToken()
    token.cancel()
    with pytest.raises(Cancelled):
        try:
            token.check()
        except Exception:
            pytest.fail("Cancelled не должен ловиться обработчиком except Exception")


def test_progress_is_throttled_but_final_update_is_kept():
    events, reporter = collecting_reporter()
    with reporter.stage("x", "X", total=1000) as stage:
        for done in range(1, 1001):
            stage.update(done)
    progress = [e for e in events if isinstance(e, StageProgress)]
    assert len(progress) < 50
    assert progress[-1].done == 1000


def test_messages_and_missing_sink():
    events, reporter = collecting_reporter()
    reporter.warning("осторожно")
    reporter.info("к сведению")
    assert events == [Message("warning", "осторожно"), Message("info", "к сведению")]
    Reporter().warning("никто не слушает")  # без приёмника событие просто отбрасывается
