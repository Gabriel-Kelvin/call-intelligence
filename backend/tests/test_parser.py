from backend.app.transcript_parser import parse_questions, parse_transcript


def test_parser_preserves_source_metadata():
    raw = """Expert 1 - Dr. Ada Stone
Market: United Kingdom

[00:01] Interviewer: What changed?
[00:07] Dr. Ada Stone: Adoption accelerated after training improved.
"""
    record, chunks, documents = parse_transcript("uk-call.txt", raw)
    assert record.title == "Dr. Ada Stone"
    assert record.market == "United Kingdom"
    assert chunks[-1].timestamp == "00:07"
    assert chunks[-1].content == "Adoption accelerated after training improved."
    assert documents[-1].metadata["id"] == chunks[-1].id


def test_question_parser_handles_numbered_and_bulleted_guides():
    guide = "Research guide\n1. What drives adoption?\n- How does price matter?\nNotes"
    assert parse_questions(guide) == ["What drives adoption?", "How does price matter?"]
