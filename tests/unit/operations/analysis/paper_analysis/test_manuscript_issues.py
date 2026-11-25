import json

import pytest

from osa_tool.operations.analysis.paper_analysis.manuscript_issues import inspect_manuscript


def test_manuscript_review_discovers_without_pairs_and_preserves_uncertain_edit():
    claim = {"claim_id": "c1", "claim": "The axis is labelled T.", "section_id": "p1",
             "original_text": "The axis is labelled T. T is also used for temperature."}
    issue = {"evidence": [{"claim_id": "c1", "quote": "labelled T"}],
             "problem": "The symbol has two meanings.", "suggested_edit": None}
    calls = []

    class Handler:
        def send_and_parse(self, prompt, parser, instruction):
            calls.append(json.loads(prompt))
            return parser(json.dumps([issue]))

    assert inspect_manuscript([claim], Handler(), lambda query: [{"snippet": "T = 1"}]) == [dict(issue, repository_context=[{"snippet": "T = 1"}])]
    assert calls[0] == [{k: claim[k] for k in ("claim_id", "claim", "section_id")}]
    assert calls[1]["sources"][0]["original_text"] == claim["original_text"]
    assert calls[1]["code"] == [{"snippet": "T = 1"}]


def test_manuscript_review_rejects_an_untraceable_diagnosis():
    class Handler:
        def send_and_parse(self, prompt, parser, instruction):
            return parser(json.dumps([{"evidence": [{"claim_id": "unknown", "quote": "10"}],
                                      "problem": "Disagreement", "suggested_edit": None}]))

    with pytest.raises(ValueError, match="quote"):
        inspect_manuscript([{"claim_id": "c1", "claim": "score 10"}], Handler())
