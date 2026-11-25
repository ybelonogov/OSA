import asyncio
import json

from osa_tool.operations.analysis.paper_analysis.visual_manuscript import consolidate, discover, groups, verify


class ScriptedHandler:
    def __init__(self, *responses):
        self.responses = iter(responses)
        self.images = []

    async def async_request_with_images(self, prompt, instruction, *, images):
        self.images.append(images)
        return json.dumps(next(self.responses))


def claim(cid, text, page):
    return {"claim_id": cid, "claim": text, "sources": [{"metadata": {"pages": [page]}}]}


def test_visual_discovery_indexes_distant_mentions_and_rejects_fabricated_quotes():
    claims = [claim("a", "Figure 2 reports 10.", 1), claim("b", "Ablation.", 2),
              claim("c", "Experiments.", 3), claim("d", "Figure 2 reports 12.", 4)]
    pages = {p: f"page{p}.png" for p in range(1, 5)}
    focus, local = next(groups(claims, pages))
    assert focus == [1, 2, 3] and "d" in local
    bad = {"problem": "Mismatched value", "suggested_edit": None,
           "evidence": [{"claim_id": "a", "quote": "999"}]}
    good = dict(bad, evidence=[{"claim_id": "a", "quote": "10"}, {"claim_id": "d", "quote": "12"}])
    h = ScriptedHandler([bad], [good], [dict(good, index=0)])
    proposals = asyncio.run(discover(h, focus, local, pages))
    assert proposals == [good]
    findings = asyncio.run(verify(h, focus, local, pages, proposals))
    assert h.images[-1] == list(pages.values())
    assert findings == [dict(good, repository_context=[])]


def test_editor_can_remove_an_unrelated_flag_but_cannot_add_a_new_one():
    cs = {"x": claim("x", "Label X", 1), "y": claim("y", "Label Y", 1)}
    issue = {"evidence": [{"claim_id": "x", "quote": "X"}, {"claim_id": "y", "quote": "Y"}],
             "problem": "Undefined label", "suggested_edit": None, "repository_context": []}
    handler = ScriptedHandler([{"index": 0, "claim_ids": ["stranger"]}], [{"index": 0, "claim_ids": ["x"]}])
    got = asyncio.run(consolidate(handler, [issue], cs))
    assert got == [dict(issue, evidence=[issue["evidence"][0]])]
    assert len(issue["evidence"]) == 2
