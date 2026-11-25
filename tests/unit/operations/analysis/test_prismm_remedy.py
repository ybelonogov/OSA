import hashlib
import json

from osa_tool.operations.analysis import prismm_remedy


def test_paired_remedy_preserves_shuffled_options_and_hides_gold(tmp_path, monkeypatch):
    folder = tmp_path / "paper_0"
    question = {"question": "Choose the repair", "parts": [{"text": "Results"}],
                "options": [{"letter": x, "content": x} for x in "CABD"]}
    hashes = {}
    for mode in ("pdf", "code"):
        payload = dict(question)
        if mode == "code":
            payload["repository_context"] = ["model.py"]
        raw = json.dumps(payload).encode()
        (folder / mode).mkdir(parents=True)
        (folder / mode / "input.json").write_bytes(raw)
        hashes[mode] = hashlib.sha256(raw).hexdigest()
    (folder / "gold_evaluation_only.json").write_text('{"answer":"D"}')
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"cases": [{"uid": "paper_0", "input_sha256": hashes, "images": []}]}))
    monkeypatch.setattr(prismm_remedy, "host_command", lambda: "codex")

    def predict(executable, model, payload, images):
        request = json.loads(payload)
        assert request.pop("repository_context", []) in ([], ["model.py"])
        assert request == question
        return "B"

    monkeypatch.setattr(prismm_remedy, "choose", predict)
    output = tmp_path / "out.json"
    prismm_remedy.run_remedy(manifest, "gpt-6.2-sol", output)
    result = json.loads(output.read_text())
    assert [(row["mode"], row["answer"]) for row in result["predictions"]] == [("pdf", "B"), ("code", "B")]
    assert prismm_remedy.score_remedy(manifest, output) == {"cases": 1, "correct": {"pdf": 0, "code": 0}}
    pdf = (folder / "pdf/input.json").read_text()
    focused = {"paper_0": {"input_key": prismm_remedy.input_key(pdf, []), "snippets": []}}
    prismm_remedy.run_remedy(manifest, "gpt-6.2-sol", output, focused)
    result = json.loads(output.read_text())
    assert result["predictions"][-1] == {"uid": "paper_0", "mode": "focused_code", "answer": "B", "decision": "no_code"}
    assert prismm_remedy.score_remedy(manifest, output)["correct"] == {"pdf": 0, "code": 0, "focused_code": 0}


def test_focus_retrieves_candidates_from_pinned_lines_for_evidence_review(tmp_path):
    import subprocess
    from osa_tool.operations.analysis.remedy_context import snippets

    def git(*args):
        return subprocess.check_output(["git", "-C", str(tmp_path), *args], text=True).strip()

    git("init", "-q")
    source = tmp_path / "train.py"
    source.write_text("metric = 'WER'\nnormalizer = 'U-WER'\n")
    git("add", "train.py")
    git("-c", "user.name=Test", "-c", "user.email=test@example.org", "commit", "-qm", "fixture")
    sha = git("rev-parse", "HEAD")
    source.write_text("unrelated working copy")
    repo = {"root": tmp_path, "sha": sha, "url": "example"}
    assert snippets(repo, ["unrelated", "working"]) == []
    assert snippets(repo, ["unmatched"]) == []
    assert snippets(repo, ["u_wer"])[0][1]["path"] == "train.py"
    hit = snippets(repo, ["metric", "normalizer"])[0][1]
    assert (hit["commit"], hit["path"], hit["line_start"], hit["line_end"]) == (sha, "train.py", 1, 2)
    assert "U-WER" in hit["snippet"]


def test_remedy_context_discards_related_snippet_without_discriminating_evidence(tmp_path, monkeypatch):
    from osa_tool.operations.analysis import remedy_context

    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"cases": [{"uid": "p_0", "paper_id": "p"}]}))
    registry = tmp_path / "repos.json"
    registry.write_text(json.dumps({"papers": [{"pid": "p", "repos": [{}]}]}))
    pdf = json.dumps({"parts": [{"text": "The figure reports WER"}], "options": ["axis label", "scale"]})
    monkeypatch.setattr(remedy_context, "read_inputs", lambda *args: (pdf, "", []))
    candidate = {"path": "evaluate.py", "snippet": "import wer"}
    monkeypatch.setattr(remedy_context, "snippets", lambda *args: [(0, candidate)])

    def complete(executable, model, prompt, images):
        if prompt.startswith(remedy_context.QUERY):
            return json.dumps({"issue": "Which metric does the axis represent?", "anchors": ["WER"]})
        assert "import wer" in prompt
        return json.dumps({"reason": "Import does not identify the plotted metric", "matches": []})

    monkeypatch.setattr(remedy_context, "complete", complete)
    result = remedy_context.build_context(manifest, registry, "test", "test")["p_0"]
    assert result["candidate_count"] == 1
    assert result["snippets"] == []


def test_edit_switch_requires_exact_action_and_traceable_evidence(monkeypatch):
    option = {"target": "Table 1 caption", "edit_statement": "change seconds to milliseconds"}
    payload = json.dumps({
        "parts": [{"part": 1, "text": "Table 1 reports milliseconds."}],
        "options": [{"letter": "B", "content": json.dumps(option)}],
        "repository_context": [{"snippet": "runtime_ms = runtime_seconds * 1000"}],
    })
    review = {"same_issue": True, "relation": "direction_from_specification",
              "remaining_discrepancy": "Headers would conflict with the specified unit.", "choice": "B", "added_fact": "Timing is emitted in ms",
              "alternative_limitation": "Changing the header would conflict with the recorded unit.", "reason": "The caption alone labels the measured values as seconds.",
              "target": option["target"], "edit": option["edit_statement"], "source_index": 0,
              "quote": "runtime_seconds * 1000", "paper_part": 1, "paper_fact": "Headers label the same values in ms."}
    def assess(executable, model, prompt, images):
        if prompt.startswith(prismm_remedy.PAPER_DISCREPANCY):
            assert json.loads(prompt.removeprefix(prismm_remedy.PAPER_DISCREPANCY)) == json.loads(payload)["parts"]
            return json.dumps({"observations": "Timing units disagree.", "intended_reference": "Specification in text"})
        return json.dumps(review)
    monkeypatch.setattr(prismm_remedy, "complete", assess)
    def result():
        return prismm_remedy.review_change("codex", "test", payload, [], "A", "B")[0]
    assert result() == "B"
    for key, invalid in (("quote", "missing code"), ("edit", "edit something else"),
                         ("added_fact", ""), ("alternative_limitation", ""), ("same_issue", False),
                         ("relation", "unrelated"), ("remaining_discrepancy", ""),
                         ("target", "main text"), ("paper_part", 2), ("source_index", -1)):
        original = review[key]
        review[key] = invalid
        assert result() == "A"
        review[key] = original
    review.clear()
    review.update(choice=None, reason="The intended unit is ambiguous.")
    assert result() == "A"


def test_remedy_review_hides_candidate_origin(monkeypatch):
    prompts = []
    def complete(executable, model, prompt, images):
        if prompt.startswith(prismm_remedy.PAPER_DISCREPANCY):
            return json.dumps({"observations": "Ambiguous", "intended_reference": "unknown"})
        prompts.append(prompt)
        return '{"choice":null,"reason":"ambiguous"}'
    monkeypatch.setattr(prismm_remedy, "complete", complete)
    for baseline, proposed in (("A", "D"), ("D", "A")):
        answer, review = prismm_remedy.review_change("codex", "test", '{"parts":[]}', [], baseline, proposed)
        assert answer == baseline
        assert review["decision"] == "keep"
    assert prompts[0] == prompts[1]
