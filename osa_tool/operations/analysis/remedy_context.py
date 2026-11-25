"""Retrieve and validate revision-pinned evidence for a PRISMM Remedy question."""
import json
import math
import re
import subprocess
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

from osa_tool.operations.analysis.paper_claims.repository_context import _SOURCES
from osa_tool.operations.analysis.prismm_remedy import complete, input_key, read_inputs

QUERY = """Read the supplied fragments, images and all four competing remedies as hypotheses.
Describe what could discriminate among the plausible discrepancies. Consider measurement code,
plot/table generation, metric normalization, model/config defaults and experiment inventories.
Do not infer an answer or assume a hypothesis is true. Return JSON {"issue":"...","anchors":[...]}.
Supply up to twelve SHORT literal search terms likely to occur in the authors' repository,
including concrete names and plausible code aliases for math notation. Use no generic words
alone. If code cannot distinguish even in principle (e.g. a purely missing caption), use [].
Use only supplied inputs, no tools.\n"""
FILTER = """Select source excerpts only if they contain a concrete fact bearing on the disputed
object under the same conditions: implementation, actual configuration, result, plotting, or
legend generation. A mention of the same model, baseline or metric alone is insufficient.
Avoid using one run to establish absence elsewhere. Read all alternatives neutrally.
Do not select an answer. Return JSON {"reason":"brief","matches":[{"id":0,"quote":"verbatim source"}]}.
Keep at most three complementary excerpts; use [] if none actually help. No tools.\n"""
EXTENSIONS = (*_SOURCES, "*.md", "*.tex", "*.csv", "*.json", "*.ipynb", "*.tsv")


def term_pattern(term):
    words = re.findall(r"[^\W_]+", term.casefold())
    if not words:
        return None
    # Handle the punctuation and spacing differences between paper and code.
    return r"[ _./-]*".join(words)


def snippets(repo, terms):
    sha = repo["sha"]
    if not re.fullmatch(r"[0-9a-f]{40}", sha):
        raise ValueError("Expected a pinned Git revision")
    patterns = list(dict.fromkeys(filter(None, map(term_pattern, terms))))
    if not patterns:
        return []
    git = ["git", "-c", "core.longpaths=true", "-C", str(repo["root"])]
    scope = repo.get("scope", "").strip("/")
    paths = [f"{scope}/{pat}" if scope else pat for pat in EXTENSIONS]
    command = [*git, "grep", "-n", "-z", "-I", "-i", "-E", "-m", "80", "-e", "|".join(patterns), sha, "--", *paths]
    result = subprocess.run(command, capture_output=True)
    if result.returncode not in (0, 1):
        raise ValueError(f"Cannot read pinned repository {repo['url']} at {sha}")
    compiled = [re.compile(r"(?<![a-z0-9])" + p + r"(?![a-z0-9])") for p in patterns]
    hits = {}
    for record in result.stdout.splitlines():
        fields = record.split(b"\0", 2)
        if len(fields) != 3 or not fields[1].isdigit() or len(fields[2]) > 900:
            continue
        path = fields[0].decode(errors="replace").removeprefix(sha + ":")
        text = fields[2].decode(errors="replace").casefold()
        matched = {i for i, p in enumerate(compiled) if p.search(text)}
        if matched:
            hits.setdefault(path, []).append((int(fields[1]), matched))
    frequency = Counter(term for records in hits.values() for term in set.union(*(m for _, m in records)))
    ranked = []
    for path, records in hits.items():
        path_terms = {i for i, p in enumerate(compiled) if p.search(path.casefold())}
        for pos, (start, matched) in enumerate(records):
            local = matched | path_terms
            for end, nearby in records[pos+1:pos+12]:
                if end > start+12:
                    break
                local.update(nearby)
            if not any(any(ch.isalpha() for ch in patterns[i].replace("[ _./-]*", "")) for i in local):
                continue
            score = sum(1 + math.log((1+len(hits))/(1+frequency[i])) for i in local)
            ranked.append((-score, path, start))
    excerpts = []
    used = Counter()
    for rank, path, line in sorted(ranked):
        if used[path] >= 2 or any(s["path"] == path and abs(s["line_start"]+3-line) < 14 for _, s in excerpts):
            continue
        lines = subprocess.check_output([*git, "cat-file", "blob", f"{sha}:{path}"]).decode(errors="replace").splitlines()
        lo, hi = max(1, line-3), min(len(lines), line+12)
        while hi > line and len("\n".join(lines[lo-1:hi])) > 1800:
            hi -= 1
        excerpt = "\n".join(lines[lo-1:hi])
        if len(excerpt) <= 1800:
            excerpts.append((rank, dict(repository=repo["url"], commit=sha, path=path,
                                        line_start=lo, line_end=hi, snippet=excerpt)))
            used[path] += 1
        if len(excerpts) == 12:
            break
    return excerpts


def build_context(manifest, registry, model, executable):
    repos = {p["pid"]: p["repos"] for p in json.loads(registry.read_bytes())["papers"]}
    cases = json.loads(manifest.read_bytes())["cases"]

    def ask(instruction, data, images, validate):
        prompt = instruction + json.dumps(data, ensure_ascii=False)
        for attempt in range(5):
            try:
                return validate(json.loads(complete(executable, model, prompt, images)))
            except (ValueError, TypeError, KeyError, subprocess.SubprocessError):
                if attempt == 4:
                    raise

    def prepare(case):
        pdf, _, images = read_inputs(manifest, case)
        paper = json.loads(pdf)

        def validate_query(query):
            anchors = query["anchors"]
            if not isinstance(anchors, list) or not all(isinstance(t, str) for t in anchors):
                raise ValueError("Invalid search anchors")
            return query["issue"], list(dict.fromkeys(t.strip().lower() for t in anchors
                                                      if 1 < len(t.strip()) <= 64 and t.isprintable()))[:12]

        issue, terms = ask(QUERY, paper, images, validate_query)
        ranked = []
        for repo in repos[case["paper_id"]]:
            ranked.extend(snippets(repo, terms))
        candidates = [s for _, s in sorted(ranked, key=lambda r:r[0])[:16]]

        def validate_selection(response):
            selected = {}
            for match in response["matches"]:
                i, quote = match["id"], match["quote"]
                if type(i) is not int or not 0 <= i < len(candidates) or not quote.strip() or quote not in candidates[i]["snippet"]:
                    raise ValueError("Evidence citation did not match the pinned snippet")
                selected[i] = candidates[i]
            if len(selected) > 3:
                raise ValueError("Too many excerpts")
            return response["reason"], list(selected.values())

        reason, selected = "No search anchors or source candidates", []
        if candidates:
            evidence = [dict(id=i, path=s["path"], source=s["snippet"]) for i, s in enumerate(candidates)]
            reason, selected = ask(FILTER, dict(paper=paper, issue=issue, candidates=evidence), images, validate_selection)
        return case["uid"], dict(input_key=input_key(pdf, images), issue=issue, anchors=terms,
                                 candidate_count=len(candidates), reason=reason, snippets=selected)

    with ThreadPoolExecutor(max_workers=6) as pool:
        return dict(pool.map(prepare, cases))
