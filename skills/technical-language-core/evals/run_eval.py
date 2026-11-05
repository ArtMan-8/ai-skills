#!/usr/bin/env python3
"""Run the full technical-language-core eval and build a comparison table.

The harness does not depend on a specific model or CLI. Generation and judging
go through one command template with two placeholders:

  {system_file}  path to the system prompt
  {prompt_file}  path to the user prompt

Example:
  python3 evals/run_eval.py \
      --runner-command 'your-llm --system {system_file} --prompt {prompt_file}'
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import random
import re
import shlex
import subprocess
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

SKILL_ROOT = Path(__file__).resolve().parents[1]
CASES_PATH = SKILL_ROOT / "evals" / "cases.jsonl"
JUDGE_PROMPT = Path(__file__).resolve().parent / "judge.md"
DEFAULT_OUT_ROOT = SKILL_ROOT / "evals" / "evals-runs"

ALL_SKILLS = ["ours", "baseline"]
TARGETS = ("partner-a", "partner-b", "partner-c")


def log(msg: str) -> None:
    print(msg, flush=True)


def load_cases() -> list[dict]:
    cases = [json.loads(line) for line in CASES_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not cases:
        raise SystemExit(f"no cases in {CASES_PATH}")
    return cases


def read_parts(paths: list[Path]) -> str:
    chunks = []
    for path in paths:
        if not path.exists():
            raise SystemExit(f"missing skill file: {path}")
        chunks.append(f"\n\n--- {path.name} ---\n\n{path.read_text(encoding='utf-8')}")
    return "\n".join(chunks)


def build_prompts(out: Path) -> dict[str, Path]:
    prompts = out / "prompts"
    prompts.mkdir(parents=True, exist_ok=True)
    wrapper = (
        "Ты — языковой ассистент с установленным навыком (Skill). Ниже приведён полный текст навыка. "
        "Следуй его инструкциям при обработке запроса пользователя. Не обсуждай сам навык, не добавляй "
        "служебных пояснений. Верни только тот результат, который требует навык и запрос пользователя.\n\n"
        "===== НАЧАЛО НАВЫКА =====\n"
    )
    footer = "\n===== КОНЕЦ НАВЫКА =====\n"

    ours_parts = [
        SKILL_ROOT / "SKILL.md",
        SKILL_ROOT / "references" / "rules.md",
        SKILL_ROOT / "references" / "english.md",
        SKILL_ROOT / "references" / "exceptions.md",
        SKILL_ROOT / "references" / "terminology.md",
        SKILL_ROOT / "references" / "examples.md",
    ]
    (prompts / "ours.md").write_text(wrapper + read_parts(ours_parts) + footer, encoding="utf-8")

    baseline = (
        "Ты — опытный редактор технического текста. Выполни запрос пользователя: напиши, отредактируй, "
        "проверь или оцени именно текст. Не вызывай инструменты и не пытайся читать файлы — у тебя нет "
        "доступа к рабочей папке; работай с тем, что есть в запросе. Не проверяй техническую истинность "
        "утверждений, если об этом не просят. Верни только результат, который просит пользователь.\n"
    )
    (prompts / "baseline.md").write_text(baseline, encoding="utf-8")
    return {name: prompts / f"{name}.md" for name in ALL_SKILLS}


def select_tasks(cases: list[dict], skills: list[str]) -> list[dict]:
    return [{"case": case, "skill": skill} for case in cases for skill in skills]


def output_path(out: Path, skill: str, case_id: str) -> Path:
    d = out / "raw" / skill
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{case_id}.txt"


def render_command(template: str, system_file: Path, prompt_file: Path) -> list[str]:
    """Turn a tool-agnostic command template into argv.

    Supported placeholders: `{system_file}` and `{prompt_file}`. The template is
    split with shlex and executed without a shell.
    """
    return [
        arg.replace("{system_file}", str(system_file.resolve()))
           .replace("{prompt_file}", str(prompt_file.resolve()))
        for arg in shlex.split(template)
    ]


def run_one(task: dict, out: Path, prompts: dict[str, Path],
            runner_cmd: str, timeout: int) -> dict:
    case = task["case"]
    skill = task["skill"]
    path = output_path(out, skill, case["id"])
    if path.exists() and path.read_text(encoding="utf-8").strip():
        return {"case": case["id"], "skill": skill, "status": "cached"}

    prompt_file = out / "workdir" / f"prompt-{skill}-{case['id']}.txt"
    prompt_file.write_text(case["prompt"], encoding="utf-8")
    cmd = render_command(runner_cmd, prompts[skill], prompt_file)
    start = time.time()
    try:
        proc = subprocess.run(cmd, cwd=out / "workdir", capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        path.with_suffix(".timeout").write_text("timeout\n", encoding="utf-8")
        return {"case": case["id"], "skill": skill, "status": "timeout"}
    elapsed = round(time.time() - start, 1)
    text = proc.stdout.strip()
    if proc.returncode != 0 or not text:
        err = out / "raw" / skill / f"{case['id']}.err"
        err.write_text(proc.stderr or f"returncode={proc.returncode}\n", encoding="utf-8")
        return {"case": case["id"], "skill": skill, "status": "error", "elapsed": elapsed}
    path.write_text(text + "\n", encoding="utf-8")
    return {"case": case["id"], "skill": skill, "status": "ok", "elapsed": elapsed}


def run_all(tasks: list[dict], out: Path, prompts: dict[str, Path],
            workers: int, runner_cmd: str, timeout: int) -> list[dict]:
    results = []
    total = len(tasks)
    done = 0
    with cf.ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = [pool.submit(run_one, task, out, prompts, runner_cmd, timeout) for task in tasks]
        for fut in cf.as_completed(futures):
            r = fut.result()
            results.append(r)
            done += 1
            if done % 20 == 0 or r["status"] != "cached":
                log(f"[{done}/{total}] {r['case']} {r['skill']}: {r['status']}")
    return results


def parse_json(text: str) -> dict | None:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def judge_one(case: dict, outputs: dict[str, str], out: Path, runner_cmd: str,
              timeout: int) -> dict:
    jf = out / "judge" / f"{case['id']}.json"
    jf.parent.mkdir(parents=True, exist_ok=True)
    if jf.exists():
        try:
            return json.loads(jf.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    labels = list("ABCDE")[: len(outputs)]
    order = list(outputs.keys())
    random.Random("map-" + case["id"]).shuffle(order)
    mapping = {label: skill for label, skill in zip(labels, order)}
    blocks = []
    for label, skill in mapping.items():
        blocks.append(f"## Ответ {label}\n\n{outputs[skill].strip()}\n")
    prompt = (
        f"## Кейс {case['id']}\n"
        f"Режим: {case['mode']}\n"
        f"Язык: {case['lang']}\n"
        f"Запрос пользователя: {case['prompt']}\n"
        f"must: {case['must']}\n"
        f"must_not: {case['must_not']}\n\n"
        + "\n".join(blocks)
        + f"\nОцени каждый ответ ({', '.join(labels)}). Верни только JSON с ключами {labels}."
    )
    judge_prompt_file = out / "judge" / f"{case['id']}.prompt.txt"
    judge_prompt_file.write_text(prompt, encoding="utf-8")
    cmd = render_command(runner_cmd, JUDGE_PROMPT, judge_prompt_file)
    try:
        proc = subprocess.run(cmd, cwd=out / "workdir", capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        (out / "judge" / f"{case['id']}.raw.txt").write_text("timeout\n", encoding="utf-8")
        return {
            "id": case["id"],
            "suite": case["suite"],
            "target": case["targets"][0],
            "lang": case["lang"],
            "mapping": mapping,
            "parsed": None,
            "returncode": None,
            "stderr": "timeout",
            "timeout": True,
        }
    raw = proc.stdout.strip()
    (out / "judge" / f"{case['id']}.raw.txt").write_text(raw + "\n", encoding="utf-8")
    parsed = parse_json(raw)
    result = {
        "id": case["id"],
        "suite": case["suite"],
        "target": case["targets"][0],
        "lang": case["lang"],
        "mapping": mapping,
        "parsed": parsed,
        "returncode": proc.returncode,
        "stderr": proc.stderr[-300:] if not parsed else "",
    }
    jf.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def judge_all(cases: list[dict], tasks: list[dict], out: Path, runner_cmd: str,
              workers: int, timeout: int) -> list[dict]:
    by_case: dict[str, list[str]] = defaultdict(list)
    for task in tasks:
        if task["skill"] not in by_case[task["case"]["id"]]:
            by_case[task["case"]["id"]].append(task["skill"])
    case_by_id = {c["id"]: c for c in cases}
    jobs = []
    for cid, skills in by_case.items():
        outputs = {}
        for skill in skills:
            path = output_path(out, skill, cid)
            if path.exists() and path.read_text(encoding="utf-8").strip():
                outputs[skill] = path.read_text(encoding="utf-8")
        if outputs:
            jobs.append((case_by_id[cid], outputs))
    results = []
    with cf.ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = [pool.submit(judge_one, case, outputs, out, runner_cmd, timeout)
                   for case, outputs in jobs]
        for i, fut in enumerate(cf.as_completed(futures), 1):
            r = fut.result()
            results.append(r)
            log(f"[judge {i}/{len(futures)}] {r['id']}: {'ok' if r['parsed'] else 'PARSE_FAIL'}")
    return results


def build_rows(judged: list[dict]) -> list[dict]:
    rows = []
    for item in judged:
        parsed = item.get("parsed") or {}
        for label, skill in item["mapping"].items():
            v = parsed.get(label) or {}
            rows.append({
                "case": item["id"],
                "suite": item["suite"],
                "target": item["target"],
                "lang": item["lang"],
                "skill": skill,
                "verdict": v.get("verdict", "UNCERTAIN"),
                "quality": v.get("quality"),
                "critical": bool(v.get("critical")),
                "violated": v.get("violated", ""),
                "notes": v.get("notes", ""),
            })
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    import csv
    fields = ["case", "suite", "target", "lang", "skill", "verdict", "quality", "critical", "violated", "notes"]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fields})


def aggregate(rows: list[dict], key: str | None = None) -> dict:
    groups = defaultdict(list)
    for row in rows:
        groups[row[key] if key else "all"].append(row)
    out = {}
    for name, rs in groups.items():
        cnt = Counter(r["verdict"] for r in rs)
        qs = [r["quality"] for r in rs if isinstance(r["quality"], (int, float))]
        out[name] = {
            "cases": len({r["case"] for r in rs}),
            "verdicts": len(rs),
            "pass": cnt["PASS"],
            "fail": cnt["FAIL"],
            "uncertain": cnt["UNCERTAIN"],
            "critical": sum(1 for r in rs if r["critical"]),
            "avg_quality": (sum(qs) / len(qs)) if qs else None,
            "q3": sum(1 for q in qs if q == 3),
        }
    return out


def markdown_table(headers: list[str], rows: list[list]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(x) for x in row) + " |")
    return "\n".join(lines)


def build_benchmark(cases: list[dict], judged: list[dict], rows: list[dict], out: Path,
                    meta: dict) -> None:
    skills = [s for s in ALL_SKILLS if any(r["skill"] == s for r in rows)]
    overall = aggregate(rows, "skill")
    by_suite = aggregate(rows, "suite")
    target_rows = [r for r in rows if r["suite"] == "strengths"]
    by_target = aggregate(target_rows, "target") if target_rows else {}

    lines = ["# Eval benchmark", ""]
    lines += [
        f"- Дата: {meta['timestamp']}",
        f"- Кейсов: {len({r['case'] for r in rows})}; оценок: {len(rows)}",
        "",
        "> Наборы `core`, `strengths` и `language` проверяют разные свойства текста. "
        "Доли PASS между наборами напрямую не сравниваются.",
        "",
        "## Общая таблица",
        "",
    ]
    table_rows = []
    for skill in skills:
        a = overall[skill]
        q = f"{a['avg_quality']:.2f}" if a["avg_quality"] is not None else "—"
        table_rows.append([skill, a["cases"], a["pass"], a["fail"], a["uncertain"], a["critical"], q, a["q3"]])
    lines.append(markdown_table(
        ["Навык", "Кейсов", "PASS", "FAIL", "UNCERTAIN", "Крит.", "Ср. качество", "«3»"],
        table_rows,
    ))
    lines += ["", "## По наборам", ""]
    suite_rows = []
    for skill in skills:
        for suite in ("core", "strengths", "language"):
            key = suite
            if key not in by_suite:
                continue
            # aggregate() groups by suite across all skills; recompute per skill+suite
            rs = [r for r in rows if r["skill"] == skill and r["suite"] == suite]
            if not rs:
                continue
            cnt = Counter(r["verdict"] for r in rs)
            qs = [r["quality"] for r in rs if isinstance(r["quality"], (int, float))]
            suite_rows.append([
                skill, suite, len({r["case"] for r in rs}), cnt["PASS"], cnt["FAIL"],
                cnt["UNCERTAIN"], sum(1 for r in rs if r["critical"]),
                f"{sum(qs)/len(qs):.2f}" if qs else "—",
            ])
    lines.append(markdown_table(
        ["Навык", "Набор", "Кейсов", "PASS", "FAIL", "UNCERTAIN", "Крит.", "Ср. качество"],
        suite_rows,
    ))
    if by_target:
        lines += ["", "## Strength по target", ""]
        trows = []
        for target in TARGETS:
            if target not in by_target:
                continue
            for skill in skills:
                rs = [r for r in rows if r["skill"] == skill and r["suite"] == "strengths" and r["target"] == target]
                if not rs:
                    continue
                cnt = Counter(r["verdict"] for r in rs)
                trows.append([target, skill, len(rs), cnt["PASS"], cnt["FAIL"], cnt["UNCERTAIN"],
                              sum(1 for r in rs if r["critical"])])
        lines.append(markdown_table(["Target", "Навык", "Кейсов", "PASS", "FAIL", "UNCERTAIN", "Крит."], trows))
    lines += ["", "## Матрица по кейсам", ""]
    cases_sorted = sorted({r["case"] for r in rows})
    headers = ["Кейс"] + skills
    matrix_rows = []
    for cid in cases_sorted:
        row = [cid]
        for skill in skills:
            r = next((x for x in rows if x["case"] == cid and x["skill"] == skill), None)
            row.append(r["verdict"][0] if r else "—")
        matrix_rows.append(row)
    lines.append(markdown_table(headers, matrix_rows))
    failures = [r for r in rows if r["verdict"] != "PASS" or r["critical"]]
    lines += ["", "## Не PASS и критические", ""]
    if failures:
        lines.append(markdown_table(
            ["Кейс", "Навык", "Вердикт", "Кач.", "Крит.", "Нарушено", "Заметка"],
            [[r["case"], r["skill"], r["verdict"], r["quality"], r["critical"],
              r["violated"][:80], r["notes"][:160]] for r in failures[:80]],
        ))
        if len(failures) > 80:
            lines.append(f"\n… ещё {len(failures) - 80} строк — в `verdicts.csv`.")
    else:
        lines.append("Нет.")
    (out / "benchmark.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_csv(out / "verdicts.csv", rows)

    import csv

    def stats(rs: list[dict]) -> list:
        cnt = Counter(r["verdict"] for r in rs)
        qs = [r["quality"] for r in rs if isinstance(r["quality"], (int, float))]
        return [
            len({r["case"] for r in rs}), cnt["PASS"], cnt["FAIL"], cnt["UNCERTAIN"],
            sum(1 for r in rs if r["critical"]),
            f"{sum(qs)/len(qs):.2f}" if qs else "",
            sum(1 for q in qs if q == 3),
        ]

    with (out / "benchmark.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scope", "name", "skill", "cases", "pass", "fail", "uncertain",
                    "critical", "avg_quality", "q3"])
        for skill in skills:
            rs = [r for r in rows if r["skill"] == skill]
            w.writerow(["overall", "all", skill, *stats(rs)])
        for skill in skills:
            for suite in ("core", "strengths", "language"):
                rs = [r for r in rows if r["skill"] == skill and r["suite"] == suite]
                if rs:
                    w.writerow(["suite", suite, skill, *stats(rs)])
        for skill in skills:
            for target in TARGETS:
                rs = [r for r in rows if r["skill"] == skill and r["suite"] == "strengths" and r["target"] == target]
                if rs:
                    w.writerow(["target", target, skill, *stats(rs)])

    (out / "benchmark.json").write_text(json.dumps({
        "meta": meta,
        "overall": overall,
        "by_suite": by_suite,
        "by_target": by_target,
        "rows": rows,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n" + "\n".join(lines[:60]) + "\n")
    print(f"full report: {out / 'benchmark.md'}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runner-command", required=True,
                    help="command template; placeholders: {system_file}, {prompt_file}")
    ap.add_argument("--out", default=None,
                    help=f"output directory (default: {DEFAULT_OUT_ROOT}/<timestamp>)")
    ap.add_argument("--workers", type=int, default=6, help="parallel calls (default: 6)")
    ap.add_argument("--timeout", type=int, default=300, help="per-call timeout in seconds (default: 300)")
    args = ap.parse_args()

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    out = (Path(args.out) if args.out else DEFAULT_OUT_ROOT / timestamp).resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / "workdir").mkdir(exist_ok=True)
    cases = load_cases()
    skills = ALL_SKILLS

    log(f"out: {out}")
    log(f"skills: {', '.join(skills)}")
    prompts = build_prompts(out)
    tasks = select_tasks(cases, skills)
    log(f"tasks: {len(tasks)}")

    meta = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "skills": skills,
        "tasks": len(tasks),
    }
    (out / "run.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    results = run_all(tasks, out, prompts, args.workers, args.runner_command, args.timeout)
    ok = sum(1 for r in results if r["status"] in ("ok", "cached"))
    log(f"generations done: {ok}/{len(results)}")
    if ok == 0:
        raise SystemExit("no successful generations")

    judged = judge_all(cases, tasks, out, args.runner_command, args.workers, args.timeout)
    rows = build_rows(judged)
    if not rows:
        raise SystemExit("no judge rows")
    build_benchmark(cases, judged, rows, out, meta)


if __name__ == "__main__":
    main()
