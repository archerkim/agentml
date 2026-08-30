"""
Запускает AIDE с встроенным early-stop по плато вместо жёсткого range(steps).

Почему не просто agent.steps в config.yaml: нативный цикл AIDE (aide/run.py) - это
безусловный `while global_step < cfg.agent.steps: agent.step(...)`, без проверки
плато. Он остановится либо на cfg.agent.steps, либо никогда раньше. Здесь используются
ровно те же примитивы, что и в aide/run.py (Agent, Journal, Interpreter, save_run,
journal2report) - тот же путь исполнения, только цикл заменён на plateau-aware.

Правило совпадает по духу с baseline.py:run_fm (best/bad-counter с patience) и с
формулировкой из спеки соревнования: "validation score has not improved by more than
epsilon over the last N consecutive iterations".

Останов по первому сработавшему условию:
  1. плато: bad_steps >= patience (нет прироста best-so-far > epsilon N раз подряд)
  2. wall-clock ceiling (--wall_clock_sec)
  3. cfg.agent.steps (жёсткий потолок итераций из config.yaml)

НЕ считается плато: степы, где journal.get_best_node(only_good=True) is None (все
попытки пока багованные) - в этом случае просто продолжаем, не трогая bad_steps.

Использование:
    python3 run_with_early_stop.py --config ./config.yaml
    python3 run_with_early_stop.py --config ./config.yaml --epsilon 0.002 --patience 3 \
        --wall_clock_sec 21600
"""
import argparse
import atexit
import json
import os
import shutil
import time
from pathlib import Path


def _load_env_file(path: Path):
    """Мини-загрузчик .env (KEY=VALUE построчно) - чтобы не тянуть python-dotenv
    ради одной переменной. Не перезаписывает уже выставленные переменные окружения."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip().strip('"').strip("'")
        if k and v and v != "PASTE_YOUR_KEY_HERE":
            os.environ.setdefault(k, v)


_load_env_file(Path(__file__).parent / ".env")

# NVIDIA Nemotron 3 Ultra через build.nvidia.com: aideml==0.2.2 определяет провайдера по
# префиксу имени модели (aide/backend/__init__.py:determine_provider) и распознаёт только
# gpt-/o<N>/codex-, claude-, gemini- - "nvidia/nemotron-3-ultra-550b-a55b" ни под один
# паттерн не попадает. determine_provider() проверяет OPENAI_BASE_URL: если задана,
# запрос идёт через backend_openai.py по обычному Chat Completions API (custom-client
# ветка, не официальный OpenAI Responses API) на этот base_url с ключом OPENAI_API_KEY.
# Ставим значение по умолчанию здесь, а не требуем от пользователя вписывать его в .env -
# там нужно только вставить ключ.
os.environ.setdefault("OPENAI_BASE_URL", "https://integrate.api.nvidia.com/v1")

from omegaconf import OmegaConf

# aideml==0.2.2 (проверено на установленном пакете): agent.py не экспортирует
# add_task_metric/determine_task_metric - это API из GitHub main, которого ещё нет
# на PyPI. Journal() в этой версии тоже не принимает metric_maximize - направление
# метрики (maximize/minimize) LLM сообщает САМ, per-node, через review_func_spec
# (lower_is_better) при каждом вызове parse_exec_result. Нашу задачу (GAUC/nDCG@5/
# primary) метрика всегда maximize - зашито ниже как константа, а не берётся из journal.
from aide.agent import Agent
from aide.interpreter import Interpreter
from aide.journal import Journal, Node
from aide.journal2report import journal2report
from aide.utils.config import load_task_desc, prep_agent_workspace, prep_cfg, save_run

# Ещё один подтверждённый баг в aideml==0.2.2 (aide/utils/config.py):
#   if current_index := int(p.name.split("-")[0]) > max_index:
# из-за приоритета операторов это парсится как
#   current_index := (int(...) > max_index)
# то есть current_index становится bool (True/False), а не числом, и
# max_index = current_index присваивает max_index булево True (== 1) при первом же
# попадании - после чего функция ВСЕГДА возвращает 2 при наличии хотя бы одной
# существующей нумерованной папки, независимо от реального максимума. Проверено
# эмпирически: с папками "0-..." и "2-..." она снова вернула 2 - и запуск упал на
# copytree() с AssertionError, натолкнувшись на уже занятую workspaces/2-.../input.
# Патчим модульную функцию (не bound method - не через multiprocessing, пиклинг тут
# не участвует, обычный monkeypatch по имени в aide.utils.config.__dict__ работает).
import aide.utils.config as _cfgmod  # noqa: E402


def _get_next_logindex_fixed(dir) -> int:
    max_index = -1
    for p in dir.iterdir():
        try:
            current_index = int(p.name.split("-")[0])
        except ValueError:
            continue
        if current_index > max_index:
            max_index = current_index
    return max_index + 1


_cfgmod._get_next_logindex = _get_next_logindex_fixed

# aide/backend/backend_gemini.py (и остальные backend_*.py) строят openai.OpenAI(...)
# без явного timeout - SDK по умолчанию ждёт ~600с на один запрос, прежде чем backoff
# вообще попробует повторить. На практике это значит, что один зависший/перегруженный
# (503) запрос к модели может блокировать весь прогон на 10 минут за попытку - именно
# так и произошло на смоук-тесте с gemini-3.7-flash. Патчим дефолтный timeout здесь,
# а не трогаем сам site-packages/aideml, чтобы это пережило переустановку пакета.
LLM_REQUEST_TIMEOUT_SEC = 180
import openai as _openai_module  # noqa: E402

_orig_openai_init = _openai_module.OpenAI.__init__


def _patched_openai_init(self, *a, **kw):
    kw.setdefault("timeout", LLM_REQUEST_TIMEOUT_SEC)
    return _orig_openai_init(self, *a, **kw)


# ФАКТИЧЕСКИ ПРИМЕНЯЕМ патч - до этой правки функция была определена, но никогда не
# присваивалась обратно в openai.OpenAI.__init__, так что таймаут ни разу не применялся
# ни на одном прогоне этой сессии (проверено grep'ом по файлу - только объявление,
# без присваивания). Обнаружено при добавлении OpenRouter-патча ниже.
_openai_module.OpenAI.__init__ = _patched_openai_init


# aideml==0.2.2 (aide/backend/backend_openrouter.py) хардкодит на КАЖДЫЙ запрос
# extra_body={"provider": {"order": ["Fireworks"], "ignore": ["Together", "DeepInfra",
# "Hyperbolic"]}} - принудительно ограничивает OpenRouter только апстримом Fireworks.
# Это было подобрано авторами aideml под какую-то другую модель и не имеет отношения к
# z-ai/glm-5.2:free (бесплатный слот, хостится у Z.ai, Fireworks его почти наверняка не
# раздаёт) - с этим ограничением запрос, скорее всего, просто не найдёт провайдера.
# provider_to_query_func - это словарь функций-ссылок, собранный в aide/backend/__init__.py
# ПРИ ИМПОРТЕ; патчить нужно именно его (а не backend_openrouter.query "на месте") -
# иначе aide.backend.query() продолжит вызывать старую функцию, захваченную в словаре
# до патча.
import aide.backend as _aide_backend_pkg  # noqa: E402
from aide.backend import backend_openrouter as _backend_openrouter  # noqa: E402
from funcy import notnone as _notnone, select_values as _select_values  # noqa: E402


def _patched_openrouter_query(system_message, user_message, func_spec=None, **model_kwargs):
    _backend_openrouter._setup_openrouter_client()
    filtered_kwargs: dict = _select_values(_notnone, model_kwargs)

    if func_spec is not None:
        raise NotImplementedError(
            "We are not supporting function calling in OpenRouter for now."
        )

    messages = [
        {"role": "user", "content": message}
        for message in [system_message, user_message]
        if message
    ]

    t0 = time.time()
    completion = _backend_openrouter.backoff_create(
        _backend_openrouter._client.chat.completions.create,
        _backend_openrouter.OPENAI_TIMEOUT_EXCEPTIONS,
        messages=messages,
        **filtered_kwargs,
    )
    req_time = time.time() - t0

    output = completion.choices[0].message.content
    in_tokens = completion.usage.prompt_tokens
    out_tokens = completion.usage.completion_tokens

    info = {
        "system_fingerprint": completion.system_fingerprint,
        "model": completion.model,
        "created": completion.created,
    }
    return output, req_time, in_tokens, out_tokens, info


_aide_backend_pkg.provider_to_query_func["openrouter"] = _patched_openrouter_query


# КРИТИЧНЫЙ ПАТЧ, подтверждён эмпирически на реальном прогоне (aideml==0.2.2,
# aide/interpreter.py:138 в установленном пакете): Interpreter._run_session()
# исполняет код кандидата через exec(compile(code, ...), global_scope), где
# global_scope стартует как {} - ПУСТОЙ dict без ключа "__name__". Проверено
# напрямую: exec(code, {}) оставляет __name__ равным 'builtins' (через
# автоматически подставляемый __builtins__), а не '__main__' и не NameError.
# Значит для ЛЮБОГО кандидата, использующего стандартную идиому
# `if __name__ == "__main__": main()` (а её использует практически любой
# LLM-сгенерированный Python-скрипт), эта проверка молча False, main() ни разу
# не вызывается - без исключения, почти без вывода, is_buggy=False - и
# feedback-модель потом сочиняет правдоподобный, но полностью выдуманный отчёт
# и метрику по этому пустому выводу. Живой прогон подтвердил: "лучший"
# кандидат с primary=0.6714 на самом деле упал бы на build_features() при
# реальном исполнении (проверено запуском того же кода напрямую) - вместо
# этого main() тихо не вызвался, и метрика 0.6714 полностью выдумана моделью.
#
# Патчим здесь, а не в site-packages/aide/interpreter.py: (а) это правка кода
# aideml==0.2.2, а не нашего проекта - трогать сторонний пакет напрямую не
# нужно без явного запроса; (б) unconditional top-level monkeypatch переживает
# spawn-multiprocessing на macOS (child-процесс переисполняет модуль __main__
# при unpickling target, поэтому патч применяется заново и там же).
from aide.interpreter import Interpreter as _Interpreter
from aide.interpreter import exception_summary as _exception_summary


def _run_session(self, code_inq, result_outq, event_outq) -> None:
    # ВАЖНО: имя функции должно быть буквально "_run_session" (совпадать с именем
    # оригинального метода), а не "_patched_run_session" - macOS использует spawn
    # для multiprocessing, и Process(target=self._run_session) пиклит bound method
    # по __name__ функции, реконструируя в child-процессе через getattr(instance,
    # func.__name__). Если имена не совпадают, unpickling падает с AttributeError
    # (проверено эмпирически - именно так это и сломалось при первой версии патча).
    self.child_proc_setup(result_outq)

    global_scope: dict = {"__name__": "__main__"}  # <-- единственная реальная правка
    while True:
        code = code_inq.get()
        os.chdir(str(self.working_dir))
        with open(self.agent_file_name, "w") as f:
            f.write(code)

        event_outq.put(("state:ready",))
        try:
            exec(compile(code, self.agent_file_name, "exec"), global_scope)
        except BaseException as e:
            tb_str, e_cls_name, exc_info, exc_stack = _exception_summary(
                e, self.working_dir, self.agent_file_name, self.format_tb_ipython
            )
            result_outq.put(tb_str)
            if e_cls_name == "KeyboardInterrupt":
                e_cls_name = "TimeoutError"
            event_outq.put(("state:finished", e_cls_name, exc_info, exc_stack))
        else:
            event_outq.put(("state:finished", None, None, None))

        os.remove(self.agent_file_name)
        result_outq.put("<|EOF|>")


_Interpreter._run_session = _run_session


# ============================================================================
# FEATURE 1: Automated candidate verification (anti-fabrication check)
# ============================================================================
# Every fabrication/leakage bug found this session (the __name__ silent-skip bug,
# the leaky user-author affinity feature) was caught by manually re-executing a
# candidate in isolation and cross-checking its recorded metric. This automates
# exactly that check, but ONLY against valid.csv (never test.csv/held_out_test) -
# it must not become a channel for indirect test-set tuning inside the search loop.
import re
import subprocess

from aide.utils.metric import WorstMetricValue


def setup_verify_workspace(cfg):
    """Persistent, reused verification workspace - copy input/ data ONCE, not per check."""
    verify_dir = cfg.log_dir / "_verify_workspace"
    (verify_dir / "input").mkdir(parents=True, exist_ok=True)
    (verify_dir / "working").mkdir(parents=True, exist_ok=True)
    for item in Path(cfg.data_dir).iterdir():
        dest = verify_dir / "input" / item.name
        if not dest.exists():
            if item.is_dir():
                shutil.copytree(item, dest)
            else:
                shutil.copy(item, dest)
    return verify_dir


_METRIC_RE = re.compile(r"primary[^0-9\-]{0,30}(-?[0-9]*\.?[0-9]+)", re.IGNORECASE)


def verify_candidate(node, verify_dir, timeout_sec):
    """Re-execute node.code in an isolated copy and sanity-check the result.
    Returns (ok: bool, reason: str). Never touches test.csv/held_out_test - only
    re-derives a valid-set metric the same way the candidate itself already does."""
    runfile = verify_dir / "verify_run.py"
    runfile.write_text(node.code)
    try:
        result = subprocess.run(
            ["python3", str(runfile.name)],
            cwd=str(verify_dir), capture_output=True, text=True, timeout=timeout_sec,
        )
    except subprocess.TimeoutExpired:
        return False, "verification re-run timed out"
    finally:
        runfile.unlink(missing_ok=True)

    if result.returncode != 0:
        return False, f"verification re-run crashed (exit {result.returncode}): {result.stderr[-300:]}"

    term_out = result.stdout
    if len(term_out.strip()) < 150:
        # exact signature of the __name__ silent-skip fabrication bug this session found:
        # a script whose main() never actually ran prints almost nothing.
        return False, f"verification re-run produced suspiciously little output ({len(term_out.strip())} chars) - possible silent no-op"

    # Use the LAST match, not the first: many candidates (including the seeded FM
    # baseline itself) print a "primary" value every epoch as training progress, then
    # a final summary line - .search() would grab epoch 1's (low, pre-convergence)
    # value instead of the real final result. Caught empirically: baseline_seed_code.py
    # false-flagged as a mismatch (0.5869 epoch-1 vs 0.6015 final) before this fix.
    matches = _METRIC_RE.findall(term_out)
    if not matches:
        return False, "verification re-run finished but no 'primary' metric found in its output"
    recomputed = float(matches[-1])
    recorded = node.metric.value
    if recorded is not None and abs(recomputed - recorded) > 0.01:
        return False, f"recomputed primary {recomputed:.4f} disagrees with recorded {recorded:.4f} (diff > 0.01)"

    return True, f"verified: recomputed primary {recomputed:.4f} matches recorded {recorded:.4f}"


# ============================================================================
# FEATURE 2: Cross-run persistent memory (findings.jsonl)
# ============================================================================
# Every python3 run_with_early_stop.py invocation starts a fresh Journal with zero
# memory of what prior runs found - this whole session, *I* manually re-briefed each
# new run ("run 7 found LambdaRank+ensemble works, run 9 found single-seed doesn't...").
# This automates that: append a compact record after each run, load+inject prior
# records into task_desc before the next run's Agent is constructed.
FINDINGS_PATH = Path(__file__).parent / "findings.jsonl"


# ============================================================================
# FEATURE 4: Research-notes injection
# ============================================================================
# HONEST LIMITATION: this bare Python script has no web-search API configured, so it
# cannot autonomously do literature research itself the way the orchestrating assistant
# did this session (WebSearch/WebFetch tool calls). What IS automatable is the
# INJECTION side: research_notes.md is loaded and appended to task_desc automatically
# at run start, same as findings.jsonl - the file just has to be populated by whoever
# has real search access (the assistant, or the user) between runs, not by this script.
RESEARCH_NOTES_PATH = Path(__file__).parent / "research_notes.md"


def load_research_notes():
    if not RESEARCH_NOTES_PATH.exists():
        return ""
    return RESEARCH_NOTES_PATH.read_text().strip()


def load_prior_findings(max_entries=8):
    if not FINDINGS_PATH.exists():
        return []
    records = []
    for line in FINDINGS_PATH.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records[-max_entries:]


def format_findings_block(records):
    if not records:
        return ""
    lines = [
        "\n\n## Findings from prior automated runs (persistent cross-run memory)\n",
        "Each line is one prior run's outcome. Use this to avoid repeating what's "
        "already been tried and to build on what worked - do not treat this as "
        "exhaustive, but weight it heavily when choosing what to try next.\n",
        "| run | best valid primary | approach (from winning candidate's own plan) |",
        "|---|---|---|",
    ]
    for r in records:
        plan = (r.get("winning_plan") or "").replace("\n", " ").strip()
        if len(plan) > 160:
            plan = plan[:157] + "..."
        lines.append(f"| {r.get('exp_name', '?')} | {r.get('best_valid_metric')} | {plan or '(no candidate succeeded)'} |")
    return "\n".join(lines) + "\n"


def append_finding(exp_name, best_valid_metric, winning_plan, stop_reason):
    FINDINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    rec = {
        "exp_name": exp_name,
        "best_valid_metric": best_valid_metric,
        "winning_plan": winning_plan,
        "stop_reason": stop_reason,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with open(FINDINGS_PATH, "a") as f:
        f.write(json.dumps(rec) + "\n")


# ============================================================================
# FEATURE 3: Automated ensemble search action
# ============================================================================
# AIDE's search_policy() only knows draft/debug/improve-one-lineage - it has no notion
# of "combine my top-K diverse candidates". This session's single best result was
# exactly that (a validated blend of 3 architecturally different models beating every
# individual one), done entirely by hand. This automates it: after the search loop
# ends, pick the best candidate from each distinct lineage (genuine architectural
# diversity, not near-duplicate children of the same parent), and deterministically
# generate+run a blending script - NOT asking the LLM to write the blend logic, since
# this session repeatedly found the weaker code model unreliable at multi-part
# instructions; the harness building it directly is more robust.
ENSEMBLE_SCRIPT_TEMPLATE = '''import sys, subprocess, itertools, os
sys.path.insert(0, "./input")
import numpy as np
import pandas as pd
from evaluate import evaluate

SOURCE_SCRIPTS = {source_scripts!r}  # list of (name, code) tuples, embedded by the harness

valid_preds, test_preds = {{}}, {{}}
for name, code in SOURCE_SCRIPTS:
    fname = f"_ens_{{name}}.py"
    with open(fname, "w") as f:
        f.write(code)
    print(f"--- running source candidate: {{name}} ---")
    subprocess.run(["python3", fname], check=True)
    valid_preds[name] = np.load("./working/valid_scores.npy")
    test_preds[name] = np.load("./working/test_scores.npy")
    os.remove(fname)

valid = pd.read_csv("./input/valid.csv")
test = pd.read_csv("./input/test.csv")
uva, yva = valid["user_id"].to_numpy(), valid["long_view"].to_numpy()
uta = test["user_id"].to_numpy()

def rank_norm(x, u):
    return pd.Series(x).groupby(u).rank(pct=True).values

names = list(valid_preds.keys())
valid_ranks = {{n: rank_norm(valid_preds[n], uva) for n in names}}
test_ranks = {{n: rank_norm(test_preds[n], uta) for n in names}}

grid = [0.0, 0.2, 0.25, 0.33, 0.4, 0.5, 0.6, 0.8, 1.0]
best_w, best_p = None, -1.0
for combo in itertools.product(grid, repeat=len(names) - 1):
    w = list(combo) + [1.0 - sum(combo)]
    if w[-1] < -0.001 or w[-1] > 1.001:
        continue
    w[-1] = max(0.0, w[-1])
    blend = sum(wi * valid_ranks[n] for wi, n in zip(w, names))
    r = evaluate(uva, yva, blend)
    if r["primary"] > best_p:
        best_p, best_w = r["primary"], w

print("Best blend weights:", dict(zip(names, best_w)))
final_valid_scores = sum(wi * valid_ranks[n] for wi, n in zip(best_w, names))
final_test_scores = sum(wi * test_ranks[n] for wi, n in zip(best_w, names))
r = evaluate(uva, yva, final_valid_scores)
print(f"Ensemble Validation Results -> GAUC: {{r['GAUC']:.4f}}, nDCG@5: {{r['nDCG@5']:.4f}}, primary: {{r['primary']:.4f}}")

os.makedirs("./working", exist_ok=True)
submission = pd.DataFrame({{
    "row_id": test["row_id"].astype(int),
    "user_id": test["user_id"].astype(int),
    "video_id": test["video_id"].astype(int),
    "score": final_test_scores,
}})
submission.to_csv("./working/submission.csv", index=False)
np.save("./working/valid_scores.npy", final_valid_scores)
np.save("./working/test_scores.npy", final_test_scores)
print("Saved ensemble submission to ./working/submission.csv successfully.")
'''


def get_root(node):
    while node.parent is not None:
        node = node.parent
    return node


def select_diverse_topk(journal, k=3, require_npy_hint=True):
    """Best node per distinct lineage (root ancestor), top-k by metric. Only considers
    nodes whose code looks like it saves the .npy arrays the ensemble script needs -
    older/incompatible nodes are silently skipped rather than crashing the ensemble."""
    good = [n for n in journal.good_nodes if n.metric is not None and n.metric.value is not None]
    if require_npy_hint:
        good = [n for n in good if "valid_scores.npy" in n.code and "test_scores.npy" in n.code]
    best_per_root = {}
    for n in good:
        root_id = get_root(n).id
        if root_id not in best_per_root or n.metric.value > best_per_root[root_id].metric.value:
            best_per_root[root_id] = n
    candidates = sorted(best_per_root.values(), key=lambda n: n.metric.value, reverse=True)
    return candidates[:k]


def run_ensemble_step(journal, agent, interpreter, k=3):
    """Build and evaluate an ensemble of the top-k architecturally-diverse good nodes.
    Returns the new Node if it ran successfully, else None (never raises - a failed
    ensemble attempt should not crash the whole run)."""
    diverse = select_diverse_topk(journal, k=k)
    if len(diverse) < 2:
        print(f"  ensemble step skipped: only {len(diverse)} distinct npy-compatible good node(s) available (need >=2)")
        return None

    print(f"  ensemble step: blending {len(diverse)} diverse candidates "
          f"(steps {[n.step for n in diverse]}, metrics {[round(n.metric.value, 4) for n in diverse]})")
    source_scripts = [(f"node{n.step}", n.code) for n in diverse]
    ensemble_code = ENSEMBLE_SCRIPT_TEMPLATE.format(source_scripts=source_scripts)

    try:
        exec_result = interpreter.run(ensemble_code, reset_session=True)
    except Exception as e:
        print(f"  ensemble step failed to execute: {e}")
        return None

    ens_node = Node(
        code=ensemble_code,
        plan=(
            f"Automated ensemble (harness-generated, not LLM-written): rank-normalized "
            f"weighted blend of the best candidate from each of {len(diverse)} distinct "
            f"lineages (steps {[n.step for n in diverse]}), weights grid-searched on "
            f"valid.csv only."
        ),
    )
    agent.parse_exec_result(node=ens_node, exec_result=exec_result)
    journal.append(ens_node)
    print(f"  ensemble result: is_buggy={ens_node.is_buggy} metric={ens_node.metric}")
    return ens_node


# ============================================================================
# FEATURE 5: Git-style diff/patch editing for Agent._improve()
# ============================================================================
# aide/agent.py's _improve() prompt ALREADY tells the model "only propose a single
# actionable improvement... atomic" - but the RESPONSE format is still "rewrite the
# whole file" (extract_code() pulls one full code block). The instruction to be
# minimal exists; the mechanism to actually stay minimal doesn't. This is exactly
# why several candidates this session silently dropped working features (multi-seed
# ensembling, correct dtype casting) when "improving" something - regenerating
# everything from scratch risks losing what already worked, even when only asked to
# change one thing. Real coding agents (Aider, Claude Code's own Edit tool) use
# SEARCH/REPLACE-style targeted edits over full rewrites for exactly this reason -
# unified diffs with line numbers are less reliable for LLMs to produce correctly
# than exact-text search/replace blocks.
#
# Only _improve() is patched, not _draft() (nothing to diff against yet) or _debug()
# (a crash could be anywhere in the file; full context is more appropriate there).
import re as _re

from aide.backend import query as _aide_query
from aide.utils.response import extract_text_up_to_code as _extract_text_up_to_code
from aide.utils.response import wrap_code as _wrap_code

_ORIGINAL_IMPROVE = Agent._improve

_SR_BLOCK_RE = _re.compile(r"<{7} SEARCH\n(.*?)\n={7}\n(.*?)\n>{7} REPLACE", _re.DOTALL)

_DIFF_RESPONSE_FORMAT = {
    "Response format": (
        "Your response should be a brief outline/sketch of the SINGLE targeted change you are "
        "making (3-5 sentences), followed by one or more SEARCH/REPLACE blocks in this EXACT "
        "format:\n\n"
        "<<<<<<< SEARCH\n"
        "<the exact existing code to find - copy it character-for-character from the code above>\n"
        "=======\n"
        "<the new code that replaces it>\n"
        ">>>>>>> REPLACE\n\n"
        "Do NOT rewrite the whole file. Only include SEARCH/REPLACE blocks for the specific lines "
        "that need to change. Keep each SEARCH block as short as possible while still being "
        "uniquely identifiable in the code (include a few lines of surrounding context only if "
        "needed to make the match unambiguous - it must match EXACTLY ONE location in the code). "
        "You may include multiple SEARCH/REPLACE blocks if the change touches multiple places."
    )
}


def _apply_search_replace(original_code, diff_text):
    """Returns (new_code, error). error is None on success, new_code is None on failure."""
    blocks = _SR_BLOCK_RE.findall(diff_text)
    if not blocks:
        return None, "no SEARCH/REPLACE blocks found in response"
    new_code = original_code
    for search, replace in blocks:
        count = new_code.count(search)
        if count == 0:
            return None, f"SEARCH block not found verbatim in code (first 150 chars): {search[:150]!r}"
        if count > 1:
            return None, f"SEARCH block matches {count} locations, ambiguous (first 150 chars): {search[:150]!r}"
        new_code = new_code.replace(search, replace, 1)
    return new_code, None


def _improve_diff(self, parent_node: Node) -> Node:
    prompt = {
        "Introduction": (
            "You are a Kaggle grandmaster attending a competition. You are provided with a "
            "previously developed solution below and should improve it by making a SINGLE, "
            "TARGETED edit - not rewriting the file. First outline a brief plan in natural "
            "language for how the solution can be improved, then express that improvement as "
            "a small, precise patch (not a full rewrite)."
        ),
        "Task description": self.task_desc,
        "Memory": self.journal.generate_summary(),
        "Instructions": {},
    }
    prompt["Previous solution"] = {"Code": _wrap_code(parent_node.code)}
    prompt["Instructions"] |= _DIFF_RESPONSE_FORMAT
    prompt["Instructions"] |= {
        "Solution improvement sketch guideline": [
            "You should be very specific and should only propose a single actionable improvement.",
            "This improvement should be atomic so we can experimentally evaluate the effect of the proposed change.",
            "Take the Memory section into consideration when proposing the improvement.",
            "The solution sketch should be 3-5 sentences.",
            "Don't suggest to do EDA.",
        ],
    }

    for attempt in range(3):
        completion_text = _aide_query(
            system_message=prompt, user_message=None,
            model=self.acfg.code.model, temperature=self.acfg.code.temp,
        )
        plan = _extract_text_up_to_code(completion_text) or ""
        new_code, error = _apply_search_replace(parent_node.code, completion_text)
        if new_code is not None:
            return Node(plan=plan, code=new_code, parent=parent_node)
        print(f"  [diff-improve] patch attempt {attempt + 1}/3 failed: {error} - retrying...")

    # Graceful fallback: couldn't get a valid patch after retries. Falling back to the
    # ORIGINAL full-rewrite _improve() for just this one attempt is strictly safer than
    # either crashing or silently returning a broken/unpatched node - matches this
    # session's established pattern (verification failures fall back to debug-eligible,
    # not a hard stop).
    print("  [diff-improve] giving up on patch format after 3 attempts, falling back to full-rewrite _improve()")
    return _ORIGINAL_IMPROVE(self, parent_node)


Agent._improve = _improve_diff


def main():
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit(
            "OPENAI_API_KEY не задан. Откройте aide_task/.env, вставьте ключ из "
            "build.nvidia.com вместо PASTE_YOUR_KEY_HERE и сохраните файл."
        )

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="./config.yaml", help="путь к нашему config.yaml (см. схему aide/utils/config.py)")
    ap.add_argument("--epsilon", type=float, default=0.002, help="порог значимого прироста primary")
    ap.add_argument("--patience", type=int, default=3, help="N - сколько подряд шагов без прироста > epsilon до останова")
    ap.add_argument("--wall_clock_sec", type=float, default=21600, help="хард-потолок по времени, сек (6ч по умолчанию)")
    ap.add_argument("--seed_baseline", type=str, default="./baseline_seed_code.py",
                     help="код официального FM baseline, засеивается как node 0 журнала перед стартом поиска "
                          "(чтобы _improve() мог реально строить на нём, а не только знать его число). "
                          "Пустая строка - не засеивать.")
    args = ap.parse_args()

    cfg = prep_cfg(OmegaConf.load(args.config))
    print(f'Starting run "{cfg.exp_name}" (early-stop: epsilon={args.epsilon}, patience={args.patience}, '
          f'wall_clock_sec={args.wall_clock_sec}, step_cap={cfg.agent.steps})')

    task_desc = load_task_desc(cfg)
    prior_findings = load_prior_findings()
    if prior_findings:
        task_desc = task_desc + format_findings_block(prior_findings)
        print(f"Injected {len(prior_findings)} prior-run findings into task_desc from {FINDINGS_PATH}")

    research_notes = load_research_notes()
    if research_notes:
        task_desc = task_desc + "\n\n## Research notes (literature findings)\n\n" + research_notes + "\n"
        print(f"Injected {len(research_notes)} chars of research notes into task_desc from {RESEARCH_NOTES_PATH}")

    print("Preparing agent workspace (copying and extracting files) ...")
    prep_agent_workspace(cfg)

    journal = Journal()
    agent = Agent(task_desc=task_desc, cfg=cfg, journal=journal)
    interpreter = Interpreter(cfg.workspace_dir, **OmegaConf.to_container(cfg.exec))

    # Засеиваем journal официальным FM baseline (baseline.py, адаптирован под layout
    # ./input/./working в baseline_seed_code.py) как node 0, ДО начала поиска. Раньше
    # агент только знал ЧИСЛО baseline (0.5946 test) из description.md и всегда писал
    # решение с нуля - "baseline как starting point" не работало, т.к. AIDE's _draft()
    # не принимает seed-код напрямую. Правильный путь: узел с parent=None (считается
    # draft_node) и реальной evaluated-метрикой в journal ДО старта - тогда
    # search_policy()'s _improve(greedy_node) может выбрать именно baseline, если он
    # окажется текущим лучшим (что реально бывает - baseline валид ~0.60 обгонял
    # многие свежие черновики агента в этой сессии).
    if args.seed_baseline:
        seed_path = Path(args.seed_baseline)
        if seed_path.exists():
            print(f"Seeding journal with official FM baseline from {seed_path} ...")
            seed_code = seed_path.read_text()
            seed_result = interpreter.run(seed_code, reset_session=True)
            seed_node = Node(
                code=seed_code,
                plan=(
                    "Official FM baseline (organizer-provided reference, baseline.py "
                    "adapted to this task's train/valid/test layout). Seeded as the "
                    "starting point for the search - improve on this directly via "
                    "targeted edits where possible, not just as a score to beat."
                ),
            )
            agent.parse_exec_result(node=seed_node, exec_result=seed_result)
            journal.append(seed_node)
            print(f"  baseline seed: is_buggy={seed_node.is_buggy} metric={seed_node.metric}")
        else:
            print(f"  --seed_baseline path {seed_path} not found, skipping seed (fresh search).")

    global_step = len(journal)

    def cleanup():
        if global_step == 0:
            shutil.rmtree(cfg.workspace_dir, ignore_errors=True)

    atexit.register(cleanup)

    # В aideml==0.2.2 нет глобального journal.metric_maximize - LLM сообщает
    # направление per-node (lower_is_better в review_func_spec). Наша задача
    # (GAUC/nDCG@5/primary) всегда maximize - это захардкожено в description.md
    # инструкцией "primary - чем выше, тем лучше", и здесь для сравнения best-so-far.
    def improved(cur, prev):
        return (cur - prev) > args.epsilon

    log_path = cfg.log_dir / "early_stop_log.jsonl"
    cfg.log_dir.mkdir(parents=True, exist_ok=True)
    verify_dir = setup_verify_workspace(cfg)
    print(f"Verification workspace ready at {verify_dir} (re-checks every new-best candidate against valid.csv only)")

    best_so_far = None
    bad_steps = 0
    stop_reason = None
    t_start = time.monotonic()

    while global_step < cfg.agent.steps:
        agent.step(exec_callback=interpreter.run)
        save_run(cfg, journal)
        global_step = len(journal)
        elapsed = time.monotonic() - t_start

        best_node = journal.get_best_node(only_good=True)
        cur_best = best_node.metric.value if best_node is not None else None

        # Candidate would become the new tracked best - verify it BEFORE trusting it.
        if cur_best is not None and (best_so_far is None or improved(cur_best, best_so_far)):
            print(f"  candidate for new best (primary={cur_best:.4f}) - re-executing in isolation to verify...")
            ok, reason = verify_candidate(best_node, verify_dir, timeout_sec=cfg.exec.timeout)
            print(f"  verification: {'PASSED' if ok else 'FAILED'} - {reason}")
            if not ok:
                # Same treatment aideml itself gives a failed candidate: mark buggy so
                # get_best_node/search_policy naturally exclude it (and it becomes
                # eligible for debug on a later step, which might genuinely fix it).
                # Only THIS step's candidate gets verified before acceptance - if
                # invalidating it exposes another unverified node as the new best, it
                # gets checked on a LATER step when it would next be accepted, not
                # immediately re-checked here (keeps per-step cost bounded).
                best_node.is_buggy = True
                best_node.analysis = (best_node.analysis or "") + f"\n[AUTO-VERIFY FAILED] {reason}"
                best_node.metric = WorstMetricValue()
                save_run(cfg, journal)
                best_node = journal.get_best_node(only_good=True)
                cur_best = best_node.metric.value if best_node is not None else None

        # FIX (task #6): AIDE's search_policy() forces the first num_drafts steps to be
        # independent fresh drafts (search_cfg.num_drafts gate in aide/agent.py), and can
        # ALSO fall back to another fresh draft later if all current good_nodes disappear
        # (e.g. our own auto-verification just invalidated the only good candidate) - both
        # are intentional EXPLORATION, not failed improvement attempts. With a low
        # patience (the real competition spec uses N=3), counting these against bad_steps
        # risks a plateau-stop DURING mandatory drafting, before the search ever reaches
        # _improve(). Confirmed by inspecting our own run logs: with num_drafts=5 and
        # patience=3, 3 non-improving drafts alone would exhaust the budget.
        #
        # IMPORTANT: checking len(journal.draft_nodes) <= num_drafts would be WRONG here -
        # that count stays fixed at num_drafts for the rest of the run once the initial
        # phase ends (no more draft-type nodes get created in the common case), so a "<="
        # check would stay permanently true and silently disable patience-counting for the
        # entire rest of the run, not just the draft phase. Check whether the node THIS
        # STEP actually produced is itself a draft instead (parent is None) - correct
        # regardless of why search_policy chose to draft.
        still_drafting = journal.nodes[-1].parent is None

        if cur_best is None:
            note = "все попытки пока багованные - плато не считается"
        elif best_so_far is None or improved(cur_best, best_so_far):
            best_so_far = cur_best
            bad_steps = 0
            note = "новый лучший результат (проверено)"
        elif still_drafting:
            note = f"без прироста, но ещё в фазе обязательных черновиков ({len(journal.draft_nodes)}/{cfg.agent.search.num_drafts}) - в bad_steps не считается"
        else:
            bad_steps += 1
            note = f"без значимого прироста ({bad_steps}/{args.patience})"

        rec = {"step": global_step, "elapsed_sec": round(elapsed, 1),
               "cur_best": cur_best, "best_so_far": best_so_far,
               "bad_steps": bad_steps, "note": note}
        print(f"[step {global_step}/{cfg.agent.steps}] best_so_far={best_so_far} "
              f"elapsed={elapsed:.0f}s | {note}")
        with open(log_path, "a") as f:
            f.write(json.dumps(rec) + "\n")

        if elapsed >= args.wall_clock_sec:
            stop_reason = f"wall-clock ceiling {args.wall_clock_sec}s reached at step {global_step}"
            break
        if cur_best is not None and bad_steps >= args.patience:
            stop_reason = (f"plateau: no improvement > {args.epsilon} over last "
                            f"{args.patience} consecutive iterations (stopped at step {global_step})")
            break

    if stop_reason is None:
        stop_reason = f"step cap {cfg.agent.steps} reached"
    print(f"\nRun stopped: {stop_reason}")
    print(f"Best validation metric: {best_so_far}")

    print("\nAttempting automated ensemble of diverse good candidates...")
    ens_node = run_ensemble_step(journal, agent, interpreter, k=3)
    if ens_node is not None and not ens_node.is_buggy:
        save_run(cfg, journal)
        true_best = journal.get_best_node(only_good=True)
        if true_best is ens_node:
            print(f"  ensemble IS the new best (primary={ens_node.metric.value:.4f}), promoted.")
            best_so_far = ens_node.metric.value
        else:
            print(f"  ensemble did not beat the existing best (primary={ens_node.metric.value:.4f} "
                  f"vs {true_best.metric.value:.4f}), keeping the existing best.")

    interpreter.cleanup_session()

    with open(cfg.log_dir / "stop_reason.json", "w") as f:
        json.dump({"stop_reason": stop_reason, "best_valid_metric": best_so_far,
                    "total_steps": global_step, "elapsed_sec": round(time.monotonic() - t_start, 1)}, f, indent=2)

    # Use the TRUE best node (not the epsilon-gated best_so_far tracker, which has
    # repeatedly understated the real best this session) for the persisted finding.
    true_best_node = journal.get_best_node(only_good=True)
    append_finding(
        exp_name=cfg.exp_name,
        best_valid_metric=(true_best_node.metric.value if true_best_node else None),
        winning_plan=(true_best_node.plan if true_best_node else None),
        stop_reason=stop_reason,
    )
    print(f"Appended finding to {FINDINGS_PATH} for future runs")

    if cfg.generate_report:
        print("Generating final report from journal...")
        report = journal2report(journal, task_desc, cfg.report)
        report_file_path = cfg.log_dir / "report.md"
        with open(report_file_path, "w") as f:
            f.write(report)
        print("Report written to file:", report_file_path)


if __name__ == "__main__":
    main()
