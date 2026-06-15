#!/usr/bin/env python3
"""runs/*/run.log (+ run_resume*.log) → summary.md + total.csv."""
import csv
import json
import re
import glob
import os
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(SCRIPT_DIR, "..")
DOURI_ROOT = os.path.normpath(os.path.join(PROJECT_ROOT, "..", "douri"))

CSV_COLUMNS = [
    "run_dir", "algo", "env", "ep", "gap", "maxgap", "N", "gamma",
    "eval_ep", "IDM", "ACTOR", "done", "subgoal", "critic",
    "tasks_idm", "tasks_actor",
]

RUN_TARGETS = [
    {
        "name": "Pathbridger",
        "runs_dir": os.path.join(PROJECT_ROOT, "runs"),
        "out_md": os.path.join(PROJECT_ROOT, "docs", "runs_results_summary.md"),
        "out_csv": os.path.join(PROJECT_ROOT, "docs", "runs_results_total.csv"),
    },
    {
        "name": "douri",
        "runs_dir": os.path.join(DOURI_ROOT, "runs"),
        "out_md": os.path.join(PROJECT_ROOT, "docs", "douri_runs_results_summary.md"),
        "out_csv": os.path.join(PROJECT_ROOT, "docs", "douri_runs_results_total.csv"),
    },
]


def parse_kv(line):
    out = {}
    for m in re.finditer(r"(\w+)=([^\s]+)", line):
        k, v = m.group(1), m.group(2)
        if k not in out:
            out[k] = v
    return out


def yaml_get(text, key):
    m = re.search(rf"^\s*{re.escape(key)}:\s*(\S+)", text, re.MULTILINE)
    return m.group(1).rstrip(",") if m else ""


def log_paths(run_dir):
    """run.log 다음 run_resume*.log(파일명 순)을 이어서 읽는다."""
    paths = []
    main = os.path.join(run_dir, "run.log")
    if os.path.isfile(main):
        paths.append(main)
    paths.extend(sorted(glob.glob(os.path.join(run_dir, "run_resume*.log"))))
    return paths


def parse_config(run_dir):
    params = {
        "gap": "", "maxgap": "", "N": "", "gamma": "",
        "subgoal": "", "critic": "",
    }
    cfg = os.path.join(run_dir, "config_used.yaml")
    if os.path.isfile(cfg):
        with open(cfg, errors="replace") as f:
            text = f.read()
        params["gap"] = yaml_get(text, "subgoal_value_gap_scale")
        params["maxgap"] = yaml_get(text, "subgoal_value_weight_max")
        params["N"] = yaml_get(text, "plan_candidates")
        params["gamma"] = yaml_get(text, "discount")
        params["subgoal"] = yaml_get(text, "subgoal_distribution")

    flags_path = os.path.join(run_dir, "flags.json")
    if os.path.isfile(flags_path):
        with open(flags_path) as f:
            data = json.load(f)
        dyn = data.get("dynamics", {})
        critic = data.get("critic_agent", {})
        top = data.get("flags", {})
        if not params["gap"]:
            params["gap"] = str(dyn.get("subgoal_value_gap_scale", ""))
        if not params["maxgap"]:
            params["maxgap"] = str(dyn.get("subgoal_value_weight_max", ""))
        if not params["N"]:
            params["N"] = str(top.get("plan_candidates", ""))
        if not params["gamma"]:
            params["gamma"] = str(critic.get("discount", dyn.get("discount", "")))
        if not params["subgoal"]:
            params["subgoal"] = str(dyn.get("subgoal_distribution", ""))
        params["critic"] = str(
            critic.get("critic_type", "") or critic.get("algorithm", "")
        )
    return params


def classify_algo(info):
    subgoal = info.get("subgoal", "").lower()
    critic = info.get("critic", "").lower()
    if subgoal == "flow":
        return None
    if critic == "dqc":
        return "DQC"
    if "trl" in critic:
        return "TRL"
    return critic.upper() if critic else "?"


def fmt_num(v):
    if v in ("", "None", "null"):
        return "-"
    try:
        f = float(v)
        return str(int(f)) if f == int(f) else str(f)
    except ValueError:
        return v


def parse_logs(paths):
    info = {
        "env": "", "train_epochs": "",
        "subgoal": "", "critic": "",
        "gap": "", "maxgap": "",
        "final_eval_epoch": "", "idm_mean": "", "actor_mean": "",
        "idm_tasks": "", "actor_tasks": "",
        "done": False,
    }
    eval_blocks = []
    cur = None

    for path in paths:
        with open(path, "r", errors="replace") as f:
            for line in f:
                line = line.rstrip("\n")
                if " run_setup " in line:
                    kv = parse_kv(line)
                    info["env"] = kv.get("env", info["env"])
                    te = kv.get("train_epochs", "")
                    if te.isdigit():
                        prev = info["train_epochs"]
                        info["train_epochs"] = str(max(int(te), int(prev))) if prev.isdigit() else te
                elif " subgoal " in line and "mode=" in line:
                    kv = parse_kv(line)
                    info["subgoal"] = kv.get("mode", info["subgoal"])
                    if kv.get("value_gap_scale"):
                        info["gap"] = kv["value_gap_scale"]
                    if kv.get("value_weight_max"):
                        info["maxgap"] = kv["value_weight_max"]
                elif " critic_actor " in line and "type=" in line:
                    kv = parse_kv(line)
                    info["critic"] = kv.get("type", info["critic"])
                    if kv.get("discount"):
                        info["gamma"] = kv["discount"]
                elif re.search(r"\bepoch=\d+ dyn=", line):
                    pass
                elif "=== EVAL START" in line:
                    m = re.search(r"epoch=(\d+)", line)
                    cur = {"epoch": m.group(1) if m else "", "idm": "", "actor": "",
                           "idm_tasks": [], "actor_tasks": []}
                elif cur is not None:
                    if "idm env_success_rate_mean=" in line:
                        cur["idm"] = line.split("=")[-1].strip()
                    elif "actor env_success_rate_mean=" in line:
                        cur["actor"] = line.split("=")[-1].strip()
                    elif re.search(r"idm task_\d+ env=", line):
                        cur["idm_tasks"].append(line.split("=")[-1].strip())
                    elif re.search(r"actor task_\d+ env=", line):
                        cur["actor_tasks"].append(line.split("=")[-1].strip())
                    elif "=== EVAL END" in line:
                        eval_blocks.append(cur)
                        cur = None
                if " done run_dir=" in line:
                    info["done"] = True

    if eval_blocks:
        last = eval_blocks[-1]
        info["final_eval_epoch"] = last["epoch"]
        info["idm_mean"] = last["idm"]
        info["actor_mean"] = last["actor"]
        info["idm_tasks"] = ",".join(last["idm_tasks"])
        info["actor_tasks"] = ",".join(last["actor_tasks"])
    return info


def fmt_actor(v):
    return v if v else "-"


def to_f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return -1.0


def best_by_env(rows, algo=None):
    subset = [r for r in rows if algo is None or r["algo"] == algo]
    envs = {}
    for r in subset:
        envs.setdefault(r["env"], []).append(r)
    return [(env, max(envs[env], key=lambda r: to_f(r["idm_mean"])))
            for env in sorted(envs)]


def write_best_table(lines, title, pairs):
    lines += [f"### {title}", ""]
    lines += ["| env | gap | maxgap | N | γ | IDM | ACTOR | run_dir |"]
    lines += ["| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for env, best in pairs:
        lines.append("| " + " | ".join([
            env, fmt_num(best["gap"]), fmt_num(best["maxgap"]),
            fmt_num(best["N"]), fmt_num(best["gamma"]),
            best["idm_mean"] or "-", fmt_actor(best["actor_mean"]),
            best["run_dir"],
        ]) + " |")
    lines.append("")


def collect_rows(runs_dir):
    rows = []
    skipped_flow = []
    if not os.path.isdir(runs_dir):
        return rows, skipped_flow
    for d in sorted(glob.glob(os.path.join(runs_dir, "*"))):
        if not os.path.isdir(d):
            continue
        paths = log_paths(d)
        if not paths:
            continue
        info = parse_config(d)
        info.update(parse_logs(paths))
        info["run_dir"] = os.path.basename(d)
        algo = classify_algo(info)
        if algo is None:
            skipped_flow.append(info["run_dir"])
            continue
        info["algo"] = algo
        rows.append(info)
    return rows, skipped_flow


def row_to_csv_record(r):
    def csv_num(key):
        v = r.get(key, "")
        return fmt_num(v) if v != "" else ""

    return {
        "run_dir": r["run_dir"],
        "algo": r["algo"],
        "env": r["env"],
        "ep": r["train_epochs"],
        "gap": csv_num("gap"),
        "maxgap": csv_num("maxgap"),
        "N": csv_num("N"),
        "gamma": csv_num("gamma"),
        "eval_ep": r["final_eval_epoch"],
        "IDM": r["idm_mean"],
        "ACTOR": r["actor_mean"],
        "done": "1" if r["done"] else "0",
        "subgoal": r["subgoal"],
        "critic": r["critic"],
        "tasks_idm": r["idm_tasks"],
        "tasks_actor": r["actor_tasks"],
    }


def write_csv(rows, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for r in rows:
            writer.writerow(row_to_csv_record(r))


def write_markdown(rows, skipped_flow, target):
    n_trl = sum(1 for r in rows if r["algo"] == "TRL")
    n_dqc = sum(1 for r in rows if r["algo"] == "DQC")
    csv_name = os.path.basename(target["out_csv"])

    lines = [
        f"# {target['name']} runs/ 결과 요약",
        "",
        f"자동 생성: {datetime.now().strftime('%Y-%m-%d %H:%M')} · `scripts/summarize_runs.py`",
        f"소스: `{target['runs_dir']}`",
        "",
        "**포함:** TRL + DQC (`subgoal=diag_gaussian`). **제외:** flow subgoal only.",
        "",
        f"총 **{len(rows)}** runs — TRL **{n_trl}**, DQC **{n_dqc}**"
        + (f" · flow 제외 **{len(skipped_flow)}**" if skipped_flow else "") + ".",
        "",
        "파라미터: **gap**, **maxgap**, **N**, **γ** · 성공률 = 마지막 `EVAL END`.",
        f"resume 로그(`run_resume*.log`)가 있으면 `run.log`에 이어 붙여 읽습니다.",
        f"전체 표 CSV: [`{csv_name}`]({csv_name}).",
        "",
        "## 환경별 best (IDM)",
        "",
    ]
    trl_best = best_by_env(rows, "TRL")
    dqc_best = best_by_env(rows, "DQC")
    if trl_best:
        write_best_table(lines, "TRL", trl_best)
    if dqc_best:
        write_best_table(lines, "DQC", dqc_best)
    if not trl_best and not dqc_best:
        lines.append("_해당 runs 없음._")
        lines.append("")

    header = ["run_dir", "algo", "env", "ep", "gap", "maxgap", "N", "γ",
              "eval_ep", "IDM", "ACTOR", "done"]
    for algo in ("TRL", "DQC"):
        subset = [r for r in rows if r["algo"] == algo]
        if not subset:
            continue
        lines += [f"## {algo}", ""]
        lines += ["| " + " | ".join(header) + " |"]
        lines += ["| " + " | ".join(["---"] * len(header)) + " |"]
        for r in subset:
            lines.append("| " + " | ".join([
                r["run_dir"], r["algo"], r["env"], r["train_epochs"],
                fmt_num(r["gap"]), fmt_num(r["maxgap"]), fmt_num(r["N"]), fmt_num(r["gamma"]),
                r["final_eval_epoch"], r["idm_mean"], fmt_actor(r["actor_mean"]),
                "✅" if r["done"] else "⏳",
            ]) + " |")
        lines.append("")

    detail_hdr = ["run_dir", "algo", "gap", "maxgap", "N", "γ",
                  "eval_ep", "IDM", "ACTOR", "tasks IDM", "tasks ACTOR"]
    lines += ["## 환경별 상세 (task별)", ""]
    envs = {}
    for r in rows:
        envs.setdefault(r["env"], []).append(r)
    for env in sorted(envs):
        lines += [f"### {env}", ""]
        lines += ["| " + " | ".join(detail_hdr) + " |"]
        lines += ["| " + " | ".join(["---"] * len(detail_hdr)) + " |"]
        for r in envs[env]:
            lines.append("| " + " | ".join([
                r["run_dir"], r["algo"],
                fmt_num(r["gap"]), fmt_num(r["maxgap"]), fmt_num(r["N"]), fmt_num(r["gamma"]),
                r["final_eval_epoch"], r["idm_mean"], fmt_actor(r["actor_mean"]),
                r["idm_tasks"], r["actor_tasks"],
            ]) + " |")
        lines.append("")

    if skipped_flow:
        lines += ["## 제외 (flow subgoal)", ""]
        for name in skipped_flow:
            lines.append(f"- `{name}`")
        lines.append("")

    os.makedirs(os.path.dirname(target["out_md"]), exist_ok=True)
    with open(target["out_md"], "w") as f:
        f.write("\n".join(lines) + "\n")


def summarize_target(target):
    if not os.path.isdir(target["runs_dir"]):
        print(f"skip {target['name']}: missing {target['runs_dir']}")
        return
    rows, skipped_flow = collect_rows(target["runs_dir"])
    write_markdown(rows, skipped_flow, target)
    write_csv(rows, target["out_csv"])
    n_trl = sum(1 for r in rows if r["algo"] == "TRL")
    n_dqc = sum(1 for r in rows if r["algo"] == "DQC")
    print(
        f"{target['name']}: wrote {target['out_md']} + {target['out_csv']} "
        f"({len(rows)} runs: TRL={n_trl} DQC={n_dqc}, flow excluded={len(skipped_flow)})"
    )


def main():
    for target in RUN_TARGETS:
        summarize_target(target)


if __name__ == "__main__":
    main()
