import argparse
import glob
import html
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


CSS = """
<style>
body { color: #1f2328; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
.muted { color: #57606a; }
.callout { border-left: 4px solid #0969da; background: #f6f8fa; padding: 12px 14px; margin: 12px 0 18px; }
.grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px; margin: 12px 0 18px; }
.kpi { border: 1px solid #d0d7de; border-radius: 6px; padding: 10px 12px; }
.kpi .label { color: #57606a; font-size: 12px; margin-bottom: 4px; }
.kpi .value { font-weight: 650; font-size: 18px; overflow-wrap: anywhere; }
table { border-collapse: collapse; width: 100%; font-size: 13px; }
th, td { border: 1px solid #d0d7de; padding: 6px 8px; text-align: left; vertical-align: top; }
th { background: #f6f8fa; }
img { max-width: 100%; border: 1px solid #d0d7de; border-radius: 6px; }
</style>
"""


def parse_args():
    parser = argparse.ArgumentParser(description="Export a compact TRQAM evaluation report from CSV logs.")
    parser.add_argument("--run-dir", type=Path, default=None, help="Specific run directory containing eval.csv.")
    parser.add_argument("--latest-glob", default=None, help="Glob pattern for run directories; newest directory is used.")
    parser.add_argument("--output-name", default="report.html", help="Report filename to write inside the run directory.")
    return parser.parse_args()


def resolve_run_dir(run_dir, latest_glob):
    if run_dir is not None:
        if not run_dir.is_dir():
            raise FileNotFoundError(f"Run directory does not exist: {run_dir}")
        return run_dir

    if not latest_glob:
        raise ValueError("Provide either --run-dir or --latest-glob")

    candidates = [Path(path) for path in glob.glob(latest_glob) if Path(path).is_dir()]
    candidates = [path for path in candidates if (path / "eval.csv").exists() or (path / "flags.json").exists()]
    if not candidates:
        raise FileNotFoundError(f"No run directories matched: {latest_glob}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def read_csv(path):
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def read_flags(path):
    if not path.exists():
        return {}
    with path.open() as f:
        return json.load(f)


def fmt(value):
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def last_value(df, key, default="n/a"):
    if df.empty or key not in df.columns:
        return default
    return fmt(df.iloc[-1][key])


def table_html(df, empty_message, max_rows=5):
    if df.empty:
        return f'<p class="muted">{html.escape(empty_message)}</p>'
    return df.tail(max_rows).to_html(index=False, border=0, escape=True, float_format=lambda x: f"{x:.6g}")


def card(label, value):
    return (
        f'<div class="kpi"><div class="label">{html.escape(str(label))}</div>'
        f'<div class="value">{html.escape(str(value))}</div></div>'
    )


def plot_metrics(df, title, candidates, out_dir):
    metrics = [
        metric
        for metric in candidates
        if metric in df.columns and pd.api.types.is_numeric_dtype(df[metric])
    ]
    if df.empty or not metrics or "step" not in df.columns:
        return None

    fig, ax = plt.subplots(figsize=(9, 4))
    df.plot(x="step", y=metrics, marker="o", ax=ax)
    ax.set_title(title)
    ax.set_xlabel("step")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    filename = title.lower().replace(" ", "_").replace("/", "_") + ".png"
    path = out_dir / filename
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def main():
    args = parse_args()
    run_dir = resolve_run_dir(args.run_dir, args.latest_glob)

    eval_df = read_csv(run_dir / "eval.csv")
    offline_df = read_csv(run_dir / "offline_agent.csv")
    online_df = read_csv(run_dir / "online_agent.csv")
    flags = read_flags(run_dir / "flags.json")

    fig_dir = run_dir / "figures"
    fig_dir.mkdir(exist_ok=True)
    figures = [
        plot_metrics(eval_df, "Evaluation metrics", ["success", "episode.return", "episode.length", "episode.final_reward"], fig_dir),
        plot_metrics(offline_df, "Offline agent diagnostics", ["actor/flow_loss", "actor/fast_loss", "actor/path_kl", "critic/critic_loss", "dual/lambda"], fig_dir),
        plot_metrics(online_df, "Online agent diagnostics", ["actor/flow_loss", "actor/fast_loss", "actor/path_kl", "critic/critic_loss", "dual/lambda"], fig_dir),
    ]
    figures = [path for path in figures if path is not None]

    cards = "".join(
        [
            card("Environment", flags.get("env_name", run_dir.parent.name)),
            card("Seed", flags.get("seed", "n/a")),
            card("Offline steps", flags.get("offline_steps", "n/a")),
            card("Online steps", flags.get("online_steps", "n/a")),
            card("Success", last_value(eval_df, "success")),
            card("Return", last_value(eval_df, "episode.return")),
            card("Episode length", last_value(eval_df, "episode.length")),
            card("Final eval step", last_value(eval_df, "step")),
        ]
    )

    flags_table = pd.DataFrame(sorted(flags.items()), columns=["Item", "Value"]).to_html(
        index=False,
        border=0,
        escape=True,
    )

    figure_html = ""
    for figure in figures:
        rel = figure.relative_to(run_dir).as_posix()
        title = figure.stem.replace("_", " ").title()
        figure_html += f"<h3>{html.escape(title)}</h3><img src=\"{html.escape(rel)}\" alt=\"{html.escape(title)}\">"
    if not figure_html:
        figure_html = '<p class="muted">No figures were generated.</p>'

    body = f"""
<h1>TRQAM Evaluation Report</h1>
<p class="muted">Generated at: {html.escape(time.strftime("%Y-%m-%d %H:%M:%S"))}</p>
<div class="callout">This report was generated from CSV logs in <code>{html.escape(str(run_dir))}</code>.</div>
<h2>Key Metrics</h2>
<div class="grid">{cards}</div>
<h2>Final Evaluation Log</h2>
{table_html(eval_df, "No evaluation logs.", max_rows=1)}
<h2>Recent Offline Logs</h2>
{table_html(offline_df, "No offline logs.", max_rows=5)}
<h2>Recent Online Logs</h2>
{table_html(online_df, "No online logs.", max_rows=5)}
<h2>Configuration</h2>
{flags_table}
<h2>Figures</h2>
{figure_html}
"""

    report = f"<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\"><title>TRQAM Evaluation Report</title>{CSS}</head><body style=\"max-width: 980px; margin: 32px auto; padding: 0 20px;\">{body}</body></html>"
    report_path = run_dir / args.output_name
    report_path.write_text(report, encoding="utf-8")

    print(f"run_dir={run_dir}")
    print(f"report={report_path}")
    if not eval_df.empty:
        print("final_eval=" + eval_df.tail(1).to_json(orient="records"))


if __name__ == "__main__":
    main()
