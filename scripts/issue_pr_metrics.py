#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "click>=8.0,<9.0",
#   "pandas>=2.0,<3.0",
#   "pydantic>=2.0.0,<3.0",
# ]
# ///
from __future__ import annotations

import json
import os
import re
import subprocess
from collections.abc import Iterable
from datetime import datetime, timedelta, timezone
from typing import Any, cast

import click
import pandas as pd
from pydantic import BaseModel, ConfigDict


# ----------------------------
# Data model
# ----------------------------
class Node(BaseModel):
    """Node representing an Issue or PullRequest from GitHub GraphQL API.

    Attributes
    ----------
    typename : str
        The GraphQL __typename field (e.g., "Issue" or "PullRequest")
    url : str
        The URL of the issue or pull request
    created_at : datetime
        The timestamp when the issue/PR was created
    closed_at : datetime | None
        The timestamp when the issue/PR was closed, or None if still open
    merged_at : datetime | None
        The timestamp when the PR was merged, or None if not merged (Issues always None)
    """

    model_config = ConfigDict(frozen=True)

    typename: str
    url: str
    created_at: datetime
    closed_at: datetime | None
    merged_at: datetime | None


# ----------------------------
# Validation (security hardening)
# ----------------------------
_RE_REPO = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")

# Allow typical labels but reject query-breaking/control chars.
# If you need emojis etc., consider a stricter escape strategy instead of allowlisting.
_RE_LABEL = re.compile(r"^[\w .\-\/]{1,50}$", re.UNICODE)

_MAX_LABELS = 20
_MAX_DAYS = 365
# GitHub Search APIs typically cap usable results at ~1000 per search.
_MAX_LIMIT = 1000


def _has_control_chars(s: str) -> bool:
    # Control chars including newlines are disallowed
    return any(ord(c) < 32 or ord(c) == 127 for c in s)


def validate_repo(repo: str) -> str:
    repo = repo.strip()
    if _has_control_chars(repo) or not _RE_REPO.fullmatch(repo):
        raise click.ClickException("Invalid --repo. Expected owner/name using [A-Za-z0-9_.-].")
    return repo


def validate_days(days: int) -> int:
    if days < 1 or days > _MAX_DAYS:
        raise click.ClickException(f"--days must be between 1 and {_MAX_DAYS}")
    return days


def validate_limit(limit: int) -> int:
    if limit < 1 or limit > _MAX_LIMIT:
        raise click.ClickException(f"--limit must be between 1 and {_MAX_LIMIT}")
    return limit


def validate_labels(labels: list[str]) -> list[str]:
    out: list[str] = []
    for label in labels:
        label = label.strip()
        if not label:
            continue
        if _has_control_chars(label) or '"' in label:
            raise click.ClickException(f"Invalid label (contains quotes/control chars): {label!r}")
        if not _RE_LABEL.fullmatch(label):
            raise click.ClickException(
                f"Invalid label: {label!r}. Allowed chars: letters/digits/underscore/space/dot/dash/slash (max 50)."
            )
        out.append(label)
    if len(out) > _MAX_LABELS:
        raise click.ClickException(f"Too many exclude labels (max {_MAX_LABELS})")
    return out


# ----------------------------
# GraphQL search (via gh)
# ----------------------------
GQL = """
query($q: String!, $cursor: String) {
  search(type: ISSUE, query: $q, first: 100, after: $cursor) {
    pageInfo { hasNextPage endCursor }
    nodes {
      __typename
      ... on Issue { url createdAt closedAt }
      ... on PullRequest { url createdAt closedAt mergedAt }
    }
  }
}
""".strip()


def iso_to_dt(s: str | None) -> datetime | None:
    if s is None:
        return None
    return datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(timezone.utc)


def dt_utc_date_str(dt: datetime) -> str:
    # dt must be UTC-aware; we treat the date part as UTC.
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%d")


def build_exclude_query(labels: list[str]) -> str:
    # labels have been validated; safe to quote
    return " ".join([f'-label:"{label}"' for label in labels])


def _looks_like_rate_limit(msg: str) -> bool:
    m = msg.lower()
    return (
        "rate limit" in m
        or "api rate limit" in m
        or "secondary rate limit" in m
        or "abuse detection" in m
    )


def _run_gh(args: list[str], timeout_sec: int = 60) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            args,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout_sec,
        )
    except subprocess.TimeoutExpired as e:
        raise click.ClickException(
            f"gh command timed out after {timeout_sec}s: {' '.join(args[:4])} ..."
        ) from e


def ensure_repo_exists(repo: str) -> None:
    proc = _run_gh(["gh", "repo", "view", repo, "--json", "nameWithOwner"], timeout_sec=30)
    if proc.returncode != 0:
        raise click.ClickException(
            f"repo not found or no access: {repo}\ngh error: {(proc.stderr or '').strip()}"
        )


def gh_graphql(query: str, variables: dict[str, Any]) -> dict[str, Any]:
    # Use -f key=value. Values are validated (no control chars/quotes in labels; repo safe).
    args = ["gh", "api", "graphql", "-f", f"query={query}"]
    for k, v in variables.items():
        if v is None:
            continue
        if isinstance(v, str) and _has_control_chars(v):
            raise click.ClickException(f"Refusing control characters in variable {k}")
        args += ["-f", f"{k}={v}"]

    proc = _run_gh(args, timeout_sec=90)
    if proc.returncode != 0:
        msg = (proc.stderr or "").strip() or "gh api graphql failed"
        hint = ""
        if _looks_like_rate_limit(msg):
            hint = (
                "\n\nHint: you may be rate-limited. Try:\n"
                "- reduce --days\n"
                "- reduce --limit (e.g. 500)\n"
                "- retry later\n"
            )
        raise click.ClickException(msg + hint)

    out = (proc.stdout or "").strip()
    if not out:
        raise click.ClickException("gh api graphql returned empty output")

    try:
        parsed = json.loads(out)
    except json.JSONDecodeError as err:
        raise click.ClickException("gh api graphql returned non-JSON output") from err

    if not isinstance(parsed, dict):
        raise click.ClickException(
            f"gh api graphql returned unexpected JSON type: {type(parsed).__name__}"
        )
    data = cast(dict[str, Any], parsed)

    if data.get("errors"):
        err_txt = json.dumps(data["errors"], ensure_ascii=False)
        hint = ""
        if _looks_like_rate_limit(err_txt):
            hint = (
                "\n\nHint: you may be rate-limited. Try:\n"
                "- reduce --days\n"
                "- reduce --limit (e.g. 500)\n"
                "- retry later\n"
            )
        raise click.ClickException("GraphQL errors: " + err_txt + hint)

    return data


def gql_search_all(search_query: str, limit: int) -> list[Node]:
    """
    Search all matching nodes (Issue / PullRequest) via GraphQL search with pagination.
    Stops when `limit` nodes have been collected (after dedupe).
    """
    nodes: list[Node] = []
    cursor: str | None = None

    while True:
        variables: dict[str, Any] = {"q": search_query}
        if cursor is not None:
            variables["cursor"] = cursor

        data = gh_graphql(GQL, variables)
        search = data["data"]["search"]

        for n in search["nodes"]:
            created = iso_to_dt(n.get("createdAt"))
            if not created:
                continue
            nodes.append(
                Node(
                    typename=n.get("__typename", ""),
                    url=n["url"],
                    created_at=created,
                    closed_at=iso_to_dt(n.get("closedAt")),
                    merged_at=iso_to_dt(n.get("mergedAt")),
                )
            )

        # bound pagination work to avoid runaway in huge repos
        if len(nodes) >= max(limit * 2, limit + 200):
            break

        page = search["pageInfo"]
        if not page["hasNextPage"]:
            break
        cursor = page["endCursor"]

    # dedupe by url, then cap to limit
    seen = set()
    uniq: list[Node] = []
    for it in nodes:
        if it.url in seen:
            continue
        seen.add(it.url)
        uniq.append(it)
        if len(uniq) >= limit:
            break
    return uniq


def durations_seconds(pairs: Iterable[tuple[datetime, datetime]]) -> list[float]:
    out: list[float] = []
    for a, b in pairs:
        sec = (b - a).total_seconds()
        if sec >= 0:
            out.append(sec)
    return out


def _quantiles(s: pd.Series, qs: list[float]) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    if len(s) == 0:
        for q in qs:
            out[f"p{int(q * 100)}_sec"] = None
        return out
    for q in qs:
        out[f"p{int(q * 100)}_sec"] = float(s.quantile(q))
    return out


def stats_full(seconds: list[float]) -> dict[str, Any]:
    """
    Summary stats for raw + 1.5*IQR filtered.
    - count, mean, std, min, max
    - p50/p75/p90/p95
    - iqr_* equivalents
    - removed
    - notes: list[str] (e.g., 'iqr_zero_no_filter')
    """
    qs = [0.50, 0.75, 0.90, 0.95]

    if not seconds:
        base: dict[str, Any] = {
            "count": 0,
            "mean_sec": None,
            "std_sec": None,
            "min_sec": None,
            "max_sec": None,
            "removed": 0,
            "notes": [],
        }
        base.update(_quantiles(pd.Series([], dtype="float64"), qs))
        base.update({f"iqr_{k}": None for k in ["mean_sec", "std_sec", "min_sec", "max_sec"]})
        base.update({f"iqr_{k}": None for k in [f"p{int(q * 100)}_sec" for q in qs]})
        return base

    s = pd.Series(seconds, dtype="float64")

    # raw stats
    raw_std = float(s.std(ddof=1)) if len(s) >= 2 else float("nan")

    q1 = float(s.quantile(0.25))
    q3 = float(s.quantile(0.75))
    iqr = q3 - q1

    notes: list[str] = []

    # IQR filtering safety: if iqr==0, filtering can "explode" and remove almost everything.
    if iqr == 0:
        kept = s
        removed = 0
        notes.append("iqr_zero_no_filter")
    else:
        low = q1 - 1.5 * iqr
        high = q3 + 1.5 * iqr
        kept = s[(s >= low) & (s <= high)]
        removed = int(len(s) - len(kept))

    kept_std = float(kept.std(ddof=1)) if len(kept) >= 2 else float("nan")

    def nan_to_none(x: float) -> float | None:
        return None if pd.isna(x) else float(x)

    out: dict[str, Any] = {
        "count": int(len(s)),
        "mean_sec": float(s.mean()),
        "std_sec": nan_to_none(raw_std),
        "min_sec": float(s.min()),
        "max_sec": float(s.max()),
        "removed": removed,
        "notes": notes,
    }
    out.update(_quantiles(s, qs))

    # filtered stats
    out["iqr_mean_sec"] = float(kept.mean()) if len(kept) else None
    out["iqr_std_sec"] = nan_to_none(kept_std)
    out["iqr_min_sec"] = float(kept.min()) if len(kept) else None
    out["iqr_max_sec"] = float(kept.max()) if len(kept) else None
    # filtered quantiles
    q_kept = _quantiles(kept, qs)
    for k, v in q_kept.items():
        out[f"iqr_{k}"] = v

    # heuristic warning if removal seems excessive
    if len(s) >= 10 and removed / len(s) > 0.2:
        out["notes"].append("high_outlier_removal")

    return out


def fmt_days(x: float | None) -> str:
    return "N/A" if x is None else f"{x / 86400:.2f}d"


# ----------------------------
# Fixed-width table printing
# ----------------------------
def _cell_left(s: str, w: int) -> str:
    return (s[:w]).ljust(w)


def _cell_right(s: str, w: int) -> str:
    return (s[:w]).rjust(w)


def _row(cells: list[str], widths: list[int], align_right: list[bool]) -> str:
    parts = []
    for v, w, r in zip(cells, widths, align_right, strict=True):
        parts.append(_cell_right(v, w) if r else _cell_left(v, w))
    return " | ".join(parts)


def _sep(widths: list[int]) -> str:
    return "-+-".join("-" * w for w in widths)


def emit_counts_fixed(
    total: dict[str, int], created: dict[str, int], completed: dict[str, int]
) -> None:
    net = {
        "issues": created["issues"] - completed["issues"],
        "prs": created["prs"] - completed["prs"],
    }
    headers = ["Type", "Total", "Created", "Closed/Merged", "Net"]
    widths = [8, 7, 7, 13, 5]
    right = [False, True, True, True, True]

    click.echo("### Counts")
    click.echo(
        "This table summarizes the volume of Issues and Pull Requests during the time window.\n"
        "- Total: items that were open at any point during the window.\n"
        "- Created: items created during the window.\n"
        "- Closed/Merged: Issues closed during the window; PRs merged during the window (merged only).\n"
        "- Net: Created − Closed/Merged.\n"
    )
    click.echo(_row(headers, widths, right))
    click.echo(_sep(widths))
    click.echo(
        _row(
            [
                "Issues",
                str(total["issues"]),
                str(created["issues"]),
                str(completed["issues"]),
                str(net["issues"]),
            ],
            widths,
            right,
        )
    )
    click.echo(
        _row(
            ["PRs", str(total["prs"]), str(created["prs"]), str(completed["prs"]), str(net["prs"])],
            widths,
            right,
        )
    )
    click.echo()


def emit_stats_table_fixed(title: str, rows: list[tuple[str, dict[str, Any]]]) -> None:
    if title:
        click.echo(title)

    headers = [
        "Type",
        "Count",
        "Mean",
        "Std",
        "Min",
        "Max",
        "p50",
        "p75",
        "p90",
        "p95",
        "Mean(IQR)",
        "Max(IQR)",
        "p90(IQR)",
        "Removed",
        "Notes",
    ]
    widths = [20, 7, 10, 10, 10, 10, 8, 8, 8, 8, 12, 10, 10, 8, 14]
    right = [False] + [True] * 13 + [False]

    click.echo(_row(headers, widths, right))
    click.echo(_sep(widths))

    for label, st in rows:
        notes = ",".join(st.get("notes", [])[:2])  # keep it short in table
        click.echo(
            _row(
                [
                    label,
                    str(st["count"]),
                    fmt_days(st["mean_sec"]),
                    fmt_days(st["std_sec"]),
                    fmt_days(st["min_sec"]),
                    fmt_days(st["max_sec"]),
                    fmt_days(st.get("p50_sec")),
                    fmt_days(st.get("p75_sec")),
                    fmt_days(st.get("p90_sec")),
                    fmt_days(st.get("p95_sec")),
                    fmt_days(st.get("iqr_mean_sec")),
                    fmt_days(st.get("iqr_max_sec")),
                    fmt_days(st.get("iqr_p90_sec")),
                    str(st["removed"]),
                    notes or "-",
                ],
                widths,
                right,
            )
        )
    click.echo()


# ----------------------------
# CLI
# ----------------------------
@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--repo", required=True, help="GitHub repository (owner/name)")
@click.option("--days", type=int, default=7, show_default=True, help="Lookback window in days.")
@click.option(
    "--exclude-label", "exclude_label_multi", multiple=True, help="Exclude label (repeatable)."
)
@click.option(
    "--exclude-labels", "exclude_labels_csv", default="", help="Comma-separated exclude labels."
)
@click.option("--token", default="", help="Optional. Uses gh auth login if omitted.")
@click.option(
    "--format",
    "out_format",
    type=click.Choice(["md", "table", "json"]),
    default="md",
    show_default=True,
)
@click.option(
    "--limit", type=int, default=1000, show_default=True, help="Max items fetched per query."
)
def cli(
    repo: str,
    days: int,
    exclude_label_multi: tuple[str, ...],
    exclude_labels_csv: str,
    token: str,
    out_format: str,
    limit: int,
) -> None:
    if token.strip():
        os.environ["GH_TOKEN"] = token.strip()

    repo = validate_repo(repo)
    days = validate_days(days)
    limit = validate_limit(limit)

    ensure_repo_exists(repo)

    labels: list[str] = list(exclude_label_multi)
    if exclude_labels_csv.strip():
        labels += [x.strip() for x in exclude_labels_csv.split(",") if x.strip()]
    labels = validate_labels(labels)

    now = datetime.now(timezone.utc)
    start = now - timedelta(days=days)

    # GitHub search date operators are typically day-granularity.
    # To avoid missing "today" items with created:<YYYY-MM-DD, we use an exclusive end boundary as tomorrow (UTC).
    start_d = dt_utc_date_str(start)
    end_d = dt_utc_date_str(now)
    end_excl_d = (now.date() + timedelta(days=1)).strftime("%Y-%m-%d")

    ex_q = build_exclude_query(labels)

    def q(s: str) -> str:
        # search query is what GitHub interprets; labels/repo validated to avoid query breakage
        return (s + (" " + ex_q if ex_q else "")).strip()

    # 1) Created in range (day-granularity)
    created_issues = gql_search_all(
        q(f"repo:{repo} is:issue created:{start_d}..{end_d}"), limit=limit
    )
    created_prs = gql_search_all(q(f"repo:{repo} is:pr created:{start_d}..{end_d}"), limit=limit)

    # 2) Completed in range (Issues closed; PRs merged only)
    closed_issues = gql_search_all(
        q(f"repo:{repo} is:issue closed:{start_d}..{end_d}"), limit=limit
    )
    merged_prs = gql_search_all(
        q(f"repo:{repo} is:pr is:merged merged:{start_d}..{end_d}"), limit=limit
    )

    closed_issue_secs = durations_seconds(
        [(n.created_at, n.closed_at) for n in closed_issues if n.closed_at]
    )
    merged_pr_secs = durations_seconds(
        [(n.created_at, n.merged_at) for n in merged_prs if n.merged_at]
    )

    closed_issue_stats = stats_full(closed_issue_secs)
    merged_pr_stats = stats_full(merged_pr_secs)

    # 3) Open at any point in window:
    # created < end_excl AND (open now OR closed >= start)
    issues_open_now = gql_search_all(
        q(f"repo:{repo} is:issue is:open created:<{end_excl_d}"), limit=limit
    )
    issues_closed_after_start = gql_search_all(
        q(f"repo:{repo} is:issue closed:>={start_d} created:<{end_excl_d}"), limit=limit
    )
    issues_in_window = list(
        {n.url: n for n in (issues_open_now + issues_closed_after_start)}.values()
    )

    prs_open_now = gql_search_all(
        q(f"repo:{repo} is:pr is:open created:<{end_excl_d}"), limit=limit
    )
    prs_closed_after_start = gql_search_all(
        q(f"repo:{repo} is:pr closed:>={start_d} created:<{end_excl_d}"), limit=limit
    )
    prs_in_window = list({n.url: n for n in (prs_open_now + prs_closed_after_start)}.values())

    def issue_end(n: Node) -> datetime:
        return min(n.closed_at or now, now)

    def pr_end(n: Node) -> datetime:
        # include close-without-merge as an end for "age"
        return min(n.merged_at or n.closed_at or now, now)

    issue_age_secs = durations_seconds([(n.created_at, issue_end(n)) for n in issues_in_window])
    pr_age_secs = durations_seconds([(n.created_at, pr_end(n)) for n in prs_in_window])

    issue_age_stats = stats_full(issue_age_secs)
    pr_age_stats = stats_full(pr_age_secs)

    total = {"issues": issue_age_stats["count"], "prs": pr_age_stats["count"]}
    created = {"issues": len(created_issues), "prs": len(created_prs)}
    completed = {"issues": len(closed_issues), "prs": len(merged_prs)}

    payload = {
        "meta": {
            "repo": repo,
            "now_utc": now.isoformat().replace("+00:00", "Z"),
            "range_utc": {
                "start_ts": start.isoformat().replace("+00:00", "Z"),
                "end_ts": now.isoformat().replace("+00:00", "Z"),
            },
            "search_window_note": (
                "GitHub search date operators are day-granularity. "
                "This report uses date-rounded windows for search queries, and uses created:<(tomorrow UTC) to include today's items."
            ),
            "search_window_days": {
                "start_day_utc": start_d,
                "end_day_utc": end_d,
                "end_exclusive_day_utc": end_excl_d,
            },
            "days": days,
            "exclude_labels": labels,
            "limit": limit,
        },
        "counts": {
            "total_open_in_window": total,
            "created": created,
            "completed": {"issues_closed": completed["issues"], "prs_merged": completed["prs"]},
            "net": {
                "issues": created["issues"] - completed["issues"],
                "prs": created["prs"] - completed["prs"],
            },
        },
        "completed_lead_time": {"issues_closed": closed_issue_stats, "prs_merged": merged_pr_stats},
        "open_during_window_age": {"issues": issue_age_stats, "prs": pr_age_stats},
    }

    if out_format == "json":
        click.echo(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    # Shared header
    header_lines = [
        "## Issue / PR Metrics",
        "",
        f"- Repo: `{repo}`",
        f"- Range (UTC timestamps): `{start.isoformat().replace('+00:00', 'Z')}` → `{now.isoformat().replace('+00:00', 'Z')}`",
        f"- Search days (UTC, day-granularity): `{start_d}` → `{end_d}` (end-exclusive: `{end_excl_d}`)",
        f"- Exclude labels: `{', '.join(labels) if labels else 'none'}`",
        f"- Limit per query: `{limit}`",
        "",
        "> Note: GitHub search date operators are day-granularity; this report rounds to days for search queries.",
        "",
    ]

    if out_format == "table":
        click.echo("\n".join(header_lines))
        emit_counts_fixed(total, created, completed)
        click.echo("### Completed Work (Lead Time)")
        click.echo(
            "Lead Time = created_at → closed_at (Issues) / merged_at (PRs).\n"
            "PR Completed count is merged-only.\n"
            "IQR filtering uses the 1.5×IQR rule; if IQR==0, filtering is disabled (note: iqr_zero_no_filter).\n"
        )
        emit_stats_table_fixed(
            "", [("Issues (closed)", closed_issue_stats), ("PRs (merged)", merged_pr_stats)]
        )
        click.echo("### Open During Window (Age)")
        click.echo(
            "Age = created_at → min(closed_at / merged_at / now).\n"
            "Items included: created < end_exclusive_day AND (open now OR closed after start). "
            "PRs closed-without-merge are included here and treated as ended at closed_at.\n"
            "IQR filtering uses the 1.5×IQR rule; if IQR==0, filtering is disabled.\n"
        )
        emit_stats_table_fixed("", [("Issues", issue_age_stats), ("PRs", pr_age_stats)])
        return

    click.echo("\n".join(header_lines))
    click.echo("### Counts")
    click.echo(
        "This table summarizes volume during the time window.\n\n"
        "- **Total (open during window)**: items open at any point in the window.\n"
        "- **Created**: items created during the window.\n"
        "- **Closed/Merged**: Issues closed during the window; PRs merged during the window (**merged-only**).\n"
        "- **Net**: Created − Closed/Merged.\n"
    )
    click.echo("| Type | Total (open during window) | Created | Closed/Merged | Net |")
    click.echo("|---|---:|---:|---:|---:|")
    click.echo(
        f"| Issues | {total['issues']} | {created['issues']} | {completed['issues']} | {created['issues'] - completed['issues']} |"
    )
    click.echo(
        f"| PRs | {total['prs']} | {created['prs']} | {completed['prs']} | {created['prs'] - completed['prs']} |\n"
    )

    def emit_md_stats(title: str, rows: list[tuple[str, dict[str, Any]]], extra_note: str) -> None:
        click.echo(title)
        click.echo(extra_note + "\n")
        click.echo(
            "Columns: Mean/Std/Min/Max and p50/p75/p90/p95 are raw stats; "
            "Mean(IQR)/Max(IQR)/p90(IQR) are after 1.5×IQR outlier removal. "
            "If IQR==0, filtering is disabled and note `iqr_zero_no_filter` is set.\n"
        )
        click.echo(
            "| Type | Count | Mean | Std | Min | Max | p50 | p75 | p90 | p95 | Mean(IQR) | Max(IQR) | p90(IQR) | Removed | Notes |"
        )
        click.echo("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
        for label, st in rows:
            notes = ",".join(st.get("notes", [])) if st.get("notes") else "-"
            click.echo(
                f"| {label} | {st['count']} | {fmt_days(st['mean_sec'])} | {fmt_days(st['std_sec'])} | "
                f"{fmt_days(st['min_sec'])} | {fmt_days(st['max_sec'])} | "
                f"{fmt_days(st.get('p50_sec'))} | {fmt_days(st.get('p75_sec'))} | {fmt_days(st.get('p90_sec'))} | {fmt_days(st.get('p95_sec'))} | "
                f"{fmt_days(st.get('iqr_mean_sec'))} | {fmt_days(st.get('iqr_max_sec'))} | {fmt_days(st.get('iqr_p90_sec'))} | "
                f"{st['removed']} | {notes} |"
            )
        click.echo()

    emit_md_stats(
        "### Completed Work (Lead Time: created → closed/merged)",
        [("Issues (closed)", closed_issue_stats), ("PRs (merged)", merged_pr_stats)],
        "Lead Time = created_at → closed_at (Issues) / merged_at (PRs). PRs here are merged-only.",
    )
    emit_md_stats(
        "### Open During Window (Age: created → min(closed/merged/now, now))",
        [("Issues", issue_age_stats), ("PRs", pr_age_stats)],
        "Age = created_at → min(closed_at / merged_at / now). PRs closed-without-merge are included and end at closed_at.",
    )


if __name__ == "__main__":
    cli()
