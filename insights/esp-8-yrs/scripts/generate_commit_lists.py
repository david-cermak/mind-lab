#!/usr/bin/env python3
"""Generate deduplicated commit lists and area summary from git repositories."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from collections import OrderedDict, defaultdict
from pathlib import Path

DEFAULT_REPO_PRIORITY = ("esp-lwip", "esp-protocols", "esp-idf")

AREA_DESCRIPTIONS: dict[str, str] = {
    "lwIP / TCP/IP Stack": (
        "Core lwIP stack work: sockets, netconn, DHCP/DHCP server, DNS, IPv4/IPv6, "
        "SNTP, pbuf handling, and upstream lwIP porting in esp-lwip and esp-idf."
    ),
    "esp_netif / Network Interface": (
        "esp_netif abstraction layer: interface lifecycle, SLIP, bridge, glue to lwIP, "
        "tcpip_adapter removal, custom/vanilla-lwip integration, and esp_eth drivers."
    ),
    "mDNS": (
        "Multicast DNS: service discovery, hostname registration, custom netif support, "
        "socket-based API, fuzzing/CI for mdns component, and protocol fixes."
    ),
    "MQTT / Mosquitto": (
        "MQTT client library and embedded Mosquitto broker component: connection handling, "
        "TLS, examples, and broker/client bug fixes."
    ),
    "Modem / PPP / Cellular": (
        "esp_modem cellular drivers (SIM7600, BG96, etc.), PPP/CMUX/DTE-DCE layers, "
        "AT command handling, and modem examples."
    ),
    "ASIO": (
        "Boost.Asio port for ESP-IDF: async socket I/O, SSL streams, executor integration, "
        "and compatibility with lwIP sockets."
    ),
    "WiFi Remote": (
        "esp_wifi_remote component: remote WiFi control, RPC/host-target split, "
        "buffer management, and co-processor WiFi offload."
    ),
    "EPPP (Ethernet over PPP)": (
        "EPPP tunneling over serial/PPP links: link setup, channel multiplexing, "
        "examples, and transport integration."
    ),
    "Transport Layer (TCP Transport & WebSocket)": (
        "Higher-level transport abstractions: esp_tcp_transport (TLS, proxy, WS upgrade) "
        "and esp_websocket_client connection/reconnect logic."
    ),
    "CI / Testing & Maintenance": (
        "CI pipelines, GitLab runners, fuzzer builds, example tests, version bumps, "
        "merge commits, build/Kconfig fixes, and general maintenance."
    ),
}

SUMMARY_AREA_ORDER = [
    "lwIP / TCP/IP Stack",
    "esp_netif / Network Interface",
    "mDNS",
    "MQTT / Mosquitto",
    "Modem / PPP / Cellular",
    "ASIO",
    "WiFi Remote",
    "EPPP (Ethernet over PPP)",
    "Transport Layer (TCP Transport & WebSocket)",
    "CI / Testing & Maintenance",
]

SUMMARY_FOLD: dict[str, str] = {
    "Ethernet (esp_eth)": "esp_netif / Network Interface",
    "Other Components & Fixes": "CI / Testing & Maintenance",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate commit lists and area summary from git repositories.",
    )
    parser.add_argument(
        "--repo",
        action="append",
        default=[],
        metavar="PATH:LABEL",
        help="Repository path and label (repeatable). Example: ext/esp-idf:esp-idf",
    )
    parser.add_argument(
        "--ref",
        action="append",
        default=[],
        metavar="LABEL=REF",
        help="Git ref per repo label (default: HEAD). Example: esp-idf=master",
    )
    parser.add_argument(
        "--email",
        action="append",
        default=[],
        help="Author or committer email to match (repeatable).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("."),
        help="Directory for list.txt, list_non_merge.txt, and summary.md (default: .).",
    )
    parser.add_argument(
        "--dedupe-priority",
        default=",".join(DEFAULT_REPO_PRIORITY),
        help="Comma-separated repo labels for dedupe preference (default: esp-lwip,esp-protocols,esp-idf).",
    )
    parser.add_argument(
        "--no-summary",
        action="store_true",
        help="Skip writing summary.md.",
    )
    parser.add_argument(
        "--short-hash-len",
        type=int,
        default=8,
        help="Length of short commit hash in output (default: 8).",
    )
    return parser.parse_args()


def git_log(repo_path: Path, ref: str) -> list[str]:
    fmt = "%H|%h|%s|%ae|%ce"
    result = subprocess.run(
        ["git", "-C", str(repo_path), "log", ref, f"--format={fmt}"],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.splitlines()


def git_ref_name(repo_path: Path, ref: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo_path), "rev-parse", "--abbrev-ref", ref],
        capture_output=True,
        text=True,
        check=True,
    )
    name = result.stdout.strip()
    return ref if name == "HEAD" else name


def is_merge_title(subject: str) -> bool:
    return subject.startswith("Merge ")


def categorize(subject: str) -> str:
    s = subject.lower()
    if re.search(r"\b(mdns|mdsn)\b", s):
        return "mDNS"
    if re.search(r"\b(mqtt|mosq|mosquitto)\b", s):
        return "MQTT / Mosquitto"
    if re.search(r"\b(modem|ppp|cmux|dte|cellular|sim7600|bg96|sim70|esp_modem)\b", s):
        return "Modem / PPP / Cellular"
    if re.search(r"\b(asio)\b", s):
        return "ASIO"
    if re.search(r"\b(wifi_remote|wifi\.remote|esp_wifi_remote)\b", s):
        return "WiFi Remote"
    if re.search(r"\b(eppp)\b", s):
        return "EPPP (Ethernet over PPP)"
    if re.search(r"\b(esp_eth|ethernet)\b", s) or subject.startswith("esp_eth:"):
        return "Ethernet (esp_eth)"
    if re.search(r"\b(lwip|socket|dhcp|dhcps|netconn|tcpip|dns|igmp|nd6|pbuf|sntp|tcp/|udp)\b", s):
        return "lwIP / TCP/IP Stack"
    if re.search(r"\b(esp_netif|esp-netif)\b", s) or (
        re.search(r"\bnetif\b", s) and "mdns" not in s
    ):
        return "esp_netif / Network Interface"
    if re.search(r"\b(tcp_transport|transport)\b", s):
        return "TCP Transport"
    if re.search(r"\b(ws_client|websocket)\b", s):
        return "WebSocket Client"
    if (
        re.search(r"\b(ci\b|/ci|fuzzer|pytest|\.gitlab|test/)", s)
        or subject.startswith("CI")
        or subject.startswith("ci")
        or s.startswith("merge branch")
    ):
        return "CI / Testing & Maintenance"
    if re.search(r"\b(example|examples|doc|readme|bump|version|release|cmake|kconfig|common)\b", s):
        return "CI / Testing & Maintenance"
    return "Other Components & Fixes"


def categorize_for_summary(subject: str) -> str:
    area = categorize(subject)
    if area in ("TCP Transport", "WebSocket Client"):
        return "Transport Layer (TCP Transport & WebSocket)"
    return area


def collect_commits(
    repos: list[tuple[Path, str]],
    refs: dict[str, str],
    emails: set[str],
    short_hash_len: int,
    dedupe_priority: list[str],
) -> tuple[list[dict], list[str]]:
    by_title: dict[str, list[dict]] = defaultdict(list)
    order: list[str] = []
    scanned: list[str] = []

    for repo_path, label in repos:
        ref = refs.get(label, "HEAD")
        ref_display = git_ref_name(repo_path, ref)
        scanned.append(f"{label} ({ref_display})")

        for line in git_log(repo_path, ref):
            if not line.strip():
                continue
            full_hash, short_hash, subject, author_email, committer_email = line.split("|", 4)
            if author_email not in emails and committer_email not in emails:
                continue
            commit = {
                "subject": subject,
                "short_hash": short_hash[:short_hash_len],
                "repo": label,
            }
            by_title[subject].append(commit)
            if subject not in order:
                order.append(subject)

    priority = {label: idx for idx, label in enumerate(dedupe_priority)}
    deduped: list[dict] = []
    for subject in order:
        best = min(by_title[subject], key=lambda c: priority.get(c["repo"], 999))
        deduped.append(best)

    return deduped, scanned


def format_line(commit: dict) -> str:
    return f"{commit['subject']} {commit['short_hash']} {commit['repo']}"


def write_summary(
    path: Path,
    deduped: list[dict],
    emails: set[str],
    scanned: list[str],
    dedupe_priority: str,
) -> None:
    areas: dict[str, list[dict]] = defaultdict(list)
    for commit in deduped:
        area = categorize_for_summary(commit["subject"])
        target = SUMMARY_FOLD.get(area, area)
        areas[target].append(commit)

    counts = {area: len(areas.get(area, [])) for area in SUMMARY_AREA_ORDER}
    total = len(deduped)
    non_merge = sum(1 for c in deduped if not is_merge_title(c["subject"]))
    merge_count = total - non_merge

    lines = ["# Commit Summary by Area\n"]
    lines.append(
        f"Total deduplicated commits: **{total}** "
        f"(author or committer: {' / '.join(sorted(emails))})\n"
    )
    lines.append(f"Repositories scanned: {', '.join(scanned)}.\n")
    lines.append(
        f"Duplicate titles across repos are deduplicated "
        f"(preferring {dedupe_priority.replace(',', ' > ')}).\n"
    )

    lines.append("## Commits per area\n")
    lines.append("| Area | Commits | % |")
    lines.append("|------|--------:|--:|")
    for area in SUMMARY_AREA_ORDER:
        count = counts[area]
        pct = (100.0 * count / total) if total else 0.0
        lines.append(f"| {area} | {count} | {pct:.1f}% |")
    lines.append(f"| **Total** | **{total}** | **100%** |")
    lines.append("")
    lines.append(
        f"Non-merge commits: **{non_merge}** · Merge commits: **{merge_count}**\n"
    )

    for area in SUMMARY_AREA_ORDER:
        count = counts[area]
        desc = AREA_DESCRIPTIONS[area]
        lines.append(f"## {area} ({count} commits)\n")
        lines.append(f"{desc}\n")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    if not args.repo:
        print("error: at least one --repo PATH:LABEL is required", file=sys.stderr)
        return 2
    if not args.email:
        print("error: at least one --email is required", file=sys.stderr)
        return 2

    repos: list[tuple[Path, str]] = []
    for spec in args.repo:
        if ":" not in spec:
            print(f"error: invalid --repo '{spec}', expected PATH:LABEL", file=sys.stderr)
            return 2
        path_str, label = spec.rsplit(":", 1)
        repo_path = Path(path_str).resolve()
        if not repo_path.is_dir():
            print(f"error: repository not found: {repo_path}", file=sys.stderr)
            return 2
        repos.append((repo_path, label))

    refs: dict[str, str] = {}
    for spec in args.ref:
        if "=" not in spec:
            print(f"error: invalid --ref '{spec}', expected LABEL=REF", file=sys.stderr)
            return 2
        label, ref = spec.split("=", 1)
        refs[label] = ref

    emails = set(args.email)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    dedupe_priority = [p.strip() for p in args.dedupe_priority.split(",") if p.strip()]
    deduped, scanned = collect_commits(
        repos, refs, emails, args.short_hash_len, dedupe_priority
    )

    list_path = output_dir / "list.txt"
    list_path.write_text("\n".join(format_line(c) for c in deduped) + "\n", encoding="utf-8")

    non_merge = [c for c in deduped if not is_merge_title(c["subject"])]
    non_merge_path = output_dir / "list_non_merge.txt"
    non_merge_path.write_text("\n".join(format_line(c) for c in non_merge) + "\n", encoding="utf-8")

    if not args.no_summary:
        write_summary(
            output_dir / "summary.md",
            deduped,
            emails,
            scanned,
            args.dedupe_priority,
        )

    print(f"Wrote {len(deduped)} commits to {list_path}")
    print(f"Wrote {len(non_merge)} non-merge commits to {non_merge_path}")
    if not args.no_summary:
        print(f"Wrote summary to {output_dir / 'summary.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
