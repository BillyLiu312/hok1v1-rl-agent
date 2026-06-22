#!/usr/bin/env python3
"""
Offline code sync helper.

Create a copy-pasteable text package on one machine, then apply it on another
machine without git, network access, or third-party Python packages.
"""

from __future__ import annotations

import argparse
import base64
import fnmatch
import hashlib
import json
import os
import shutil
import sys
import tarfile
import tempfile
import time
import zlib
from pathlib import Path, PurePosixPath


MARKER_BEGIN = "-----BEGIN OFFLINE SYNC PACKAGE-----"
MARKER_END = "-----END OFFLINE SYNC PACKAGE-----"
FORMAT_VERSION = 1

V1_2_REQUIRED_FILES = [
    "agent_ppo/conf/conf.py",
    "agent_ppo/conf/evaluation_config.py",
    "agent_ppo/conf/monitor_builder.py",
    "agent_ppo/conf/opponent_schedule.py",
    "agent_ppo/conf/summoner_skill.py",
    "agent_ppo/conf/train_env_conf.toml",
    "agent_ppo/feature/reward_process.py",
    "agent_ppo/feature/feature_process/hero_process.py",
    "agent_ppo/feature/feature_process/organ_process.py",
    "agent_ppo/feature/feature_process/soldier_process.py",
    "agent_ppo/workflow/train_workflow.py",
    "utils/analyze_run_records.py",
    "utils/analyze_training_logs.py",
    "utils/build_experiment_report.py",
    "utils/checkpoint_matrix.py",
    "utils/compare_experiment_reports.py",
    "utils/evaluate_v1_2_candidate.py",
    "utils/evaluation_config_export.py",
    "utils/evaluation_matrix.py",
    "utils/select_checkpoint.py",
    "utils/run_metadata_summary.py",
    "utils/summoner_skill_grid.py",
    "utils/summoner_skill_results.py",
    "utils/training_recorder.py",
    "utils/v1_2_experiment_plan.py",
    "utils/v1_2_launch_manifest.py",
    "utils/v1_2_preflight.py",
    "docs/README.md",
    "docs/v1.1-training-analysis.md",
    "docs/v1.2-implementation-plan.md",
    "docs/v1.2-runbook.md",
]

PRESET_INCLUDE_PATTERNS = {
    "v1.2": [
        "agent_ppo/conf/**",
        "agent_ppo/feature/**",
        "agent_ppo/workflow/**",
        "utils/**",
        "tests/**",
        "docs/README.md",
        "docs/v1.1-training-analysis.md",
        "docs/v1.2-implementation-plan.md",
        "docs/v1.2-runbook.md",
    ],
}

DEFAULT_EXCLUDES = [
    ".git/**",
    ".git",
    ".DS_Store",
    ".vscode/**",
    ".offline_sync_backups/**",
    "__pycache__/**",
    "*.pyc",
    "*.pyo",
    ".venv/**",
    "venv/**",
    "logs/**",
    "output/**",
    "*.log",
    "*.pt",
    "*.pth",
    "*.ckpt",
]


def repo_root() -> Path:
    return Path.cwd().resolve()


def to_posix(path: Path) -> str:
    return path.as_posix()


def clean_rel_path(raw: str) -> PurePosixPath:
    rel = PurePosixPath(raw)
    if rel.is_absolute() or ".." in rel.parts:
        raise ValueError(f"unsafe path in package: {raw}")
    return rel


def matches_any(path: str, patterns: list[str]) -> bool:
    basename = PurePosixPath(path).name
    for pattern in patterns:
        if fnmatch.fnmatch(path, pattern):
            return True
        if "/" not in pattern and fnmatch.fnmatch(basename, pattern):
            return True
        if pattern.endswith("/**"):
            directory = pattern[:-3].rstrip("/")
            if directory and directory in PurePosixPath(path).parts:
                return True
    return False


def load_gitignore_patterns(root: Path) -> list[str]:
    gitignore = root / ".gitignore"
    if not gitignore.exists():
        return []

    patterns: list[str] = []
    for raw_line in gitignore.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or line.startswith("!"):
            continue
        line = line.lstrip("/")
        if line.endswith("/"):
            patterns.append(f"{line}**")
        else:
            patterns.append(line)
    return patterns


def should_include(
    path: Path,
    root: Path,
    include_patterns: list[str],
    exclude_patterns: list[str],
) -> bool:
    rel = to_posix(path.relative_to(root))
    if include_patterns and not matches_any(rel, include_patterns):
        return False
    return not matches_any(rel, exclude_patterns)


def iter_files(
    root: Path,
    include_patterns: list[str],
    exclude_patterns: list[str],
) -> list[Path]:
    files: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if should_include(path, root, include_patterns, exclude_patterns):
            files.append(path)
    return sorted(files, key=lambda p: to_posix(p.relative_to(root)))


def preset_include_patterns(preset: str | None) -> list[str]:
    if not preset:
        return []
    return PRESET_INCLUDE_PATTERNS.get(preset, [])


def v1_2_readiness(root: Path, include_patterns: list[str], exclude_patterns: list[str]) -> dict:
    included_paths = {to_posix(path.relative_to(root)) for path in iter_files(root, include_patterns, exclude_patterns)}
    missing = [path for path in V1_2_REQUIRED_FILES if path not in included_paths]
    return {
        "preset": "v1.2",
        "included_count": len(included_paths),
        "required_count": len(V1_2_REQUIRED_FILES),
        "missing": missing,
        "ready": not missing,
    }


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_tar(root: Path, files: list[Path]) -> bytes:
    with tempfile.NamedTemporaryFile(suffix=".tar") as temp:
        with tarfile.open(fileobj=temp, mode="w") as tar:
            for path in files:
                rel = to_posix(path.relative_to(root))
                tar.add(path, arcname=rel, recursive=False)
        temp.seek(0)
        return temp.read()


def wrap_payload(payload: bytes, line_width: int = 76) -> str:
    encoded = base64.b64encode(payload).decode("ascii")
    lines = [encoded[i : i + line_width] for i in range(0, len(encoded), line_width)]
    return "\n".join([MARKER_BEGIN, *lines, MARKER_END, ""])


def extract_payload(text: str) -> bytes:
    begin = text.find(MARKER_BEGIN)
    end = text.find(MARKER_END)
    if begin == -1 or end == -1 or end <= begin:
        raise ValueError("offline sync markers were not found")

    body = text[begin + len(MARKER_BEGIN) : end]
    compact = "".join(line.strip() for line in body.splitlines() if line.strip())
    return base64.b64decode(compact)


def make_package(
    root: Path,
    include_patterns: list[str],
    exclude_patterns: list[str],
    note: str,
) -> tuple[str, dict]:
    files = iter_files(root, include_patterns, exclude_patterns)
    tar_bytes = build_tar(root, files)
    compressed = zlib.compress(tar_bytes, level=9)
    manifest = {
        "format": FORMAT_VERSION,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "root_name": root.name,
        "note": note,
        "file_count": len(files),
        "uncompressed_bytes": len(tar_bytes),
        "compressed_bytes": len(compressed),
        "files": [
            {
                "path": to_posix(path.relative_to(root)),
                "size": path.stat().st_size,
                "sha256": file_sha256(path),
            }
            for path in files
        ],
    }
    envelope = json.dumps(manifest, ensure_ascii=False, sort_keys=True).encode("utf-8")
    envelope += b"\n\n"
    envelope += compressed
    return wrap_payload(envelope), manifest


def read_package(path: str | None) -> tuple[dict, bytes]:
    if path and path != "-":
        text = Path(path).read_text(encoding="utf-8", errors="replace")
    else:
        text = sys.stdin.read()

    payload = extract_payload(text)
    manifest_bytes, compressed = payload.split(b"\n\n", 1)
    manifest = json.loads(manifest_bytes.decode("utf-8"))
    if manifest.get("format") != FORMAT_VERSION:
        raise ValueError(f"unsupported package format: {manifest.get('format')}")
    return manifest, zlib.decompress(compressed)


def make_backup(path: Path, backup_root: Path, root: Path) -> bool:
    if not path.exists():
        return False
    rel = path.relative_to(root)
    backup_path = backup_root / rel
    backup_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, backup_path)
    return True


def unpack_tar(
    root: Path,
    tar_bytes: bytes,
    manifest: dict,
    dry_run: bool,
    backup: bool,
) -> dict:
    expected = {item["path"]: item for item in manifest["files"]}
    backup_root = root / ".offline_sync_backups" / time.strftime("%Y%m%d-%H%M%S")
    changed: list[str] = []
    unchanged: list[str] = []
    backed_up = False

    with tempfile.NamedTemporaryFile(suffix=".tar") as temp:
        temp.write(tar_bytes)
        temp.flush()
        temp.seek(0)
        with tarfile.open(fileobj=temp, mode="r") as tar:
            for member in tar.getmembers():
                if not member.isfile():
                    continue

                rel = clean_rel_path(member.name)
                rel_str = rel.as_posix()
                target = root / rel
                extracted = tar.extractfile(member)
                data = extracted.read() if extracted else b""
                digest = hashlib.sha256(data).hexdigest()
                if expected[rel_str]["sha256"] != digest:
                    raise ValueError(f"checksum mismatch for {rel_str}")

                if target.exists() and target.read_bytes() == data:
                    unchanged.append(rel_str)
                    continue

                changed.append(rel_str)
                if dry_run:
                    continue

                if backup:
                    backed_up = make_backup(target, backup_root, root) or backed_up
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)

    return {
        "changed": changed,
        "unchanged": unchanged,
        "backup_dir": str(backup_root) if backed_up and not dry_run else "",
    }


def cmd_pack(args: argparse.Namespace) -> int:
    root = repo_root()
    excludes = DEFAULT_EXCLUDES + load_gitignore_patterns(root) + args.exclude
    includes = preset_include_patterns(args.preset) + args.include
    if args.preset == "v1.2":
        readiness = v1_2_readiness(root, includes, excludes)
        if not readiness["ready"]:
            raise ValueError(f"v1.2 package is missing required files: {', '.join(readiness['missing'])}")
    package, manifest = make_package(root, includes, excludes, args.note)

    if args.output:
        Path(args.output).write_text(package, encoding="utf-8")
        print(f"Wrote package: {args.output}")
    else:
        print(package)

    print(
        f"Packed {manifest['file_count']} files, "
        f"{manifest['compressed_bytes']} compressed bytes.",
        file=sys.stderr,
    )
    return 0


def cmd_check(args: argparse.Namespace) -> int:
    root = repo_root()
    excludes = DEFAULT_EXCLUDES + load_gitignore_patterns(root) + args.exclude
    includes = preset_include_patterns(args.preset) + args.include
    if args.preset != "v1.2":
        raise ValueError("only --preset v1.2 is currently supported for check")

    readiness = v1_2_readiness(root, includes, excludes)
    print(f"preset: {readiness['preset']}")
    print(f"included_files: {readiness['included_count']}")
    print(f"required_files: {readiness['required_count']}")
    print(f"ready: {str(readiness['ready']).lower()}")
    if readiness["missing"]:
        print("missing:")
        for path in readiness["missing"]:
            print(f"  {path}")
    return 0 if readiness["ready"] else 1


def cmd_apply(args: argparse.Namespace) -> int:
    root = repo_root()
    manifest, tar_bytes = read_package(args.package)
    result = unpack_tar(
        root=root,
        tar_bytes=tar_bytes,
        manifest=manifest,
        dry_run=args.dry_run,
        backup=not args.no_backup,
    )

    action = "Would update" if args.dry_run else "Updated"
    print(f"{action} {len(result['changed'])} files; {len(result['unchanged'])} already matched.")
    for path in result["changed"][:50]:
        print(f"  {path}")
    if len(result["changed"]) > 50:
        print(f"  ... and {len(result['changed']) - 50} more")
    if result["backup_dir"]:
        print(f"Backups saved in: {result['backup_dir']}")
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    manifest, _tar_bytes = read_package(args.package)
    print(f"Package from: {manifest.get('root_name', '(unknown)')}")
    print(f"Created at: {manifest.get('created_at', '(unknown)')}")
    if manifest.get("note"):
        print(f"Note: {manifest['note']}")
    print(f"Files: {manifest['file_count']}")
    for item in manifest["files"]:
        print(f"{item['size']:>8}  {item['path']}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create or apply a copy-pasteable offline code sync package."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    pack = subparsers.add_parser("pack", help="create an offline sync package")
    pack.add_argument(
        "-o",
        "--output",
        help="write package text to this file instead of printing it",
    )
    pack.add_argument(
        "--include",
        action="append",
        default=[],
        help="only include paths matching this glob; can be repeated, e.g. 'agent_ppo/**'",
    )
    pack.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="exclude paths matching this glob; can be repeated",
    )
    pack.add_argument("--preset", choices=sorted(PRESET_INCLUDE_PATTERNS), default=None, help="include a curated file set")
    pack.add_argument("--note", default="", help="optional note stored in package metadata")
    pack.set_defaults(func=cmd_pack)

    check_cmd = subparsers.add_parser("check", help="check a curated package preset before syncing")
    check_cmd.add_argument("--preset", choices=sorted(PRESET_INCLUDE_PATTERNS), required=True, help="preset to check")
    check_cmd.add_argument(
        "--include",
        action="append",
        default=[],
        help="additional include glob; can be repeated",
    )
    check_cmd.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="additional exclude glob; can be repeated",
    )
    check_cmd.set_defaults(func=cmd_check)

    apply_cmd = subparsers.add_parser("apply", help="apply an offline sync package")
    apply_cmd.add_argument(
        "package",
        nargs="?",
        default="-",
        help="package text file, or '-' / omitted to read pasted text from stdin",
    )
    apply_cmd.add_argument(
        "--dry-run",
        action="store_true",
        help="preview changed files without writing anything",
    )
    apply_cmd.add_argument(
        "--no-backup",
        action="store_true",
        help="do not copy overwritten files into .offline_sync_backups first",
    )
    apply_cmd.set_defaults(func=cmd_apply)

    list_cmd = subparsers.add_parser("list", help="show files inside a package")
    list_cmd.add_argument(
        "package",
        nargs="?",
        default="-",
        help="package text file, or '-' / omitted to read pasted text from stdin",
    )
    list_cmd.set_defaults(func=cmd_list)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except Exception as exc:
        print(f"offline_sync.py: error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
