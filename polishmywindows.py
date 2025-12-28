import argparse
import dataclasses
import datetime as _dt
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


APP_DIR_NAME = ".PolishMyWindows"
DEFAULT_RULES_FILE = "rules.json"
UNDO_LOG_NAME = "undo.jsonl"


@dataclasses.dataclass(frozen=True)
class MoveAction:
    src: str
    dst: str
    timestamp_utc: str


def _utc_now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _load_rules(rules_path: Path) -> dict:
    with rules_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _is_hidden_or_system(p: Path) -> bool:
    # Cross-platform fallback: on Windows, hidden files often start with '.' or have attributes.
    # We avoid ctypes here; name-based skip is good enough for MVP.
    name = p.name
    if name.startswith("."):
        return True
    return False


def _safe_rel(p: Path, root: Path) -> str:
    try:
        return str(p.relative_to(root))
    except Exception:
        return str(p)


def _iter_files(root: Path, recursive: bool) -> Iterable[Path]:
    if recursive:
        for p in root.rglob("*"):
            if p.is_file():
                yield p
    else:
        for p in root.iterdir():
            if p.is_file():
                yield p


def _keyword_match(name_lower: str, keywords: List[str]) -> bool:
    for kw in keywords:
        if kw and kw.lower() in name_lower:
            return True
    return False


def _choose_category(p: Path, rules: dict) -> str:
    ext = p.suffix.lower()
    name_lower = p.name.lower()

    categories: Dict[str, dict] = rules.get("categories", {})
    order: List[str] = rules.get("category_order", list(categories.keys()))

    # First pass: extension match in order
    for cat in order:
        spec = categories.get(cat, {})
        exts = [e.lower() for e in spec.get("extensions", [])]
        if ext and ext in exts:
            return cat

    # Second pass: keyword match in order
    for cat in order:
        spec = categories.get(cat, {})
        kws = spec.get("name_keywords", [])
        if kws and _keyword_match(name_lower, kws):
            return cat

    return "Other" if "Other" in categories else (order[-1] if order else "Other")


def _month_bucket(p: Path) -> str:
    # Prefer modified time
    ts = p.stat().st_mtime
    dt = _dt.datetime.fromtimestamp(ts)
    return f"{dt.year:04d}-{dt.month:02d}"


def _resolve_conflict(dst: Path, mode: str) -> Path:
    if not dst.exists():
        return dst

    if mode == "skip":
        return dst

    if mode != "rename":
        raise ValueError(f"Unsupported conflict_mode: {mode}")

    stem = dst.stem
    suffix = dst.suffix
    parent = dst.parent
    for i in range(1, 10_000):
        candidate = parent / f"{stem} ({i}){suffix}"
        if not candidate.exists():
            return candidate

    raise RuntimeError(f"Could not resolve filename conflict for: {dst}")


def _ensure_app_dir(root: Path) -> Path:
    d = root / APP_DIR_NAME
    d.mkdir(parents=True, exist_ok=True)
    return d


def _append_undo(app_dir: Path, action: MoveAction) -> None:
    log_path = app_dir / UNDO_LOG_NAME
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(dataclasses.asdict(action), ensure_ascii=False) + "\n")


def plan_moves(
    root: Path,
    rules: dict,
    dest_root: Optional[Path],
    recursive: bool,
) -> List[Tuple[Path, Path, str]]:
    behavior = rules.get("behavior", {})
    grouping = rules.get("grouping", {})

    skip_exts = set(e.lower() for e in behavior.get("skip_extensions", []))
    skip_hidden = bool(behavior.get("skip_hidden", True))

    moves: List[Tuple[Path, Path, str]] = []
    for src in _iter_files(root, recursive=recursive):
        if skip_hidden and _is_hidden_or_system(src):
            continue
        if src.suffix.lower() in skip_exts:
            continue
        if src.parent.name == APP_DIR_NAME:
            continue

        cat = _choose_category(src, rules)

        base = dest_root if dest_root is not None else root
        target_dir = base / cat

        if grouping.get("enabled") and cat in set(grouping.get("for_categories", [])):
            if grouping.get("mode") == "year_month":
                target_dir = target_dir / _month_bucket(src)

        dst = target_dir / src.name
        moves.append((src, dst, cat))

    return moves


def apply_moves(
    root: Path,
    moves: List[Tuple[Path, Path, str]],
    rules: dict,
    dry_run: bool,
) -> int:
    behavior = rules.get("behavior", {})
    conflict_mode = behavior.get("conflict_mode", "rename")

    app_dir = _ensure_app_dir(root)

    changed = 0
    for src, dst, _cat in moves:
        if src.resolve() == dst.resolve():
            continue

        dst.parent.mkdir(parents=True, exist_ok=True)
        final_dst = _resolve_conflict(dst, conflict_mode)

        if final_dst.exists() and conflict_mode == "skip":
            continue

        if dry_run:
            changed += 1
            continue

        shutil.move(str(src), str(final_dst))
        _append_undo(
            app_dir,
            MoveAction(src=str(src), dst=str(final_dst), timestamp_utc=_utc_now_iso()),
        )
        changed += 1

    return changed


def cmd_organize(args: argparse.Namespace) -> int:
    root = Path(args.path).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        print(f"Path is not a directory: {root}", file=sys.stderr)
        return 2

    rules_path = Path(args.rules).expanduser().resolve()
    if not rules_path.exists():
        print(f"Rules file not found: {rules_path}", file=sys.stderr)
        return 2

    rules = _load_rules(rules_path)

    dest_root = Path(args.dest).expanduser().resolve() if args.dest else None
    if dest_root is not None and not dest_root.exists():
        print(f"Destination folder not found: {dest_root}", file=sys.stderr)
        return 2

    moves = plan_moves(root=root, rules=rules, dest_root=dest_root, recursive=args.recursive)

    if args.limit is not None:
        moves = moves[: max(0, args.limit)]

    if args.print_plan:
        for src, dst, cat in moves:
            print(f"[{cat}] {_safe_rel(src, root)} -> {_safe_rel(dst, root if dest_root is None else dest_root)}")

    dry_run = not args.apply
    changed = apply_moves(root=root, moves=moves, rules=rules, dry_run=dry_run)

    if dry_run:
        print(f"Dry-run complete. Planned moves: {changed}")
        print("Run again with --apply to actually move files.")
    else:
        print(f"Done. Moved: {changed}")
        print(f"Undo log: {root / APP_DIR_NAME / UNDO_LOG_NAME}")

    return 0


def _read_undo_actions(log_path: Path) -> List[MoveAction]:
    actions: List[MoveAction] = []
    if not log_path.exists():
        return actions

    with log_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            actions.append(MoveAction(**obj))

    return actions


def cmd_undo(args: argparse.Namespace) -> int:
    root = Path(args.path).expanduser().resolve()
    app_dir = root / APP_DIR_NAME
    log_path = app_dir / UNDO_LOG_NAME

    actions = _read_undo_actions(log_path)
    if not actions:
        print("Nothing to undo.")
        return 0

    # Undo in reverse order
    to_undo = actions[-args.steps :] if args.steps is not None else list(reversed(actions))
    to_undo = list(reversed(to_undo))  # ensure reverse chronological

    undone = 0
    for action in to_undo:
        src = Path(action.src)
        dst = Path(action.dst)

        if not dst.exists():
            continue

        src.parent.mkdir(parents=True, exist_ok=True)

        # Avoid overwriting anything on undo
        final_src = src
        if final_src.exists():
            final_src = _resolve_conflict(final_src, "rename")

        if args.dry_run:
            undone += 1
            continue

        shutil.move(str(dst), str(final_src))
        undone += 1

    if args.dry_run:
        print(f"Dry-run undo complete. Would undo: {undone}")
        return 0

    # Rewrite the log excluding undone steps (best-effort)
    remaining = actions[: max(0, len(actions) - undone)]
    tmp = log_path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for action in remaining:
            f.write(json.dumps(dataclasses.asdict(action), ensure_ascii=False) + "\n")
    tmp.replace(log_path)

    print(f"Undo complete. Undone: {undone}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="PolishMyWindows",
        description="Intelligent file organizer for Windows (safe dry-run by default).",
    )
    sub = p.add_subparsers(dest="command", required=True)

    org = sub.add_parser("organize", help="Plan and (optionally) move files into category folders")
    org.add_argument("path", help="Folder to organize")
    org.add_argument("--rules", default=DEFAULT_RULES_FILE, help="Path to rules.json")
    org.add_argument("--dest", default=None, help="Optional destination root (defaults to PATH)")
    org.add_argument("--recursive", action="store_true", help="Include files in subfolders")
    org.add_argument("--print-plan", action="store_true", help="Print planned moves")
    org.add_argument("--limit", type=int, default=None, help="Limit number of moves (for testing)")
    org.add_argument("--apply", action="store_true", help="Actually move files (otherwise dry-run)")
    org.set_defaults(func=cmd_organize)

    undo = sub.add_parser("undo", help="Undo recent moves using the undo log")
    undo.add_argument("path", help="Same folder you organized")
    undo.add_argument("--steps", type=int, default=None, help="Undo last N moves (default: all)")
    undo.add_argument("--dry-run", action="store_true", help="Preview undo actions")
    undo.set_defaults(func=cmd_undo)

    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
