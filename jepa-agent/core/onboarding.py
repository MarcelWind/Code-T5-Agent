"""Project onboarding — detect project type, languages, configure agent for any codebase.

Detection strategies:
1. Walk up from target file to find project root (.git/, package.json, etc.)
2. Scan config files to determine primary programming languages
3. Generate a project profile persisted as .jepa-project.json
"""

import json
import os
from pathlib import Path
from typing import Optional


# ── Project root indicators (ordered by specificity) ──

_ROOT_INDICATORS = [
    ".git",
    "pyproject.toml",
    "package.json",
    "Cargo.toml",
    "go.mod",
    "composer.json",
    "Gemfile",
    "pom.xml",
    "setup.py",
    "setup.cfg",
    "project.clj",
    "mix.exs",
    "CMakeLists.txt",
    "Makefile",
    "Rakefile",
    "stack.yaml",
    "dub.json",
    "pubspec.yaml",
    "Package.swift",
    "gradlew",
    "sln",
]

# ── Language detectors ──


def _check_file_exists(root: str, *parts: str) -> bool:
    return os.path.isfile(os.path.join(root, *parts))


def _check_dir_exists(root: str, *parts: str) -> bool:
    return os.path.isdir(os.path.join(root, *parts))


def _detect_from_config(root: str) -> list[dict]:
    """Read project config files to determine primary languages.

    Checks both `root` and first-level subdirectories for config files
    (handles monorepo layouts where project lives in a subdir).
    """
    detected = []

    # Collect candidate config dirs: root + first-level subdirs
    config_dirs = [root]
    try:
        for entry in os.listdir(root):
            sub = os.path.join(root, entry)
            if os.path.isdir(sub) and not entry.startswith(".") and not entry.startswith("_"):
                config_dirs.append(sub)
    except OSError:
        pass

    def _any_config(*files: str) -> bool:
        for d in config_dirs:
            if any(_check_file_exists(d, f) for f in files):
                return True
        return False

    # Python
    if _any_config("pyproject.toml", "setup.py", "setup.cfg", "requirements.txt", "Pipfile"):
        detected.append({
            "language": "python",
            "detected_by": "config",
            "confidence": 0.9,
            "extensions": [".py"],
        })

    # JavaScript / TypeScript
    if _any_config("package.json"):
        # Find which dir has package.json to check for TS
        pkg_root = root
        for d in config_dirs:
            if _check_file_exists(d, "package.json"):
                pkg_root = d
                break
        lang, exts, conf = _check_ts_in_package(pkg_root)
        detected.append({
            "language": lang,
            "detected_by": "package.json",
            "confidence": conf,
            "extensions": exts,
        })

    # Rust
    if _any_config("Cargo.toml"):
        detected.append({
            "language": "rust",
            "detected_by": "Cargo.toml",
            "confidence": 0.95,
            "extensions": [".rs"],
        })

    # Go
    if _any_config("go.mod"):
        detected.append({
            "language": "go",
            "detected_by": "go.mod",
            "confidence": 0.95,
            "extensions": [".go"],
        })

    # Ruby
    if _any_config("Gemfile"):
        detected.append({
            "language": "ruby",
            "detected_by": "Gemfile",
            "confidence": 0.9,
            "extensions": [".rb"],
        })

    # PHP
    if _any_config("composer.json"):
        detected.append({
            "language": "php",
            "detected_by": "composer.json",
            "confidence": 0.9,
            "extensions": [".php"],
        })

    # Java
    if _any_config("pom.xml", "build.gradle", "gradlew"):
        detected.append({
            "language": "java",
            "detected_by": "build config",
            "confidence": 0.85,
            "extensions": [".java"],
        })

    # C/C++
    if _any_config("CMakeLists.txt"):
        detected.append({
            "language": "cpp",
            "detected_by": "CMakeLists.txt",
            "confidence": 0.85,
            "extensions": [".cpp", ".c", ".h", ".hpp"],
        })

    return detected


def _check_ts_in_package(root: str) -> tuple[str, list[str], float]:
    """Check package.json for TypeScript dependency."""
    try:
        pkg_path = os.path.join(root, "package.json")
        with open(pkg_path, "r", encoding="utf-8") as f:
            pkg = json.load(f)
        all_deps = json.dumps({**pkg.get("dependencies", {}), **pkg.get("devDependencies", {})})
        if "typescript" in all_deps:
            return "typescript", [".ts", ".tsx", ".js", ".jsx"], 0.9
    except (json.JSONDecodeError, OSError):
        pass
    return "javascript", [".js", ".jsx", ".mjs"], 0.9


def _detect_from_source_dirs(root: str, existing_languages: list[str]) -> list[dict]:
    """Scan source directories for file extensions to confirm/expand languages."""
    extension_map = {
        ".py": ("python", 0.6),
        ".js": ("javascript", 0.6),
        ".jsx": ("javascript", 0.6),
        ".ts": ("typescript", 0.6),
        ".tsx": ("typescript", 0.6),
        ".rs": ("rust", 0.6),
        ".go": ("go", 0.6),
        ".rb": ("ruby", 0.6),
        ".php": ("php", 0.6),
        ".java": ("java", 0.6),
        ".c": ("c", 0.5),
        ".cpp": ("cpp", 0.5),
        ".h": ("c", 0.4),
        ".hpp": ("cpp", 0.4),
    }

    src_dirs = ["src", "source", "lib", "app", "components", "pages"]
    existing_set = {d["language"] for d in existing_languages}
    found: dict[str, float] = {}

    for d in src_dirs:
        src_path = os.path.join(root, d)
        if not os.path.isdir(src_path):
            continue
        try:
            for entry in os.listdir(src_path):
                ext = os.path.splitext(entry)[1].lower()
                if ext in extension_map:
                    lang, conf = extension_map[ext]
                    if lang not in found or found[lang] < conf:
                        found[lang] = conf
        except OSError:
            continue

    detected = []
    for lang, confidence in found.items():
        if lang in existing_set:
            continue
        rev_exts = [k for k, v in extension_map.items() if v[0] == lang]
        detected.append({
            "language": lang,
            "detected_by": "source scan",
            "confidence": confidence,
            "extensions": list(set(rev_exts)),
        })
    return detected


# ── Public API ──

PROJECT_CONFIG_FILE = ".jepa-project.json"


def detect_project_root(path: str) -> Optional[str]:
    """Walk up from `path` to find the project root directory.

    Looks for known project root indicators (.git, pyproject.toml, etc.).
    Returns absolute path to root, or None (falls back to file's directory).
    """
    abs_path = os.path.abspath(path)
    start_dir = os.path.dirname(abs_path) if os.path.isfile(abs_path) else abs_path

    current = os.path.abspath(start_dir)
    while True:
        for indicator in _ROOT_INDICATORS:
            if os.path.exists(os.path.join(current, indicator)):
                return current
        parent = os.path.dirname(current)
        if parent == current:
            break
        current = parent

    # Fallback
    return os.path.dirname(abs_path)


def detect_project_languages(root: str) -> list[dict]:
    """Detect primary programming languages in a project.

    Returns list of dicts:
        {"language": str, "detected_by": str, "confidence": float, "extensions": list[str]}
    """
    config_detected = _detect_from_config(root)
    source_detected = _detect_from_source_dirs(root, config_detected)
    return config_detected + source_detected


def generate_project_profile(root: str, languages: list[dict]) -> dict:
    """Compile a project profile dict."""
    primary = max(languages, key=lambda d: d.get("confidence", 0)) if languages else {"language": "unknown", "confidence": 0, "extensions": []}

    all_exts = set()
    for lang_info in languages:
        all_exts.update(lang_info.get("extensions", []))

    return {
        "project_root": root,
        "project_name": os.path.basename(root),
        "languages": languages,
        "primary_language": primary["language"],
        "all_extensions": sorted(all_exts),
    }


def load_project_profile(root: str) -> Optional[dict]:
    """Load existing .jepa-project.json from project root."""
    path = os.path.join(root, PROJECT_CONFIG_FILE)
    if os.path.isfile(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return None
    return None


def save_project_profile(profile: dict) -> bool:
    """Save project profile to .jepa-project.json at project root."""
    path = os.path.join(profile["project_root"], PROJECT_CONFIG_FILE)
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(profile, f, indent=2)
        return True
    except OSError:
        return False


def format_profile_markdown(profile: dict) -> str:
    """Format project profile as markdown for vault storage."""
    lines = [
        f"# Project Profile: {profile['project_name']}",
        "",
        f"**Root:** `{profile['project_root']}`",
        "",
        "## Languages",
        "",
    ]
    for lang_info in profile.get("languages", []):
        confidence_pct = lang_info.get("confidence", 0) * 100
        exts = ", ".join(lang_info.get("extensions", []))
        lines.append(f"- **{lang_info['language']}** (confidence: {confidence_pct:.0f}%)")
        lines.append(f"  - Detected by: {lang_info['detected_by']}")
        lines.append(f"  - Extensions: {exts}")
        lines.append("")

    lines.append(f"**Primary:** {profile.get('primary_language', 'unknown')}")
    lines.append(f"**Extensions:** {', '.join(profile.get('all_extensions', []))}")
    lines.append("")
    lines.append("---")
    lines.append("*Auto-generated by JEPA onboarding.*")

    return "\n".join(lines)
