"""Pre-release checks for osiworx-pack (ComfyUI Registry / ComfyUI-Manager).

usage: python scripts/check_release.py        (run from anywhere; exits 1 while blockers remain)
Checks pyproject.toml against the registry spec, required files, leftover TODO placeholders, the Manager list entry,
that every .py file parses, that each module exposes literal NODE_CLASS_MAPPINGS (what the Manager scanner reads),
and that the bundled workflows are valid JSON.
"""
import ast
import json
import re
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
blockers, warnings, ok = [], [], []


def check(cond, good, bad, blocker=True):
    (ok if cond else (blockers if blocker else warnings)).append(good if cond else bad)


# --- pyproject.toml -------------------------------------------------------------------------------------------
py = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
proj, comfy = py.get("project", {}), py.get("tool", {}).get("comfy", {})
name = proj.get("name", "")
check(bool(re.fullmatch(r"[A-Za-z][A-Za-z0-9._-]*", name)) and len(name) < 100 and not re.search(r"[._-]{2}", name),
      f"name '{name}' is valid", f"name '{name}' violates the registry rules")
check("comfyui" not in name.lower(), "name does not contain 'ComfyUI'", "name should not contain 'ComfyUI'", blocker=False)
check(bool(re.fullmatch(r"\d+\.\d+\.\d+", proj.get("version", ""))), f"version {proj.get('version')} is X.Y.Z",
      "version must be X.Y.Z")
check(bool(proj.get("description")), "description present", "description missing")
lic = proj.get("license")
lic_todo = (not lic) or ("TODO" in json.dumps(lic))
check(not lic_todo, f"license set: {lic}", "license is not decided yet (TODO)")
if isinstance(lic, dict) and "file" in lic:
    check((ROOT / lic["file"]).exists(), f"license file {lic['file']} exists", f"license file {lic['file']} is missing")
repo = proj.get("urls", {}).get("Repository", "")
check(repo.startswith("https://github.com/") and "TODO" not in repo, f"repository {repo}", "Repository URL still a placeholder")
check("TODO" not in comfy.get("PublisherId", "TODO"), f"PublisherId {comfy.get('PublisherId')}", "PublisherId still a placeholder")
check(bool(comfy.get("DisplayName")), "DisplayName present", "DisplayName missing", blocker=False)
check(bool(comfy.get("requires-comfyui")), f"requires-comfyui {comfy.get('requires-comfyui')}", "requires-comfyui missing", blocker=False)

# --- files ----------------------------------------------------------------------------------------------------
for f in ("__init__.py", "README.md", "requirements.txt", ".comfyignore", ".gitignore", "CHANGELOG.md",
          ".github/workflows/publish_action.yml", "manager/custom-node-list-entry.json"):
    check((ROOT / f).exists(), f"{f} present", f"{f} missing")
check(not list(ROOT.rglob("*.safetensors")) and not list(ROOT.rglob("*.ckpt")), "no model files in the repo",
      "model files found in the repo folder")
big = [p for p in ROOT.rglob("*") if p.is_file() and p.stat().st_size > 2_000_000 and "__pycache__" not in p.parts]
check(not big, "no file over 2 MB", f"large files: {[str(p.relative_to(ROOT)) for p in big]}", blocker=False)

# --- python sources --------------------------------------------------------------------------------------------
for p in ROOT.rglob("*.py"):
    if "__pycache__" in p.parts:
        continue
    try:
        tree = ast.parse(p.read_text(encoding="utf-8"))
    except SyntaxError as e:
        blockers.append(f"{p.relative_to(ROOT)} does not parse: {e}")
        continue
    if p.parent != ROOT and "scripts" not in p.parts and "tests" not in p.parts and p.name != "__init__.py" and "NODE_CLASS_MAPPINGS" in p.read_text(encoding="utf-8"):
        has = any(isinstance(n, ast.Assign) and any(getattr(t, "id", "") == "NODE_CLASS_MAPPINGS" for t in n.targets)
                  and isinstance(n.value, ast.Dict) for n in tree.body)
        check(has, f"{p.relative_to(ROOT)} has a literal NODE_CLASS_MAPPINGS", f"{p.relative_to(ROOT)}: no literal NODE_CLASS_MAPPINGS (Manager scanner)")

# --- Manager list entry -----------------------------------------------------------------------------------------
entry = json.loads((ROOT / "manager" / "custom-node-list-entry.json").read_text(encoding="utf-8"))
need = {"author", "title", "reference", "files", "install_type", "description"}
check(need <= set(entry), "Manager entry has all required keys", f"Manager entry misses {need - set(entry)}")
check("TODO" not in json.dumps(entry), "Manager entry has no placeholders", "Manager entry still has placeholders (author/repo)")
check(entry.get("reference") == repo or "TODO" in repo, "Manager entry points at the same repo as pyproject", "Manager entry URL differs from pyproject Repository")

# --- workflows --------------------------------------------------------------------------------------------------
for w in sorted((ROOT / "workflows").glob("*.json")):
    try:
        data = json.loads(w.read_text(encoding="utf-8"))
        types = {n["type"] for n in data["nodes"]}
        ok.append(f"workflow {w.name}: valid JSON, {len(data['nodes'])} nodes")
        check(any(t.startswith("LayerStream") for t in types), f"{w.name} uses a LayerStream node", f"{w.name} has no LayerStream node", blocker=False)
    except Exception as e:
        blockers.append(f"workflow {w.name} invalid: {e}")

print("OK:")
for m in ok:
    print("  +", m)
print("WARNINGS:" if warnings else "WARNINGS: none")
for m in warnings:
    print("  ~", m)
print("BLOCKERS:" if blockers else "BLOCKERS: none")
for m in blockers:
    print("  !", m)
sys.exit(1 if blockers else 0)
