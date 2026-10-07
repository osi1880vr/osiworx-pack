# Releasing osiworx-pack (ComfyUI Registry + ComfyUI-Manager)

How listing works today (checked against docs.comfy.org and the ComfyUI-Manager README):
- **ComfyUI Registry** (registry.comfy.org) is the primary route; ComfyUI-Manager reads it ("Officially supports registry.comfy.org").
  You publish with `comfy node publish` or the included GitHub Action.
- **ComfyUI-Manager list**: additionally, add an entry to `custom-node-list.json` in the ComfyUI-Manager repository by pull request
  (entry prepared in `manager/custom-node-list-entry.json`). Use this route if you also want to be listed via the git URL.

## What is already prepared in this folder
- `pyproject.toml` (registry spec: name, version, description, classifiers, requires-comfyui, urls, `[tool.comfy]`)
- `requirements.txt` (empty on purpose), `.gitignore`, `.comfyignore`
- `.github/workflows/publish_action.yml` (publishes on a push to `main` that changes `pyproject.toml`)
- `manager/custom-node-list-entry.json`, `CHANGELOG.md`, READMEs, workflows, and `scripts/check_release.py`

Run `python scripts/check_release.py` any time. It lists what is still open (currently only the owner decisions below).

## What only the owner can do (open blockers)
1. ~~Choose a license~~ Done: MIT (`LICENSE`, copyright holder "osi1880vr").
2. ~~Create the public GitHub repository~~ Done: https://github.com/osi1880vr/osiworx-pack (URLs and author are filled in).
3. **Create a Registry publisher** at https://registry.comfy.org (publisher id is global and permanent). Put it in `[tool.comfy] PublisherId`.
4. **Create a publishing API key** for that publisher, and either:
   - store it as the GitHub repository secret `REGISTRY_ACCESS_TOKEN` (the included Action then publishes), or
   - run `comfy node publish` locally (needs `comfy-cli`).
5. Optional: `Icon` (square, up to 400x400 px) and `Banner` (21:9) URLs in `[tool.comfy]`.

## Release steps
1. `python scripts/check_release.py` -> no blockers.
2. Bump `version` in `pyproject.toml` (semantic versioning, X.Y.Z) and add a line to `CHANGELOG.md`.
3. Commit and push to `main` (the Action publishes when `pyproject.toml` changed), or run `comfy node publish`.
4. For the Manager git-URL list: copy `manager/custom-node-list-entry.json` into the `custom_nodes` array of `custom-node-list.json` in a fork of
   ComfyUI-Manager, enable "Use local DB" in Manager and check that the "Install custom nodes" dialog still loads (a stray comma breaks the file),
   then open the pull request.
5. After listing, test the real install path on a clean ComfyUI: install from Manager, restart, load both workflows, check that the
   `LayerStream: ...` nodes exist.

## Things reviewers / users will care about (be upfront in the repo)
- The nodes **monkeypatch ComfyUI internals** (`DiTBlock`/`FinalLayer` for H3, `SingleStreamBlock`/`LastLayer` for Krea2) at import time. They
  behave as stock unless the node is connected, but they can break when ComfyUI changes those classes. Keep `requires-comfyui` accurate and
  re-test on each ComfyUI release.
- Each module is import-safe: no MiniMax H3 / Krea2 code in the user's ComfyUI -> that node is skipped with a warning, nothing else breaks.
- The registry name must not contain "ComfyUI" and is immutable after creation (`osiworx-pack`).
- Tested only on Windows + NVIDIA CUDA (RTX 4090 24 GB, ComfyUI 0.39.0). The classifiers say so; widen them only after testing.
- Do not ship model files or large media in the repo (the checker flags `*.safetensors` and files over 2 MB).
- Security: the code does no network access, no file writes and no subprocess calls; it only patches model forward passes. Say that in the
  PR description, it is what Manager reviewers look for.
