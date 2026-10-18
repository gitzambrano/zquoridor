"""Internal read-only snapshots of training files on existing Colab runtimes.

The training runner owns configuration. This module never allocates a VM,
opens a kernel, executes a cell, or changes a remote file.
"""
from collections import Counter
import hashlib
import json
from pathlib import Path, PurePosixPath
from urllib.parse import quote

from colab.cli import backend

ALLOWED = {"positions.jsonl", "targets.jsonl", "targets.npz", "manifest.json", "games.jsonl.gz"}
PREFIXES = ("selfplay_", "wall_drops_", "gumbel_sims_", "claustro_search_games_")


def snapshot(worker_ids, destination, download=False, latest_only=False):
    """List every selected directory and optionally copy stable file versions."""
    import requests
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    records, accounts = [], []
    previous = {}
    ledger = destination / "downloads.jsonl"
    if ledger.exists():
        previous = {row["path"]: row for row in map(json.loads, ledger.read_text(encoding="utf-8").splitlines())}
    for worker in worker_ids:
        summary = {"worker": worker, "status": "pending", "files": 0}
        accounts.append(summary)
        try:
            assignments = backend.client(worker).list_assignments()
            if len(assignments) != 1:
                summary["status"] = "no_existing_runtime" if not assignments else "runtime_selection_required"
                continue
            assignment = assignments[0]
            root = "content/drive/MyDrive/zquoridor_data"
            try:
                folders = backend.contents(assignment, root)["content"]
            except requests.HTTPError as error:
                if error.response.status_code != 404:
                    raise
                root = "drive/MyDrive/zquoridor_data"
                folders = backend.contents(assignment, root)["content"]
            stack = []
            for folder in folders:
                name = folder["name"]
                if folder["type"] == "directory" and name.startswith(PREFIXES) and (not latest_only or "v401" in name):
                    stack.append((folder["path"], name))
            while stack:
                path, kind = stack.pop()
                for entry in backend.contents(assignment, path)["content"]:
                    if entry["type"] == "directory":
                        stack.append((entry["path"], kind))
                        continue
                    name = entry["name"]
                    if name not in ALLOWED and not name.endswith((".bin", ".meta")):
                        continue
                    relative = PurePosixPath(entry["path"]).relative_to(root)
                    if ".." in relative.parts:
                        raise ValueError("Invalid runtime file path.")
                    record = dict(worker=worker, path=f"worker-{worker}/{relative.as_posix()}",
                                  remote_path=entry["path"], title=name, kind=kind,
                                  size=entry.get("size"), modified=entry.get("last_modified"))
                    record["local"] = str((destination / "files" / record["path"]).resolve())
                    summary["files"] += 1
                    records.append(record)
                    if not download or not record["size"]:
                        continue
                    old = previous.get(record["path"])
                    local = Path(record["local"])
                    if old and old.get("modified") == record["modified"] and local.exists() and old.get("sha256") == _sha(local):
                        previous[record["path"]] = old
                        continue
                    local.parent.mkdir(parents=True, exist_ok=True)
                    temporary = local.with_suffix(local.suffix + ".part")
                    proxy = assignment.runtime_proxy_info
                    with requests.get(proxy.url.rstrip("/") + "/files/" + quote(entry["path"], safe="/"),
                                      params={"colab-runtime-proxy-token": proxy.token},
                                      headers={"X-Colab-Runtime-Proxy-Token": proxy.token},
                                      timeout=(20, 60), stream=True) as response:
                        response.raise_for_status()
                        with temporary.open("wb") as output:
                            for block in response.iter_content(1 << 20):
                                output.write(block)
                    after = backend.contents(assignment, entry["path"], content=False)
                    if after.get("last_modified") != record["modified"] or temporary.stat().st_size != record["size"]:
                        temporary.unlink(missing_ok=True)
                        record["status"] = "changed_during_copy"
                        continue
                    temporary.replace(local)
                    record["sha256"] = _sha(local)
                    previous[record["path"]] = dict(record)
                    _ledger(ledger, previous)
            summary["status"] = "complete"
        except Exception as error:
            # HTTP errors can contain runtime proxy credentials in their URLs.
            summary["status"] = backend.error_status(error)
            summary["error_type"] = type(error).__name__
    report = dict(schema="zquoridor.cloud_snapshot.v1", accounts=accounts, files=records,
                  counts_by_kind=dict(Counter(r["kind"] for r in records)),
                  latest_only=latest_only, download=download,
                  complete=all(a["status"] == "complete" and a["files"] for a in accounts))
    (destination / "inventory.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    return report


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ledger(path, records):
    temporary = path.with_suffix(".tmp")
    temporary.write_text("".join(json.dumps(r)+"\n" for r in records.values()), encoding="utf-8")
    temporary.replace(path)
