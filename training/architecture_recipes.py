"""Frozen-reference surprise caches that preserve independent head evidence."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

try:
    from .student_model import Student, encode_features
    from .compute_surprise_weights import compute_policy_kl_batch
    from .mix_experimental_datasets import _sha256, _atomic_json
except ImportError:
    from student_model import Student, encode_features
    from compute_surprise_weights import compute_policy_kl_batch
    from mix_experimental_datasets import _sha256, _atomic_json


def apply_reference_surprise(data, dataset, reference, cache, architecture, hidden,
                            alpha=.5, cap=4., device="cpu", batch_size=2048):
    """Serialize cache construction across local campaign processes."""
    from colab.control import file_lease
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    with file_lease(cache/"construction.lock"):
        return _apply_reference_surprise(data, dataset, reference, cache, architecture, hidden,
                                         alpha, cap, device, batch_size)


def _apply_reference_surprise(data, dataset, reference, cache, architecture, hidden,
                            alpha=.5, cap=4., device="cpu", batch_size=2048):
    """Compute KL once and adjust policy evidence using training-only scales."""
    if not np.isfinite(alpha) or alpha < 0 or not np.isfinite(cap) or cap <= 0:
        raise ValueError("Surprise alpha must be nonnegative. The KL cap must be positive.")
    if alpha == 0:
        return {"enabled": False}
    dataset, reference, cache = Path(dataset), Path(reference), Path(cache)
    source = dataset/"dataset.manifest.json" if dataset.is_dir() else dataset
    identity = dict(dataset_sha256=_sha256(source), reference_sha256=_sha256(reference),
                    recipe_sha256=_sha256(Path(__file__)),
                    model_sha256=_sha256(Path(__file__).with_name("student_model.py")),
                    architecture=architecture, hidden=hidden, qat=True)
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
    folder = cache/key
    folder.mkdir(parents=True, exist_ok=True)
    manifest_file, kl_file = folder/"kl.manifest.json", folder/"kl.npy"
    n = len(data["value"])
    if manifest_file.exists():
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
        if manifest["identity"] != identity or manifest["sha256"] != _sha256(kl_file):
            raise ValueError("The surprise cache failed identity validation.")
        kl = np.load(kl_file, mmap_mode="r")
        if kl.shape != (n,):
            raise ValueError("The surprise cache has an incorrect row count.")
    else:
        model = Student(architecture, hidden, qat=True)
        model.load_float(reference)
        model.to(device).eval().requires_grad_(False)
        temporary = folder/"kl.partial.npy"
        kl = np.lib.format.open_memmap(temporary, mode="w+", dtype="float32", shape=(n,))
        fields = ("own_pawn", "opp_pawn", "walls_h", "walls_v", "own_dist", "opp_dist",
                  "walls_left_own", "walls_left_opp")
        with torch.inference_mode():
            for start in range(0, n, batch_size):
                end = min(start+batch_size,n)
                block = {field: np.asarray(data[field][start:end]) for field in fields}
                features = encode_features(block, np.arange(end-start), architecture)
                target = torch.from_numpy(np.asarray(data["policy"][start:end], np.float32)).to(device)
                kl[start:end] = compute_policy_kl_batch(model, torch.from_numpy(features).to(device), target).cpu().numpy()
        kl.flush()
        kl._mmap.close()
        temporary.replace(kl_file)
        _atomic_json(manifest_file, dict(complete=True, identity=identity, samples=n, sha256=_sha256(kl_file)))
        kl = np.load(kl_file, mmap_mode="r")
    variant = folder/("alpha_"+str(alpha)+"_cap_"+str(cap))
    variant.mkdir(exist_ok=True)
    variant_manifest = variant/"recipe.json"
    if variant_manifest.exists():
        existing = json.loads(variant_manifest.read_text(encoding="utf-8"))
        for field, expected in existing["weight_sha256"].items():
            weight_file = variant/(field+".npy")
            if _sha256(weight_file) != expected:
                raise ValueError("A cached surprise weight failed hash validation.")
            data[field] = np.load(weight_file, mmap_mode="r")
        return existing
    scales = {}
    for head in ("policy", "aux"):
        field = head+"_sample_weight"
        original = data.get(field)
        base_mass, boosted_mass = 0., 0.
        for start in range(0,n,32768):
            end = min(start+32768,n)
            train = ~np.asarray(data["is_val"][start:end],dtype=bool)
            weight = np.array(data["weight"][start:end],dtype=np.float64,copy=True)
            if original is not None:
                weight *= np.asarray(original[start:end],dtype=np.float64)
            boost = 1 + alpha*np.minimum(np.asarray(kl[start:end]),cap)
            base_mass += weight[train].sum()
            boosted_mass += (weight*boost)[train].sum()
        scale = base_mass/boosted_mass if boosted_mass > 0 else 1.
        scales[head] = float(scale)
        weights = np.lib.format.open_memmap(variant/(field+".npy"),mode="w+",dtype="float32",shape=(n,))
        for start in range(0,n,32768):
            end = min(start+32768,n)
            before = 1 if original is None else np.asarray(original[start:end])
            weights[start:end] = before*(1+alpha*np.minimum(np.asarray(kl[start:end]),cap))*scale
        weights.flush()
        data[field] = weights
    result = dict(enabled=True, alpha=alpha, cap=cap, scales=scales, identity=identity,
                  kl_sha256=_sha256(kl_file), cache=str(folder),
                  value_weights_changed=False, normalization="training_only_per_head_evidence_mass")
    result["weight_sha256"] = {head+"_sample_weight":_sha256(variant/(head+"_sample_weight.npy")) for head in ("policy","aux")}
    _atomic_json(variant_manifest, result)
    return result
