"""Internal orchestration for the shared architecture campaign."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import shutil

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT/"training") not in sys.path:
    sys.path.insert(0, str(ROOT/"training"))
from architecture_data import inventory, prepare_ledgers
from architecture_models import SPECS, BASE
from mix_experimental_datasets import mix_datasets, _atomic_json

DEFAULTS = {
    "command": "plan",
    "suite": "experimental/experiments/architecture-lab-v401/suite.json",
    "experiments": [],
    "data_roots": ["data/teaching", "experimental/experiments/production-central-finetune-20261002/new_replay"],
    "downloads": ["artifacts/architecture_campaign/cloud/downloads.jsonl"],
    "sources_file": "",
    "output": "experimental/experiments/architecture-lab-v401/artifacts",
    "cloud": "none",
    "cloud_latest_only": True,
    "worker_ids": [1, 2, 3, 4, 5],
    "require_five_workers": True,
    "pool_fractions": {"historical": .25, "replay": .15, "teacher": .15,
                       "wall": .10, "gumbel": .15, "temporal": .20},
    "chunk_size": 8192,
    "epochs": 40,
    "batch_size": 2048,
    "device": "auto",
    "cpu_threads": 4,
    "seed": 20261018,
    "resume": True,
    "init_from": "data/nnue/nnue_weights.bin",
    "training_overrides": {},
}


def path(value):
    result = Path(value)
    return result if result.is_absolute() else ROOT/result


def selected_experiments(cfg):
    suite_path = path(cfg["suite"])
    suite = json.loads(suite_path.read_text(encoding="utf-8"))
    selected = cfg["experiments"] or suite["experiments"]
    if len(selected) != len(set(selected)) or not set(selected) <= set(suite["experiments"]):
        raise ValueError("Unknown or repeated experiment names.")
    return [json.loads((suite_path.parent/name/"experiment.json").read_text(encoding="utf-8")) for name in selected]


def training_config(cfg, experiment, dataset):
    from run_experiment import CONFIG
    variant = experiment["architecture"]
    spec = SPECS.get(variant, {})
    result = dict(CONFIG, data=str(dataset), out_dir=str(path(cfg["output"])/experiment["id"]),
                  architecture=variant, hidden=512, init_architecture=BASE, init_hidden=512,
                  init_from=str(reference_path(cfg)), from_scratch=spec.get("transfer") == "distillation",
                  qat=True, epochs=cfg["epochs"], batch_size=cfg["batch_size"], device=cfg["device"],
                  cpu_threads=cfg["cpu_threads"], seed=cfg["seed"], resume=cfg["resume"],
                  lr=1e-5, min_lr=1e-7, warmup_epochs=0, trunk_lr_scale=.1, patience=0,
                  checkpoint_every=5, aux_policy_enabled=True, aux_policy_lr_scale=1.,
                  aux_policy_temperature=2., aux_policy_weight=.15, max_sample_weight=1e9,
                  build=False, benchmark=False)
    result.update(experiment.get("training", {}))
    overrides = cfg.get("training_overrides", {})
    protected = {"architecture", "data", "out_dir", "init_from", "init_architecture", "init_hidden", "build", "benchmark"}
    if set(overrides) & protected or set(overrides) - result.keys():
        raise ValueError("Training overrides contain an unknown or protected field.")
    result.update(overrides)
    return result


def reference_path(cfg):
    frozen = path(cfg["output"])/"reference/nnue_weights.bin"
    return frozen if frozen.exists() else path(cfg["init_from"])


def resolve_sources(rows, cfg):
    selected = [row for row in rows if row["status"] == "available"]
    active = set(row["pool"] for row in selected)
    total = sum(cfg["pool_fractions"][pool] for pool in active)
    if total <= 0:
        raise ValueError("No eligible training datasets are available.")
    return [dict(row, fraction=None, pool_fraction=cfg["pool_fractions"][row["pool"]]/total) for row in selected]


def cloud_snapshot(cfg):
    if cfg["cloud"] not in ("none", "inventory", "download"):
        raise ValueError("cloud must be none, inventory, or download")
    if cfg["cloud"] == "none":
        return
    from colab.cli.settings import cli_python
    destination = ROOT/"artifacts/architecture_campaign/cloud"
    # Use the installed CLI environment. It owns the Colab client dependency.
    code = ("from colab.cli.data_snapshot import snapshot; import json; "
            "r=snapshot("+repr(cfg["worker_ids"])+", "+repr(str(destination))+", download="+
            repr(cfg["cloud"] == "download")+", latest_only="+repr(cfg["cloud_latest_only"])+"); "
            "print(json.dumps({k:v for k,v in r.items() if k!='files'},indent=2))")
    subprocess.run([cli_python(), "-c", code], check=True, cwd=ROOT)


def run(cfg):
    cfg = copy.deepcopy(cfg)
    if cfg["command"] not in ("plan", "initialize", "prepare", "train"):
        raise ValueError("The campaign command must be plan, initialize, prepare, or train.")
    output = path(cfg["output"])
    cloud_snapshot(cfg)
    rows = inventory(cfg["data_roots"])
    ledgers = [path(value) for value in cfg["downloads"] if path(value).exists()]
    workers = set()
    for ledger in ledgers:
        records = [json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines() if line.strip()]
        keys = {record["path"].replace("\\", "/") for record in records if Path(record["local"]).is_file()
                and Path(record["local"]).stat().st_size > 0}
        for record in records:
            key = record["path"].replace("\\", "/")
            pair = key.rsplit("/",1)[0]+"/targets.jsonl"
            paired_rich = record["title"] == "positions.jsonl" and pair in keys
            paired_shard = (record["title"].endswith(".bin") and not record["title"].endswith(".meta.bin")
                            and any(key[:-4]+suffix in keys for suffix in (".meta", ".meta.bin")))
            if (record.get("worker") in cfg["worker_ids"] and "v401" in record.get("kind", "")
                    and key in keys and (paired_rich or paired_shard)):
                workers.add(record["worker"])
    missing = sorted(set(cfg["worker_ids"]) - workers)
    experiments = selected_experiments(cfg)
    dataset = output/"dataset"
    plan = dict(schema="zquoridor.architecture_campaign.v1", command=cfg["command"],
                configuration=cfg, sources=rows, download_ledgers=[str(p) for p in ledgers],
                latest_workers_materialized=sorted(workers), missing_latest_workers=missing,
                training_ready=not missing or not cfg["require_five_workers"],
                dataset=str(dataset), jobs=[dict(experiment=e, config=training_config(cfg,e,dataset)) for e in experiments],
                production_modified=False, training_started=False)
    _atomic_json(output/"plan.json", plan)
    if cfg["command"] == "plan":
        return plan
    if cfg["command"] == "initialize":
        import torch
        from architecture_models import make_model, SCHEMA
        from student_model import Student
        torch.set_num_threads(cfg["cpu_threads"])
        frozen = output/"reference/nnue_weights.bin"
        if not frozen.exists():
            frozen.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path(cfg["init_from"]), frozen)
        reference = Student(BASE,512,qat=True)
        reference.load_float(frozen)
        initialized = []
        for experiment in experiments:
            torch.manual_seed(cfg["seed"])
            model = make_model(experiment["architecture"],512,qat=True)
            transfer = SPECS.get(experiment["architecture"], {}).get("transfer", "exact")
            if transfer == "exact":
                model.warm_start(reference)
            folder = output/experiment["id"]
            folder.mkdir(parents=True, exist_ok=True)
            torch.save(dict(schema=SCHEMA, architecture=model.architecture, hidden=512, qat=True,
                            model=model.state_dict()), folder/"initial_model.pt")
            record = dict(id=experiment["id"], transfer=transfer,
                          parameters=sum(p.numel() for p in model.parameters()),
                          training_started=False, checkpoint=str(folder/"initial_model.pt"))
            _atomic_json(folder/"initialization.json", record)
            initialized.append(record)
        return dict(status="initialized_without_training", experiments=initialized,
                    missing_latest_workers=missing, training_started=False)
    if missing and cfg["require_five_workers"]:
        raise ValueError("Latest v4.01 snapshots are missing for workers: "+", ".join(map(str,missing))+
                         ". Materialize their download ledgers before preparation or training.")
    if ledgers:
        rows.extend(prepare_ledgers(ledgers, output/"prepared", reference_path(cfg), cfg["seed"]))
    if cfg["sources_file"]:
        sources = json.loads(path(cfg["sources_file"]).read_text(encoding="utf-8"))
    else:
        sources = resolve_sources(rows, cfg)
    _atomic_json(output/"sources.json", sources)
    mixture = mix_datasets(sources, dataset, cfg["chunk_size"], cfg["resume"])
    _atomic_json(output/"preparation.json", dict(sources=rows, mixture=mixture))
    if cfg["command"] == "prepare":
        return dict(status="prepared", dataset=str(dataset), samples=mixture["samples"])
    from run_experiment import train
    reports = []
    for experiment in experiments:
        train_cfg = training_config(cfg, experiment, dataset)
        _atomic_json(output/"status.json", dict(stage="training", experiment=experiment["id"], completed=reports))
        report = train(train_cfg)
        reports.append(dict(id=experiment["id"], report=report))
    result = dict(status="training_complete_evaluation_pending", experiments=reports, promoted=False)
    _atomic_json(output/"status.json", result)
    return result
