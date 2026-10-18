"""Select a large training batch from measured CUDA memory demand."""
import gc
import hashlib
import json
from pathlib import Path

import torch


def largest_batch(minimum, maximum, probe):
    """Double the batch until the memory budget fails or the ceiling succeeds."""
    selected, attempts, candidate = None, [], min(minimum, maximum)
    while True:
        fits, peak = probe(candidate)
        attempts.append(dict(batch_size=candidate, fits=bool(fits), peak_bytes=int(peak)))
        if not fits:
            break
        selected = candidate
        if candidate == maximum:
            break
        candidate = min(candidate * 2, maximum)
    if selected is None:
        raise RuntimeError("The minimum training batch exceeds the CUDA memory budget. Reduce batch_size_min.")
    return selected, attempts


def calibrate(model, config, device, folder, sample_count, run_probe):
    """Measure full forward, backward, and Adam state allocation without retention."""
    requested = dict(batch_size=config['batch_size'], minimum=config['batch_size_min'],
                     maximum=config['batch_size_max'], fraction=config['gpu_memory_fraction'])
    if requested['minimum'] <= 0 or requested['maximum'] < requested['minimum']:
        raise ValueError("Batch limits must be positive and ordered.")
    if not 0 < requested['fraction'] < 1:
        raise ValueError("gpu_memory_fraction must be between zero and one.")
    if not config['auto_batch_size'] or not str(device).startswith('cuda'):
        return dict(enabled=False, batch_size=config['batch_size'], reason='fixed_batch_or_cpu')
    props = torch.cuda.get_device_properties(device)
    identity = dict(requested=requested, architecture=config['architecture'], hidden=config['hidden'],
                    qat=config['qat'], train_scope=config['train_scope'],
                    distillation_weight=config['distillation_weight'],
                    gpu=props.name, total_bytes=props.total_memory, samples=sample_count,
                    model_sha256=hashlib.sha256(Path(__file__).with_name('architecture_models.py').read_bytes()).hexdigest(),
                    trainer_sha256=hashlib.sha256(Path(__file__).with_name('run_experiment.py').read_bytes()).hexdigest(),
                    calibration_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    key = hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()[:16]
    cache = Path(folder).parent/'batch_capacity'/(config['architecture']+'_'+key+'.json')
    cache.parent.mkdir(parents=True,exist_ok=True)
    if cache.is_file():
        saved = json.loads(cache.read_text(encoding='utf-8'))
        if saved['identity'] != identity:
            raise ValueError("The batch calibration settings changed. Select a new output directory.")
        return saved
    free, total = torch.cuda.mem_get_info(device)
    # Include other processes in the budget. Reserve space for allocator variation.
    external = total-free-torch.cuda.memory_reserved(device)
    budget = min(int(total*requested['fraction'])-max(0,external), int(free*.95)+torch.cuda.memory_allocated(device))
    saved_state = {key:value.detach().cpu().clone() for key,value in model.state_dict().items()}
    mode = model.training
    active = getattr(model,'distillation_active',False)
    model.distillation_active = getattr(model,'distillation_teacher',None) is not None

    def probe(batch):
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        try:
            run_probe(batch)
            torch.cuda.synchronize(device)
            peak = max(torch.cuda.max_memory_allocated(device),torch.cuda.max_memory_reserved(device))
            return peak <= budget, peak
        except torch.cuda.OutOfMemoryError:
            return False, total
        finally:
            model.zero_grad(set_to_none=True)
            model.load_state_dict(saved_state)
            gc.collect()
            torch.cuda.empty_cache()

    try:
        device_index = torch.device(device).index
        if device_index is None:
            device_index = torch.cuda.current_device()
        with torch.random.fork_rng(devices=[device_index]):
            batch, attempts = largest_batch(requested['minimum'], min(requested['maximum'],sample_count), probe)
    finally:
        model.train(mode)
        model.distillation_active = active
    result = dict(enabled=True, identity=identity, batch_size=batch, attempts=attempts,
                  budget_bytes=budget, optimizer_updates_retained=False)
    temporary = cache.with_suffix('.tmp')
    temporary.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    temporary.replace(cache)
    print(json.dumps(dict(batch_calibration=result)),flush=True)
    return result
