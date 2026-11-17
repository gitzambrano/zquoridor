"""Check that a native INT8 export matches its float NNUE student.

The check builds ``tools/teacher/student_probe.cpp`` with the export flags,
evaluates the probe positions natively and in Python, and writes
``parity.json`` in the experiment folder. A logit mismatch above 0.002 raises
an assertion error.
"""
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from training.student_model import Student, encode_features


def parity(out):
    architecture = json.loads((out/'student.architecture.json').read_text())
    probe = out/'native_probe'
    subprocess.run(['g++','-O2','-std=c++17','-I'+str(ROOT/'src'),*architecture['cpp_flags'],
                    str(ROOT/'tools/teacher/student_probe.cpp'),'-o',str(probe)],check=True)
    rows = [json.loads(line) for line in subprocess.check_output([str(probe),str(out/'student_int8.bin')],text=True).splitlines() if line.startswith('{')]
    keys = ('own_pawn','opp_pawn','walls_h','walls_v','own_dist','opp_dist','walls_left_own','walls_left_opp')
    data = {key:np.array([row[key] for row in rows],dtype=np.uint64 if key in ('walls_h','walls_v') else np.int64) for key in keys}
    model = Student(architecture['architecture'],architecture['hidden'],qat=True);model.load_float(out/'student.bin')
    with torch.no_grad():
        value,policy = model(torch.from_numpy(encode_features(data,np.arange(len(rows)),model.architecture)))
    native_value = np.array([row['value'] for row in rows]);native_policy = np.array([row['policy'] for row in rows])
    valid = np.ones_like(native_policy,dtype=bool);valid[data['walls_left_own']==0,81:]=False
    np.testing.assert_allclose(value.numpy(),native_value,atol=.002,rtol=0)
    np.testing.assert_allclose(policy.numpy()[valid],native_policy[valid],atol=.002,rtol=0)
    result = {'status':'passed','positions_perspectives':len(rows),
              'value_logit_max_abs_error':float(abs(value.numpy()-native_value).max()),
              'policy_logit_max_abs_error':float(abs(policy.numpy()[valid]-native_policy[valid]).max())}
    (out/'parity.json').write_text(json.dumps(result,indent=2)+'\n')
    return result
