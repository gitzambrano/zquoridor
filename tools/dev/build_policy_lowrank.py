#!/usr/bin/env python3
import argparse, json, re, struct
from pathlib import Path
import numpy as np

NUM_FEATURES=354
HIDDEN=256
POLICY_OUT=209
VALUE_HIDDEN=32
RANK=48

def parse_weights(path):
    data=Path(path).read_bytes()
    off=0
    QA,QB=struct.unpack_from('<ii',data,off); off+=8
    off += NUM_FEATURES*HIDDEN*2
    off += HIDDEN*2
    off += HIDDEN*VALUE_HIDDEN
    off += VALUE_HIDDEN*4
    off += VALUE_HIDDEN
    off += 4
    need=POLICY_OUT*HIDDEN
    wp=np.frombuffer(data,dtype=np.int8,count=need,offset=off).astype(np.float64).reshape(POLICY_OUT,HIDDEN)
    off += need
    bp=np.frombuffer(data,dtype='<i4',count=POLICY_OUT,offset=off).copy()
    off += POLICY_OUT*4
    if off != len(data):
        raise SystemExit(f'layout mismatch: parsed {off}, file {len(data)}')
    return QA,QB,wp,bp

def arr2cpp(name,a):
    rows=[]
    for row in a:
        rows.append('    {'+','.join(f'{x:.9g}f' for x in row)+'}')
    return f'inline constexpr float {name}[{a.shape[0]}][{a.shape[1]}] = {{\n'+',\n'.join(rows)+'\n};\n'

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('weights')
    ap.add_argument('--nnue',default='src/nnue.hpp')
    ap.add_argument('--header',default='src/policy_lowrank_r48.inc')
    ap.add_argument('--report',default='policy-lowrank-r48.json')
    args=ap.parse_args()
    QA,QB,M,bp=parse_weights(args.weights)
    U,S,Vt=np.linalg.svd(M,full_matrices=False)
    r=RANK
    root=np.sqrt(S[:r])
    B=U[:,:r]*root[None,:]
    A=root[:,None]*Vt[:r,:]
    Mr=B@A
    fro=np.linalg.norm(M)
    rel=float(np.linalg.norm(M-Mr)/(fro+1e-30))
    energy=float(np.sum(S[:r]**2)/np.sum(S**2))
    # Spectral sweep for deciding follow-up ranks from the same Gen8 matrix.
    sweep={}
    den=float(np.sum(S**2))
    for rr in (16,32,48,64,96,128):
        sweep[str(rr)]={'energy':float(np.sum(S[:rr]**2)/den),'relative_fro_error':float(np.sqrt(max(0.0,1-np.sum(S[:rr]**2)/den)))}
    header='// Generated from Gen8 quantized policy matrix by build_policy_lowrank.py\n#pragma once\ninline constexpr int POLICY_LR_RANK = 48;\n'+arr2cpp('POLICY_LR_A',A.astype(np.float32))+arr2cpp('POLICY_LR_B',B.astype(np.float32))
    Path(args.header).write_text(header)
    p=Path(args.nnue)
    text=p.read_text()
    if '#include "policy_lowrank_r48.inc"' not in text:
        marker='constexpr int POLICY_OUT = N * N + WS * WS * 2;'
        text=text.replace(marker,marker+'\n#include "policy_lowrank_r48.inc"',1)
    pat=r'inline void forwardPolicyQuant\(const AccumulatorQuant& acc, std::array<float, POLICY_OUT>& out\) \{.*?\n\}\n\n// ========================================================================='
    repl=r'''inline void forwardPolicyQuant(const AccumulatorQuant& acc, std::array<float, POLICY_OUT>& out) {
    auto& W = weightsQuant();
    alignas(32) std::array<uint8_t, HIDDEN> a;
    for (int i = 0; i < HIDDEN; i++) a[i] = screluQuant(acc.v[i], W.QA);

    // Experiment policy-lowrank-r48: SVD factorization of the fixed Gen8
    // quantized policy matrix. No activation is inserted between factors,
    // so this remains one linear map, approximated at rank 48.
    alignas(32) std::array<float, POLICY_LR_RANK> h{};
    for (int k = 0; k < POLICY_LR_RANK; k++) {
        float s = 0.f;
        const float* row = POLICY_LR_A[k];
        for (int i = 0; i < HIDDEN; i++) s += (float)a[i] * row[i];
        h[k] = s;
    }
    const float invScale = 1.0f / (float)((int64_t)W.QA * (int64_t)W.QB);
    for (int o = 0; o < POLICY_OUT; o++) {
        float s = (float)W.bp[o];
        const float* row = POLICY_LR_B[o];
        for (int k = 0; k < POLICY_LR_RANK; k++) s += h[k] * row[k];
        out[o] = s * invScale;
    }
}

// ========================================================================='''
    new,n=re.subn(pat,repl,text,flags=re.S)
    if n!=1:
        raise SystemExit(f'forwardPolicyQuant patch count={n}')
    p.write_text(new)
    report={'rank':r,'qa':QA,'qb':QB,'dense_macs':POLICY_OUT*HIDDEN,'lowrank_macs':HIDDEN*r+r*POLICY_OUT,'mac_reduction':1-(HIDDEN*r+r*POLICY_OUT)/(POLICY_OUT*HIDDEN),'energy':energy,'relative_fro_error':rel,'spectral_sweep':sweep}
    Path(args.report).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
if __name__=='__main__': main()
