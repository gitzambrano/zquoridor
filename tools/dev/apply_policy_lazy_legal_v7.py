#!/usr/bin/env python3
from pathlib import Path

p = Path('src/nnue.hpp')
text = p.read_text()
old = '''inline void forwardPolicyQuant(const AccumulatorQuant& acc, std::array<float, POLICY_OUT>& out) {
    auto& W = weightsQuant();
    alignas(32) std::array<uint8_t, HIDDEN> a;
    for (int i = 0; i < HIDDEN; i++) a[i] = screluQuant(acc.v[i], W.QA);

    // perf/speed-elo-100: o laço quente era escalar com acumulador int64 e
    // um branch `if (ai == 0) continue` que impedia vetorização automática
    // -- sozinho, era ~60% do tempo do caminho MCAB+NNUE (perfilado com
    // benchmarks/profile_mcab.cpp). Agora: acumulador int32 SEM branch --
    // o limite superior da soma é 255*127*256 ~= 8.3M << INT32_MAX, então
    // o inteiro acumulado é EXATAMENTE o mesmo do código antigo em int64,
    // e a divisão final em double é idêntica bit a bit. O produto uint8 x
    // int8 -> int32 agora autovetoriza (AVX2: vpmovzx/vpmovsx + pmulld).
    const double qaqb = (double)((int64_t)W.QA * (int64_t)W.QB);
    for (int o = 0; o < POLICY_OUT; o++) {
        const int8_t* row = W.wp[o].data();
        const uint8_t* av = a.data();
        int32_t s = W.bp[o];
        for (int i = 0; i < HIDDEN; i++)
            s += (int32_t)av[i] * (int32_t)row[i];
        out[o] = (float)((double)s / qaqb);   // des-escala final idêntica, ver forwardValueQuant
    }
}
'''
new = '''inline void forwardPolicyQuant(const AccumulatorQuant& acc, std::array<float, POLICY_OUT>& out) {
    auto& W = weightsQuant();
    alignas(32) std::array<uint8_t, HIDDEN> a;
    for (int i = 0; i < HIDDEN; i++) a[i] = screluQuant(acc.v[i], W.QA);

    // exp/policy-lazy-legal-v7: quando o lado desta perspectiva não possui
    // mais muros, legalMoves/MCAB jamais podem consumir os 128 logits de
    // muro. Calculamos somente os 81 destinos de peão, com o MESMO produto
    // int32 e a MESMA desescala da baseline. Os slots inalcançáveis ficam
    // zerados defensivamente para evitar lixo caso algum diagnóstico leia
    // o array inteiro. Com >=1 muro o caminho é exatamente o baseline.
    const int outputs = (acc.ownWallsLeftBucket == 0) ? (N * N) : POLICY_OUT;
    if (outputs < POLICY_OUT) out.fill(0.f);

    const double qaqb = (double)((int64_t)W.QA * (int64_t)W.QB);
    for (int o = 0; o < outputs; o++) {
        const int8_t* row = W.wp[o].data();
        const uint8_t* av = a.data();
        int32_t s = W.bp[o];
        for (int i = 0; i < HIDDEN; i++)
            s += (int32_t)av[i] * (int32_t)row[i];
        out[o] = (float)((double)s / qaqb);
    }
}
'''
if old not in text:
    raise SystemExit('baseline forwardPolicyQuant block not found')
p.write_text(text.replace(old, new, 1))
print('applied exact zero-wall policy-row pruning v7')
