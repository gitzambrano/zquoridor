"""
read_selfplay.py -- leitura do dataset binario de self-play gerado por
`selfplay` (harness C++, ver selfplay.hpp).

O arquivo e um array cru de structs `TrainingSample` (packed, 27 bytes por
posicao). `load_selfplay` abre um unico arquivo via memmap (zero-copy).
`load_multi_selfplay`/`MultiFileSelfPlay` fazem o mesmo para varios
arquivos ao mesmo tempo SEM concatenar em RAM: cada shard continua
memory-mapped e so e lido do disco quando um indice/slice especifico e
efetivamente acessado. Isso e o que permite treinar (train_nnue.py /
train_nnue_numpy.py) em chunks limitados por um orcamento de RAM/VRAM em
vez de materializar o dataset inteiro de uma vez.

Uso:
    python3 read_selfplay.py data/selfplay_001.bin [data/selfplay_002.bin ...]
"""
import glob
import os
import sys
import numpy as np

# Layout deve casar byte a byte com TrainingSample em selfplay.hpp.
# '<' = little-endian, sem padding.
SAMPLE_DTYPE = np.dtype([
    ("own_pawn",       "<u1"),   # 0..80
    ("opp_pawn",       "<u1"),   # 0..80
    ("walls_h",        "<u8"),   # bitboard 64 bits
    ("walls_v",        "<u8"),   # bitboard 64 bits
    ("walls_left_own", "<i1"),
    ("walls_left_opp", "<i1"),
    ("search_score",   "<i2"),   # evalSimple no momento do lance
    ("game_result",    "<i1"),   # +1 vitoria do mover / -1 derrota
    ("policy_target",  "<u2"),   # 0..208: indice do lance jogado
    ("own_dist",       "<u1"),   # distancia BFS (shortestPathLen) ate a meta
    ("opp_dist",       "<u1"),
])
assert SAMPLE_DTYPE.itemsize == 27

# Bucket one-hot da distancia BFS -- mesma constante/semantica de
# DIST_BUCKETS em nnue.hpp: 0..19 exato, 20 = "20 ou mais".
DIST_BUCKETS = 21


def dist_bucket(dist: np.ndarray) -> np.ndarray:
    return np.minimum(dist.astype(np.int64), DIST_BUCKETS - 1)


def load_selfplay(path: str) -> np.ndarray:
    """Carrega um arquivo como array estruturado numpy (zero-copy via mmap)."""
    return np.memmap(path, dtype=SAMPLE_DTYPE, mode="r")


def expand_data_paths(spec) -> list:
    """Resolve a especificacao de dados num vetor ordenado de caminhos, sem
    duplicatas. `spec` pode ser:
      - uma string com um ou mais caminhos separados por virgula;
      - cada token pode ser um arquivo .bin direto, um diretorio (todos os
        *.bin dentro, ordenados) ou um glob (`data/gen/shard_*.bin`);
      - uma lista/tupla de strings, cada uma tratada com as mesmas regras
        acima (permite `--data` repetido na linha de comando).
    """
    if isinstance(spec, (list, tuple)):
        tokens = []
        for item in spec:
            tokens.extend(str(item).split(","))
    else:
        tokens = str(spec).split(",")

    out = []
    for token in tokens:
        token = token.strip()
        if not token:
            continue
        if os.path.isdir(token):
            matches = sorted(glob.glob(os.path.join(token, "*.bin")))
        else:
            matches = sorted(glob.glob(token))
        if not matches:
            matches = [token]  # caminho literal; erro explicito adiante se nao existir
        for m in matches:
            if m not in out:
                out.append(m)
    return out


class MultiFileSelfPlay:
    """Visao concatenada e preguicosa sobre varios shards de self-play.

    Ao contrario de `np.concatenate([load_selfplay(p) for p in paths])`,
    que copia todos os shards para um unico array em RAM, esta classe
    mantem cada shard como memmap e so le do disco os indices realmente
    pedidos. Suporta `len()`, indexacao inteira, slices e indexacao "fancy"
    (array/lista de indices), sempre devolvendo um array estruturado denso
    (copiado) apenas para os indices pedidos -- ideal para treino em
    chunks limitados por orcamento de RAM.
    """

    def __init__(self, paths):
        self.paths = list(paths)
        if not self.paths:
            raise ValueError("MultiFileSelfPlay: lista de caminhos vazia")
        self._maps = [load_selfplay(p) for p in self.paths]
        self._lens = [len(m) for m in self._maps]
        self._offsets = np.concatenate([[0], np.cumsum(self._lens)])
        self.dtype = SAMPLE_DTYPE

    def __len__(self):
        return int(self._offsets[-1])

    def sizes(self):
        return list(zip(self.paths, self._lens))

    def _fancy(self, idx: np.ndarray) -> np.ndarray:
        idx = np.asarray(idx, dtype=np.int64)
        n = len(self)
        idx = np.where(idx < 0, idx + n, idx)
        if idx.size and ((idx < 0).any() or (idx >= n).any()):
            raise IndexError("indice fora do intervalo em MultiFileSelfPlay")
        out = np.empty(len(idx), dtype=self.dtype)
        file_ids = np.searchsorted(self._offsets, idx, side="right") - 1
        for fi in np.unique(file_ids):
            mask = file_ids == fi
            local = idx[mask] - self._offsets[fi]
            out[mask] = self._maps[fi][local]
        return out

    def __getitem__(self, key):
        n = len(self)
        if isinstance(key, slice):
            start, stop, step = key.indices(n)
            return self._fancy(np.arange(start, stop, step))
        if isinstance(key, (np.ndarray, list, tuple)):
            return self._fancy(np.asarray(key))
        if isinstance(key, (int, np.integer)):
            k = int(key) + (n if key < 0 else 0)
            if not (0 <= k < n):
                raise IndexError("indice fora do intervalo em MultiFileSelfPlay")
            fi = int(np.searchsorted(self._offsets, k, side="right") - 1)
            return self._maps[fi][k - self._offsets[fi]]
        raise TypeError(f"tipo de indice nao suportado: {type(key)!r}")


def load_multi_selfplay(spec):
    """Atalho: expande `spec` (string/lista, ver `expand_data_paths`) e
    devolve (paths, MultiFileSelfPlay)."""
    paths = expand_data_paths(spec)
    if not paths:
        raise ValueError(f"nenhum arquivo .bin encontrado para: {spec!r}")
    return paths, MultiFileSelfPlay(paths)


def unpack_wall_bits(walls_u64: np.ndarray) -> np.ndarray:
    """Expande uma coluna de bitboards uint64 (N,) em (N, 64) bits 0/1."""
    bits = ((walls_u64[:, None] >> np.arange(64, dtype=np.uint64)) & 1).astype(np.float32)
    return bits


def to_dense_features(batch: np.ndarray) -> np.ndarray:
    """Converte um batch de TrainingSample nas 332 features esparsas one-hot
    (81 peao proprio + 81 peao oponente + 64 muro H + 64 muro V + 21 bucket
    dist. propria + 21 bucket dist. oponente) como matriz densa float32
    (N, 332)."""
    n = len(batch)
    x = np.zeros((n, 332), dtype=np.float32)
    x[np.arange(n), batch["own_pawn"]] = 1.0
    x[np.arange(n), 81 + batch["opp_pawn"]] = 1.0
    x[:, 162:162 + 64] = unpack_wall_bits(batch["walls_h"])
    x[:, 162 + 64:162 + 128] = unpack_wall_bits(batch["walls_v"])
    own_bucket = dist_bucket(batch["own_dist"])
    opp_bucket = dist_bucket(batch["opp_dist"])
    x[np.arange(n), 290 + own_bucket] = 1.0
    x[np.arange(n), 290 + DIST_BUCKETS + opp_bucket] = 1.0
    return x


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    paths = expand_data_paths(sys.argv[1:])
    ds = MultiFileSelfPlay(paths)
    print(f"arquivos: {len(paths)}")
    for p, n in ds.sizes():
        print(f"  - {p}: {n:,} posicoes")
    data = ds[:]
    print(f"total de posicoes: {len(data)}")
    print(f"bytes/posicao: {SAMPLE_DTYPE.itemsize} (total: {len(data) * SAMPLE_DTYPE.itemsize:,} bytes)")
    print(f"distribuicao de game_result: +1={np.sum(data['game_result'] == 1)}  "
          f"0={np.sum(data['game_result'] == 0)}  -1={np.sum(data['game_result'] == -1)}")
    print(f"search_score: min={data['search_score'].min()} max={data['search_score'].max()} "
          f"media={data['search_score'].astype(np.float64).mean():.2f}")
    print(f"policy_target: min={data['policy_target'].min()} max={data['policy_target'].max()} "
          f"(esperado 0..208)")
    print(f"own_dist: min={data['own_dist'].min()} max={data['own_dist'].max()} "
          f"media={data['own_dist'].astype(np.float64).mean():.2f}")
    print(f"opp_dist: min={data['opp_dist'].min()} max={data['opp_dist'].max()} "
          f"media={data['opp_dist'].astype(np.float64).mean():.2f}")
    feats = to_dense_features(data[:8])
    print(f"exemplo: to_dense_features(data[:8]).shape = {feats.shape} (esperado (8, 332))")
