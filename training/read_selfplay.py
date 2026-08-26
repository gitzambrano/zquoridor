"""
read_selfplay.py -- leitura do dataset binario de self-play gerado por
`selfplay` (harness C++, ver selfplay.hpp).

Dois formatos suportados (deteccao automatica por `load_selfplay`):
  - 32 bytes/posicao (SAMPLE_DTYPE, formato atual 2026-08): inclui `mover`,
    `own_cat_total`, `opp_cat_total` e own_pawn/opp_pawn/walls_*/policy_target
    JA ESPELHADOS para a perspectiva canonica do mover.
  - 27 bytes/posicao (SAMPLE_DTYPE_LEGACY, formato anterior): sem os 5 bytes
    extras acima; os campos comuns tem os mesmos offsets. `load_selfplay`
    detecta automaticamente e faz upcast para SAMPLE_DTYPE (os 3 campos novos
    ficam com valor 0), permitindo que o resto do pipeline use sempre
    SAMPLE_DTYPE independentemente do formato no disco.

`load_multi_selfplay`/`MultiFileSelfPlay` fazem o mesmo para varios
arquivos ao mesmo tempo SEM concatenar em RAM: cada shard continua
carregado (ou upcast) uma unica vez e so e lido do disco quando um
indice/slice especifico e efetivamente acessado.

Uso:
    python3 read_selfplay.py data/selfplay_001.bin [data/selfplay_002.bin ...]
"""
import glob
import os
import sys
import numpy as np

# Layout deve casar byte a byte com TrainingSample em selfplay.hpp (e com
# a copia duplicada em teste/arena.cpp -- ver nota lá sobre --bin-file).
# '<' = little-endian, sem padding.
#
# ATENÇÃO (mudança 2026-08): own_pawn/opp_pawn/walls_h/walls_v abaixo são
# gravados por selfplay.hpp JÁ ESPELHADOS pra perspectiva canônica do
# mover (mirroredPawnCell/mirrorWallBitboard, nnue.hpp) -- não são mais
# coordenada crua do tabuleiro. Datasets .bin gravados ANTES desta
# mudança tem 27 bytes/amostra (SAMPLE_DTYPE_LEGACY abaixo), sem os 3 campos
# novos (mover/own_cat_total/opp_cat_total) e com as 4 colunas acima em
# coordenada CRUA (sem espelho). load_selfplay detecta o formato automaticamente
# e faz upcast para SAMPLE_DTYPE; o restante do pipeline sempre usa SAMPLE_DTYPE.
#
# ATENÇÃO 2 (mudança 2026-08, mesma leva): o campo antigo `search_score`
# (int16, avaliação HEURÍSTICA evalSimple no momento do lance) foi
# substituído por `nnue_eval` (uint16, avaliação da PRÓPRIA NNUE, escala
# fixa 0..65535 == probabilidade 0.0..1.0 de vitória das BRANCAS --
# perspectiva ABSOLUTA de cor, não do mover; ver nota completa em
# TrainingSample::evalNNUE em selfplay.hpp e a tabela em status.md/
# CLAUDE.md "Avaliação: o que cada estágio usa"). Mesmo offset/tamanho (2
# bytes) do campo antigo -- arquivos .bin gravados ANTES desta mudança
# continuam lidos sem erro, só que o conteúdo desse campo ainda é a
# heurística antiga, não uma probabilidade. train_nnue.py NÃO tenta
# detectar automaticamente qual é qual: cada fonte em DATA_SOURCES_DEFAULT
# tem seu próprio `k` (peso do resultado real vs. desta avaliação
# gravada); fontes antigas devem usar k=1.0, o que zera completamente a
# contribuição deste campo, seja lá o que ele contiver.
EV_SCALE = 65535.0

# --- formato legado (27 bytes/amostra, gerado antes de 2026-08) --------------
SAMPLE_DTYPE_LEGACY = np.dtype([
    ("own_pawn",       "<u1"),   # 0..80, coordenada crua (SEM espelho)
    ("opp_pawn",       "<u1"),   # 0..80, coordenada crua (SEM espelho)
    ("walls_h",        "<u8"),   # bitboard, coordenada crua (SEM espelho)
    ("walls_v",        "<u8"),   # bitboard, coordenada crua (SEM espelho)
    ("walls_left_own", "<i1"),
    ("walls_left_opp", "<i1"),
    ("nnue_eval",      "<i2"),   # NOME NOVO, CONTEÚDO ANTIGO: neste formato pré-histórico
                                  # (anterior à própria NNUE existir) é sempre o heurístico
                                  # evalSimple, nunca uma avaliação de rede -- use k=1.0.
    ("game_result",    "<i1"),
    ("policy_target",  "<u2"),   # 0..208, coordenada crua (SEM espelho)
    ("own_dist",       "<u1"),
    ("opp_dist",       "<u1"),
])
assert SAMPLE_DTYPE_LEGACY.itemsize == 27

# --- formato atual (32 bytes/amostra, 2026-08+) ------------------------------
SAMPLE_DTYPE = np.dtype([
    ("own_pawn",       "<u1"),   # 0..80, já espelhado
    ("opp_pawn",       "<u1"),   # 0..80, já espelhado
    ("walls_h",        "<u8"),   # bitboard 64 bits, já espelhado
    ("walls_v",        "<u8"),   # bitboard 64 bits, já espelhado
    ("walls_left_own", "<i1"),
    ("walls_left_opp", "<i1"),
    ("nnue_eval",      "<u2"),   # avaliação da própria NNUE, 0..65535 == 0.0..1.0 (perspectiva
                                  # BRANCAS) -- ver ATENÇÃO 2 acima. Substitui o antigo search_score.
    ("game_result",    "<i1"),   # +1 vitoria do mover / -1 derrota
    ("policy_target",  "<u2"),   # 0..208: indice do lance jogado, já espelhado
    ("own_dist",       "<u1"),   # distancia BFS (shortestPathLen) ate a meta
    ("opp_dist",       "<u1"),
    # Campos novos (2026-08, mesma leva de mudança de formato):
    ("mover",          "<u1"),   # 0/1 -- identidade FÍSICA de quem jogou (não confundir com
                                  # own/opp, que são sempre relativos ao mover). Não é feature de
                                  # entrada da NNUE -- serve pra reconstruir o tabuleiro real fora
                                  # do referencial espelhado (debug/visualização/ferramentas externas)
                                  # E pra reprojetar nnue_eval (perspectiva brancas) pra perspectiva
                                  # do mover na hora de misturar com game_result (ver train_nnue.py).
    ("own_cat_total",  "<i2"),   # soma do calor de corredor (cat.hpp) do mover sobre as 81 casas --
    ("opp_cat_total",  "<i2"),   # candidato a feature futura, NÃO usado em to_dense_features/
                                  # to_chunk_tensors ainda (ver plano-additional.md).
])
assert SAMPLE_DTYPE.itemsize == 32

# Campos comuns entre SAMPLE_DTYPE_LEGACY e SAMPLE_DTYPE (mesmos nomes, mesmos
# offsets -- inclui "nnue_eval", que em ambos os dtypes ocupa os mesmos 2
# bytes; só muda o que esse campo SIGNIFICA entre um formato e outro, não
# onde ele mora).
_LEGACY_COMMON_FIELDS = [n for n, _ in SAMPLE_DTYPE_LEGACY.descr]


def _upcast_legacy(arr27: np.ndarray) -> np.ndarray:
    """Converte um array no formato legado (27 bytes) para SAMPLE_DTYPE (32
    bytes). Os campos comuns sao copiados; os 3 campos novos (mover,
    own_cat_total, opp_cat_total) ficam zerados. A copia e inevitavel
    (tamanhos diferentes), mas ocorre uma unica vez por arquivo em
    load_selfplay, antes do memmap/lazy-read -- o restante do pipeline e
    identico para os dois formatos."""
    out = np.zeros(len(arr27), dtype=SAMPLE_DTYPE)
    for field in _LEGACY_COMMON_FIELDS:
        out[field] = arr27[field]
    return out

# Bucket one-hot da distancia BFS -- mesma constante/semantica de
# DIST_BUCKETS em nnue.hpp: 0..19 exato, 20 = "20 ou mais".
DIST_BUCKETS = 21

# Muros restantes (feature nova, 2026-08) -- mesma constante/semantica de
# WALLS_LEFT_BUCKETS em nnue.hpp: 0..WALLS_PER_PLAYER, one-hot.
WALLS_PER_PLAYER = 10
WALLS_LEFT_BUCKETS = WALLS_PER_PLAYER + 1  # 11
N, WS = 9, 8
NUM_FEATURES = N * N + N * N + WS * WS * 2 + 2 * DIST_BUCKETS + 2 * WALLS_LEFT_BUCKETS  # 354


def dist_bucket(dist: np.ndarray) -> np.ndarray:
    return np.minimum(dist.astype(np.int64), DIST_BUCKETS - 1)


# --- per-game ply index -----------------------------------------------------
# selfplay.hpp writes one game for each fwrite call, so the samples of one
# game stay contiguous in the file and keep their ply order. That makes the
# per-game ply index recoverable without a ply field in the record.
#
# A ply 0 record is the initial position: no wall on the board, ten walls for
# each player, and a BFS distance of 8 for each player (both pawns still sit
# on their start rows). By ply 1 the first player has already moved, so
# opp_dist differs, unless move 1 placed a wall, which makes walls_h or
# walls_v non-zero. Either way, ply 1 cannot match the mask.
#
# Validated on 2026-08-25 over four shards and 9000 games: the mover field
# alternates strictly inside 100 percent of the detected games, and no game
# is degenerate or exceeds the 300 ply safety cut. See
# docs/superpowers/specs/2026-08-25-policy-visit-distribution-design.md.

def game_start_mask(arr: np.ndarray) -> np.ndarray:
    """Boolean mask that marks the first record of every game in `arr`."""
    return (
        (arr["walls_h"] == 0)
        & (arr["walls_v"] == 0)
        & (arr["walls_left_own"] == 10)
        & (arr["walls_left_opp"] == 10)
        & (arr["own_dist"] == 8)
        & (arr["opp_dist"] == 8)
    )


def ply_index(arr: np.ndarray) -> np.ndarray:
    """Per-record ply index inside its own game, as int32.

    `arr` must hold whole games in file order. A record before the first
    detected game start gets its ply counted from the start of the array,
    which only happens for a truncated file.
    """
    n = len(arr)
    if n == 0:
        return np.zeros(0, dtype=np.int32)
    starts = game_start_mask(arr)
    # Index of the most recent game start at or before each position.
    pos = np.arange(n, dtype=np.int64)
    last_start = np.maximum.accumulate(np.where(starts, pos, -1))
    # No start seen yet: count from 0 instead of from -1.
    last_start = np.where(last_start < 0, 0, last_start)
    return (pos - last_start).astype(np.int32)


def _detect_format_by_size(size: int):
    """Devolve (dtype, itemsize, ambig) para um arquivo de `size` bytes.
    `ambig=True` quando o tamanho e divisivel por ambos 27 e 32 (multiplo
    de mmc(27,32)=864) e a leitura do conteudo e necessaria para desempatar."""
    ok27 = size % 27 == 0
    ok32 = size % 32 == 0
    if ok27 and ok32:
        return None, None, True   # ambiguo: precisa inspecionar conteudo
    if ok27:
        return SAMPLE_DTYPE_LEGACY, 27, False
    if ok32:
        return SAMPLE_DTYPE, 32, False
    # Tamanho nao divisivel por nenhum: usa 32 como fallback (o caller ja
    # filtrou arquivos vazios/muito pequenos em expand_data_paths)
    return SAMPLE_DTYPE, 32, False


def _sniff_format(path: str, size: int) -> tuple:
    """Para arquivos divisiveis por ambos 27 e 32 (ambiguos), amostra os
    primeiros min(N, 100) registros em cada interpretacao e escolhe o formato
    cujos campos ficam dentro do range valido (own_pawn 0..80, policy 0..208).
    Se ambas passam ou ambas falham, prefere 27 (legado) como conservador --
    jamais descarta amostras validas silenciosamente."""
    n_probe = min(size // 32, 100)  # 32 >= 27, entao cabe em ambos
    if n_probe == 0:
        return SAMPLE_DTYPE_LEGACY, 27  # arquivo pequeno demais pra amostrar

    probe32 = np.memmap(path, dtype=SAMPLE_DTYPE,        mode="r", shape=(size // 32,))[:n_probe]
    probe27 = np.memmap(path, dtype=SAMPLE_DTYPE_LEGACY, mode="r", shape=(size // 27,))[:n_probe]

    ok32 = (int(probe32["policy_target"].max()) <= 208 and int(probe32["own_pawn"].max()) <= 80)
    ok27 = (int(probe27["policy_target"].max()) <= 208 and int(probe27["own_pawn"].max()) <= 80)

    if ok32 and not ok27:
        return SAMPLE_DTYPE, 32
    # Nos demais casos (so 27 ok, nenhum ok, ou ambos ok) usa legado --
    # e a escolha conservadora: se o arquivo for realmente novo e esse
    # caminho nunca deveria acontecer (mmc=864 e raro E arquivo novo=32-byte).
    return SAMPLE_DTYPE_LEGACY, 27


def load_selfplay(path: str, quiet: bool = False) -> np.ndarray:
    """Carrega um arquivo como array estruturado numpy. Detecta automaticamente
    o formato (27 ou 32 bytes/amostra) e sempre devolve SAMPLE_DTYPE (32
    bytes): arquivos legados sao upcast em memoria (copia unica por arquivo;
    os 3 campos novos ficam zerados). Arquivos no formato atual sao
    memory-mapped (zero-copy).

    Deteccao de formato:
      - Se o tamanho e divisivel so por 27: formato legado.
      - Se o tamanho e divisivel so por 32: formato atual.
      - Se divisivel por ambos (multiplo de 864): amostra o conteudo para
        decidir (own_pawn/policy_target devem ser <= 80/208).

    `quiet=True` suprime o aviso de formato legado (use quando o chamador
    emitir um aviso consolidado, como MultiFileSelfPlay)."""
    size = os.path.getsize(path)
    dtype, itemsize, ambig = _detect_format_by_size(size)
    if ambig:
        dtype, itemsize = _sniff_format(path, size)
    num_samples = size // itemsize
    if num_samples == 0:
        return np.empty(0, dtype=SAMPLE_DTYPE)
    raw = np.memmap(path, dtype=dtype, mode="r", shape=(num_samples,))
    if itemsize == 27:
        if not quiet:
            sys.stderr.write(
                f"[read_selfplay] aviso: formato legado (27 bytes) em {os.path.basename(path)} "
                f"-- upcast para 32 bytes (mover/cat zerados). "
                f"Regere o dataset para usar o formato atual.\n"
            )
        return _upcast_legacy(raw)
    return raw


def _is_legacy_file(path: str) -> bool:
    """Retorna True se o arquivo e formato legado (27 bytes/amostra)."""
    size = os.path.getsize(path)
    if size == 0:
        return False
    _, itemsize, ambig = _detect_format_by_size(size)
    if ambig:
        _, itemsize = _sniff_format(path, size)
    return itemsize == 27




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
            if not matches:
                raise ValueError(
                    f"diretorio '{token}' nao contem nenhum arquivo .bin "
                    f"(verifique se os shards de self-play estao numa subpasta, "
                    f"ex: '{os.path.join(token, 'selfplay')}')"
                )
        else:
            matches = sorted(glob.glob(token))
            if not matches:
                matches = [token]  # caminho literal; erro explicito adiante se nao existir
        for m in matches:
            if m not in out:
                if os.path.exists(m) and os.path.getsize(m) < SAMPLE_DTYPE.itemsize:
                    sys.stderr.write(f"[read_selfplay] aviso: ignorando arquivo de selfplay vazio/incompleto: {m}\n")
                    continue
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
        # Carrega todos os shards suprimindo o aviso por arquivo; emite um
        # unico aviso consolidado ao final se houver arquivos legados.
        self._maps = [load_selfplay(p, quiet=True) for p in self.paths]
        n_legacy = sum(1 for p in self.paths if _is_legacy_file(p))
        if n_legacy > 0:
            sys.stderr.write(
                f"[read_selfplay] aviso: {n_legacy} de {len(self.paths)} arquivo(s) "
                f"no formato legado (27 bytes/amostra, coordenadas sem espelho de "
                f"perspectiva) -- upcast automatico para 32 bytes aplicado. "
                f"Regere o dataset com selfplay.exe para garantir perspectiva correta.\n"
            )
        self._lens = [len(m) for m in self._maps]
        self._offsets = np.concatenate([[0], np.cumsum(self._lens)])
        self.dtype = SAMPLE_DTYPE


    def __len__(self):
        return int(self._offsets[-1])

    def sizes(self):
        return list(zip(self.paths, self._lens))

    def ply_index(self) -> np.ndarray:
        """Per-record ply index over the whole concatenated view, as int32.

        The index is built one shard at a time, in file order, because a
        game never crosses a shard boundary. The result therefore lines up
        with the global index space of this view, so the caller can permute
        or subsample it exactly like `k_by_source` in train_nnue.py.

        The array is built once and cached.
        """
        if getattr(self, "_ply_index", None) is None:
            self._ply_index = np.concatenate(
                [ply_index(m) for m in self._maps]
            ).astype(np.int32) if self._maps else np.zeros(0, dtype=np.int32)
        return self._ply_index

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
    """Converte um batch de TrainingSample nas NUM_FEATURES features
    esparsas one-hot (81 peao proprio + 81 peao oponente + 64 muro H + 64
    muro V + 21 bucket dist. propria + 21 bucket dist. oponente + 11
    bucket muros restantes proprios + 11 bucket muros restantes do
    oponente = 354) como matriz densa float32 (N, NUM_FEATURES).
    own_pawn/opp_pawn/walls_h/walls_v já chegam espelhados pra perspectiva
    canônica do mover (ver nota em SAMPLE_DTYPE acima) -- nada a
    transformar aqui, só espalhar em one-hot. Mantida em sincronia manual
    com to_chunk_tensors em train_nnue.py (mesma ressalva de duplicação
    de NUM_FEATURES em três lugares, ver nota lá)."""
    n = len(batch)
    x = np.zeros((n, NUM_FEATURES), dtype=np.float32)
    x[np.arange(n), batch["own_pawn"]] = 1.0
    x[np.arange(n), 81 + batch["opp_pawn"]] = 1.0
    x[:, 162:162 + 64] = unpack_wall_bits(batch["walls_h"])
    x[:, 162 + 64:162 + 128] = unpack_wall_bits(batch["walls_v"])
    own_bucket = dist_bucket(batch["own_dist"])
    opp_bucket = dist_bucket(batch["opp_dist"])
    x[np.arange(n), 290 + own_bucket] = 1.0
    x[np.arange(n), 290 + DIST_BUCKETS + opp_bucket] = 1.0
    wl_base = 290 + 2 * DIST_BUCKETS  # 332
    own_wl_bucket = np.clip(batch["walls_left_own"].astype(np.int64), 0, WALLS_LEFT_BUCKETS - 1)
    opp_wl_bucket = np.clip(batch["walls_left_opp"].astype(np.int64), 0, WALLS_LEFT_BUCKETS - 1)
    x[np.arange(n), wl_base + own_wl_bucket] = 1.0
    x[np.arange(n), wl_base + WALLS_LEFT_BUCKETS + opp_wl_bucket] = 1.0
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
    ev_prob = data["nnue_eval"].astype(np.float64) / EV_SCALE
    print(f"nnue_eval (prob. vitoria brancas): min={ev_prob.min():.4f} max={ev_prob.max():.4f} "
          f"media={ev_prob.mean():.4f}")
    print(f"policy_target: min={data['policy_target'].min()} max={data['policy_target'].max()} "
          f"(esperado 0..208)")
    print(f"own_dist: min={data['own_dist'].min()} max={data['own_dist'].max()} "
          f"media={data['own_dist'].astype(np.float64).mean():.2f}")
    print(f"opp_dist: min={data['opp_dist'].min()} max={data['opp_dist'].max()} "
          f"media={data['opp_dist'].astype(np.float64).mean():.2f}")
    feats = to_dense_features(data[:8])
    print(f"exemplo: to_dense_features(data[:8]).shape = {feats.shape} (esperado (8, {NUM_FEATURES}))")
