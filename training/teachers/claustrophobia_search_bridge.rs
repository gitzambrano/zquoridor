//! Companion binary compiled inside a pinned Claustrophobia checkout.
//!
//! It replays ZQuoridor move histories with Claustrophobia's own rules, runs the
//! pinned teacher net plus MCTS, and emits one JSON object per input row with
//! root visit counts and root value. The Python adapter converts action indices
//! into ZQuoridor's canonical frame and stores compact soft targets.

use std::fs::File;
use std::io::{BufRead, BufReader};

use quoridor::mcts::{run_mcts_batched, BatchedConfig};
#[cfg(feature = "nn")]
use quoridor::nn::TchEvaluator;
#[cfg(not(feature = "nn"))]
mod zq_ipc_eval;
#[cfg(not(feature = "nn"))]
use zq_ipc_eval::TchEvaluator;
use quoridor::{generate_all_moves, GameState, Move, MoveBuf};

fn parse_move(input: &str, state: &GameState) -> Option<Move> {
    let t = input.trim().to_ascii_lowercase();
    let b = t.as_bytes();
    let mut buf = MoveBuf::new();
    generate_all_moves(state, &mut buf);
    let candidate = if b.len() == 3
        && (b'a'..=b'i').contains(&b[0])
        && (b'1'..=b'9').contains(&b[1])
    {
        let col = (b[0] - b'a') as usize;
        let row = (b[1] - b'1') as usize;
        if row >= 8 || col >= 8 || (b[2] != b'h' && b[2] != b'v') {
            return None;
        }
        Move::wall((row * 8 + col) as u8, b[2] == b'h')
    } else if b.len() == 2
        && (b'a'..=b'i').contains(&b[0])
        && (b'1'..=b'9').contains(&b[1])
    {
        let col = (b[0] - b'a') as usize;
        let row = (b[1] - b'1') as usize;
        Move::step((row * 9 + col) as u8)
    } else {
        return None;
    };
    buf.as_slice().iter().copied().find(|&m| m == candidate)
}

fn replay(history: &str, sample_id: &str) -> Result<GameState, String> {
    // V3 snapshots are already mirrored to the mover.  Reconstruct them as
    // side zero so the emitted actions remain in the V3 action frame.  The
    // record has no repetition history, therefore MCTS starts with none.
    if history.starts_with("@state ") {
        let fields: Vec<&str> = history.split_whitespace().collect();
        if fields.len() != 7 { return Err("state requires six fields".into()); }
        let own: u8 = fields[1].parse().map_err(|_| "invalid own pawn")?;
        let opp: u8 = fields[2].parse().map_err(|_| "invalid opponent pawn")?;
        let wh: u64 = fields[3].parse().map_err(|_| "invalid horizontal walls")?;
        let wv: u64 = fields[4].parse().map_err(|_| "invalid vertical walls")?;
        let ow: u8 = fields[5].parse().map_err(|_| "invalid own reserve")?;
        let pw: u8 = fields[6].parse().map_err(|_| "invalid opponent reserve")?;
        if own >= 81 || opp >= 81 || own == opp || ow > 10 || pw > 10
            || wh.count_ones() + wv.count_ones() != 20 - ow as u32 - pw as u32 {
            return Err("invalid canonical state resources or pawns".into());
        }
        let mut state = GameState::with_pawns(own, opp, 0);
        for (walls, horizontal) in [(wh, true), (wv, false)] {
            for slot in 0..64 {
                if (walls >> slot) & 1 == 0 { continue; }
                let mv = Move::wall(slot as u8, horizontal);
                let mut legal = MoveBuf::new();
                generate_all_moves(&state, &mut legal);
                if !legal.as_slice().contains(&mv) { return Err("illegal wall topology".into()); }
                state.setup_wall(slot, horizontal);
            }
        }
        state.walls_left = [ow, pw];
        return Ok(state);
    }
    let mut state = GameState::start();
    if history.trim().is_empty() {
        return Ok(state);
    }
    for (ply, token) in history.split_whitespace().enumerate() {
        if state.is_terminal() {
            return Err(format!("{sample_id}: terminal before ply {ply}"));
        }
        let mv = parse_move(token, &state)
            .ok_or_else(|| format!("{sample_id}: illegal move `{token}` at ply {ply}"))?;
        state.make_move(mv);
    }
    Ok(state)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 || args.len() > 5 {
        eprintln!("usage: zq_search_bridge <positions.tsv> <checkpoint.pt> <sims> [cpuct]");
        std::process::exit(2);
    }
    let sims: u32 = args[3].parse()?;
    if sims == 0 {
        return Err("sims must be positive".into());
    }
    let cpuct: f64 = if args.len() == 5 { args[4].parse()? } else { 1.5 };
    // The public v1.3.1 champion is a 20-plane net. Keep this local to the
    // bridge so callers cannot accidentally evaluate it with the 16-plane path.
    std::env::set_var("QUORIDOR_PLANES", "20");
    let evaluator = TchEvaluator::new(&args[2])?;
    let mut cfg = BatchedConfig::new(8);
    cfg.eval_tt = true;
    cfg.solver = true;

    let input = BufReader::new(File::open(&args[1])?);
    for (line_no, line) in input.lines().enumerate() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let (sample_id, history) = line
            .split_once('\t')
            .ok_or_else(|| format!("input line {} lacks a tab", line_no + 1))?;
        let state = replay(history, sample_id)?;
        if state.is_terminal() {
            return Err(format!("{sample_id}: search position is terminal").into());
        }
        let result = run_mcts_batched(state, &evaluator, sims, cpuct, cfg);
        print!(
            "{{\"id\":\"{}\",\"side_to_move\":{},\"sims\":{},\"best_action\":{},\"root_value\":{:.9},\"visits\":[",
            sample_id,
            state.side,
            sims,
            result.best_action,
            result.root_value_side_to_move()
        );
        for (i, (action, visits)) in result.visit_counts.iter().enumerate() {
            if i != 0 {
                print!(",");
            }
            print!("[{},{}]", action, visits);
        }
        println!("]}}");
    }
    Ok(())
}
