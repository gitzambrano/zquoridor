//! Keep one Claustrophobia model loaded for all searches in one benchmark game.
use std::io::{self, BufRead, Write};
use std::thread;
use std::time::{Duration, Instant};

use quoridor::mcts::{run_mcts_batched, BatchedConfig};
#[cfg(feature = "nn")]
use quoridor::nn::TchEvaluator;
#[cfg(not(feature = "nn"))]
mod zq_ipc_eval;
#[cfg(not(feature = "nn"))]
use zq_ipc_eval::TchEvaluator;
use quoridor::{generate_all_moves, index_to_move, GameState, Move, MoveBuf};

fn parse_move(input: &str, state: &GameState) -> Option<Move> {
    let text = input.trim().to_ascii_lowercase();
    let bytes = text.as_bytes();
    let candidate = if bytes.len() == 2
        && (b'a'..=b'i').contains(&bytes[0])
        && (b'1'..=b'9').contains(&bytes[1])
    {
        let col = bytes[0] - b'a';
        let row = bytes[1] - b'1';
        Move::step(row * 9 + col)
    } else if bytes.len() == 3
        && (b'a'..=b'h').contains(&bytes[0])
        && (b'1'..=b'8').contains(&bytes[1])
        && (bytes[2] == b'h' || bytes[2] == b'v')
    {
        let col = bytes[0] - b'a';
        let row = bytes[1] - b'1';
        Move::wall(row * 8 + col, bytes[2] == b'h')
    } else {
        return None;
    };
    let mut moves = MoveBuf::new();
    generate_all_moves(state, &mut moves);
    moves.as_slice().iter().copied().find(|&item| item == candidate)
}

fn replay(history: &str) -> Result<GameState, String> {
    let mut state = GameState::start();
    for (ply, token) in history.split_whitespace().enumerate() {
        if state.is_terminal() {
            return Err(format!("terminal position before ply {ply}"));
        }
        let chosen = parse_move(token, &state)
            .ok_or_else(|| format!("illegal history move `{token}` at ply {ply}"))?;
        state.make_move(chosen);
    }
    if state.is_terminal() {
        return Err("cannot search a terminal position".to_string());
    }
    Ok(state)
}

fn move_text(chosen: Move) -> String {
    if chosen.is_wall() {
        let slot = chosen.slot();
        format!(
            "{}{}{}",
            (b'a' + (slot % 8) as u8) as char,
            slot / 8 + 1,
            if chosen.horiz() { 'h' } else { 'v' }
        )
    } else {
        let destination = chosen.dest() as usize;
        format!(
            "{}{}",
            (b'a' + (destination % 9) as u8) as char,
            destination / 9 + 1
        )
    }
}

fn json_error(message: &str) -> String {
    let escaped = message
        .replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
        .replace('\r', "\\r");
    format!("{{\"error\":\"{escaped}\"}}")
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 5 && args.len() != 6 {
        eprintln!("usage: zq_benchmark_bridge <checkpoint.pt> <move-time-ms> <cpuct> <cpu|gpu> [max-sims]");
        std::process::exit(2);
    }
    let move_time_ms: u64 = args[2].parse()?;
    let cpuct: f64 = args[3].parse()?;
    let max_sims: u32 = if args.len() == 6 { args[5].parse()? } else { 4096 };
    if move_time_ms == 0 || max_sims == 0 || cpuct <= 0.0 {
        return Err("move-time-ms, max-sims, and cpuct must be positive".into());
    }
    if args[4] != "cpu" && args[4] != "gpu" {
        return Err("device must be cpu or gpu".into());
    }
    // The public checkpoint uses the 20-plane encoder.
    std::env::set_var("QUORIDOR_PLANES", "20");
    std::env::set_var("QUORIDOR_DEVICE", &args[4]);
    let evaluator = TchEvaluator::new(&args[1])?;
    let mut config = BatchedConfig::new(8);
    config.eval_tt = true;
    config.solver = true;

    // Measure a small deterministic search once. The bridge then keeps each
    // search below the clock and sleeps until the complete move budget ends.
    // The cap protects positions that are slower than the start position.
    let calibration_sims = 8u32.min(max_sims);
    let calibration_start = Instant::now();
    let _ = run_mcts_batched(GameState::start(), &evaluator, calibration_sims, cpuct, config);
    let calibration_ms = calibration_start.elapsed().as_secs_f64() * 1000.0;
    let sims_per_ms = calibration_sims as f64 / calibration_ms.max(1.0);

    println!("ready");
    io::stdout().flush()?;
    for line in io::stdin().lock().lines() {
        let line = line?;
        if line == "quit" {
            break;
        }
        let Some((command, history)) = line.split_once('\t') else {
            println!("{}", json_error("expected a tab after the command"));
            io::stdout().flush()?;
            continue;
        };
        if command != "position" {
            println!("{}", json_error("unknown command"));
            io::stdout().flush()?;
            continue;
        }
        match replay(history) {
            Ok(state) => {
                let started = Instant::now();
                // Reserve one quarter of the clock for position variance. The
                // remaining time is spent as a deterministic MCTS search.
                let search_budget_ms = (move_time_ms as f64 * 0.75).max(1.0);
                let sims = ((sims_per_ms * search_budget_ms).floor() as u32)
                    .clamp(1, max_sims);
                let result = run_mcts_batched(state, &evaluator, sims, cpuct, config);
                match index_to_move(&state, result.best_action) {
                    Some(chosen) if parse_move(&move_text(chosen), &state).is_some() => {
                        let search_ms = started.elapsed().as_secs_f64() * 1000.0;
                        let remaining_ms = move_time_ms as f64 - search_ms;
                        if remaining_ms > 0.0 {
                            thread::sleep(Duration::from_secs_f64(remaining_ms / 1000.0));
                        }
                        let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
                        println!(
                            "{{\"bestmove\":\"{}\",\"move_time_ms\":{},\"sims\":{},\"search_ms\":{:.3},\"elapsed_ms\":{:.3},\"root_value\":{:.9}}}",
                            move_text(chosen), move_time_ms, sims, search_ms, elapsed_ms,
                            result.root_value_side_to_move()
                        );
                    }
                    _ => println!("{}", json_error("search returned an illegal action")),
                }
            }
            Err(message) => println!("{}", json_error(&message)),
        }
        io::stdout().flush()?;
    }
    Ok(())
}
