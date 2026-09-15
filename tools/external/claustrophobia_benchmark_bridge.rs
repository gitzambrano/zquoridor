//! Keep one Claustrophobia model loaded for all searches in one benchmark game.
use std::io::{self, BufRead, Write};

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
    if args.len() != 5 {
        eprintln!("usage: zq_benchmark_bridge <checkpoint.pt> <sims> <cpuct> <cpu|gpu>");
        std::process::exit(2);
    }
    let sims: u32 = args[2].parse()?;
    let cpuct: f64 = args[3].parse()?;
    if sims == 0 || cpuct <= 0.0 {
        return Err("sims and cpuct must be positive".into());
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
                let result = run_mcts_batched(state, &evaluator, sims, cpuct, config);
                match index_to_move(&state, result.best_action) {
                    Some(chosen) if parse_move(&move_text(chosen), &state).is_some() => {
                        println!(
                            "{{\"bestmove\":\"{}\",\"sims\":{},\"root_value\":{:.9}}}",
                            move_text(chosen), sims, result.root_value_side_to_move()
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
