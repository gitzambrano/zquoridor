//! Keep one Claustrophobia model loaded for all searches in one benchmark game.
use std::io::{self, BufRead, Write};
use std::time::{Duration, Instant};

use quoridor::mcts::{run_mcts_batched, run_mcts_batched_until, BatchedConfig};
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

// The upstream fixed-count API has no game-clock manager. This engine-side
// extension owns its clock policy; the arena sends complete clocks unchanged.
fn clock_budget(remaining_ms: u64, increment_ms: u64, ply: usize, walls: u8) -> u64 {
    let reserve = (remaining_ms / 20).clamp(50, 250).min(remaining_ms.saturating_sub(1));
    let spendable = remaining_ms.saturating_sub(reserve).max(1);
    let horizon: usize = if walls == 0 { 12 } else { 30 };
    let own_moves = horizon.saturating_sub(ply / 6).max(8) as u64;
    (spendable / own_moves + increment_ms.saturating_mul(4) / 5).clamp(1, spendable)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 5 && args.len() != 6 {
        eprintln!("usage: zq_benchmark_bridge <checkpoint.pt> <move-time-ms> <cpuct> <cpu|gpu> [max-sims]");
        std::process::exit(2);
    }
    let move_time_ms: u64 = args[2].parse()?;
    let cpuct: f64 = args[3].parse()?;
    // Accept the obsolete positional cap only for adapter compatibility. It
    // never limits clock searches; the protocol records the actual work done.
    let _obsolete_cap: u32 = if args.len() == 6 { args[5].parse()? } else { u32::MAX };
    if move_time_ms == 0 || !cpuct.is_finite() || cpuct <= 0.0 {
        return Err("move-time-ms and cpuct must be positive and finite".into());
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

    // Warm model profiling outside the game clock. No throughput estimate is
    // used to decide when any later search stops.
    for _ in 0..3 {
        let _ = run_mcts_batched(GameState::start(), &evaluator, 128, cpuct, config);
    }

    println!("ready");
    io::stdout().flush()?;
    for line in io::stdin().lock().lines() {
        let line = line?;
        if line == "quit" {
            break;
        }
        let Some((command, payload)) = line.split_once('\t') else {
            println!("{}", json_error("expected a tab after the command"));
            io::stdout().flush()?;
            continue;
        };
        let request = (|| -> Result<(u64, &str, Option<[u64; 3]>), String> {
            match command {
                "position" => match payload.split_once('\t') {
                    Some((budget, history)) => {
                        let ms = budget.parse::<u64>().map_err(|_| "invalid move-time-ms")?;
                        if ms == 0 { return Err("move-time-ms must be positive".into()); }
                        Ok((ms, history, None))
                    }
                    None => Ok((move_time_ms, payload, None)),
                },
                "clock" => {
                    let fields: Vec<&str> = payload.splitn(4, '\t').collect();
                    if fields.len() != 4 { return Err("clock needs both remaining clocks, increment, and history".into()); }
                    let white = fields[0].parse::<u64>().map_err(|_| "invalid white clock")?;
                    let black = fields[1].parse::<u64>().map_err(|_| "invalid black clock")?;
                    let increment = fields[2].parse::<u64>().map_err(|_| "invalid increment")?;
                    if white == 0 || black == 0 { return Err("remaining clocks must be positive".into()); }
                    Ok((0, fields[3], Some([white, black, increment])))
                }
                _ => Err("unknown command".into()),
            }
        })();
        let (requested_ms, history, clocks) = match request {
            Ok(request) => request,
            Err(message) => {
                println!("{}", json_error(&message));
                io::stdout().flush()?;
                continue;
            }
        };
        match replay(history) {
            Ok(state) => {
                let request_move_time_ms = match clocks {
                    Some(clock) => clock_budget(clock[state.side as usize], clock[2],
                        history.split_whitespace().count(), state.walls_left[0] + state.walls_left[1]),
                    None => requested_ms,
                };
                let started = Instant::now();
                let deadline = started + Duration::from_millis(request_move_time_ms);
                let result = run_mcts_batched_until(state, &evaluator, deadline, cpuct, config);
                match index_to_move(&state, result.best_action) {
                    Some(chosen) if parse_move(&move_text(chosen), &state).is_some() => {
                        let search_ms = started.elapsed().as_secs_f64() * 1000.0;
                        let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
                        let mut visits = [0u32; quoridor::ACTION_COUNT];
                        for &(action, count) in &result.visit_counts {
                            visits[action] = count;
                        }
                        let visit_json = visits.iter().map(u32::to_string).collect::<Vec<_>>().join(",");
                        let clock = clocks.unwrap_or([0, 0, 0]);
                        let stop_reason = if result.root_is_proven() { "solved" }
                            else if started >= deadline || elapsed_ms >= request_move_time_ms as f64 { "deadline" }
                            else { "overflow_guard" };
                        println!(
                            "{{\"protocol\":\"deadline-and-engine-clock-v2\",\"mode\":\"{}\",\"bestmove\":\"{}\",\"move_time_ms\":{},\"sims\":{},\"root_visits\":{},\"search_ms\":{:.3},\"elapsed_ms\":{:.3},\"overshoot_ms\":{:.3},\"stop_reason\":\"{}\",\"white_ms\":{},\"black_ms\":{},\"increment_ms\":{},\"clock_policy\":\"remaining-horizon-plus-increment-v1\",\"root_value\":{:.9},\"root_value_perspective\":\"side-to-move\",\"policy_frame\":\"claustrophobia-canonical-209\",\"side_to_move\":{},\"best_action\":{},\"visit_counts\":[{}]}}",
                            if clocks.is_some() { "game-clock" } else { "movetime" },
                            move_text(chosen), request_move_time_ms, result.total_simulations,
                            result.completed_root_visits(), search_ms, elapsed_ms,
                            (elapsed_ms - request_move_time_ms as f64).max(0.0), stop_reason,
                            clock[0], clock[1], clock[2], result.root_value_side_to_move(),
                            state.side, result.best_action, visit_json
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
