//! Temporary companion binary compiled inside a pinned Claustrophobia checkout.
//!
//! Input TSV: `<sample-id>\t<space separated move history>`.
//! Output binary: raw little-endian f32 tensors, one 20x9x9 tensor per input row.
//! The bridge deliberately uses Claustrophobia's own GameState, move generator,
//! and encode20() so ZQuoridor never reimplements the teacher's feature planes.

use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};

use quoridor::{encode20, generate_all_moves, GameState, Move, MoveBuf, TENSOR_LEN20};

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
    if args.len() != 3 {
        eprintln!("usage: zq_encode_bridge <input.tsv> <output.f32>");
        std::process::exit(2);
    }
    let input = BufReader::new(File::open(&args[1])?);
    let mut output = BufWriter::new(File::create(&args[2])?);
    let mut count = 0usize;

    for (line_no, line) in input.lines().enumerate() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let (sample_id, history) = line
            .split_once('\t')
            .ok_or_else(|| format!("input line {} lacks a tab", line_no + 1))?;
        let state = replay(history, sample_id)?;
        let tensor = encode20(&state);
        debug_assert_eq!(tensor.len(), TENSOR_LEN20);
        for value in tensor {
            output.write_all(&value.to_le_bytes())?;
        }
        count += 1;
    }
    output.flush()?;
    eprintln!("encoded {count} positions with Claustrophobia encode20");
    Ok(())
}
