extern crate quoridor_clock as clocked;
extern crate quoridor_orig as original;

use std::thread;
use std::time::{Duration, Instant};

fn uniform_original(states: &[original::GameState]) -> Vec<(Vec<(usize, f32)>, f32)> {
    states
        .iter()
        .map(|state| {
            let legal = original::legal_mask(state);
            let count = legal.iter().filter(|&&is_legal| is_legal).count() as f32;
            let priors = legal
                .iter()
                .enumerate()
                .filter_map(|(action, &is_legal)| {
                    is_legal.then_some((action, 1.0 / count))
                })
                .collect();
            (priors, 0.0)
        })
        .collect()
}

fn uniform_clocked(states: &[clocked::GameState]) -> Vec<(Vec<(usize, f32)>, f32)> {
    states
        .iter()
        .map(|state| {
            let legal = clocked::legal_mask(state);
            let count = legal.iter().filter(|&&is_legal| is_legal).count() as f32;
            let priors = legal
                .iter()
                .enumerate()
                .filter_map(|(action, &is_legal)| {
                    is_legal.then_some((action, 1.0 / count))
                })
                .collect();
            (priors, 0.0)
        })
        .collect()
}

fn fixed_count_equivalence() {
    for simulations in [1, 128, 512] {
        let original_state = original::GameState::start();
        let clocked_state = clocked::GameState::start();
        let original_evaluator = |states: &[original::GameState]| uniform_original(states);
        let clocked_evaluator = |states: &[clocked::GameState]| uniform_clocked(states);
        let original_result = original::mcts::run_mcts_batched(
            original_state,
            &original_evaluator,
            simulations,
            1.5,
            original::mcts::BatchedConfig::new(8),
        );
        let clocked_result = clocked::mcts::run_mcts_batched(
            clocked_state,
            &clocked_evaluator,
            simulations,
            1.5,
            clocked::mcts::BatchedConfig::new(8),
        );

        assert_eq!(original_result.best_action, clocked_result.best_action);
        assert_eq!(original_result.visit_counts, clocked_result.visit_counts);
        assert_eq!(original_result.total_simulations, clocked_result.total_simulations);
        assert_eq!(
            original_result.root_value_side_to_move().to_bits(),
            clocked_result.root_value_side_to_move().to_bits()
        );
        assert_eq!(
            clocked_result.completed_root_visits(),
            simulations,
            "fixed-count root visits must preserve the original API contract"
        );
    }
}

fn expired_deadline_returns_a_legal_prior_fallback() {
    let state = clocked::GameState::start();
    let evaluator = |states: &[clocked::GameState]| uniform_clocked(states);
    let deadline = Instant::now() - Duration::from_secs(1);
    let result = clocked::mcts::run_mcts_batched_until(
        state,
        &evaluator,
        deadline,
        1.5,
        clocked::mcts::BatchedConfig::new(8),
    );

    assert_eq!(result.total_simulations, 0);
    assert_eq!(result.completed_root_visits(), 0);
    assert!(!result.visit_counts.is_empty());
    assert!(clocked::legal_mask(&state)[result.best_action]);
}

fn positive_deadline_stops_after_a_complete_wave() {
    let state = clocked::GameState::start();
    let slow_evaluator = |states: &[clocked::GameState]| {
        thread::sleep(Duration::from_millis(12));
        uniform_clocked(states)
    };
    let budget = Duration::from_millis(80);
    let started = Instant::now();
    let result = clocked::mcts::run_mcts_batched_until(
        state,
        &slow_evaluator,
        started + budget,
        1.5,
        clocked::mcts::BatchedConfig::new(4),
    );
    let elapsed = started.elapsed();
    let root_child_visits: u32 = result.visit_counts.iter().map(|&(_, n)| n).sum();

    assert!(result.total_simulations > 0, "a positive budget should complete work");
    assert_eq!(root_child_visits, result.total_simulations);
    assert_eq!(result.completed_root_visits(), result.total_simulations);
    assert!(elapsed >= budget, "the search should stop at a completed wave");
    assert!(
        elapsed < Duration::from_secs(2),
        "deadline stop should bound evaluator overshoot to a wave"
    );
    assert!(result.total_simulations < 100_000, "the emergency cap must not bind");
}

fn main() {
    fixed_count_equivalence();
    expired_deadline_returns_a_legal_prior_fallback();
    positive_deadline_stops_after_a_complete_wave();
    println!("fixed_count=PASS expired_deadline=PASS timed_wave=PASS");
}
