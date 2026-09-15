//! Upstream MCTS evaluator with a persistent Python TorchScript transport.
use std::cell::RefCell;
use std::io::{Read, Write, BufReader, BufWriter};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use quoridor::{encode20, legal_mask_into, GameState, ACTION_COUNT};
use quoridor::mcts::BatchEvaluator;

struct Transport {
    child: Child,
    input: BufWriter<ChildStdin>,
    output: BufReader<ChildStdout>,
}

pub struct TchEvaluator { transport: RefCell<Transport> }

impl TchEvaluator {
    pub fn new(checkpoint: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let worker = std::env::current_exe()?.parent().unwrap().join("zq_inference_worker.py");
        let python = std::env::var("ZQ_PYTHON").unwrap_or_else(|_| "python".into());
        let device = std::env::var("QUORIDOR_DEVICE").unwrap_or_else(|_| "cpu".into());
        let mut command = Command::new(python);
        command.arg(worker).arg(checkpoint).arg(device)
            .stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::inherit());
        #[cfg(windows)] {
            use std::os::windows::process::CommandExt;
            command.creation_flags(0x08000000);
        }
        let mut child = command.spawn()?;
        let input = BufWriter::new(child.stdin.take().unwrap());
        let mut output = BufReader::new(child.stdout.take().unwrap());
        let mut ready = [0u8; 4];
        if let Err(error) = output.read_exact(&mut ready) {
            let _ = child.kill(); let _ = child.wait(); return Err(error.into());
        }
        if &ready != b"ZQIP" {
            let _ = child.kill(); let _ = child.wait(); return Err("invalid inference handshake".into());
        }
        Ok(Self { transport: RefCell::new(Transport { child, input, output }) })
    }
}

impl BatchEvaluator for TchEvaluator {
    fn evaluate_batch(&self, states: &[GameState]) -> Vec<(Vec<(usize, f32)>, f32)> {
        if states.is_empty() { return Vec::new(); }
        let mut transport = self.transport.borrow_mut();
        let n = states.len();
        transport.input.write_all(&(n as u32).to_le_bytes()).expect("inference batch header");
        let mut masks = Vec::with_capacity(n);
        for state in states {
            for value in encode20(state) {
                transport.input.write_all(&value.to_le_bytes()).expect("inference tensor");
            }
            let mut legal = [false; ACTION_COUNT];
            legal_mask_into(state, &mut legal);
            masks.push(legal);
        }
        for mask in &masks {
            for &legal in mask {
                transport.input.write_all(&[legal as u8]).expect("inference mask");
            }
        }
        transport.input.flush().expect("inference flush");
        let mut bytes = vec![0u8; (n * ACTION_COUNT + n) * 4];
        transport.output.read_exact(&mut bytes).expect("inference output");
        let values: Vec<f32> = bytes.chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0],b[1],b[2],b[3]])).collect();
        (0..n).map(|i| {
            let priors = (0..ACTION_COUNT).filter(|&j| masks[i][j])
                .map(|j| (j, values[i * ACTION_COUNT + j])).collect();
            (priors, values[n * ACTION_COUNT + i])
        }).collect()
    }
}

impl Drop for Transport {
    fn drop(&mut self) { let _ = self.child.kill(); let _ = self.child.wait(); }
}
