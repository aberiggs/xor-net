use rand::RngExt;

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

// TODO: Consider making x param actually sig_x (a1, a2, or a3) to avoid re-computation
fn deriv_sigmoid(x: f32) -> f32 {
    let fx = sigmoid(x);
    fx * (1.0 - fx)
}

fn mse_loss(y_hat: f32, y: f32) -> f32 {
    (y_hat - y).powi(2)
}

fn deriv_mse_loss(y_hat: f32, y: f32) -> f32 {
    2.0 * (y_hat - y)
}

/// A simple XOR network that represents the following network:
///
/// Input (size 2) -> Hidden (size 2) -> Output (size 1)
///
struct XORNetwork {
    w1: f32,
    w2: f32,
    w3: f32,
    w4: f32,
    w5: f32,
    w6: f32,
    b1: f32,
    b2: f32,
    b3: f32,
}

// Data involved in the forward pass of the XOR network
struct XORNetworkForwardData {
    x1: f32,
    x2: f32,
    z1: f32,
    z2: f32,
    z3: f32,
    a1: f32,
    a2: f32,
    a3: f32,
}

impl XORNetwork {
    fn new() -> XORNetwork {
        let mut rng = rand::rng();
        XORNetwork {
            w1: rng.random_range(-1.0..1.0),
            w2: rng.random_range(-1.0..1.0),
            w3: rng.random_range(-1.0..1.0),
            w4: rng.random_range(-1.0..1.0),
            w5: rng.random_range(-1.0..1.0),
            w6: rng.random_range(-1.0..1.0),
            b1: rng.random_range(-1.0..1.0),
            b2: rng.random_range(-1.0..1.0),
            b3: rng.random_range(-1.0..1.0),
        }
    }

    fn forward(&self, x1: f32, x2: f32) -> XORNetworkForwardData {
        let z1 = x1 * self.w1 + x2 * self.w2 + self.b1;
        let z2 = x1 * self.w3 + x2 * self.w4 + self.b2;
        let a1 = sigmoid(z1);
        let a2 = sigmoid(z2);
        let z3 = a1 * self.w5 + a2 * self.w6 + self.b3;
        let a3 = sigmoid(z3);

        XORNetworkForwardData {
            x1,
            x2,
            z1,
            z2,
            z3,
            a1,
            a2,
            a3,
        }
    }

    /// Backpropagation step for the XOR network. Computes gradients and updates weights
    fn backward(&mut self, forward_data: XORNetworkForwardData, y: f32, learning_rate: f32) {
        // dL/da3
        let dl_da3 = deriv_mse_loss(forward_data.a3, y);

        // da3/dz3 - a3 is activation with sigmoid
        let da3_dz3 = deriv_sigmoid(forward_data.z3);

        // dz3/dw5 - remember z3 = a1 * w5 + a2 * w6 + b3
        let dz3_dw5 = forward_data.a1;

        // dL/dw5 : w5 -> z3 -> a3 -> L
        let dl_dw5 = dl_da3 * da3_dz3 * dz3_dw5;

        // dz3/dw6 - remember z3 = a1 * w5 + a2 * w6 + b3
        let dz3_dw6 = forward_data.a2;

        // dL/dw6 : w6 -> z3 -> a3 -> L
        let dl_dw6 = dl_da3 * da3_dz3 * dz3_dw6;

        // dz3/db3 - remember z3 = a1 * w5 + a2 * w6 + b3
        let dz3_db3 = 1.0;

        // dL/db3 : b3 -> z3 -> a3 -> L
        let dl_db3 = dl_da3 * da3_dz3 * dz3_db3;

        // dz3_da1 - remember z3 = a1 * w5 + a2 * w6 + b3
        let dz3_da1 = self.w5;

        // da1_dz1 - a1 is activation with sigmoid
        let da1_dz1 = deriv_sigmoid(forward_data.z1);

        // dz1_dw1 - remember z1 = x1 * w1 + x2 * w2 + b1
        let dz1_dw1 = forward_data.x1;

        // dL/dw1 : w1 -> z1 -> a1 -> z3 -> a3 -> L
        let dl_dw1 = dl_da3 * da3_dz3 * dz3_da1 * da1_dz1 * dz1_dw1;

        // dz1_dw2 - remember z1 = x1 * w1 + x2 * w2 + b1
        let dz1_dw2 = forward_data.x2;

        // dL/dw2 : w2 -> z1 -> a1 -> z3 -> a3 -> L
        let dl_dw2 = dl_da3 * da3_dz3 * dz3_da1 * da1_dz1 * dz1_dw2;

        // dz1_db1 - remember z1 = x1 * w1 + x2 * w2 + b1
        let dz1_db1 = 1.0;

        // dL/db1 : b1 -> z1 -> a1 -> z3 -> a3 -> L
        let dl_db1 = dl_da3 * da3_dz3 * dz3_da1 * da1_dz1 * dz1_db1;

        // dz3_da2 - remember z3 = a1 * w5 + a2 * w6 + b3
        let dz3_da2 = self.w6;

        // da2_dz2 - a2 is activation with sigmoid
        let da2_dz2 = deriv_sigmoid(forward_data.z2);

        // dz2_dw3 - remember z2 = x1 * w3 + x2 * w4 + b2
        let dz2_dw3 = forward_data.x1;

        // dL/dw3 : w3 -> z2 -> a2 -> z3 -> a3 -> L
        let dl_dw3 = dl_da3 * da3_dz3 * dz3_da2 * da2_dz2 * dz2_dw3;

        // dz2_dw4 - remember z2 = x1 * w3 + x2 * w4 + b2
        let dz2_dw4 = forward_data.x2;

        // dL/dw4 : w4 -> z2 -> a2 -> z3 -> a3 -> L
        let dl_dw4 = dl_da3 * da3_dz3 * dz3_da2 * da2_dz2 * dz2_dw4;

        // dz2_db2 - remember z2 = x1 * w3 + x2 * w4 + b2
        let dz2_db2 = 1.0;

        // dL/db2 : b2 -> z2 -> a2 -> z3 -> a3 -> L
        let dl_db2 = dl_da3 * da3_dz3 * dz3_da2 * da2_dz2 * dz2_db2;

        // Finally, we can update the weights.
        self.w1 -= learning_rate * dl_dw1;
        self.w2 -= learning_rate * dl_dw2;
        self.w3 -= learning_rate * dl_dw3;
        self.w4 -= learning_rate * dl_dw4;
        self.w5 -= learning_rate * dl_dw5;
        self.w6 -= learning_rate * dl_dw6;
        self.b1 -= learning_rate * dl_db1;
        self.b2 -= learning_rate * dl_db2;
        self.b3 -= learning_rate * dl_db3;
    }

    fn predict(&self, x1: bool, x2: bool) -> bool {
        let forward_data = self.forward(x1 as i8 as f32, x2 as i8 as f32);

        forward_data.a3 > 0.5
    }

    fn train(&mut self, epochs: usize, learning_rate: f32) {
        let data_set: Vec<(f32, f32, f32)> = vec![
            (0.0, 0.0, 0.0),
            (0.0, 1.0, 1.0),
            (1.0, 0.0, 1.0),
            (1.0, 1.0, 0.0),
        ];

        for i in 0..epochs {
            let mut loss = 0.0;
            for &(a, b, y) in &data_set {
                let forward_data = self.forward(a, b);
                loss += mse_loss(forward_data.a3, y as f32);
                self.backward(forward_data, y, learning_rate);
            }

            if i % 100_000 == 0 {
                println!(
                    "[Epoch {}] Average loss: {}",
                    i + 1,
                    loss / data_set.len() as f32
                );
            }
        }
    }
}

fn main() {
    let mut net = XORNetwork::new();
    net.train(1_000_000, 0.001);

    println!("\nTraining complete! Testing...");
    println!("0 ^ 0 = {}", net.predict(false, false));
    println!("0 ^ 1 = {}", net.predict(false, true));
    println!("1 ^ 0 = {}", net.predict(true, false));
    println!("1 ^ 1 = {}", net.predict(true, true));
}
