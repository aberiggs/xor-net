use ndarray::{Array1, Array2, array};
use rand::RngExt;

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn main() {
    let mut rng = rand::rng();

    // Input
    let x: Array1<f32> = array![1.0, 0.0];

    // W1: (2, 2)
    let w1 = Array2::from_shape_fn((2, 2), |_| rng.random_range(-1.0..1.0));

    // b1: (2,)
    let b1 = Array1::from_shape_fn(2, |_| rng.random_range(-1.0..1.0));

    // W2: (2, 1)
    let w2 = Array2::from_shape_fn((2, 1), |_| rng.random_range(-1.0..1.0));

    // b2: (1,)
    let b2 = Array1::from_shape_fn(1, |_| rng.random_range(-1.0..1.0));

    // Do foward pass
    let z1 = x.dot(&w1) + &b1;
    let h = z1.mapv(sigmoid);

    let z2 = h.dot(&w2) + &b2;
    let y_hat = z2.mapv(sigmoid);

    println!("Input: {:?}", x);
    println!("Hidden: {:?}", h);
    println!("Output: {:?}", y_hat)
}
