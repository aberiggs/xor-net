# xor-net

This is a mini-project where I implemented a neural network for XOR by hand in Rust.

It uses a network made up of 2 input neurons, 1 hidden layer with 2 neurons, and 1 output neuron, all with sigmoid activations and manual backpropagation:

```mermaid
graph LR
    b1[(b1)] --> h1
    b3[(b3)] --> y
    x1((x1)) -->|w1| h1((h1))
    x2((x2)) -->|w2| h1
    x1 -->|w3| h2((h2))
    x2 -->|w4| h2
    h1 -->|w5| y((y_hat))
    h2 -->|w6| y
    b2[(b2)] --> h2

```

- forward pass computes `z` and `a` values
- backward pass applies chain rule for all weights/biases
- gradient descent updates parameters each step

The goal here is understanding the mechanics of neural networks, not performance.

## Run

```bash
cargo run
```

The program trains on the 4 XOR samples and prints predictions after training.

## Notes

- Training can be sensitive to random initialization.
- Some runs converge to the correct XOR mapping, others can get stuck.
- If it fails to converge, try running it again!
