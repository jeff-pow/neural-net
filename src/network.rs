use ndarray::{Array, Array1, Array2, Axis};
use rand::{
    distributions::{Distribution, Uniform},
    rngs::StdRng,
    seq::SliceRandom,
    SeedableRng,
};

pub type Data = (Array1<f32>, Array1<f32>);

#[derive(Clone, Debug, PartialEq)]
pub struct Network {
    pub weights: Vec<Array2<f32>>,
    pub biases: Vec<Array1<f32>>,
    pub sizes: Vec<usize>,
}

impl Network {
    pub fn new(sizes: Vec<usize>) -> Self {
        let mut rng: StdRng = SeedableRng::seed_from_u64(12037);
        let between = Uniform::from(-1.0..1.0);
        let num_layers = sizes.len();

        let mut biases = Vec::new();
        let mut weights = Vec::new();
        for l in 1..num_layers {
            let weight_arr = (0..sizes[l] * sizes[l - 1])
                .map(|_| between.sample(&mut rng))
                .collect();
            let bias_arr = (0..sizes[l]).map(|_| between.sample(&mut rng)).collect();

            let weight_matrix =
                Array::from_shape_vec((sizes[l], sizes[l - 1]), weight_arr).unwrap();
            let bias_matrix = Array1::from_shape_vec(sizes[l], bias_arr).unwrap();

            weights.push(weight_matrix);
            biases.push(bias_matrix);
        }
        Self {
            weights,
            biases,
            sizes,
        }
    }

    pub fn train(
        &mut self,
        training_data: &mut [Data],
        epochs: usize,
        mini_batch_size: usize,
        lr: f32,
    ) {
        let mut rng: StdRng = SeedableRng::seed_from_u64(1);
        training_data.shuffle(&mut rng);
        for i in 0..epochs {
            for data in training_data.chunks(mini_batch_size) {
                self.update_mini_batch(data, lr)
            }
            println!(
                "Epoch: {}, Training Accuracy: {:.2}",
                i,
                self.evaluate(training_data)
            );
        }
    }

    fn update_mini_batch(&mut self, mini_batch: &[Data], lr: f32) {
        let mut biases_change: Vec<Array1<f32>> = Vec::new();
        for b in &self.biases {
            biases_change.push(Array1::zeros(b.len()));
        }
        let mut weights_change: Vec<Array2<f32>> = Vec::new();
        for b in &self.weights {
            weights_change.push(Array2::zeros(b.raw_dim()));
        }
        for (x, y) in mini_batch {
            let (delta_biases, delta_weights) = self.backward(x, y);
            for (db, dw) in delta_biases.iter().zip(&delta_weights) {
                if self.evaluate(&[(x.clone(), y.clone())]) != 1.0 {
                    // dbg!(&self.feed_forward(x));
                    // dbg!(&y);
                    assert_ne!(db, Array1::zeros(db.len()));
                    assert_ne!(dw, Array2::zeros((dw.nrows(), dw.ncols())));
                }
            }
            biases_change = biases_change
                .iter()
                .zip(delta_biases.iter())
                .map(|(nb, dnb)| nb + dnb)
                .collect();
            weights_change = weights_change
                .iter()
                .zip(delta_weights.iter())
                .map(|(nw, dnw)| nw + dnw)
                .collect();
        }

        self.weights
            .iter_mut()
            .flatten()
            .zip(weights_change.iter().flatten())
            .for_each(|(w, nw)| *w -= lr / mini_batch.len() as f32 * nw);
        let b = self.biases.clone();
        self.biases
            .iter_mut()
            .flatten()
            .zip(biases_change.iter().flatten())
            .for_each(|(b, nb)| *b -= lr / mini_batch.len() as f32 * nb);
        if b == self.biases {
            (biases_change
                .iter()
                .flatten()
                .for_each(|&x| assert!(x == 0.)));
        }
    }

    pub fn backward(
        &self,
        input: &Array1<f32>,
        target_output: &Array1<f32>,
    ) -> (Vec<Array1<f32>>, Vec<Array2<f32>>) {
        let mut activation = input.clone();
        let mut activations = vec![activation.clone()];
        let mut zs = Vec::new();

        // Forward pass
        for (w, b) in self.weights.iter().zip(self.biases.iter()) {
            let z = w.dot(&activation) + b;
            zs.push(z.clone());
            // activation = sigmoid(&z);
            activation = z.mapv(activate);
            activations.push(activation.clone());
        }

        // Find the error of the output layer using the derivative of the cost and the derivative of the activation function on the unactivated input
        // to the final layer.
        // BP1
        let mut delta = cost_derivative(activations.last().unwrap(), target_output)
            * zs.last().unwrap().mapv(activate_prime);

        let mut delta_biases = Vec::new();
        let mut delta_weights = Vec::new();

        // BP3
        delta_biases.push(delta.clone());
        // BP4
        delta_weights.push(fat_cross(&delta, &activations[activations.len() - 2]));
        assert_eq!(
            delta_weights.last().unwrap().shape(),
            self.weights.last().unwrap().shape()
        );
        assert_eq!(
            delta_biases.last().unwrap().shape(),
            self.biases.last().unwrap().shape()
        );

        for l in 1..self.num_layers() - 1 {
            delta = self.weights[l].t().dot(&delta) * zs[l - 1].mapv(activate_prime);
            delta_weights.push(fat_cross(&delta, &activations[l - 1]));
            delta_biases.push(delta.clone());
        }
        // l = 1, l - 1 = 0
        // delta = (self.weights[1].t().dot(&delta)) * zs[0].mapv(activate_prime);
        // delta_weights.push(fat_cross(&delta, &activations[0]));
        // delta_biases.push(delta);
        assert_eq!(delta_biases.last().unwrap().shape(), self.biases[0].shape());
        assert_eq!(
            delta_weights.last().unwrap().shape(),
            self.weights[0].shape()
        );

        delta_biases.reverse();
        delta_weights.reverse();

        for (w, dw) in self.weights.iter().zip(delta_weights.iter()) {
            assert_eq!(w.shape(), dw.shape());
        }
        for (b, db) in self.biases.iter().zip(delta_biases.iter()) {
            assert_eq!(b.shape(), db.shape());
        }

        (delta_biases, delta_weights)
    }

    fn num_layers(&self) -> usize {
        self.sizes.len()
    }

    // Double vertical bars denoting the magnitude of the vector
    /// Cost = 1 / 2n * sum (||(prediction - actual)|| ** 2)
    pub fn cost(&self, test_data: &[Data]) -> f32 {
        0.5 * test_data
            .iter()
            .map(|(x, y)| {
                let prediction = self.feed_forward(x);
                (y - &prediction).mapv(|x| x.powi(2)).sum()
            })
            .sum::<f32>()
            / test_data.len() as f32
    }

    pub fn feed_forward(&self, input: &Array1<f32>) -> Array1<f32> {
        let mut a = input.to_owned();
        for (w, b) in self.weights.iter().zip(self.biases.iter()) {
            a = (w.dot(&a) + b).mapv(activate);
        }
        a
    }

    /// Returns accuracy of the network on mnist digit data as a percentage
    pub fn evaluate(&self, test_data: &[Data]) -> f32 {
        let mut count = 0;
        for (x, y) in test_data {
            let eval = self.feed_forward(x);
            count += u32::from(activated(&eval) == activated(y));
        }
        count as f32 / test_data.len() as f32 * 100.
    }
}

fn activated(vec: &Array1<f32>) -> usize {
    let mut max = vec[0];
    for &i in vec.iter() {
        max = max.max(i);
    }
    vec.iter().position(|&x| x == max).unwrap()
}

fn activate(x: f32) -> f32 {
    // x.max(0.0)
    1. / (1. + (-x).exp())
}

fn activate_prime(x: f32) -> f32 {
    // f32::from(x > 0.0)
    activate(x) * (1. - activate(x))
}

fn cost_derivative(prediction: &Array1<f32>, target: &Array1<f32>) -> Array1<f32> {
    prediction - target
}

pub fn fat_cross(vec1: &Array1<f32>, vec2: &Array1<f32>) -> Array2<f32> {
    vec1.clone()
        .insert_axis(Axis(1))
        .dot(&vec2.clone().insert_axis(Axis(0)))
}
