mod network;

use mnist::*;
use ndarray::prelude::*;
use network::Network;
use rand::{rngs::StdRng, seq::SliceRandom, SeedableRng};

use crate::network::Data;

const IMAGE_SIZE: usize = 28 * 28;

fn main() {
    let mut network = Network::new(vec![784, 20, 10]);
    let (mut train_data, test_data) = load_data();

    network.train(&mut train_data, 30, 10, 3.0);

    println!(
        "Accuracy in test data: {:.2}%",
        network.evaluate(&test_data)
    );
    println!("Mean squared error: {:.2}", network.cost(&test_data));
}

/// Returns two vecs of Data, which itself is a tuple of the form (input vec, expected output vec).
/// The training data is shuffled.
fn load_data() -> (Vec<Data>, Vec<Data>) {
    let Mnist {
        trn_img,
        trn_lbl,
        tst_img,
        tst_lbl,
        ..
    } = MnistBuilder::new()
        .download_and_extract()
        .label_format_digit()
        .training_set_length(50_000)
        .validation_set_length(10_000)
        .test_set_length(10_000)
        .finalize();

    let train_images = trn_img
        .chunks(28 * 28)
        .map(|x| {
            Array1::from_shape_vec(IMAGE_SIZE, x.to_vec())
                .unwrap()
                .map(|i| *i as f32 / 256.)
        })
        .collect::<Vec<_>>();

    let mut train_labels: Vec<Array1<f32>> = vec![Array1::zeros(10); 50_000];
    for (i, label) in trn_lbl.iter().enumerate() {
        train_labels[i][usize::from(*label)] = 1.0;
    }

    let mut train_data = train_images
        .into_iter()
        .zip(train_labels)
        .collect::<Vec<Data>>();

    let mut rng: StdRng = SeedableRng::seed_from_u64(0xA37293u64);
    train_data.shuffle(&mut rng);

    /*
    let test_images = Array2::from_shape_vec((10_000, 28 * 28), tst_img.clone())
        .unwrap()
        .map(|x| *x as f32 / 256.);
    let mut test_labels: Array2<f32> = Array2::zeros((10_000, 10));
    for (i, label) in tst_lbl.iter().enumerate() {
        test_labels.row_mut(i)[usize::from(*label)] = 1.0;
    }
    let test_data = test_images
        .rows()
        .into_iter()
        .zip(test_labels.rows())
        .map(|(x, y)| {
            (
                Array1::from_iter(x.iter().cloned()),
                Array1::from_iter(y.iter().cloned()),
            )
        })
        .collect::<Vec<Data>>();
    */

    let test_data = tst_img
        .chunks(IMAGE_SIZE)
        .zip(tst_lbl.clone())
        .map(|(x, y)| {
            (
                {
                    Array1::from_shape_vec(IMAGE_SIZE, x.to_vec())
                        .unwrap()
                        .map(|i| *i as f32 / 256.)
                },
                {
                    let mut a = Array1::zeros(10);
                    a[usize::from(y)] = 1.0;
                    a
                },
            )
        })
        .collect::<Vec<Data>>();
    (train_data, test_data)
}
