use ndarray::Array1;

#[derive(Clone, Debug, PartialEq)]
pub enum Activation {
    Sigmoid,
    Relu,
    Identity,
    Softmax,
}

impl Activation {
    pub fn activate(&self, vec: &Array1<f32>) -> Array1<f32> {
        match self {
            Activation::Sigmoid => vec.mapv(Sigmoid::activate),
            Activation::Relu => vec.mapv(Relu::activate),
            Activation::Identity => vec.mapv(Identity::activate),
            Activation::Softmax => softmax(vec),
        }
    }

    pub fn activate_prime(&self, vec: &Array1<f32>) -> Array1<f32> {
        match self {
            Activation::Sigmoid => vec.mapv(Sigmoid::activate_prime),
            Activation::Relu => vec.mapv(Relu::activate_prime),
            Activation::Identity => vec.mapv(Identity::activate_prime),
            Activation::Softmax => {
                let mut ret = Array1::zeros(vec.len());
                let activated = softmax(vec);
                // dbg!(vec);
                // dbg!(&activated);
                assert!(!activated.iter().any(|x| x.is_nan()));
                for i in 0..ret.len() {
                    for j in 0..ret.len() {
                        if i == j {
                            ret[i] += activated[i] * (1.0 - activated[i]);
                        }
                        ret[i] += -activated[i] * activated[j];
                    }
                }
                assert!(!ret.iter().any(|x: &f32| x.is_nan()));
                ret
            }
        }
    }
}

fn softmax(vec: &Array1<f32>) -> Array1<f32> {
    let max = vec
        .iter()
        .max_by(|x, y| x.partial_cmp(y).expect("Numbers are comparable"))
        .unwrap();
    // Subtract highest element from each in order to restrict range of values and hopefully
    // prevent nan's in the activated vector
    let denominator = vec.iter().map(|x| (x - max).exp()).sum::<f32>();
    vec.mapv(|x| (x - max).exp() / denominator)
}

pub trait ActivationFunction {
    fn activate(x: f32) -> f32;
    fn activate_prime(x: f32) -> f32;
}

#[derive(Clone, Debug, PartialEq)]
pub struct Sigmoid;
impl ActivationFunction for Sigmoid {
    fn activate(x: f32) -> f32 {
        1. / (1. + (-x).exp())
    }

    fn activate_prime(x: f32) -> f32 {
        Self::activate(x) * (1. - Self::activate(x))
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Relu;
impl ActivationFunction for Relu {
    fn activate(x: f32) -> f32 {
        x.max(0.0)
    }

    fn activate_prime(x: f32) -> f32 {
        f32::from(x > 0.0)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Identity;
impl ActivationFunction for Identity {
    fn activate(x: f32) -> f32 {
        x
    }

    fn activate_prime(_: f32) -> f32 {
        1.0
    }
}
