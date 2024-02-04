#[derive(Clone, Debug, PartialEq)]
pub enum Activation {
    Sigmoid,
    Relu,
    Identity,
}

impl Activation {
    pub fn activate(&self) -> impl Fn(f32) -> f32 {
        match self {
            Activation::Sigmoid => Sigmoid::activate,
            Activation::Relu => Relu::activate,
            Activation::Identity => Identity::activate,
        }
    }

    pub fn activate_prime(&self) -> impl Fn(f32) -> f32 {
        match self {
            Activation::Sigmoid => Sigmoid::activate_prime,
            Activation::Relu => Relu::activate_prime,
            Activation::Identity => Identity::activate_prime,
        }
    }
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
