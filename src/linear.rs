use crate::{matrix::{Matrix, Numeric}, Layer};

pub struct Linear<T: Numeric> {
    weight: Matrix<T>,
    bias: Option<Vec<T>>,
}

impl<T: Numeric> Linear<T> {
    pub fn new(in_features: usize, out_features: usize, bias: bool) -> Self {
        Self {
            weight: Matrix::zeros(in_features, out_features),
            bias: if bias {
                Some(vec![T::ZERO; out_features])
            } else {
                None
            },
        }
    }
}

impl <T: Numeric> Layer<T> for Linear<T> {
    fn forward(&self, x: &Matrix<T>) -> Matrix<T> {
        let output = x.mul(&self.weight);
        if let Some(bias) = &self.bias {
            todo!("bias not yet implemented")
        } else {
            output
        }
    }
}