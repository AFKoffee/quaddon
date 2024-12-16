use matrix::{Matrix, Numeric};

pub mod linear;
pub mod matrix;
pub mod quantization;
pub mod rmsnorm;

pub trait Layer<T: Numeric> {
    fn forward(&self, x: &Matrix<T>) -> Matrix<T>;
}