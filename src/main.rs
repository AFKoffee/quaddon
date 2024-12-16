use quaddon::matrix::Matrix;

fn main() {
    let a: Matrix<f32> = Matrix::ones(3, 3);
    let b: Matrix<f32> = Matrix::ones(3, 3);

    let c = a.mul(&b);

    println!("{c}")
}
