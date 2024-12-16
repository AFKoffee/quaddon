use std::{
    fmt::Display,
    ops::{AddAssign, Index, IndexMut, Mul},
};

pub trait Numeric: Mul<Self, Output = Self> + AddAssign<Self> + Copy {
    const ZERO: Self;
    const ONE: Self;
}

impl Numeric for f32 {
    const ZERO: Self = 0.0;

    const ONE: Self = 1.0;
}

pub struct Storage<T: Numeric> {
    data: Vec<T>,
    rows: usize,
    cols: usize,
}

impl<T: Numeric> Storage<T> {
    pub fn new() -> Self {
        Self {
            data: Vec::new(),
            rows: 0,
            cols: 0,
        }
    }

    pub fn from_data(data: Vec<T>, rows: usize, cols: usize) -> Self {
        assert!(
            data.len() == rows * cols,
            "Error: data-dimension mismatch! Could not instanciate storage"
        );

        Self { data, rows, cols }
    }

    pub fn get_row(&self, idx: usize) -> &[T] {
        assert!(idx < self.rows, "Error: row index out of bounds!");
        let row_base = idx * self.cols;
        &self.data[row_base..row_base + self.cols]
    }

    pub fn get_row_mut(&mut self, idx: usize) -> &mut [T] {
        assert!(idx < self.rows, "Error: row index out of bounds!");
        let row_base = idx * self.cols;
        &mut self.data[row_base..row_base + self.cols]
    }
}

impl<T: Numeric> Default for Storage<T> {
    fn default() -> Self {
        Self::new()
    }
}

pub struct Matrix<T: Numeric> {
    storage: Storage<T>,
}

impl<T: Numeric> Matrix<T> {
    pub fn new() -> Self {
        Self {
            storage: Storage::new(),
        }
    }

    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            storage: Storage::from_data(vec![T::ZERO; rows * cols], rows, cols),
        }
    }

    pub fn ones(rows: usize, cols: usize) -> Self {
        Self {
            storage: Storage::from_data(vec![T::ONE; rows * cols], rows, cols),
        }
    }

    pub fn mul(&self, other: &Matrix<T>) -> Matrix<T> {
        assert_eq!(
            self.storage.cols, other.storage.rows,
            "Error: dimension mismatch! {} cols vs. {} rows",
            self.storage.cols, other.storage.rows
        );
        let mut output = Self::zeros(self.storage.rows, other.storage.cols);

        for i in 0..self.storage.rows {
            for j in 0..other.storage.cols {
                for k in 0..other.storage.rows {
                    /*
                    let first = self[i][k];
                    let second = other[k][j];
                    let output_elem = &mut output[i][j];
                    let calc = first * second;
                    *output_elem += calc;
                    */
                    output[i][j] += self[i][k] * other[k][j]
                }
            }
        }

        output
    }
}

impl<T: Numeric> Default for Matrix<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Numeric> Index<usize> for Matrix<T> {
    type Output = [T];

    fn index(&self, index: usize) -> &Self::Output {
        self.storage.get_row(index)
    }
}

impl<T: Numeric> IndexMut<usize> for Matrix<T> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        self.storage.get_row_mut(index)
    }
}

impl<T: Numeric + Display> Display for Matrix<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[")?;
        for i in 0..self.storage.rows {
            write!(
                f,
                "[{}]",
                self[i]
                    .iter()
                    .map(|elem| format!("{elem}"))
                    .fold("".to_string(), |acc, e| acc + (", ") + &e)
            )?;
            if i < self.storage.rows - 1 {
                writeln!(f)?;
            }
        }
        writeln!(f, "]")?;
        Ok(())
    }
}
