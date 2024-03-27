use std::fmt::{Display, Formatter};

/// Represents a tensor with data and shape.
///
/// A tensor is a generalization of vectors and matrices to potentially higher
/// dimensions.
/// The `Tensor` struct in this library holds a reference to its data (a slice
/// of `f32` values) and its shape (a slice of `usize` values indicating the
/// size of each dimension).
#[derive(Debug, PartialEq)]
pub struct Tensor<'a> {
    pub data: &'a [f32],
    pub shape: &'a [usize],
}

/// Errors that can occur when working with `Tensor`.
///
/// This enum defines various error types that can result from operations on
/// `Tensor` instances, such as mismatches in dimensions or indices out of
/// bounds.
#[derive(Debug, PartialEq)]
pub enum TensorError<'a> {
    DimensionMismatch {
        expected: usize,
        found: usize,
    },
    IndexOutOfBounds {
        dimension: usize,
        index: usize,
        bound: usize,
    },
    ShapeDataMismatch {
        shape_elements: usize,
        data_elements: usize,
    },
    ShapeMismatch {
        a: &'a [usize],
        b: &'a [usize],
    },
}

impl<'a> Tensor<'a> {
    /// Creates a new `Tensor` instance from data and shape slices.
    ///
    /// Validates that the total number of elements indicated by the shape
    /// matches the number of elements in the data slice.
    ///
    /// # Arguments
    ///
    /// * `data` - A slice of `f32` representing the tensor's data.
    /// * `shape` - A slice of `usize` representing the tensor's shape.
    ///
    /// # Returns
    ///
    /// An `Ok(Tensor)` instance if the shape matches the data,
    /// or a `TensorError::ShapeDataMismatch` error if they do not match.
    pub fn new(data: &'a [f32], shape: &'a [usize]) -> Result<Tensor<'a>, TensorError<'a>> {
        let shape_elements = shape.iter().product::<usize>();
        if shape_elements != data.len() {
            return Err(TensorError::ShapeDataMismatch {
                shape_elements,
                data_elements: data.len(),
            });
        }
        Ok(Tensor { data, shape })
    }

    /// Calculates the flat index for a given multi-dimensional index.
    ///
    /// # Arguments
    ///
    /// * `index` - A slice of `usize` representing the multi-dimensional index.
    ///
    /// # Returns
    ///
    /// An `Ok(usize)` representing the flat index, or a `TensorError` if the
    /// index is out of bounds or if there is a dimension mismatch.
    pub fn flat_index(&self, index: &[usize]) -> Result<usize, TensorError<'a>> {
        if index.len() != self.shape.len() {
            return Err(TensorError::DimensionMismatch {
                expected: self.shape.len(),
                found: index.len(),
            });
        }

        let mut flat_index = 0;
        let mut dimension_stride = 1; // Tells us how many elements to skip for
                                      // moving to the next dimension

        // Iterate in reverse to start from the innermost dimension, which
        // simplify dimension stride calculation
        for (i, &dimension_index) in index.iter().rev().enumerate() {
            let shape_index = self.shape.len() - 1 - i;

            if dimension_index >= self.shape[shape_index] {
                return Err(TensorError::IndexOutOfBounds {
                    dimension: shape_index,
                    index: dimension_index,
                    bound: self.shape[shape_index] - 1,
                });
            }
            flat_index += dimension_index * dimension_stride;
            dimension_stride *= self.shape[shape_index];
        }

        Ok(flat_index)
    }

    /// Retrieves the tensor element at a given multi-dimensional index.
    ///
    /// # Arguments
    ///
    /// * `index` - A slice of `usize` representing the multi-dimensional index.
    ///
    /// # Returns
    ///
    /// An `Ok(f32)` containing the value at the given index, or a `TensorError`
    /// if the index is out of bounds or if there is a dimension mismatch.
    pub fn get(&self, index: &[usize]) -> Result<f32, TensorError<'a>> {
        let flat_index = self.flat_index(index)?;
        Ok(self.data[flat_index])
    }

    /// Computes the dot product of the current tensor with another tensor.
    ///
    /// The dot product is calculated as the sum of element-wise multiplications
    /// between the two tensors.
    /// This operation requires that both tensors have the exact same shape.
    /// If the shapes do not match, a `TensorError::ShapeMismatch` error is
    /// returned.
    ///
    /// # Arguments
    ///
    /// * `tensor` - A reference to the `Tensor` instance to be dot-multiplied
    /// with the current tensor.
    ///
    /// # Returns
    ///
    /// An `Ok(f32)` containing the dot product of the two tensors if their
    /// shapes match.
    /// Otherwise, returns a `TensorError::ShapeMismatch` error detailing the
    /// mismatched shapes.
    pub fn dot(&self, tensor: &Tensor<'a>) -> Result<f32, TensorError<'a>> {
        if self.shape != tensor.shape {
            return Err(TensorError::ShapeMismatch {
                a: self.shape,
                b: tensor.shape,
            });
        }

        Ok(self
            .data
            .iter()
            .zip(tensor.data.iter())
            .map(|(&x, &y)| x * y)
            .sum())
    }
}

impl<'a> Display for TensorError<'a> {
    fn fmt(&self, formatter: &mut Formatter) -> std::fmt::Result {
        match *self {
            TensorError::DimensionMismatch { expected, found } => {
                write!(
                    formatter,
                    "Dimension mismatch: expected {}, found {}",
                    expected, found
                )
            }
            TensorError::IndexOutOfBounds {
                dimension,
                index,
                bound,
            } => {
                write!(
                    formatter,
                    "Index out of bounds: dimension {}, index {}, exceeds bound {}",
                    dimension, index, bound
                )
            }
            TensorError::ShapeDataMismatch {
                shape_elements,
                data_elements,
            } => {
                write!(
                    formatter,
                    "Shape-data mismatch: expected shape to represent {} elements, but data contains {} elements.",
                    shape_elements,
                    data_elements
                )
            }
            TensorError::ShapeMismatch { a, b } => {
                write!(
                    formatter,
                    "Shape mismatch: first tensor has shape {:?} but second one has {:?}.",
                    a, b
                )
            }
        }
    }
}
