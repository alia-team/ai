pub struct Tensor<'a> {
    pub data: &'a [f32],
    pub dimension: &'a [u8],
}

impl<'a> Tensor<'a> {
    pub fn new(data: &'a [f32], dimension: &'a [u8]) -> Tensor<'a> {
        Tensor { data, dimension }
    }
}
