use crate::cnn::conv::ConvLayer;
use crate::cnn::dense::DenseLayer;
use crate::cnn::mxpl::MxplLayer;

pub enum Layer {
    Conv(ConvLayer),
    Mxpl(MxplLayer),
    Dense(DenseLayer),
}
