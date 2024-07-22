use crate::cnn::conv2d::Conv2D;
use crate::cnn::dense::Dense;
use crate::cnn::maxpool2d::MaxPool2D;

pub enum LayerType {
    Conv2D(Conv2D),
    MaxPool2D(MaxPool2D),
    Dense(Dense),
}
