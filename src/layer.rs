use crate::conv2d::Conv2D;
use crate::dense::Dense;
use crate::maxpool2d::MaxPool2D;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
pub enum LayerType {
    Conv2D(Conv2D),
    MaxPool2D(MaxPool2D),
    Dense(Dense),
}
