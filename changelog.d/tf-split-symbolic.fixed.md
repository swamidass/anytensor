Fix TensorFlow ``split`` under ``tf.function`` when the split axis is a symbolic ``None`` dim (empty or index cuts slice instead of ``int(x.shape[axis])``).
