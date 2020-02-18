from tf_semantic_segmentation import activations
from tensorflow.keras.layers import Activation
import numpy as np

TEST_BATCH = np.ones((1, 32, 32, 3), dtype=np.float32)

for name, activation in activations.custom_objects.items():
    print("testing activation %s" % name)

    # use custom object name (keras Activation)
    act = Activation(name)
    a1 = act(TEST_BATCH)

    # use activation
    print(activation)
    print(activation.__dict__)
    a2 = activation(TEST_BATCH)
    np.testing.assert_allclose(a1, a2)
