from tf_semantic_segmentation import models
import tensorflow as tf
from tensorflow.keras import backend as K


def test_build_models():

    for name, model_fn in models.models_by_name.items():

        print("testing model %s with model fn %s" % (name, model_fn))
        # check that the model builds
        input_shape = (256, 256, 3)
        num_classes = 5
        model = models.get_model_by_name(name, {"input_shape": input_shape, "num_classes": num_classes})
        assert(model.output.get_shape().as_list(), [None, input_shape[0], input_shape[1], num_classes])
        K.clear_session()
