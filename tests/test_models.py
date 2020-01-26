from tf_semantic_segmentation import models
import tensorflow as tf


def test_build_models():

    for name, model_fn in models.models_by_name.items():

        print("testing model %s with model fn %s" % (name, model_fn))
        # check that the model builds
        input_shape = (128, 128, 3)
        num_classes = 5
        model, base_model = models.get_model_by_name(name, {"input_shape": input_shape, "num_classes": num_classes})
        print(model.output.shape, model.input.shape, base_model.output.shape)
        assert(model.output.get_shape().as_list(), [None, input_shape[0], input_shape[1], num_classes])
