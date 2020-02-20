from tf_semantic_segmentation.models.apps import resnet50, inception, mobilenet
import tensorflow as tf


def test_resnet50():
    model = resnet50.ResNet50(input_shape=(256, 256, 3), include_top=False)
    print(model.input.shape, model.output.get_shape())
    assert(model.output.shape.as_list() == [None, 1, 1, 2048])


def test_inception():
    model = inception.InceptionResNetV2(input_shape=(256, 256, 3), include_top=False)
    print(model.input.shape, model.output.get_shape())
    assert(model.output.shape.as_list() == [None, 8, 8, 1536])
    assert(model.input.shape.as_list() == [None, 256, 256, 3])


def test_mobilenet():
    model = mobilenet.MobileNet(input_shape=(256, 256, 3), include_top=False)
    print(model.input.shape, model.output.get_shape())
    assert(model.output.shape.as_list() == [None, 8, 8, 1024])
    assert(model.input.shape.as_list() == [None, 256, 256, 3])
