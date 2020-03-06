from tf_semantic_segmentation.visualizations import masks as masks_utils
import random
import imageio
import numpy as np
import streamlit as st
import os
import tensorflow as tf
import pandas as pd
from tf_semantic_segmentation.datasets import get_dataset_by_name, datasets_by_name, DataType
from tf_semantic_segmentation.processing import ColorMode
from tf_semantic_segmentation.datasets.utils import convert2tfdataset
from tf_semantic_segmentation.processing.dataset import get_preprocess_fn
from tf_semantic_segmentation.serving import predict_on_batch
from tf_semantic_segmentation.metrics.iou_score import iou_score
from tf_semantic_segmentation.metrics.f_scores import f1_score
from tf_semantic_segmentation.metrics.recall import recall
from tf_semantic_segmentation.metrics.precision import precision

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

size = (256, 256)
color_mode = ColorMode.RGB
resize_method = 'resize'
num_classes = 2

# @st.cache


def get_ds():
    ds = get_dataset_by_name('tacobinary', '/hdd/datasets/taco/')
    return ds, ds.raw()[DataType.VAL]


@st.cache
def predict(idx):
    ds, data = get_ds()
    example = data[idx]
    image, mask = ds.parse_example(example)
    print(tf.shape(image))
    print(image.shape, mask.shape)
    image, mask = get_preprocess_fn(size, color_mode, resize_method, scale_mask=False, mode='np')(image, mask, num_classes)

    images = tf.expand_dims(image, axis=0).numpy()
    masks_onehot = tf.expand_dims(mask, axis=0).numpy()
    predictions = predict_on_batch(images, model_name='0', input_name='input_1')
    predictions = np.asarray(predictions)

    print(predictions.max(), predictions.min(), predictions.dtype)
    print(predictions.shape)

    print('onehot...')
    if predictions[0].shape[-1] == 1:
        predictions[predictions > 0.7] = 1.0
        predictions[predictions <= 0.7] = 0.0

        predictions = tf.cast(predictions, tf.int32)
        predictions = tf.squeeze(predictions, axis=-1)
        predictions = tf.one_hot(predictions, 2)

    prediction_onehot = np.asarray(predictions, dtype=np.int32)
    return images, masks_onehot.astype(np.float32), prediction_onehot.astype(np.float32)


idx = st.sidebar.number_input('batch', min_value=0, max_value=100, value=0)


if st.sidebar.button('random'):
    idx = random.randint(0, 100)

images, masks_onehot, prediction_onehot = predict(idx)

with tf.device("cpu:0"):
    df = {
        "iou": iou_score()(masks_onehot, prediction_onehot).numpy(),
        "precition": precision()(masks_onehot, prediction_onehot).numpy(),
        "recall": recall()(masks_onehot, prediction_onehot).numpy(),
        "f1_score": f1_score()(masks_onehot, prediction_onehot).numpy()
    }

    for name, value in df.items():
        st.sidebar.markdown("- %s: %.2f" % (name, value))

masks = [np.argmax(mask, axis=-1).astype(np.float32) for mask in masks_onehot]

image = np.concatenate(images, axis=1) if len(images) > 1 else images[0]
mask = np.concatenate(masks, axis=1) if len(masks) > 1 else masks[0]

prediction_masks = [np.argmax(p, axis=-1).astype(np.float32) for p in prediction_onehot]
prediction = np.concatenate(prediction_masks, axis=1) if len(prediction_masks) > 1 else prediction_masks[0]

predictions_on_image = [masks_utils.overlay_classes((images[k] * 255.).astype(np.uint8), np.argmax(p, axis=-1), masks_utils.get_colors(num_classes), num_classes)
                        for k, p in enumerate(prediction_onehot)]

prediction_on_image = np.concatenate(predictions_on_image, axis=1)


st.image((image * 255.).astype(np.uint8), 'inputs', use_column_width=False)
print(mask.shape)
st.image(mask, 'masks', use_column_width=False)
st.image(prediction, 'predictions', use_column_width=False)
st.image(prediction_on_image, 'prediction_on_image', use_column_width=False)
