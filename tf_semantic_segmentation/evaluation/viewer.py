import random
import imageio
import numpy as np
import streamlit as st
import os
import tensorflow as tf
import pandas as pd
from tf_semantic_segmentation.datasets import get_dataset_by_name, datasets_by_name, DataType
from tf_semantic_segmentation.processing import loader, ColorMode
from tf_semantic_segmentation.serving import predict_on_batch
from tf_semantic_segmentation.losses.all import iou_score, f_score, precision, recall
from tf_semantic_segmentation.visualizations import masks as masks_utils

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


@st.cache
def get_loader():
    ds = get_dataset_by_name('tacobinary', '/hdd/datasets/taco/')
    return loader.DataLoader(ds, DataType.VAL, (256, 256), color_mode=ColorMode.RGB, batch_size=3)


@st.cache
def predict(idx):
    l = get_loader()
    images, masks_onehot = l[idx]
    predictions = predict_on_batch(images)
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
    print(prediction_onehot.shape)
    return images, masks_onehot.astype(np.float32), prediction_onehot.astype(np.float32)


idx = st.sidebar.number_input('batch', min_value=0, max_value=len(get_loader()) - 1, value=0)


if st.sidebar.button('random'):
    idx = random.randint(0, len(get_loader()) - 1)

images, masks_onehot, prediction_onehot = predict(idx)
masks = [np.argmax(mask, axis=-1).astype(np.float32) for mask in masks_onehot]

image = np.concatenate(images, axis=1)
mask = np.concatenate(masks, axis=1)

prediction_masks = [np.argmax(p, axis=-1).astype(np.float32) for p in prediction_onehot]
prediction = np.concatenate(prediction_masks, axis=1)
N = 2
predictions_on_image = [masks_utils.overlay_classes((images[k] * 255.).astype(np.uint8), np.argmax(p, axis=-1), masks_utils.get_colors(N), N)
                        for k, p in enumerate(prediction_onehot)]
prediction_on_image = np.concatenate(predictions_on_image, axis=1)


st.image(image, 'inputs', use_column_width=True)
st.image(mask, 'masks', use_column_width=True)
st.image(prediction, 'predictions', use_column_width=True)
st.image(prediction_on_image, 'prediction_on_image', use_column_width=True)

with tf.device("cpu:0"):
    df = {
        "iou": iou_score(masks_onehot, prediction_onehot).numpy(),
        "precition": precision(masks_onehot, prediction_onehot).numpy(),
        "recall": recall(masks_onehot, prediction_onehot).numpy(),
        "f1_score": f_score(masks_onehot, prediction_onehot).numpy()
    }

    for name, value in df.items():
        st.sidebar.markdown("- %s: %.2f" % (name, value))
