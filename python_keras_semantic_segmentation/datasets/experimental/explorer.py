import random
import imageio
import numpy as np
import streamlit as st
import os
from python_keras_semantic_segmentation.datasets import get_dataset_by_name, datasets_by_name

# TODO: write streamlit application...

name = st.sidebar.radio("Dataset", list(datasets_by_name.keys()))
if name:
    ds = get_dataset_by_name(name, os.path.join('/hdd/datasets', name.lower()))
    labels = ['all']
    labels.extend(ds.labels)
    label = st.sidebar.selectbox('Label', labels)
    gen = ds.get()()
    imageLocation = st.empty()

    image, target = next(gen)
    # st.image(image, caption='image', format='PNG')
    imageLocation.image(target, format='PNG')
