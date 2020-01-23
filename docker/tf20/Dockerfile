FROM tensorflow/tensorflow:2.0.0-gpu-py3

ADD requirements.txt requirements.txt

RUN apt-get update && apt-get install -y libsm6 libxext6 libxrender-dev libyaml-dev libpython3-dev
RUN pip install -r requirements.txt

# install tf addons
RUN pip install tensorflow-addons==0.6.0
RUN apt-get install -y fish tmux curl htop screen

WORKDIR /root

# expose tensorboard port
EXPOSE 6006

# ADD run.sh run.sh
COPY tf_semantic_segmentation/ tf_semantic_segmentation

# hack for rtx cards to work with tf 2.0, otherwise pooling operation will fail
# see: https://github.com/AlexEMG/DeepLabCut/issues/1
ENV TF_FORCE_GPU_ALLOW_GROWTH 'true'
# ARG record_tag=""
# RUN test -z "$record_tag" || python -m tf_semantic_segmentation.bin.tfrecord_download -t ${record_tag} -r /hdd/datasets/downloaded/${record_tag}
CMD fish