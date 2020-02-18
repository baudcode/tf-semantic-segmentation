from tf_semantic_segmentation.bin import train
from ..fixtures import tfrecord_reader
import pytest
import tempfile
import shutil


@pytest.mark.usefixtures('tfrecord_reader')
def test_simple_train(tfrecord_reader):
    args = train.get_args({})
    args.validation_steps = 1
    args.epochs = 1
    args.steps_per_epoch = 1
    args.batch_size = 1
    args.buffer_size = 1
    args.gpus = ""
    args.record_dir = tfrecord_reader.record_dir
    args.logdir = tempfile.mkdtemp()
    data = train.train_test_model(args)
    shutil.rmtree(args.logdir)
