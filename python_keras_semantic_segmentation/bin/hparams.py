from .train import train_test_model, get_args
from tensorboard.plugins.hparams import api as hp
import tensorflow as tf
from collections import namedtuple


if __name__ == "__main__":
    # HP_DROPOUT = hp.HParam('loss', hp.RealInterval(0.1, 0.2))
    HP_LOSS = hp.HParam('loss', hp.Discrete(['softmax_crossentropy', 'dice', 'label_smoothing']))
    HP_OPTIMIZER = hp.HParam('optimizer', hp.Discrete(['adam', 'radam', 'ranger']))
    HP_LEARNING_RATE = hp.HParam('learning_rate', hp.Discrete([1e-3, 5e-3, 1e-4]))

    """
    METRIC_ACCURACY = 'accuracy'
    with tf.summary.create_file_writer('logs/hparam_tuning').as_default():
        hp.hparams_config(
            hparams=[HP_LOSS, HP_OPTIMIZER],
            metrics=[hp.Metric(METRIC_ACCURACY, display_name='Accuracy')],
        )
    """

    session_num = 0

    for loss in HP_LOSS.domain.values:
        for lr in HP_LEARNING_RATE.domain.values:
            for optimizer in HP_OPTIMIZER.domain.values:
                hparams = {
                    HP_LEARNING_RATE: lr,
                    HP_LOSS: loss,
                    HP_OPTIMIZER: optimizer,
                }
                run_name = "run-%d" % session_num
                logdir = 'logs/hparam_tuning/' + run_name
                print('--- Starting trial: %s' % run_name)
                print({h.name: hparams[h] for h in hparams})
                args = vars(get_args())
                args['batch_size'] = 8
                args['size'] = [256, 256]
                args['model'] = 'erfnet'
                args['color_mode'] = 0
                args['batch_size'] = 8
                args['epochs'] = 100
                args['logdir'] = logdir
                args['learning_rate'] = lr
                args['optimizer'] = optimizer
                args['losses'] = [loss]

                args['record_dir'] = "/hdd/datasets/taco/records/tacobinary/"
                args = namedtuple('Args', args.keys())(*args.values())

                train_test_model(args, hparams=hparams)
                session_num += 1
