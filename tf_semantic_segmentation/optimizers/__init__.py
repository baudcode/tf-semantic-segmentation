from tensorflow.keras import optimizers


def get_optimizer_by_name(name, lr) -> optimizers.Optimizer:
    if name == 'adam':
        return optimizers.Adam(learning_rate=lr)

    elif name == 'radam':
        import tensorflow_addons as tfa
        return tfa.optimizers.RectifiedAdam(learning_rate=lr)

    elif name == 'ranger':
        import tensorflow_addons as tfa
        radam = tfa.optimizers.RectifiedAdam(learning_rate=lr)
        ranger = tfa.optimizers.Lookahead(radam, sync_period=6, slow_step_size=0.5)
        return ranger

    elif name == 'radam_warmup':
        import tensorflow_addons as tfa
        return tfa.optimizers.RectifiedAdam(
            lr=1e-3,
            total_steps=10000,
            warmup_proportion=0.1,
            min_lr=lr,
        )

    elif name == 'madgrad':
        from madgrad import MadGrad
        return MadGrad(lr=lr)
    elif name == 'novograd':
        import tensorflow_addons as tfa
        return tfa.optimizers.NovoGrad(lr=lr)

    else:
        raise Exception("unknown optimizer %s" % name)


names = ['adam', 'radam', 'ranger']

__all__ = ['names', 'get_optimizer_by_name']
