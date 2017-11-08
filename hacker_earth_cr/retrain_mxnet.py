import mxnet as mx
import cnn
import os


def save_model():
    if not os.path.exists("checkpoint_retrain"):
        os.mkdir("checkpoint_retrain")
    return mx.callback.do_checkpoint("checkpoint_retrain/checkpoint")


def retrain():
    import logging
    logging.getLogger().setLevel(logging.DEBUG)
    sym, arg_params, aux_params = mx.model.load_checkpoint("checkpoint/checkpoint", 25)
    mod = mx.mod.Module(symbol=sym, context=mx.cpu())
    batch_size = 10
    num_embed = 128
    train_iter, valid_iter, sentence_size, embed_size, vocab_size = cnn.data_iter(batch_size, num_embed)
    mod.fit(train_data=train_iter,
               eval_data=valid_iter,
               eval_metric='acc',
               optimizer='sgd',
               optimizer_params={'learning_rate': 0.01},
               initializer=mx.initializer.Uniform(0.1),
               batch_end_callback=mx.callback.Speedometer(batch_size, 20),
               num_epoch=50,
               epoch_end_callback=save_model(),
               begin_epoch=26,
            allow_missing=True,
            arg_params=arg_params,
            aux_params=aux_params)


if __name__ == '__main__':
    retrain()