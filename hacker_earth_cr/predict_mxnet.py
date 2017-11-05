import pandas as pd
import mxnet as mx
import cnn
import numpy as np
import pickle


def get_sentences():
    with open('cleaned_data_test', 'r') as f:
        sentences = pickle.load(f)
    return sentences


if __name__ == '__main__':
    data = pd.read_csv("/home/mohit/ml_data/test.csv")
    sentences = get_sentences()
    print "done"
    ids = data['User_ID']
    sym, arg_params, aux_params = mx.model.load_checkpoint("checkpoint/checkpoint", 10)
    mod = mx.mod.Module(symbol=sym, context=mx.cpu(), label_names=None)
    mod.bind(for_training=False, data_shapes=[('data', (10, 1191))],label_shapes=mod._label_shapes)
    mod.set_params(arg_params, aux_params, allow_missing=True)
    vocab = cnn.get_vocab()
    sentences = cnn.pad_sentences(sentences,[])
    x = np.array([[vocab[word] for word in sentence] for sentence in sentences])
    print "done extracting vocab"
    mod.predict(x[0])
    #out = pd.DataFrame({'User_ID': ids, 'Is_Response': preds})
    #out.to_csv('out.csv', index=False)