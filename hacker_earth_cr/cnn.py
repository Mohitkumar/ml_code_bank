from collections import Counter
import pickle
import itertools
import numpy as np
import mxnet as mx
import os

def save_model():
    if not os.path.exists("checkpoint"):
        os.mkdir("checkpoint")
    return mx.callback.do_checkpoint("checkpoint/checkpoint")

def load_document():
    with open('cleaned_data', 'r') as f1:
        document = pickle.load(f1)
    return document


def load_test_sentences():
    with open('cleaned_data_test', 'r') as f:
        sentences = pickle.load(f)
    return sentences


def pad_sentences(sentences, test_sentences):
    max_length = max(len(x) for x in sentences + test_sentences)
    padded_sentence = []
    for i in range(len(sentences)):
        sentence = sentences[i]
        num_padding = max_length - len(sentence)
        new_sentence = sentence + ["</s>"] * num_padding
        padded_sentence.append(new_sentence)
    return padded_sentence


def build_vocab(sentences):
    word_counter = Counter(itertools.chain(*sentences))
    vocab_inv = [x[0] for x in word_counter.most_common()]
    vocab = {x: i for i, x in enumerate(vocab_inv)}
    return [vocab, vocab_inv]


def build_input_data(sentences, labels, vocab):
    x = np.array([[vocab[word] for word in sentence] for sentence in sentences])
    labels = [0 if l == 'not happy' else 1 for l in labels]
    y = np.array(labels)
    return [x,y]


def get_vocab():
    with open('vocab','r') as f:
        vocab = pickle.load(f)
    return vocab


def load_data():
    doc = load_document();
    test_sentences = load_test_sentences()
    sentences = []
    labels = []
    for sent, label in doc:
        sentences.append(sent)
        labels.append(label)
    sentence_padded = pad_sentences(sentences, test_sentences)
    vocab, vocab_inv = build_vocab(sentence_padded + test_sentences)
    with open('vocab','wb') as f:
        pickle.dump(vocab, f, 2)

    x,y = build_input_data(sentence_padded, labels,vocab)
    return [x,y,vocab, vocab_inv]


def batch_iter(data, batch_size, num_epoch):
    data = np.array(data)
    data_size = len(data)
    num_batch_per_epoch = int(len(data)/batch_size) + 1
    for epoch in range(num_epoch):
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_data = data[shuffle_indices]
        for batch_num in range(num_batch_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num +1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]


def data_iter(batch_size, num_embed):
    x,y,vocab, vocab_inv = load_data()
    print type(x)
    print type(y)
    embed_size = num_embed
    sentence_size = x.shape[1]
    vocab_size = len(vocab)
    np.random.seed(10)
    shuffle_indices = np.random.permutation(len(y))
    x_shuffled = x[shuffle_indices]
    y_shuffled = y[shuffle_indices]
    x_train, x_val = x_shuffled[:-1000], x_shuffled[-1000:]
    y_train, y_val = y_shuffled[:-1000], y_shuffled[-1000:]
    print('Train/Valid split: %d/%d' % (len(y_train), len(y_val)))
    print('train shape:', x_train.shape, type(x_train))
    print('valid shape:', x_val.shape, type(x_val))
    print('sentence max words', sentence_size)
    print('embedding size', embed_size)
    print('vocab size', vocab_size)
    train = mx.io.NDArrayIter(data=x_train, label=y_train, batch_size=batch_size, shuffle=True)
    valid = mx.io.NDArrayIter(data=x_val, label=y_val, batch_size=batch_size)

    return (train, valid, sentence_size, embed_size, vocab_size)

def sym_gen(batch_size, sentence_size, num_embed,
            vocab_size, num_label=2, filter_list=[2,3,4,5], num_filter = 100, dropout=0.0):
    input_x = mx.sym.Variable('data')
    input_y = mx.sym.Variable('softmax_label')
    embed_layer = mx.sym.Embedding(data=input_x,input_dim=vocab_size,
                                   output_dim=num_embed, name='vocab_embed')
    conv_input = mx.sym.reshape(data=embed_layer,target_shape=(batch_size, 1,sentence_size, num_embed))
    pooled_output = []
    for i, filter_size in enumerate(filter_list):
        convi = mx.sym.Convolution(data=conv_input, kernel=(filter_size, num_embed),  num_filter=num_filter)
        relui = mx.sym.Activation(data=convi, act_type='relu')
        pooli = mx.sym.Pooling(data=relui, pool_type='max', kernel=(sentence_size - filter_size +1, 1), stride=(1,1))
        pooled_output.append(pooli)

    total_filters = num_filter * len(filter_list)
    concat = mx.sym.Concat(*pooled_output, dim=1)
    h_pool = mx.sym.Reshape(data=concat,target_shape=(batch_size, total_filters))
    if(dropout > 0.0):
        h_drop = mx.sym.Dropout(data=h_pool,p=dropout)
    else:
        h_drop = h_pool

    cls_weight = mx.sym.Variable('cls_weight')
    cls_bias = mx.sym.Variable('cls_bias')
    fc = mx.sym.FullyConnected(data=h_drop, weight=cls_weight, bias= cls_bias, num_hidden=num_label)
    sm =mx.sym.SoftmaxOutput(data=fc, label=input_y, name='softmax')
    return sm, ('data',), ('softmax_label',)


def train(symbol, train_iter, valid_iter, data_names, label_names, batch_size, num_epoch):
    import logging
    logging.getLogger().setLevel(logging.DEBUG)
    module = mx.mod.Module(symbol, data_names=data_names, label_names=label_names, context=mx.cpu())
    module.fit(train_data=train_iter,
               eval_data=valid_iter,
               eval_metric='acc',
               optimizer='sgd',
               optimizer_params={'learning_rate':0.01},
               initializer=mx.initializer.Uniform(0.1),
               batch_end_callback=mx.callback.Speedometer(batch_size,50),
               num_epoch=num_epoch,
               epoch_end_callback=save_model())


if __name__ == '__main__':
    batch_size = 100
    num_embed = 128
    num_filter = 128
    num_epoch = 25
    train_iter, valid_iter, sentence_size, embed_size, vocab_size = data_iter(batch_size,num_embed)
    symbol, data_names, label_names = sym_gen(batch_size,sentence_size,embed_size,vocab_size,num_filter=num_filter,dropout=0.1)
    train(symbol,train_iter,valid_iter,data_names, label_names, batch_size, num_epoch=num_epoch)