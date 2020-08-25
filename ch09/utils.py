import glob
import os
from random import shuffle
from nltk.tokenize import TreebankWordTokenizer
from gensim.models.keyedvectors import KeyedVectors


def pre_process_data(filepath):
    """
    This is dependent on your training data source but we will try to generalize it as best as possible.
    """
    positive_path = os.path.join(filepath, 'pos')
    negative_path = os.path.join(filepath, 'neg')
    
    pos_label = 1
    neg_label = 0
    
    dataset = []
    
    for filename in glob.glob(os.path.join(positive_path, '*.txt')):
        with open(filename, 'r') as f:
            dataset.append((pos_label, f.read()))
            
    for filename in glob.glob(os.path.join(negative_path, '*.txt')):
        with open(filename, 'r') as f:
            dataset.append((neg_label, f.read()))
    
    shuffle(dataset)
    
    return dataset


def tokenize_and_vectorize(dataset):
    tokenizer = TreebankWordTokenizer()
    word_vectors = KeyedVectors.load_word2vec_format('/Users/chenwang/Workspace/datasets/GoogleNews-vectors-negative300.bin.gz', binary=True, limit=200000)
    
    vectorized_data = []
    expected = []
    for sample in dataset:
        # sample 是一个二元组，sample[0] 是label, sample[1] 是评论text
        tokens = tokenizer.tokenize(sample[1])
        
        sample_vecs = []
        for token in tokens:
            try:
                sample_vecs.append(word_vectors[token])

            except KeyError:
                pass  # No matching token in the Google w2v vocab
        
        # list of list    
        vectorized_data.append(sample_vecs)

    return vectorized_data


def collect_expected(dataset):
    """ Peel of the target values from the dataset """
    expected = []
    for sample in dataset:
        expected.append(sample[0])
    return expected


def pad_trunc(data, maxlen):
    """ For a given dataset pad with zero vectors or truncate to maxlen """
    new_data = []

    # Create a vector of 0's the length of our word vectors
    
    zero_vector = []
    
    # data[0] 第一个review 所有token 的词向量
    for _ in range(len(data[0][0])):
        zero_vector.append(0.0)

    for sample in data:
 
        if len(sample) > maxlen:
            temp = sample[:maxlen]
        elif len(sample) < maxlen:
            temp = sample
            additional_elems = maxlen - len(sample)
            for _ in range(additional_elems):
                temp.append(zero_vector)
        else:
            temp = sample
        new_data.append(temp)
    return new_data
