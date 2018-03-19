import pickle
import numpy as np

from subprocess import call

data_src = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'

def download_data():
    call(['wget', data_src])
    call(['tar', '-zxvf', 'cifar-10-python.tar.gz'])
   
    
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def batch(name):
    data = unpickle('cifar-10-batches-py/' + name)
    
    # print(data[b'filenames'])

    return {
        'data': data[b'data'],
        'labels': np.array(data[b'labels'])
    }

def RGB(images):
    return images.reshape((-1, 3, 1024)).transpose((0, 2, 1)) / 255


def load_cifar(download=False):
    if download: download_data()
    batches_names = ['data_batch_' + str(i) for i in range(1,6)] 
    batches = [batch(name) for name in batches_names]
    train_data = np.vstack([b['data'] for b in batches])
    train_labels = np.hstack([b['labels'] for b in batches])
    test_batch = batch('test_batch')
    test_data = test_batch['data']
    test_labels = test_batch['labels']
    return RGB(train_data), train_labels, RGB(test_data), test_labels


