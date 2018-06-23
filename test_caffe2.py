import time
import onnx
import caffe2.python.onnx.backend as backend
import numpy as np
from tqdm import tqdm

model = onnx.load('./mnist_cnn.onnx')
onnx.checker.check_model(model)
print(onnx.helper.printable_graph(model.graph))

def load_mnist_file(filename, offset=16):
    with open(filename, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=offset)
    return data

def load_mnist():
    dirpath = '/Users/makora/data/mnist/raw/'
    files = [['train-images-idx3-ubyte','train-labels-idx1-ubyte'],
             ['t10k-images-idx3-ubyte','t10k-labels-idx1-ubyte']]

    train_data = load_mnist_file(dirpath + files[0][0])
    train_label = load_mnist_file(dirpath + files[0][1], 8)

    test_data = load_mnist_file(dirpath + files[1][0])
    test_label = load_mnist_file(dirpath + files[1][1], 8)

    return (train_data.astype(np.float32).reshape(-1, 1, 28, 28)/255., train_label), (test_data.astype(np.float32).reshape(-1, 1, 28, 28)/255., test_label)

(train_data, train_label), (test_data, test_label) = load_mnist()

rep = backend.prepare(model)

correct = 0
average_time = 0
data_len = test_data.shape[0]

pbar = tqdm(range(data_len))
for i in pbar:
    image, target = test_data[i].reshape(1, 1, 28, 28), test_label[i]
    start = time.time()
    output = rep.run(image)[0]
    average_time += (time.time() - start)
    pred = output.argmax(1)[0]
    correct += target == pred


print('Test set: Accuracy: {}/{} ({:.0f}%), Time: {}'.format(
        correct,
        data_len,
        100. * correct / data_len,
        average_time / data_len))





