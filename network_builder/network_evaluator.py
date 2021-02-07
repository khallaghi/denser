from network_builder.grammar_parser import GrammarParser
from network_builder.neural_net_builder import NeuralNetworkBuilder
import matplotlib.pyplot as plt
from keras.datasets import cifar10
import ssl


def plot(_history):
    plt.plot(_history.history['accuracy'], label='accuracy')
    plt.plot(_history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.5, 1])
    plt.legend(loc='lower right')
    plt.show()


def load_data():
    (_train_images, _train_labels), (_test_images, _test_labels) = cifar10.load_data()
    _train_images, _test_images = _train_images / 255.0, _test_images / 255.0
    return _train_images, _train_labels, _test_images, _test_labels


def get_network_fitness(network_rules):
    test_acc = 0
    test_loss = float("inf")
    try:
        ssl._create_default_https_context = ssl._create_unverified_context
        grammar_file = 'grammar.lark'
        # initial_network = sys.argv[2]

        train_images, train_labels, test_images, test_labels = load_data()

        gp = GrammarParser(grammar_file)
        network_builder = NeuralNetworkBuilder(gp, epoch=7)
        network_builder.build(network_rules)
        network_builder.compile()
        history = network_builder.fit(train_images, train_labels, test_images, test_labels)
        test_loss, test_acc = network_builder.evaluate(test_images, test_labels)

        print("Accuracy: ", test_acc)
        print("loss: ", test_loss)
    except Exception as e:
        print(e)
    return test_acc, test_loss
