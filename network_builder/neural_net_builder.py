from tensorflow.keras import models, layers
import tensorflow as tf

CONV = 'convolution'
POOL = 'pooling'
DENSE = 'classification'
FLATTEN = 'flatten'
FEATURES = 'feature'


class NeuralNetworkBuilder:
    def __init__(self, grammar_parser, epoch=10):
        self.grammar_parser = grammar_parser
        self.epoch = epoch
        self.model = models.Sequential()

    def build(self, input_rules):
        tree = self.grammar_parser.read_input(input_rules)

        for subtree in tree.children:
            if type(subtree).__name__ == 'Token':
                continue
            if subtree.data == FEATURES:
                self._iter_over_layers(subtree.children[0])
            if subtree.data == DENSE or subtree.data == FLATTEN:
                self._iter_over_layers(subtree)
        self.summary()

    def compile(self):
        self.model.compile(optimizer='adam',
                           loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                           metrics=['accuracy'])

    def summary(self):
        return self.model.summary()

    def fit(self, train_images, train_labels, test_images, test_labels):
        return self.model.fit(train_images, train_labels, epochs=self.epoch,
                              validation_data=(test_images, test_labels))

    def evaluate(self, test_images, test_labels):
        return self.model.evaluate(test_images, test_labels, verbose=2)

    def _iter_over_layers(self, subtree):
        _layer = self._gen_layer(subtree)
        self.model.add(_layer)

    def _add_flatten_layer(self):
        self.model.add(layers.Flatten())

    @staticmethod
    def _gen_layer(subtree):
        if subtree.data == CONV:
            return NeuralNetworkBuilder._gen_conv(subtree.children)
        if subtree.data == POOL:
            return NeuralNetworkBuilder._gen_pool(subtree.children)
        if subtree.data == DENSE:
            return NeuralNetworkBuilder._gen_dense(subtree.children)
        if subtree.data == FLATTEN:
            return NeuralNetworkBuilder._gen_flat()

    @staticmethod
    def _gen_conv(conv_child):
        conv_features = NeuralNetworkBuilder._get_features(conv_child)
        print("CONV", conv_features)
        if "input_shape" in conv_features:
            return layers.Conv2D(
                conv_features["filter_size"],
                conv_features["filter_shape"],
                activation=conv_features["activation"],
                input_shape=conv_features["input_shape"]
            )
        else:
            return layers.Conv2D(
                conv_features["filter_size"],
                conv_features["filter_shape"],
                activation=conv_features["activation"]
            )

    @staticmethod
    def _gen_pool(pool_child):
        pool_features = NeuralNetworkBuilder._get_features(pool_child)
        print("POOL", pool_features)
        return layers.MaxPool2D(pool_features["kernel_size"][0], pool_features["kernel_size"][1])

    @staticmethod
    def _gen_dense(dense_child):
        dense_features = NeuralNetworkBuilder._get_features(dense_child)
        print("DENSE", dense_features)
        if "activation" in dense_features:
            return layers.Dense(
                dense_features["net_size"],
                activation=dense_features["activation"]
            )
        else:
            return layers.Dense(
                dense_features["net_size"]
            )

    @staticmethod
    def _get_features(node):
        features = {}
        for _child in node:
            if type(_child).__name__ == 'Token':
                continue
            data = NeuralNetworkBuilder.get_key(_child)
            features[data[0]] = data[1]
        return features

    @staticmethod
    def _gen_flat():
        return layers.Flatten()

    @staticmethod
    def get_key(node):
        key = node.data
        data = NeuralNetworkBuilder.get_data(node.children)
        return key, data

    @staticmethod
    def get_data(_list):
        for item in _list:
            if type(item).__name__ == 'Token':
                continue
            if item.data == 'one_dim_size':
                return int(item.children[0])
            if item.data == 'two_dim_size':
                return int(item.children[0]), int(item.children[1])
            if item.data == 'three_dim_size':
                return int(item.children[0]), int(item.children[1]), int(item.children[2])
            if item.data == 'string_value':
                return str(item.children[0])
