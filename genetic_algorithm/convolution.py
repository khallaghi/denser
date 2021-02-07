class Convolution:
    def __init__(self, filter_size=32, filter_shape=(3, 3), activation='relu', input_shape=None):
        self.filter_size = filter_size
        self.filter_shape = filter_shape
        self.activation = activation
        self.input_shape = input_shape

    @staticmethod
    def stringify_two_d(_input):
        return "{}x{}".format(_input[0], _input[1])

    @staticmethod
    def stringify_three_d(_input):
        return "{}x{}x{}".format(_input[0], _input[1], _input[2])

    def __str__(self):
        gen_str = "conv::filter_size={},filter_shape={},activation={}".format(
            self.filter_size,
            self.stringify_two_d(self.filter_shape),
            self.activation
        )
        if self.input_shape is not None:
            gen_str += ",input_shape={}".format(self.stringify_three_d(self.input_shape))
        gen_str += ";"
        return gen_str
