class Classification:
    def __init__(self, net_size=64, activation='relu'):
        self.net_size = net_size
        self.activation = activation

    def __str__(self):
        gen_str = "dense::net_size={}".format(self.net_size)
        if self.activation is not None:
            gen_str += ",activation={}".format(self.activation)
        gen_str += ";"
        return gen_str
