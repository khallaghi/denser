class MaxPooling:
    def __init__(self, kernel_size=(2, 2)):
        self.kernel_size = kernel_size

    def __str__(self):
        gen_str = "pool::kernel_size={}x{};".format(self.kernel_size[0], self.kernel_size[1])
        return gen_str
