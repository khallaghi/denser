from lark import Lark


class GrammarParser:
    parser = None

    def __init__(self, grammar_file):
        self._read_grammar(grammar_file)

    def _read_grammar(self, grammar_file):
        self.parser = Lark.open(grammar_file, rel_to=__file__)

    def read_input_from_file(self, input_file):
        input_sentence = ''
        with open(input_file, 'r+') as f:
            input_sentence = ''.join(f.readlines())

        return self.parser.parse(input_sentence)

    def read_input(self, input_rules):
        return self.parser.parse(input_rules)


