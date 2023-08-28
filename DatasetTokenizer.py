from EquationTokenizer import EquationTokenizer

class DatasetTokenizer(EquationTokenizer):
    def __init__(self, x, y):
        self.x = x
        self.y = y
        super().__init__()
        self.char_set = self.create_set_char()
        self.create_tokenizer(self.char_set)

        self.x_tokenized = [self.tokenize(i) for i in self.x]
        self.y_tokenized = [self.tokenize(i) for i in self.y]

        self.x_tokenized = self.tensorize_and_pad(self.x_tokenized)
        self.max_length = self.x_tokenized.shape[1]
        self.y_tokenized = self.tensorize_and_pad_by_len(self.y_tokenized, self.max_length)


    def create_set_char(self):
        char_list_x = [j for i in self.x for j in i]
        char_list_y = [j for i in self.y for j in i]
        char_set = set(char_list_x).union(set(char_list_y))
        return char_set
    
    def sympy_to_padded_tokens(self, eq):
        """Takes in a sympy equation and outputs a tokenized padded list."""

        seq = self.sympy_to_list(eq)

        if set(seq) - self.char_set != set():
            raise('The equation contains characters not in the character set. The models output will be non-sensical.')
        
        seq = self.tokenize(seq)
        seq = self.tensorize_and_pad_by_len([seq], self.max_length)
        return seq
    
    def split(self, factor):
        split_n = int(self.x_tokenized.shape[0]*factor)

        self.x_train =  self.x_tokenized[:split_n]
        self.y_train = self.y_tokenized[:split_n]
        self.x_val = self.x_tokenized[split_n:]
        self.y_val = self.y_tokenized[split_n:]