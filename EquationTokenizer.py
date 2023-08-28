import sympy as sp
from sympy.core.numbers import Float
from sympy.core.numbers import Integer
from sympy.core.numbers import Rational
import torch
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F


class EquationTokenizer:
    """
    Tokenizer for equations.
    """
    def __init__(self):
        self.tokenizer_dict = {}
        self.dict_size = 0
        self.tokenize = None
        self.decode = None
        self.tokenizer_dict = None
        self.decode_dict = None


    def sympy_to_list(self, eq):
        """
        Converts a sympy equation to a list that will be tokenized.
        This uses prefix notation.
        """
        eq_list = []
        eq_args = eq.args
        args_len = len(eq_args)
        if args_len == 0:
            return [eq]
            
        eq_list.append(eq.func)
        eq_list.append('(')
        for ind, arg in enumerate(eq_args):
            sub_arg_list = self.sympy_to_list(arg)
            for _sub in sub_arg_list:
                if isinstance(_sub, Float) or isinstance(_sub, Integer) or isinstance(_sub, Rational): 
                    #prehaps not general enough should allow for more types
                    #the idea is we want to tokenize '12.2' as '1','2,'.','2' and not '12.2'
                    for i in str(_sub):
                        eq_list.append(i)
                else:
                    eq_list.append(_sub)

            if ind != args_len - 1:
                eq_list.append(',')

        eq_list.append(')')
        
        return eq_list

    def _parantheses_to_list(self, eq_list):
        """
        Converts a list with parantheses to a list of lists according to parentheses this is a util func.
        """
        final_list = []
        
        fin_idx = 0
        for idx, i in enumerate(eq_list):
            if idx <= fin_idx-1:
                continue

            if i == '(':
                rec_result, _fin_idx = self._parantheses_to_list(eq_list[idx+1:])
                fin_idx = idx+ _fin_idx+1
                rec_result.insert(0, eq_list[idx-1])
                final_list.append(rec_result)

            elif i == ')':
                return final_list, idx+1

            elif i!=',':
                if len(eq_list)==idx+1:
                    final_list.append(i)
                elif eq_list[idx+1] != '(':
                    final_list.append(i)

        return final_list, idx
    
    def _utils_exec_sympy(self, eq_list):
        function = eq_list[0]
        args_list = []
        for i in eq_list[1:]:
            if isinstance(i, list):
                args_list.append(self._utils_exec_sympy(i))
            else:
                args_list.append(i)

        return function(*args_list)
    
    def _regroup_numbers(self, eq_list):
        """
        Regroups numbers in a list to a their original glory.
        """
        final_list = []
        numbers_list = ['0','1','2','3','4','5','6','7','8','9','.','-','/']
        num_str = ''
        for idx, i in enumerate(eq_list):
            if i in numbers_list:
                num_str += i
                if eq_list[idx+1] not in numbers_list:
                    if '.' in num_str:
                        final_list.append(Float(num_str))
                    elif '/' in num_str:
                        final_list.append(Rational(num_str))
                    else:
                        final_list.append(Integer(num_str))
                    num_str = ''
            else: 
                final_list.append(i)

        return final_list

    def list_to_sympy(self, eq_list):
        """This function takes in a list of functions and numbers and outputs the sympy equation."""
        grouped_num_list = self._regroup_numbers(eq_list)
        parsed_list = self._parantheses_to_list(grouped_num_list)[0][0]
        return self._utils_exec_sympy(parsed_list)
    
    def sympy_to_tokens(self, sympy_eq):
        """Takes in a sympy equation and outputs a tokenized list."""
        if self.tokenize is None:
            raise('Tokenizer not created yet.')
        seq = self.tokenize(self.sympy_to_list(sympy_eq)) + [self.tokenize_dict['END']]
        return seq
    
    def tokens_to_sympy(self, tokens):
        """Takes in a sympy equation and outputs a tokenized list."""
        if self.tokenize is None:
            raise('Tokenizer not created yet.')
        decoded_seq = self.decode(tokens)
        decoded_seq = [i for i in decoded_seq if i not in ['END','PAD']]
        seq = self.list_to_sympy(self.decode(tokens))
        return seq

    def create_tokenizer(self, symbol_set):
        """Takes a set of symbols and creates a tokenizer for them."""

        #add the special tokens
        symbol_set = symbol_set.union(set(['END','PAD']))

        tokenize_dict = {symbol:idx for symbol, idx in zip(list(symbol_set), range(len(symbol_set)))}
        decode_dict = {idx:symbol for symbol, idx in zip(list(symbol_set), range(len(symbol_set)))}

        self.tokenize_dict = tokenize_dict
        self.dict_size = len(tokenize_dict)
        self.decode_dict = decode_dict

        self.tokenize = lambda x: [self.tokenize_dict[i] for i in x] + [self.tokenize_dict['END']]
        self.decode = lambda x: [self.decode_dict[i] for i in x]

        print(f'Created Tokenizer and Decoder for character set with size: {self.dict_size}')

    def tensorize_and_pad(self, list_of_token_list):
        """Takes in a list of tokenized lists and outputs a padded tensor of tensors."""
    
        pad_val = self.tokenize_dict['PAD']

        list_of_token_list = [torch.tensor(i) for i in list_of_token_list]

        output = pad_sequence(list_of_token_list, batch_first=True, padding_value=pad_val)

        return output
    
    def tensorize_and_pad_by_len(self, list_of_token_list, max_len):
        """Takes in a list of tokenized lists and outputs a padded tensor of tensors of defined length."""
    
        pad_val = self.tokenize_dict['PAD']

        list_of_token_list = [torch.tensor(i) for i in list_of_token_list]
        _extra = torch.zeros(max_len)
        list_of_token_list.append(_extra)

        output = pad_sequence(list_of_token_list, batch_first=True, padding_value=pad_val)

        return output[:-1]
