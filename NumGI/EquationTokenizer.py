from __future__ import annotations

import sympy as sp
import torch
from sympy.core.numbers import Float
from sympy.core.numbers import Integer
from sympy.core.numbers import Rational
from torch.nn.utils.rnn import pad_sequence

from NumGI.ConstantDictionaries import DEFAULT_DICT
from NumGI.ConstantDictionaries import SP_TO_TORCH


class EquationTokenizer:
    """Tokenizer for equations.

    Given a set of equations it creates a tokenizer and decoder for their unique symbols.
    This will allow us to parse equations as sequences of tokens.
    These tokens can now be processed by the transformer.
    """

    def __init__(self, useDefaultTokenizer=False):
        self.tokenizer_dict = {}
        self.dict_size = 0
        self.tokenize = None
        self.decode = None
        self.tokenizer_dict = None
        self.decode_dict = None

        if useDefaultTokenizer:
            print("Using default tokenizer.")
            self.tokenize_dict, self.decode_dict, self.tokenize, self.decode = defaultTokenizer()
            self.dict_size = len(self.tokenize_dict)
            self.char_set = set(self.tokenize_dict.keys())

        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

    def sympy_to_list(self, sympy_equation) -> list:
        """Converts a sympy equation to a list that will be tokenized.

        This uses prefix notation. of the form: [function1(arg1, function2(arg2_1, arg2_2), ...)],
        Where args can be functions with their own arguments.

        Args:
            sympy_equation: Sympy object/equation.

        Returns:
            list: Sympy equation transformed to a list format.
        """
        eq_list = []
        eq_args = sympy_equation.args
        args_len = len(eq_args)
        if args_len == 0:
            return [sympy_equation]

        eq_list.append(sympy_equation.func)
        eq_list.append("(")
        for ind, arg in enumerate(eq_args):
            sub_arg_list = self.sympy_to_list(arg)
            for _sub in sub_arg_list:
                if self.is_number(_sub):
                    # perhaps not general enough should allow for more types
                    # the idea is we want to tokenize '12.2' as '1','2,'.','2' and not '12.2'
                    for i in str(_sub):
                        eq_list.append(i)
                else:
                    eq_list.append(_sub)

            if ind != args_len - 1:
                eq_list.append(",")

        eq_list.append(")")

        return eq_list

    def sympy_to_numpy(self, sympy_equation):
        """Converts a sympy equation to a numpy function.

        This is a util func.
        """
        symbols = list(sympy_equation.free_symbols)
        return sp.lambdify(symbols, sympy_equation, "numpy"), symbols

    def sympy_to_torch(self, sympy_equation):
        """Converts a sympy equation to a pytorch function.

        This is a util func.
        """
        # simplified_eq = sympy_equation.simplify()
        simplified_eq = sympy_equation
        sympy_list = self.sympy_to_list(simplified_eq)
        grouped_num_list = self._regroup_numbers(sympy_list)
        parsed_list = self._parantheses_to_list(grouped_num_list)[0][0]

        variables = list(simplified_eq.free_symbols)
        variables = [str(i) for i in variables]

        def torch_func(**kwargs):
            return self._utils_exec_torch(parsed_list, **kwargs)

        return torch_func, variables

    def _utils_exec_torch(self, sympy_list, **kwargs):
        """Converts a sympy list to a torch function.

        This is a util func.
        """
        function = sympy_list[0]
        torch_function = SP_TO_TORCH[function]
        args_list = []
        for i in sympy_list[1:]:
            if isinstance(i, list):
                args_list.append(self._utils_exec_torch(i, **kwargs))
            elif isinstance(i, sp.Symbol):
                args_list.append(kwargs[str(i)])
            elif self.is_number(i):
                args_list.append(torch.tensor(float(i), device=self.device))
            elif i == sp.core.numbers.Pi:
                args_list.append(torch.tensor(torch.pi, device=self.device))
            else:
                raise ValueError(f"Unknown type: {type(i)}, for {i}")

        if len(args_list) > 2 and (function == sp.Add or function == sp.Mul):
            return self.call_multi_input_torch(torch_function, args_list)

        else:
            return torch_function(*args_list)

    def call_multi_input_torch(self, func, args):
        if len(args) > 2:
            return func(args[0], self.call_multi_input_torch(func, args[1:]))
        else:
            return func(args[0], args[1])

    def _parantheses_to_list(self, eq_list):
        """Converts a list with parentheses to a list of lists according to parentheses.

        This is a util func.
        """
        final_list = []

        fin_idx = 0
        for idx, i in enumerate(eq_list):
            if idx <= fin_idx - 1:
                continue

            if i == "(":
                rec_result, _fin_idx = self._parantheses_to_list(eq_list[idx + 1 :])
                fin_idx = idx + _fin_idx + 1
                rec_result.insert(0, eq_list[idx - 1])
                final_list.append(rec_result)

            elif i == ")":
                return final_list, idx + 1

            elif i != ",":
                if len(eq_list) == idx + 1:
                    final_list.append(i)
                elif eq_list[idx + 1] != "(":
                    final_list.append(i)

        return final_list, idx

    def _utils_exec_sympy(self, eq_list):
        """Takes in a graph in the shape of lists of lists and converts it to a sympy equation.

        Using a kind of AST. This is a util func.
        """
        function = eq_list[0]
        args_list = []
        for i in eq_list[1:]:
            if isinstance(i, list):
                args_list.append(self._utils_exec_sympy(i))
            else:
                args_list.append(i)

        return function(*args_list)

    def _regroup_numbers(self, eq_list):
        """Regroups numbers in a list to a their original glory.

        We had separated them to tokenize them but we now put them back together.
        This is the first step of going from list to sympy.
        """
        final_list = []
        numbers_list = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", ".", "-", "/"]
        num_str = ""
        for idx, i in enumerate(eq_list):
            if i in numbers_list:
                num_str += i
                if eq_list[idx + 1] not in numbers_list:
                    if "." in num_str:
                        final_list.append(Float(num_str))
                    elif "/" in num_str:
                        final_list.append(Rational(num_str))
                    else:
                        final_list.append(Integer(num_str))
                    num_str = ""
            else:
                final_list.append(i)

        return final_list

    def list_to_sympy(self, eq_list):
        """Takes in a list of functions and numbers and outputs the sympy equation.

        To go from list/sequence to sympy equation we need to construct the AST again.
        We do so by regrouping functions and their arguments.
        """
        grouped_num_list = self._regroup_numbers(eq_list)
        parsed_list = self._parantheses_to_list(grouped_num_list)[0][0]
        return self._utils_exec_sympy(parsed_list)

    def sympy_to_tokens(self, sympy_eq):
        """Takes in a sympy equation and outputs a tokenized list."""
        if self.tokenize is None:
            raise ("Tokenizer not created yet.")
        seq = self.tokenize(self.sympy_to_list(sympy_eq)) + [self.tokenize_dict["END"]]
        return seq

    def tokens_to_sympy(self, tokens):
        """Takes in a tokenized list and outputs a sympy equation."""
        decoded_seq = self.tokens_to_list(tokens)
        seq = self.list_to_sympy(decoded_seq)
        return seq

    def tokens_to_list(self, tokens):
        if self.tokenize is None:
            raise ("Tokenizer not created yet.")
        decoded_seq = self.decode(tokens)
        decoded_seq = self.remove_non_sympy_tokens(decoded_seq)
        return decoded_seq

    def remove_non_sympy_tokens(self, decoded_seq):
        non_sympy = ["START", "PAD"]
        res = []
        for i in decoded_seq:
            if i == "END":
                return res
            if i not in non_sympy:
                res.append(i)
        return res

    def create_tokenizer(self, symbol_set):
        """Takes a set of symbols and creates a tokenizer for them."""
        # add the special tokens
        symbol_set = symbol_set.union({"START", "END", "PAD"})

        tokenize_dict = {
            symbol: idx for symbol, idx in zip(list(symbol_set), range(len(symbol_set)))
        }
        decode_dict = {idx: symbol for symbol, idx in zip(list(symbol_set), range(len(symbol_set)))}

        self.tokenize_dict = tokenize_dict
        self.dict_size = len(tokenize_dict)
        self.decode_dict = decode_dict

        self.tokenize = (
            lambda x: [self.tokenize_dict["START"]]
            + [self.tokenize_dict[i] for i in x]
            + [self.tokenize_dict["END"]]
        )
        self.decode = lambda x: [self.decode_dict[i] for i in x]

        print(f"Created Tokenizer and Decoder for character set with size: {self.dict_size}")

    def tensorize_and_pad(self, list_of_token_list):
        """Takes in a list of tokenized lists and outputs a padded tensor of tensors."""
        pad_val = self.tokenize_dict["PAD"]

        list_of_token_list = [torch.tensor(i, device=self.device) for i in list_of_token_list]
        output = pad_sequence(list_of_token_list, batch_first=True, padding_value=pad_val)

        return output

    def tensorize_and_pad_by_len(self, list_of_token_list, max_len):
        """Takes in a list of tokenized lists and outputs a padded tensor of defined length."""
        pad_val = self.tokenize_dict["PAD"]
        list_of_token_list = [torch.tensor(i, device=self.device) for i in list_of_token_list]

        return self._pad_tensors(list_of_token_list, max_len, pad_val)

    def pad_by_len(self, list_of_token_list, max_len):
        """Takes in a list of tokenized lists and outputs a padded tensor of defined length."""
        pad_val = self.tokenize_dict["PAD"]
        list_of_token_list = [i.to(self.device) for i in list_of_token_list]

        return self._pad_tensors(list_of_token_list, max_len, pad_val)

    def _pad_tensors(self, list_of_token_list, max_len, pad_val):
        _extra = torch.zeros(max_len, device=self.device)
        list_of_token_list.append(_extra)

        output = pad_sequence(list_of_token_list, batch_first=True, padding_value=pad_val)
        return output[torch.max((output != _extra), axis=1).values]

    def is_number(self, sp_class):
        return (
            isinstance(sp_class, Float)
            or isinstance(sp_class, Integer)
            or isinstance(sp_class, Rational)
        )


def defaultTokenizer():
    """Returns a default tokenizer. Because of issues with pickling."""
    tokenize_dict = DEFAULT_DICT

    # invert tokenizer_dict into decode_dict
    decode_dict = {v: k for k, v in tokenize_dict.items()}

    tokenize = (
        lambda x: [tokenize_dict["START"]] + [tokenize_dict[i] for i in x] + [tokenize_dict["END"]]
    )
    decode = lambda x: [decode_dict[i] for i in x]

    return tokenize_dict, decode_dict, tokenize, decode
