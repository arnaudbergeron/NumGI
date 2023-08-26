import sympy as sp
from sympy.core.numbers import Float
from sympy.core.numbers import Integer

class EquationTokenizer:
    """
    Tokenizer for equations.
    """
    def __init__(self):
        self.tokenizer_dict = {}
        self.dict_size = None


    def sympy_to_list(self, eq):
        """
        Converts a sympy equation to a list that will be tokenized.
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
                if isinstance(_sub, Float) or isinstance(_sub, Integer): 
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
                if eq_list[idx+1] != '(':
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
        numbers_list = ['0','1','2','3','4','5','6','7','8','9','.']
        num_str = ''
        for idx, i in enumerate(eq_list):
            if i in numbers_list:
                num_str += i
                if eq_list[idx+1] not in numbers_list:
                    if '.' in num_str:
                        final_list.append(Float(num_str))
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
