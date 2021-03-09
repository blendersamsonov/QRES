from pyparsing import (
    Literal,
    Word,
    Group,
    Forward,
    alphas,
    alphanums,
    Regex,
    ParseException,
    CaselessKeyword,
    Suppress,
    delimitedList,
    Optional
)
import operator
from functools import reduce
import numpy as np

class Parser:
    exprStack = []
    bnf = None
    var = []
    get_var = None
    opn = {
        "+": operator.add,
        "-": operator.sub,
        "*": operator.mul,
        "/": operator.truediv,
        "^": operator.pow,
    }

    fn = {
        "sin": np.sin,
        "cos": np.cos,
        "tan": np.tan,
        "exp": np.exp,
        "abs": np.abs,
        "sqrt": np.sqrt,
        "arctan": np.arctan,
        "arccos": np.arccos,
        "arcsin": np.arcsin,
    }
    
    def __init__(self, var = [], get_var = lambda x:x):
        self.var = var
        self.get_var = get_var
        # keys = []
        # for v in var+['pi','i']:
            # keys.append(CaselessKeyword(v))

        fnumber = Regex(r"[+-]?\d+(?:\.\d*)?(?:[eE][+-]?\d+)?")
        ident = Word(alphas, alphanums + "_$")

        plus, minus, mult, div = map(Literal, "+-*/")
        lpar, rpar = map(Suppress, "()")
        addop = plus | minus
        multop = mult | div
        expop = Literal("^")

        expr = Forward()
        expr_list = delimitedList(Group(expr))
        # add parse action that replaces the function identifier with a (name, number of args) tuple
        fn_call = (ident + lpar - Group(expr_list) + rpar).setParseAction(
            lambda t: t.insert(0, (t.pop(0), len(t[0])))
        )
        atom = (
            addop[...]
            + (
                # (fn_call | reduce(lambda x, y: x | y, keys) | fnumber | ident).setParseAction(self.push_first)
                (fn_call | fnumber | ident).setParseAction(self.push_first)
                | Group(lpar + expr + rpar)
            )
        ).setParseAction(self.push_unary_minus)

        # by defining exponentiation as "atom [ ^ factor ]..." instead of "atom [ ^ atom ]...", we get right-to-left
        # exponents, instead of left-to-right that is, 2^3^2 = 2^(3^2), not (2^3)^2.
        factor = Forward()
        factor <<= atom + (expop + factor).setParseAction(self.push_first)[...]
        term = factor + (multop + factor).setParseAction(self.push_first)[...]
        expr <<= term + (addop + term).setParseAction(self.push_first)[...]
        self.bnf = expr
        
    def push_first(self, toks):
            self.exprStack.append(toks[0])

    def push_unary_minus(self, toks):
        for t in toks:
            if t == "-":
                self.exprStack.append("unary -")
            else:
                break
        
    def stack_regression(self, s, var_name='x', **get_args):
        op, num_args = s.pop(), 0
        if isinstance(op, tuple):
            op, num_args = op
        if op == "unary -":
            return -self.stack_regression(s,var_name, **get_args)
        if op in "+-*/^":
            # note: operands are pushed onto the stack in reverse order
            op2 = self.stack_regression(s,var_name, **get_args)
            op1 = self.stack_regression(s,var_name, **get_args)
            return self.opn[op](op1, op2)
        elif op == "pi":
            return np.pi
        elif op == "i":
            return 1j
        elif op in self.var:
            get_args.update({var_name:op})
            return self.get_var(**get_args)
        elif op in self.fn:
            # note: args are pushed onto the stack in reverse order
            args = reversed([self.stack_regression(s,var_name, **get_args) for _ in range(num_args)])
            return self.fn[op](*args)
        elif op[0].isalpha():
            raise Exception("invalid identifier '%s'" % op)
        else:
            # try to evaluate as int first, then as float if int fails
            try:
                return int(op)
            except ValueError:
                return float(op)
            
    def evaluate(self, string, var_name='x', **get_args):
        self.exprStack = []
        res=self.bnf.parseString(string, parseAll=True)
        print(res)
        print(self.exprStack)
        return self.stack_regression(self.exprStack,var_name,**get_args)