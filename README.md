About
==============

This repo brings about a strong analysis of scoping for Python language,
which could tell a symbol inside a context is a `freevar`, `bounded var` or `cell var`, and many other details
like, "is this closure a coroutine/generator/async generator?" and so on.

More about Python's scoping, check [code object](https://github.com/Xython/YAPyPy/blob/master/python-internals/code-object.md).

`Symbol-Resolver` could be leveraged to implement transpilers from Python to another language with hygienic scopes,
like [Tensorflow AutoGraph](https://www.tensorflow.org/guide/autograph) and [Numba AutoJit](http://numba.pydata.org/numba-doc/latest/reference/jit-compilation.html#jit-functions):

```python
from utranspiler import transpile, func
# The `func` might not be a valid python function,
# but it can be used to denote some external function.

@transpile
def ufunc(x):
    func(x)
```

In above codes, `func` is resolved to be a `global` variable, in other words,
you can then search the name `func` from global context from current module,
and make sure it's exactly the `func` object that is the expected external function.

If you're still confusing of the power of Symbol-Resolver, there is also an example for you:

```python
from utranspiler import transpile, func

@transpile
def ufunc(x):
    func = lambda x: x + 1
    func(x)
```

Or

```python
from utranspiler import transpile, func
func = lambda x: x + 1
@transpile
def ufunc(x):
    func(x)
```

Now you cannot expect `func` as an external function, but you might not want to check the symbol manually by implementing
your specific symbol analyzer.

Tensorflow team is still struggling with above problem, you can something unhygienic in following link:

https://github.com/tensorflow/tensorflow/blob/3ae375aa92fbb6155f82393735d0b98d8fb9c1b2/tensorflow/python/autograph/converters/lists.py#L129

Usage
=======

Check `test/test_simple.py`.

The `ScopeTagger` converts any AST that lead to a new context into a wrapped node named `ScopedAst`, where all
the information about current context is held.

```python
import unittest
import ast
from scoping_resolver import to_scoped_ast, ScopedAst, SymTable, ScopeTagger

mod_code = """
c = lambda x: x
def f(x):
    g(x)
    c = 2
    g(c)
"""
class TestSimple(unittest.TestCase):
    def test(self):
        mod = ast.parse(mod_code)

        # Make a new symbol table object for global context.
        # Of course, symbol tables for sub-contexts would be created
        # when analyzing the whole module AST.
        g = SymTable.global_context()

        # Get raw information of AST.
        ScopeTagger(g).visit(mod)

        # Peform analysis.
        g.analyze()

        # You can directly use `to_scoped_ast(mod)`
        # instead when you don't need a top level `g`.

        # Show representations of nested scopes:
        print(g.show_resolution())
        # [AnalyzedSymTable(bounds=set(), freevars=set(), cellvars=set()),
        #  [[AnalyzedSymTable(bounds={'x'}, freevars=set(), cellvars=set()), []],
        #   [AnalyzedSymTable(bounds={'x', 'c'}, freevars=set(), cellvars=set()), []]]]

        body = mod.body
        def_f: ScopedAst = body[1]

        # `FunctionDef` creates a new context, so it'll be wrapped inside a ScopedAst

        self.assertEqual(type(def_f), ScopedAst)
        self.assertEqual(type(def_f.node), ast.FunctionDef)
        self.assertIn('c', def_f.scope.analyzed.bounds)
```
