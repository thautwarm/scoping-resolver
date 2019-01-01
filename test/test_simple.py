import unittest
from scoping_resolver import to_scoped_ast, ScopedAst, SymTable, ScopeTagger
import ast


class TestSimple(unittest.TestCase):
    def test(self):
        mod_code = ("""
c = lambda x: x
def f(x):
    g(x)
    c = 2
    g(c)
        """)
        mod = ast.parse(mod_code)
        g = SymTable.global_context()
        ScopeTagger(g).visit(mod)
        g.analyze()
        print(g.show_resolution())

        body = mod.body
        def_f: ScopedAst = body[1]
        self.assertEqual(type(def_f), ScopedAst)
        self.assertEqual(type(def_f.node), ast.FunctionDef)
        self.assertIn('c', def_f.scope.analyzed.bounds)

