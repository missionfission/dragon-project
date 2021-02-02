import unittest
import ast
from staticfg.model import *
from staticfg.builder import CFGBuilder, is_py38_or_higher


class TestBlock(unittest.TestCase):

    def test_instanciation(self):
        block = Block(1)
        self.assertEqual(block.id, 1)
        self.assertEqual(block.statements, [])
        self.assertEqual(block.func_calls, [])
        self.assertEqual(block.predecessors, [])
        self.assertEqual(block.exits, [])

    def test_str_representation(self):
        block = Block(1)
        self.assertEqual(str(block), "empty block:1")
        tree = ast.parse("a = 1")
        block.statements.append(tree.body[0])
        self.assertEqual(str(block), "block:1@1")

    def test_repr(self):
        block = Block(1)
        self.assertEqual(repr(block), "empty block:1 with 0 exits")
        tree = ast.parse("a = 1")
        block.statements.append(tree.body[0])
        if is_py38_or_higher():
            self.assertEqual(repr(block), "block:1@1 with 0 exits, body=[\
Assign(targets=[Name(id='a', ctx=Store())], value=Constant(value=1, kind=None), type_comment=None)]")
        else:
            self.assertEqual(repr(block), "block:1@1 with 0 exits, body=[\
Assign(targets=[Name(id='a', ctx=Store())], value=Num(n=1))]")

    def test_at(self):
        block = Block(1)
        self.assertEqual(block.at(), None)
        tree = ast.parse("a = 1")
        block.statements.append(tree.body[0])
        self.assertEqual(block.at(), 1)

    def test_is_empty(self):
        block = Block(1)
        self.assertTrue(block.is_empty())
        tree = ast.parse("a = 1")
        block.statements.append(tree.body[0])
        self.assertFalse(block.is_empty())

    def test_get_source(self):
        block = Block(1)
        self.assertEqual(block.get_source(), "")
        tree = ast.parse("a = 1")
        block.statements.append(tree.body[0])
        self.assertEqual(block.get_source(), "a = 1\n")

    def test_get_calls(self):
        block = Block(1)
        self.assertEqual(block.get_calls(), "")
        block.func_calls.append("fun")
        self.assertEqual(block.get_calls(), "fun\n")


class TestLink(unittest.TestCase):

    def test_instanciation(self):
        block1 = Block(1)
        block2 = Block(2)
        with self.assertRaises(AssertionError):
            Link(2, block2)  # Source of a link must be a block.
            Link(block1, 2)  # Target of a link must be a block.

        condition = ast.parse("a == 1").body[0]
        link = Link(block1, block2, condition)
        self.assertEqual(link.source, block1)
        self.assertEqual(link.target, block2)
        self.assertEqual(link.exitcase, condition)

    def test_str_representation(self):
        block1 = Block(1)
        block2 = Block(2)
        link = Link(block1, block2)
        self.assertEqual(str(link), "link from empty block:1 to empty block:2")

    def test_repr(self):
        block1 = Block(1)
        block2 = Block(2)
        condition = ast.parse("a == 1").body[0]
        link = Link(block1, block2, condition)
        self.assertEqual(repr(link), "link from empty block:1 to empty block:2\
, with exitcase {}".format(ast.dump(condition)))

    def test_get_exitcase(self):
        block1 = Block(1)
        block2 = Block(2)
        condition = ast.parse("a == 1").body[0]
        link = Link(block1, block2, condition)
        self.assertEqual(link.get_exitcase(), "a == 1\n")


class TestCFG(unittest.TestCase):

    def test_instanciation(self):
        with self.assertRaises(AssertionError):
            CFG(2, False)  # Name of a CFG must be a string.
            CFG('cfg', 2)  # Async argument must be a boolean.

        cfg = CFG('cfg', False)
        self.assertEqual(cfg.name, 'cfg')
        self.assertFalse(cfg.asynchr)
        self.assertEqual(cfg.entryblock, None)
        self.assertEqual(cfg.finalblocks, [])
        self.assertEqual(cfg.functioncfgs, {})

    def test_str_representation(self):
        cfg = CFG('cfg', False)
        self.assertEqual(str(cfg), 'CFG for cfg')

    def test_iter(self):
        src = """\
def fib():
    a, b = 0, 1
    while True:
        yield a
        a, b = b, a + b
"""
        cfg = CFGBuilder().build_from_src("fib", src)
        expected_block_sources = [
            "def fib():...\n",
            "a, b = 0, 1\n",
            "while True:\n",
            "yield a\n",
            "a, b = b, a + b\n"
        ]
        for actual_block, expected_src in zip(cfg, expected_block_sources):
            self.assertEqual(actual_block.get_source(), expected_src)

    def test_break(self):
        src = """\
def foo():
    i = 0
    while True:
        i += 1
        if i == 3:
            break
    for j in range(3):
        i += j
        if j == 2:
            break
    return i
"""
        cfg = CFGBuilder().build_from_src("foo", src)
        expected_block_sources = [
            "def foo():...\n",
            "i = 0\n",
            "while True:\n",
            "i += 1\n"
            "if i == 3:\n",
            "for j in range(3):\n",
            "i += j\n"
            "if j == 2:\n",
            "return i\n",
            "",
        ]
        for actual_block, expected_src in zip(cfg, expected_block_sources):
            self.assertEqual(actual_block.get_source(), expected_src)

    def test_break_in_main_body(self):
        src = """\
def foo():
    i = 0
    while True:
        i += 1
        break
    for j in range(3):
        i += j
        break
    return i
"""
        cfg = CFGBuilder().build_from_src("foo", src)
        expected_block_sources = [
            "def foo():...\n",
            "i = 0\n",
            "while True:\n",
            "i += 1\n",
            "for j in range(3):\n",
            "i += j\n",
            "return i\n"
        ]
        for actual_block, expected_src in zip(cfg, expected_block_sources):
            self.assertEqual(actual_block.get_source(), expected_src)

if __name__ == "__main__":
    unittest.main()
