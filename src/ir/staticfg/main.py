import ast

from staticfg import CFGBuilder
cfg = CFGBuilder().build_from_file(
    "aes.py",
    "/home/khushal/code-anyhwperf/src/nonai_models/aes.py",
)
cfg.build_visual("exampleCFG", "pdf", show=False)
print(cfg)
# for node in cfg:
#     for i in node.statements:
#         print(ast.dump(i))

# for node in cfg.statements:
# print(node)


for node in cfg:
    print(node)
    get_source(node)
    for i in node.statements:
            print(ast.dump(i))

node_types = [ast.If, ast.For, ast.While, ast.FunctionDef, ast.Compare,ast.BinOp, ast.UnaryOp, ast.BoolOp]
# ast.Name, ast.Attribute, ast.Str, ast.Subscript