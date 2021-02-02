from staticfg import CFGBuilder

import ast

cfg = CFGBuilder().build_from_file('paillier.py','/home/khushal/code-anyhwperf/src/plugins/python-paillier/phe/paillier.py')
cfg.build_visual('exampleCFG', 'pdf', show=False)
print(cfg)
for node in cfg:
    for i in node.statements:
        print(ast.dump(i))

# for node in cfg.statements:
    # print(node)
