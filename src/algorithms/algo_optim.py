from ..scheduling import run_asap

def optim_algo(graph, algo, hw):
    bottlenecks = run_asap(hw)
    print(bottlenecks)
    
