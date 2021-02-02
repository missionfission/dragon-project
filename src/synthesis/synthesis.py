
lib_hw_dict = ['add','mult','buffer','reg','sys_array','logic','fsm']

def parser(graph):
    """
    Parse a non-AI workload graph and store the configuration as a hardware representation 
    """
    # G_hw_repr(graph)
    lib_common_ops = ['add','mult','mmv','spmv','conv']
    common_ops = set()
    for node in graph:
        if node.operator in lib_common_ops:
            common_ops.add(node.operator)


hw_allocated = {}
def allocation(H):
    for node in graph:
        hw_allocated['node.name'] = allocate(node)
    
def binding(H):
    for common_ops in graph:
        for key in hw_allocated.keys():
            if(key in common_ops):
                hw_allocated["common_ops"] = merge(hw_allocated[key], hw_allocated["common_ops"].copy())

def allocate_node(node):
    if(node.operator == 'if_else'):
        return create_fsm(node)
    if(node.operator=='loop'):
        #unroll
        # transform
        # check
        pass
    if(node.operator=='func'):
        getall = []
        for i in func:
            getall.append(allocate_node(i))
        return getall


lib_template_space = ['global_mem','local_mem', 'pes','noc','buffers']
        
def template_space(H):
    template_space = {}
    for i in lib_template_space:
        template_space[i] = template_handlers(i,hw_allocated)

