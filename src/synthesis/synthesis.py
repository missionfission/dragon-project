
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


void BaseDatapath::initBaseAddress() {
  auto opt = getGraphOpt<BaseAddressInit>();
  opt->run();
#if 0
  // TODO: writing the base addresses can cause simulation to freeze when no
  // partitioning is applied to arrays due to how writeBaseAddress() parses the
  // partition number from an array's label. So this is going to be disabled
  // for the time being, until we find a chance to fix this.
  writeBaseAddress();
#endif
}

void BaseDatapath::initDmaBaseAddress() {
  auto opt = getGraphOpt<DmaBaseAddressInit>();
  opt->run();
}

void BaseDatapath::memoryAmbiguation() {
  auto opt = getGraphOpt<MemoryAmbiguationOpt>();
  opt->run();
}

void BaseDatapath::removePhiNodes() {
  auto opt = getGraphOpt<PhiNodeRemoval>();
  opt->run();
}

void BaseDatapath::loopFlatten() {
  auto opt = getGraphOpt<LoopFlattening>();
  opt->run();
}

void BaseDatapath::removeInductionDependence() {
  auto opt = getGraphOpt<InductionDependenceRemoval>();
  opt->run();
}

void BaseDatapath::loopPipelining() {
  auto opt = getGraphOpt<GlobalLoopPipelining>();
  opt->run();
}

void BaseDatapath::perLoopPipelining() {
  auto opt = getGraphOpt<PerLoopPipelining>();
  opt->run();
}

void BaseDatapath::loopUnrolling() {
  auto opt = getGraphOpt<LoopUnrolling>();
  opt->run();
}

void BaseDatapath::fuseRegLoadStores() {
  auto opt = getGraphOpt<RegLoadStoreFusion>();
  opt->run();
}

void BaseDatapath::fuseConsecutiveBranches() {
  auto opt = getGraphOpt<ConsecutiveBranchFusion>();
  opt->run();
}

void BaseDatapath::removeSharedLoads() {
  auto opt = getGraphOpt<LoadBuffering>();
  opt->run();
}

void BaseDatapath::storeBuffer() {
  auto opt = getGraphOpt<StoreBuffering>();
  opt->run();
}

void BaseDatapath::removeRepeatedStores() {
  auto opt = getGraphOpt<RepeatedStoreRemoval>();
  opt->run();
}

void BaseDatapath::treeHeightReduction() {
  auto opt = getGraphOpt<TreeHeightReduction>();
  opt->run();
}
