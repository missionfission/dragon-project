class DDFG_Node:
    def __init__(self, operator, attributes, inputs, outputs, scope):
        self.operator = operator  # type of stencil, control, loop, mult, add, specialops etc
        self.attributes = attributes
        self.inputs = inputs
        self.outputs = outputs
        self.scope = scope

        self.compute_expense = 0
        self.static_inputs = 0
        self.write_access = 0
        self.read_access = 0

        self.mem_util = 0  # in_edge_mem including weights
        self.mem_fetch = 0

        self.in_edge_mem = 0 # only from nodes, not static inputs
        self.out_edge_mem = 0
        self.n_in_edges = 1
        self.n_out_edges = 1

        self.prev = None # [list of where the input comes from]
        self.next = None # [list of where the output gets next]
        
    @property
    def operator(self):
        return self._operator

    @operator.setter
    def operator(self, operator):
        self._operator = operator.lower()

    @property
    def attributes(self):
        return self._attributes

    @attributes.setter
    def attributes(self, attributes):
        self._attributes = attributes

    @property
    def inputs(self):
        return self._inputs

    @inputs.setter
    def inputs(self, inputs):
        self._inputs = inputs

    @property
    def outputs(self):
        return self._outputs

    @outputs.setter
    def outputs(self, outputs):
        self._outputs = outputs

    @property
    def scope(self):
        return self._scope

    @scope.setter
    def scope(self, scope):
        self._scope = scope

    def get_stats(self):
        return self.compute_expense, self.weights

    def __repr__(self):

        # text = ", ".join([str(v) for v in self.outputs])
        # text += " = " + self.operator
        # if self.attributes:
        #     text += (
        #         "["
        #         + ", ".join(
        #             [str(k) + " = " + str(v) for k, v in self.attributes.items()]
        #         )
        #         + "]"
        #     )
        # text += "(" + ", ".join([str(v) for v in self.inputs]) + ")"
        return self.operator
