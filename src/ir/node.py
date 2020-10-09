class Node:
    def __init__(self, operator, attributes, inputs, outputs, scope):
        self.operator = operator
        self.attributes = attributes
        self.inputs = inputs
        self.outputs = outputs
        self.scope = scope
        self.compute_expense = 0
        self.weights = 0
        self.write_access = 0
        self.read_access = 0
        self.mem_util = 0  # out_edge_mem + read_access
        self.in_edge_mem = 0
        self.out_edge_mem = 0
        self.next = None

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
