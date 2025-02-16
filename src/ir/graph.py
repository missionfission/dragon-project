import torch

class Graph:
    def __init__(self, name, variables, inputs, outputs, nodes, fx_graph=None):
        self.name = name
        self.variables = variables
        self.inputs = inputs
        self.outputs = outputs
        self.nodes = nodes
        self._fx_graph = fx_graph
        
        # Store edges as (source_node, target_node, edge_data)
        self._edges = []
        if fx_graph:
            self._extract_edges_from_fx()

    def _extract_edges_from_fx(self):
        """Extract edges from the FX graph structure"""
        self._edges = []
        node_map = {node.name: node for node in self.nodes}
        
        # Iterate through FX nodes to get edges
        for fx_node in self._fx_graph.nodes:
            if fx_node.op in ['output', 'placeholder']:
                continue
                
            target_node = node_map.get(fx_node.name)
            if not target_node:
                continue
                
            # Get edges from arguments
            for arg in fx_node.args:
                if not isinstance(arg, torch.fx.Node):
                    continue
                    
                source_node = node_map.get(arg.name)
                if source_node:
                    edge_data = {
                        'memory': target_node.in_edge_mem,
                        'variable': target_node.inputs[0] if target_node.inputs else None,
                        'fx_node': fx_node  # Store reference to FX node
                    }
                    self._edges.append((source_node, target_node, edge_data))

    @property
    def edges(self):
        """Get list of edges in the graph"""
        return self._edges

    def get_node_predecessors(self, node):
        """Get all predecessor nodes of the given node"""
        return [edge[0] for edge in self._edges if edge[1] == node]

    def get_node_successors(self, node):
        """Get all successor nodes of the given node"""
        return [edge[1] for edge in self._edges if edge[0] == node]

    def get_edge_data(self, source_node, target_node):
        """Get edge data between source and target nodes if it exists"""
        for edge in self._edges:
            if edge[0] == source_node and edge[1] == target_node:
                return edge[2]
        return None

    def get_in_degree(self, node):
        """Get number of incoming edges to node"""
        return len([edge for edge in self._edges if edge[1] == node])

    def get_out_degree(self, node):
        """Get number of outgoing edges from node"""
        return len([edge for edge in self._edges if edge[0] == node])

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        self._name = name

    @property
    def variables(self):
        return self._variables

    @variables.setter
    def variables(self, variables):
        self._variables = variables

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
    def nodes(self):
        return self._nodes

    @nodes.setter
    def nodes(self, nodes):
        self._nodes = nodes
        # Re-extract edges if FX graph exists
        if self._fx_graph:
            self._extract_edges_from_fx()

    def __repr__(self):
        text = self.name
        text += " (" + "\n"
        text += ",\n".join(["\t" + str(v) for v in self.inputs]) + "\n"
        text += "):" + "\n"
        text += "\n".join(["\t" + str(x) for x in self.nodes]) + "\n"
        text += "\t" + "return " + ", ".join([str(v) for v in self.outputs])
        return text
