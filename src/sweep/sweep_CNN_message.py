import sys
sys.path.append("/pool0/wynwyn/dragon-project/src/dlrm")
sys.path.append("/pool0/wynwyn/dragon-project/src")
from common_models import  resnet_18_graph, resnet_50_graph, bert_graph
from mapper.mappings import illusion_mapping
# bert_graph = bert_graph()
resnet_18_graph = resnet_18_graph()
# resnet_50_graph = resnet_50_graph()

chips = [6,6,8,12,20,32,55,96,171,308,559]
deeper = [2**x for x in range(11)]
capacity=[2,4,6,8,10,12,14,16,18,20,22]


print("{:>12} {:>12} {:>12} {:>12} {:>12}".format("message_cost", "avg_bound", "max_bound", "violate_avg_bound", "violate_max_bound"))

for i in range(11):
    illusion_mapping(resnet_18_graph, chips[i], deeper[i], 10**6*capacity[i], deeper= True, wider=False)

for i in range(11):
    illusion_mapping(resnet_18_graph, chips[i], deeper[i], 10**6*capacity[i], deeper= False, wider=True)

