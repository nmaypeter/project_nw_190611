from SeedSelection_MDAG import *
from Diffusion import *
from Evaluation import *
import time


bi = 0

dataset_name = 'toy2'
product_name = 'item_lphc'
cascade_model = 'ic'
distribution_type = ''
eva_monte_carlo = 100

ini = Initialization(dataset_name, product_name)
seed_cost_dict = ini.constructSeedCostDict()
graph_dict = ini.constructGraphDict(cascade_model)
product_list = ini.constructProductList()
num_product = len(product_list)
product_weight_list = getProductWeight(product_list, distribution_type)
total_cost = sum(seed_cost_dict[k][i] for i in seed_cost_dict[0] for k in range(num_product))
print([round(total_cost / 2**ii, 4) for ii in range(10, bi - 1, -1)])
total_budget = round(total_cost / 2**bi, 4)


# -- initialization for each budget --
start_time = time.time()
ssmdag = SeedSelectionMDAG(graph_dict, seed_cost_dict, product_list)

sub_graph, topological_list = ssmdag.generateDAG()
print(round(time.time() - start_time, 4))