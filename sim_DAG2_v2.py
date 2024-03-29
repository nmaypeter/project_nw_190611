from Diffusion import *
from SeedSelection_MIOA_v2 import *
from Evaluation import *
import time
import copy


r_flag = False

bi = 6

dataset_name = 'email'
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
ssmioa = SeedSelectionMIOA(graph_dict, seed_cost_dict, product_list)

mioa_dict = ssmioa.generateMIOA()
celf_heap = [(round(sum(mioa_dict[i][j][0] for j in mioa_dict[i]) * product_list[k][0] / (seed_cost_dict[k][i] if r_flag else 1.0), 4), k, i, 0) for k in range(num_product) for i in mioa_dict]
heap.heapify_max(celf_heap)

now_budget, now_profit = 0.0, 0.0
seed_set = [set() for _ in range(num_product)]
seed_dag_dict = [{} for _ in range(num_product)]
mep_seed_dag_dict = (celf_heap[0][0], [{} for _ in range(num_product)])
mep_seed_dag_dict[1][celf_heap[0][1]] = {celf_heap[0][2]: mioa_dict[celf_heap[0][2]]}

while now_budget < total_budget and celf_heap:
    mep_item = heap.heappop_max(celf_heap)
    mep_mg, mep_k_prod, mep_i_node, mep_flag = mep_item
    # print(mep_item)
    sc = seed_cost_dict[mep_k_prod][mep_i_node]
    seed_set_length = sum(len(seed_set[k]) for k in range(num_product))

    if round(now_budget + sc, 4) > total_budget:
        continue

    if mep_flag == seed_set_length:
        # print('new seed')
        print(mep_item)
        seed_set[mep_k_prod].add(mep_i_node)
        now_budget = round(now_budget + sc, 4)
        mep_mg *= sc if r_flag else 1.0
        now_profit = round(now_profit + mep_mg, 4)
        seed_dag_dict = mep_seed_dag_dict[1]
        mep_seed_dag_dict = (0.0, [{} for _ in range(num_product)])
    else:
        seed_exp_dag_dict = copy.deepcopy(seed_dag_dict)
        updateSeedDAGDict(seed_exp_dag_dict, mep_k_prod, mep_i_node)
        seed_set_t = seed_set[mep_k_prod].copy()
        seed_set_t.add(mep_i_node)
        dag_k_dict = ssmioa.generateDAG2(seed_set_t, {i: mioa_dict[i] for i in seed_set_t})
        seed_exp_dag_dict[mep_k_prod] = ssmioa.generateSeedDAGDict(dag_k_dict, seed_set_t)
        expected_inf = calculateExpectedInf(seed_exp_dag_dict)
        ep_t = sum(expected_inf[k] * product_list[k][0] for k in range(num_product))
        mg_t = round(ep_t - now_profit, 4)
        if r_flag:
            mg_t = safe_div(mg_t, sc)
        flag_t = seed_set_length

        if mg_t > 0:
            celf_item_t = (mg_t, mep_k_prod, mep_i_node, flag_t)
            heap.heappush_max(celf_heap, celf_item_t)
            if mg_t > mep_seed_dag_dict[0]:
                mep_seed_dag_dict = (mg_t, seed_exp_dag_dict)

distribution_type = 'm50e25'
eva = Evaluation(graph_dict, product_list)
iniW = IniWallet(dataset_name, product_name, distribution_type)
wallet_dict = iniW.constructWalletDict()

sample_pro_acc, sample_bud_acc = 0.0, 0.0
sample_sn_k_acc, sample_pnn_k_acc = [0.0 for _ in range(num_product)], [0 for _ in range(num_product)]
sample_pro_k_acc, sample_bud_k_acc = [0.0 for _ in range(num_product)], [0.0 for _ in range(num_product)]

for _ in range(eva_monte_carlo):
    pro_k_list, pnn_k_list = eva.getSeedSetProfit(seed_set, wallet_dict.copy())
    sample_pro_k_acc = [(pro_k + sample_pro_k) for pro_k, sample_pro_k in zip(pro_k_list, sample_pro_k_acc)]
    sample_pnn_k_acc = [(pnn_k + sample_pnn_k) for pnn_k, sample_pnn_k in zip(pnn_k_list, sample_pnn_k_acc)]
sample_pro_k_acc = [round(sample_pro_k / eva_monte_carlo, 4) for sample_pro_k in sample_pro_k_acc]
sample_pnn_k_acc = [round(sample_pnn_k / eva_monte_carlo, 4) for sample_pnn_k in sample_pnn_k_acc]
sample_bud_k_acc = [round(sum(seed_cost_dict[seed_set.index(sample_bud_k)][i] for i in sample_bud_k), 4) for sample_bud_k in seed_set]
sample_sn_k_acc = [len(sample_sn_k) for sample_sn_k in seed_set]
sample_pro_acc = round(sum(sample_pro_k_acc), 4)
sample_bud_acc = round(sum(sample_bud_k_acc), 4)

print('seed set: ' + str(seed_set))
print('profit: ' + str(sample_pro_acc))
print('budget: ' + str(sample_bud_acc))
print('seed number: ' + str(sample_sn_k_acc))
print('purchasing node number: ' + str(sample_pnn_k_acc))
print('ratio profit: ' + str(sample_pro_k_acc))
print('ratio budget: ' + str(sample_bud_k_acc))
print('total time: ' + str(round(time.time() - start_time, 2)) + 'sec')