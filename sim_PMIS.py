from SeedSelection import *
from Evaluation import *
import copy

bi = 0

dataset_name = 'toy2'
product_name = 'item_lphc'
cascade_model = 'ic'
distribution_type = ''
monte_carlo = 100
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
sspmis = SeedSelectionPMIS(graph_dict, seed_cost_dict, product_list, product_weight_list)
diff = Diffusion(graph_dict, product_list, product_weight_list)

s_matrix_sequence, c_matrix_sequence = [], []
celf_heap = sspmis.generateCelfHeap()

for k in range(num_product):
    # -- initialization for each sample --
    now_budget, now_profit = 0.0, 0.0
    seed_set = [set() for _ in range(num_product)]
    s_matrix, c_matrix = [[set() for _ in range(num_product)]], [0.0]

    while now_budget < total_budget and celf_heap[k]:
        mep_item = heap.heappop_max(celf_heap[k])
        mep_mg, mep_k_prod, mep_i_node, mep_flag = mep_item
        sc = seed_cost_dict[mep_k_prod][mep_i_node]
        seed_set_length = sum(len(seed_set[k]) for k in range(num_product))

        if round(now_budget + sc, 4) > total_budget:
            continue

        if mep_flag == seed_set_length:
            seed_set[mep_k_prod].add(mep_i_node)
            now_budget = round(now_budget + sc, 4)
            now_profit = round(sum([diff.getSeedSetProfit(seed_set) for _ in range(monte_carlo)]) / monte_carlo, 4)
            s_matrix.append(copy.deepcopy(seed_set))
            c_matrix.append(now_budget)
        else:
            seed_set_t = copy.deepcopy(seed_set)
            seed_set_t[mep_k_prod].add(mep_i_node)
            ep_t = round(sum([diff.getSeedSetProfit(seed_set_t) for _ in range(monte_carlo)]) / monte_carlo, 4)
            mg_t = round(ep_t - now_profit, 4)
            flag_t = seed_set_length

            if mg_t > 0:
                celf_item_t = (mg_t, mep_k_prod, mep_i_node, flag_t)
                heap.heappush_max(celf_heap[k], celf_item_t)

    s_matrix_sequence.append(s_matrix)
    c_matrix_sequence.append(c_matrix)

seed_set = sspmis.solveMultipleChoiceKnapsackProblem(total_budget, s_matrix_sequence, c_matrix_sequence)

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