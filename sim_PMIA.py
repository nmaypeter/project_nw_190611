from SeedSelection import *
from SeedSelection_PMIA import *
from Evaluation import *
import time


bi = 0

dataset_name = 'toy'
product_name = 'item_hplc'
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
sspmia = SeedSelectionPMIA(graph_dict, seed_cost_dict, product_list)

# -- initialization for each sample --
now_budget = 0.0
seed_seq = [[] for _ in range(num_product)]
mioa_dict, miia_dict = sspmia.generateMIA(seed_seq)
mioa_dict, miia_dict = [mioa_dict] * num_product, [miia_dict] * num_product

inc_inf_dict = {(k, v): 0.0 for k in range(num_product) for v in mioa_dict[k]}
ap_dict, alpha_dict, is_dict = [{} for _ in range(num_product)], [{} for _ in range(num_product)], [{} for _ in range(num_product)]
for k in range(num_product):
    for v in seed_cost_dict[k]:
        if v in miia_dict[k]:
            ap_dict[k][v], alpha_dict[k][v] = {u: 0.0 for u in miia_dict[k][v]}, {}
            ap_dict[k][v][v] = 1.0
            alpha_dict[k][v] = sspmia.updateAlpha(v, seed_seq[k], miia_dict[k][v], ap_dict[k][v])
            for u in miia_dict[k][v]:
                inc_inf_dict[(k, u)] += alpha_dict[k][v][u] * (1 - ap_dict[k][v][u]) * product_list[k][0]
        else:
            ap_dict[k][v], alpha_dict[k][v] = {v: 1.0}, {v: 0.0}

pmioa_dict, pmiia_dict = copy.deepcopy(mioa_dict), copy.deepcopy(miia_dict)
print(round(time.time() - start_time, 4), len(inc_inf_dict))
inc_inf_dict = {(k, v): inc_inf_dict[(k, v)] for k in range(num_product) for v in mioa_dict[k] if round(inc_inf_dict[(k, v)], 4) > 0.0}
while now_budget < total_budget and inc_inf_dict:
    mep_item = max(inc_inf_dict, key=inc_inf_dict.get)
    mep_k_prod, mep_i_node = mep_item
    inc_inf = inc_inf_dict[mep_item]
    del inc_inf_dict[(mep_k_prod, mep_i_node)]
    sc = seed_cost_dict[mep_k_prod][mep_i_node]

    if round(now_budget + sc, 4) > total_budget or round(inc_inf, 4) <= 0:
        continue

    mioa_dict[mep_k_prod] = pmioa_dict[mep_k_prod]
    mioa_u_set = set(mioa_dict[mep_k_prod][mep_i_node]).difference(seed_seq[mep_k_prod]) if mep_i_node in mioa_dict[mep_k_prod] else set()
    mioa_u_set.add(mep_i_node)
    for v in mioa_u_set:
        miia_v_set = set(miia_dict[mep_k_prod][v]).difference(seed_seq[mep_k_prod]) if v in miia_dict[mep_k_prod] else set()
        miia_v_set.add(mep_i_node)
        for w in miia_v_set:
            if (mep_k_prod, w) in inc_inf_dict:
                inc_inf_dict[(mep_k_prod, w)] -= alpha_dict[mep_k_prod][v][w] * (1 - ap_dict[mep_k_prod][v][w]) * product_list[mep_k_prod][0]

    print(len(inc_inf_dict), mep_item)
    seed_seq[mep_k_prod].append(mep_i_node)
    now_budget = round(now_budget + sc, 4)
    print(round(time.time() - start_time, 4), now_budget, [len(seed_seq[k]) for k in range(num_product)])

    pmioa_dict[mep_k_prod], pmiia_dict[mep_k_prod] = sspmia.generateMIA(seed_seq)
    miia_dict[mep_k_prod] = sspmia.updatePMIIA(seed_seq[mep_k_prod], pmiia_dict[mep_k_prod])

    mioa_u_set = set(mioa_dict[mep_k_prod][mep_i_node]).difference(seed_seq[mep_k_prod]) if mep_i_node in mioa_dict[mep_k_prod] else set()
    for v in mioa_u_set:
        ap_dict[mep_k_prod][v] = sspmia.updateAP(v, seed_seq[mep_k_prod], miia_dict[mep_k_prod][v])
        alpha_dict[mep_k_prod][v] = sspmia.updateAlpha(v, seed_seq, miia_dict[mep_k_prod][v], ap_dict[mep_k_prod][v])
        miia_v_set = set(miia_dict[mep_k_prod][v]).difference(seed_seq[mep_k_prod]) if v in miia_dict[mep_k_prod] else set()
        for w in miia_v_set:
            if (mep_k_prod, w) in inc_inf_dict:
                inc_inf_dict[(mep_k_prod, w)] += alpha_dict[mep_k_prod][v][w] * (1 - ap_dict[mep_k_prod][v][w]) * product_list[mep_k_prod][0]

seed_set = [set(seed_seq[k]) for k in range(num_product)]
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