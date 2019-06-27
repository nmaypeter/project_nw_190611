from SeedSelection_PMIA2 import *
from Initialization import *
import time


if __name__ == '__main__':
    start_time = time.time()
    dataset_name = 'toy'
    product_name = 'item_hplc'
    cascade_model = 'ic'

    ini = Initialization(dataset_name, product_name)
    seed_cost_dict = ini.constructSeedCostDict()
    graph_dict = ini.constructGraphDict(cascade_model)
    product_list = ini.constructProductList()

    sspmia = SeedSelectionPMIA(graph_dict, seed_cost_dict, product_list)

    seed_seq = []
    mioa_dict, miia_dict = sspmia.generateMIA(seed_seq)

    inc_inf_dict = {v: 0.0 for v in mioa_dict}
    ap_dict, alpha_dict, is_dict = {}, {}, {}
    for v in seed_cost_dict[0]:
        if v in miia_dict:
            ap_dict[v], alpha_dict[v], is_dict[v] = {u: 0.0 for u in miia_dict[v]}, {}, set()
            ap_dict[v][v] = 1.0
            alpha_dict[v] = sspmia.updateAlpha(v, seed_seq, miia_dict[v], ap_dict[v])
            for u in miia_dict[v]:
                inc_inf_dict[u] += alpha_dict[v][u] * (1 - ap_dict[v][u])
        else:
            ap_dict[v], alpha_dict[v], is_dict[v] = {v: 1.0}, {v: 0.0}, set()
    # for v in miia_dict:
    #     ap_dict[v], alpha_dict[v], is_dict[v] = {u: 0.0 for u in miia_dict[v]}, {}, set()
    #     alpha_dict[v] = sspmia.updateAlpha(v, seed_seq, miia_dict[v], ap_dict[v])
    #     for u in miia_dict[v]:
    #         inc_inf_dict[u] += alpha_dict[v][u] * (1 - ap_dict[v][u])

    pmioa_dict, pmiia_dict = mioa_dict, miia_dict
    exist_positive = lambda inc_dict: True if [i for i in inc_dict if round(inc_dict[i], 4) > 0.0 and i not in seed_seq] else False
    while exist_positive(inc_inf_dict):
    # while set(inc_inf_dict).difference(set(seed_seq)):
        u = max(inc_inf_dict, key=inc_inf_dict.get)
        print(u, inc_inf_dict[u])

        mioa_dict = pmioa_dict
        mioa_u_set = set(mioa_dict[u]).difference(seed_seq) if u in mioa_dict else set()
        mioa_u_set.add(u)
        for v in mioa_u_set:
            miia_v_set = set(miia_dict[v]).difference(seed_seq) if v in miia_dict else set()
            miia_v_set.add(u)
            for w in miia_v_set:
                inc_inf_dict[w] -= alpha_dict[v][w] * (1 - ap_dict[v][w])

        # sspmia.updateIS(is_dict, seed_seq, u, mioa_dict[u], miia_dict)
        seed_seq.append(u)

        pmioa_dict, pmiia_dict = sspmia.generateMIA(seed_seq)
        # miia_dict = sspmia.updatePMIIA(seed_seq, miia_dict, is_dict)
        miia_dict = sspmia.updatePMIIA(seed_seq, pmiia_dict)

        mioa_u_set2 = set(mioa_dict[u]).difference(seed_seq) if u in mioa_dict else set()
        for v in mioa_u_set2:
            ap_dict[v] = sspmia.updateAP(v, seed_seq, miia_dict[v])
            alpha_dict[v] = sspmia.updateAlpha(v, seed_seq, miia_dict[v], ap_dict[v])
            miia_v_set2 = set(miia_dict[v]).difference(seed_seq) if v in miia_dict else set()
            for w in miia_v_set2:
                inc_inf_dict[w] += alpha_dict[v][w] * (1 - ap_dict[v][w])