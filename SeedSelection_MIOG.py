import heap


def updateSeedMIOGDict(s_set_miog_seq, k_prod, i_node, s_set, seed_miog_dict):
    s_total_set = set(s for k in range(len(s_set)) for s in s_set[k])
    #
    for k in range(len(s_set_miog_seq)):
        s_set_miog_seq[k] = [item for item in s_set_miog_seq[k] if i_node not in item[1]]
    # a = [item for item in seed_miog_dict if not (s_total_set & set(item[1]))]
    s_set_miog_seq[k_prod] += [item for item in seed_miog_dict if not (s_total_set & set(item[1]))]

    # # s_set_miog_dict[k_prod][i_node] = {i: seed_miog_dict[i] for i in seed_miog_dict if not (s_total_set & set(seed_miog_dict[i][1]))}
    # seed_miog_seq = [item for item in seed_miog_dict if not (s_total_set & set(item[1]))]
    # s_set_miog_dict[k_prod][i_node] = {item[1][-1]: [] for item in seed_miog_seq}
    # print()
    # for item in seed_miog_seq:




def calculateExpectedInf(seed_exp_mioa_dict):
    exp_inf_dict = [{i: 0.0 for s in seed_exp_mioa_dict[k] for i in seed_exp_mioa_dict[k][s]} for k in range(len(seed_exp_mioa_dict))]
    for k in range(len(seed_exp_mioa_dict)):
        for s in seed_exp_mioa_dict[k]:
            for i in seed_exp_mioa_dict[k][s]:
                exp_inf_dict[k][i] = 1.0 - (1.0 - exp_inf_dict[k][i]) * (1.0 - seed_exp_mioa_dict[k][s][i][0])

    exp_inf = [round(sum(exp_inf_dict[k][i] for i in exp_inf_dict[k]), 4) for k in range(len(exp_inf_dict))]

    return exp_inf


def calculateMIOGExpectedInf(seed_exp_miog_dict):
    exp_inf_dict = [{item[1][-1]: 0.0 for item in seed_exp_miog_dict[k]} for k in range(len(seed_exp_miog_dict))]
    for k in range(len(seed_exp_miog_dict)):
        for item in seed_exp_miog_dict[k]:
            exp_inf_dict[k][item[1][-1]] = 1.0 - (1.0 - exp_inf_dict[k][item[1][-1]]) * (1.0 - item[0])

    exp_inf = [round(sum(exp_inf_dict[k][i] for i in exp_inf_dict[k]), 4) for k in range(len(exp_inf_dict))]

    return exp_inf


class SeedSelectionMIOG:
    def __init__(self, graph_dict, seed_cost_dict, product_list):
        ### graph_dict: (dict) the graph
        ### seed_cost_dict: (dict) the set of cost for seeds
        ### product_list: (list) the set to record products [k's profit, k's cost, k's price]
        ### num_product: (int) the kinds of products
        self.graph_dict = graph_dict
        self.seed_cost_dict = seed_cost_dict
        self.product_list = product_list
        self.num_product = len(product_list)
        self.prob_threshold = 0.001

    def generateMIOG(self):
        miog_dict = {s: [] for s in self.graph_dict}

        for source_node in self.graph_dict:
            ### source_dict[i_node] = (prob, in-neighbor)
            source_dict = {i: self.graph_dict[source_node][i] for i in self.graph_dict[source_node]}
            source_dict[source_node] = 1.0
            source_heap = [(self.graph_dict[source_node][i], i, [i]) for i in self.graph_dict[source_node]]
            heap.heapify_max(source_heap)

            # -- it will not find a better path than the existing MIP --
            # -- because if this path exists, it should be pop earlier from the heap. --
            while source_heap:
                (i_prob, i_node, i_path) = heap.heappop_max(source_heap)

                if i_prob >= source_dict[i_node]:
                    miog_dict[source_node].append((i_prob, i_path))
                    # if i_prev not in miog_dict[source_node]:
                    #     miog_dict[source_node][i_prev] = [i_node]
                    # else:
                    #     miog_dict[source_node][i_prev].append(i_node)

                    if i_node in self.graph_dict:
                        for ii_node in self.graph_dict[i_node]:
                            # -- not yet find MIP from source_node to ii_node --
                            ii_prob = round(i_prob * self.graph_dict[i_node][ii_node], 4)
                            if ii_prob >= self.prob_threshold:
                                if ii_node in source_dict:
                                    if ii_prob >= source_dict[ii_node]:
                                        source_dict[ii_node] = ii_prob
                                        heap.heappush_max(source_heap, (ii_prob, ii_node, i_path + [ii_node]))
                                else:
                                    source_dict[ii_node] = ii_prob
                                    heap.heappush_max(source_heap, (ii_prob, ii_node, i_path + [ii_node]))

        return miog_dict