import heap


def updateSeedMIOADict(s_set_mioa_dict, k_prod, i_node, s_set, seed_mioa_dict):
    s_total_set = set(s for k in range(len(s_set)) for s in s_set[k])
    if set() in s_set:
        del_list = [(k, s, i) for k in range(len(s_set_mioa_dict)) for s in s_set_mioa_dict[k] for i in s_set_mioa_dict[k][s] if i_node in s_set_mioa_dict[k][s][i][1]]
        while del_list:
            k, s, i = del_list.pop()
            del s_set_mioa_dict[k][s][i]
    else:
        s_set_mioa_dict = [{i: {j: s_set_mioa_dict[k][i][j] for j in s_set_mioa_dict[k][i] if i_node not in s_set_mioa_dict[k][i][j][1]} for i in s_set_mioa_dict[k]} for k in range(len(s_set_mioa_dict))]

    s_set_mioa_dict[k_prod][i_node] = {i: seed_mioa_dict[i] for i in seed_mioa_dict if not (s_total_set & set(seed_mioa_dict[i][1]))}

    return s_set_mioa_dict


def calculateExpectedInf(seed_exp_mioa_dict):
    exp_inf_dict = [{i: 0.0 for s in seed_exp_mioa_dict[k] for i in seed_exp_mioa_dict[k][s]} for k in range(len(seed_exp_mioa_dict))]
    for k in range(len(seed_exp_mioa_dict)):
        for s in seed_exp_mioa_dict[k]:
            for i in seed_exp_mioa_dict[k][s]:
                exp_inf_dict[k][i] = 1.0 - (1.0 - exp_inf_dict[k][i]) * (1.0 - seed_exp_mioa_dict[k][s][i][0])

    exp_inf = [round(sum(exp_inf_dict[k][i] for i in exp_inf_dict[k]), 4) for k in range(len(exp_inf_dict))]

    return exp_inf


class SeedSelectionMDAGEPW:
    def __init__(self, graph_dict, seed_cost_dict, product_list, product_weight_list):
        ### graph_dict: (dict) the graph
        ### seed_cost_dict: (dict) the set of cost for seeds
        ### product_list: (list) the set to record products [k's profit, k's cost, k's price]
        ### num_product: (int) the kinds of products
        self.graph_dict = graph_dict
        self.seed_cost_dict = seed_cost_dict
        self.product_list = product_list
        self.num_product = len(product_list)
        self.product_weight_list = product_weight_list
        self.prob_threshold = 0.001

    def generateMIOA(self):
        mioa_dict = {}

        for source_node in self.graph_dict:
            ### source_dict[i_node] = (prob, in-neighbor)
            mioa_dict[source_node] = {}
            source_dict = {i: (self.graph_dict[source_node][i], source_node) for i in self.graph_dict[source_node]}
            source_dict[source_node] = (1.0, source_node)
            source_heap = [(self.graph_dict[source_node][i], i) for i in self.graph_dict[source_node]]
            heap.heapify_max(source_heap)

            # -- it will not find a better path than the existing MIP --
            # -- because if this path exists, it should be pop earlier from the heap. --
            while source_heap:
                (i_prob, i_node) = heap.heappop_max(source_heap)
                i_prev = source_dict[i_node][1]

                # -- find MIP from source_node to i_node --
                i_path = [i_node, i_prev]
                while i_prev != source_node:
                    i_prev = source_dict[i_prev][1]
                    i_path.append(i_prev)
                i_path.pop()
                i_path.reverse()

                mioa_dict[source_node][i_node] = (i_prob, i_path)

                if i_node in self.graph_dict:
                    for ii_node in self.graph_dict[i_node]:
                        # -- not yet find MIP from source_node to ii_node --
                        if ii_node not in mioa_dict[source_node]:
                            ii_prob = round(i_prob * self.graph_dict[i_node][ii_node], 4)

                            if ii_prob >= self.prob_threshold:
                                # -- if ii_node is in heap --
                                if ii_node in source_dict:
                                    ii_prob_d = source_dict[ii_node][0]
                                    if ii_prob > ii_prob_d:
                                        source_dict[ii_node] = (ii_prob, i_node)
                                        source_heap.remove((ii_prob_d, ii_node))
                                        source_heap.append((ii_prob, ii_node))
                                        heap.heapify_max(source_heap)
                                # -- if ii_node is not in heap --
                                else:
                                    source_dict[ii_node] = (ii_prob, i_node)
                                    heap.heappush_max(source_heap, (ii_prob, ii_node))

        return mioa_dict

    def updateMIOAEPW(self, mioa_dict):
        mioa_dict = [{i: {j: (round(mioa_dict[i][j][0] * self.product_weight_list[k] ** len(mioa_dict[i][j][1]), 4), mioa_dict[i][j][1])
                          for j in mioa_dict[i]} for i in mioa_dict} for k in range(self.num_product)]
        mioa_dict = [{i: {j: mioa_dict[k][i][j] for j in mioa_dict[k][i] if mioa_dict[k][i][j][0] >= self.prob_threshold} for i in mioa_dict[k]} for k in range(self.num_product)]
        mioa_dict = [{i: mioa_dict[k][i] for i in mioa_dict[k] if mioa_dict[k][i]} for k in range(self.num_product)]

        return mioa_dict