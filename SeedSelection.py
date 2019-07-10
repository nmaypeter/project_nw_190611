from Diffusion import *
import heap
import random


class SeedSelectionNAPG:
    def __init__(self, graph_dict, seed_cost_dict, product_list, product_weight_list, r_flag):
        ### graph_dict: (dict) the graph
        ### seed_cost_dict[k][i]: (float4) the seed of i-node and k-item
        ### product_list: (list) the set to record products [k's profit, k's cost, k's price]
        ### num_product: (int) the kinds of products
        ### product_weight_list: (list) the product weight list
        self.graph_dict = graph_dict
        self.seed_cost_dict = seed_cost_dict
        self.product_list = product_list
        self.num_product = len(product_list)
        self.product_weight_list = product_weight_list
        self.r_flag = r_flag

    def generateCelfHeap(self):
        # -- calculate expected profit for all combinations of nodes and products --
        ### celf_item: (list) (mg, k_prod, i_node, flag)
        celf_heap = []

        diffap = DiffusionAccProb(self.graph_dict, self.product_list)
        for i in self.graph_dict:
            print(i)
            i_dict = diffap.buildNodeExpectedInfDict({i}, 0, i, 1)
            ei = getExpectedInf(i_dict)

            if ei > 0:
                for k in range(self.num_product):
                    mg = round(ei * self.product_list[k][0] * self.product_weight_list[k], 4)
                    if self.r_flag:
                        mg = safe_div(mg, self.seed_cost_dict[k][i])
                    celf_item = (mg, k, i, 0)
                    heap.heappush_max(celf_heap, celf_item)

        return celf_heap


def updateSeedNAPGDict(s_set_napg_dict, s_set_napg_seq, k_prod, i_node, s_set, seed_napg_seq):
    s_total_set = set(s for k in range(len(s_set)) for s in s_set[k])
    del_list = [(k, i_item) for k in range(len(s_set_napg_seq)) for i_item in s_set_napg_seq[k] if i_node in i_item[2]]

    while del_list:
        k, i_item = del_list.pop()
        i_node, i_prod, i_path = i_item
        s_set_napg_dict[k][i_node] = safe_div(s_set_napg_dict[k][i_node] - i_prod, 1.0 - i_prod)
        s_set_napg_seq[k].remove(i_item)

    new_seed_napg_seq = [i for i in seed_napg_seq if not (s_total_set & set(i[2]))]
    for i_item in new_seed_napg_seq:
        i_node, i_prod, i_path = i_item
        s_set_napg_dict[k_prod][i_node] = round(1.0 - (1.0 - s_set_napg_dict[k_prod][i_node]) * (1.0 - i_prod), 4)
    s_set_napg_seq[k_prod] += new_seed_napg_seq


class SeedSelectionNAPG2:
    def __init__(self, graph_dict, seed_cost_dict, product_list, product_weight_list, r_flag):
        ### graph_dict: (dict) the graph
        ### seed_cost_dict[k][i]: (float4) the seed of i-node and k-item
        ### product_list: (list) the set to record products [k's profit, k's cost, k's price]
        ### num_product: (int) the kinds of products
        ### product_weight_list: (list) the product weight list
        self.graph_dict = graph_dict
        self.seed_cost_dict = seed_cost_dict
        self.product_list = product_list
        self.num_product = len(product_list)
        self.product_weight_list = product_weight_list
        self.prob_threshold = 0.001
        self.r_flag = r_flag

    def generateExpectedInfDict(self):
        i_dict = {s: {i: 0.0 for i in self.seed_cost_dict[0]} for s in self.graph_dict}
        i_path_dict = {s: [] for s in self.graph_dict}

        for s in self.graph_dict:
            i_seq = [(ii, self.graph_dict[s][ii], [ii]) for ii in self.graph_dict[s] if self.graph_dict[s][ii] >= self.prob_threshold]
            while i_seq:
                for i_item in i_seq:
                    i_node, i_prod, i_path = i_item
                    i_dict[s][i_node] = round(1.0 - (1.0 - i_dict[s][i_node]) * (1.0 - i_prod), 4)
                    i_path_dict[s].append(i_item)

                i_seq = [i for i in i_seq if i[0] in self.graph_dict]
                i_seq = [(ii, round(i[1] * self.graph_dict[i[0]][ii], 4), i[2] + [ii]) for i in i_seq for ii in self.graph_dict[i[0]] if round(i[1] * self.graph_dict[i[0]][ii], 4) >= self.prob_threshold]

        return i_dict, i_path_dict

    def generateCelfHeap(self):
        # -- calculate expected profit for all combinations of nodes and products --
        ### celf_item: (list) (mg, k_prod, i_node, flag)
        celf_heap = []

        diffap = DiffusionAccProb2(self.graph_dict, self.seed_cost_dict, self.product_list)
        for i in self.graph_dict:
            print(i)
            ei = diffap.getExpectedInf([{i}])[0]

            if ei > 0:
                for k in range(self.num_product):
                    mg = round(ei * self.product_list[k][0] * self.product_weight_list[k], 4)
                    if self.r_flag:
                        mg = safe_div(mg, self.seed_cost_dict[k][i])
                    celf_item = (mg, k, i, 0)
                    heap.heappush_max(celf_heap, celf_item)

        return celf_heap


class SeedSelectionNG:
    def __init__(self, graph_dict, seed_cost_dict, product_list, product_weight_list, r_flag):
        ### graph_dict: (dict) the graph
        ### seed_cost_dict[k][i]: (float4) the seed of i-node and k-item
        ### product_list: (list) the set to record products [k's profit, k's cost, k's price]
        ### num_product: (int) the kinds of products
        ### product_weight_list: (list) the product weight list
        ### monte: (int) monte carlo times
        self.graph_dict = graph_dict
        self.seed_cost_dict = seed_cost_dict
        self.product_list = product_list
        self.num_product = len(product_list)
        self.product_weight_list = product_weight_list
        self.r_flag = r_flag
        self.monte = 100

    def generateCelfHeap(self):
        # -- calculate expected profit for all combinations of nodes and products --
        ### celf_item: (list) (mg, k_prod, i_node, flag)
        celf_heap = []

        diff = Diffusion(self.graph_dict, self.product_list, self.product_weight_list)
        for i in self.graph_dict:
            s_set = [set() for _ in range(self.num_product)]
            s_set[0].add(i)
            ep = round(sum([diff.getSeedSetProfit(s_set) for _ in range(self.monte)]) / self.monte, 4)

            if ep > 0:
                for k in range(self.num_product):
                    mg = safe_div(ep * self.product_list[k][0] * self.product_weight_list[k], self.product_list[0][0] * self.product_weight_list[0])
                    if self.r_flag:
                        mg = safe_div(mg, self.seed_cost_dict[k][i])
                    celf_item = (mg, k, i, 0)
                    heap.heappush_max(celf_heap, celf_item)

        return celf_heap


class SeedSelectionHD:
    def __init__(self, graph_dict, product_list):
        ### graph_dict: (dict) the graph
        ### product_list: (list) the set to record products [k's profit, k's cost, k's price]
        ### num_product: (int) the kinds of products
        self.graph_dict = graph_dict
        self.product_list = product_list
        self.num_product = len(product_list)

    def generateDegreeHeap(self):
        degree_heap = []

        for i in self.graph_dict:
            deg = len(self.graph_dict[i])
            for k in range(self.num_product):
                degree_item = (int(deg), k, i)
                heap.heappush_max(degree_heap, degree_item)

        return degree_heap


class SeedSelectionRandom:
    def __init__(self, graph_dict, product_list):
        ### graph_dict: (dict) the graph
        ### product_list: (list) the set to record products [k's profit, k's cost, k's price]
        ### num_product: (int) the kinds of products
        self.graph_dict = graph_dict
        self.product_list = product_list
        self.num_product = len(product_list)

    def generateRandomList(self):
        rn_list = [(k, i) for i in self.graph_dict for k in range(self.num_product)]
        random.shuffle(rn_list)

        return rn_list


class SeedSelectionPMIS:
    def __init__(self, graph_dict, seed_cost_dict, product_list, product_weight_list):
        ### graph_dict: (dict) the graph
        ### seed_cost_dict[k][i]: (float4) the seed of i-node and k-item
        ### product_list: (list) the set to record products [k's profit, k's cost, k's price]
        ### num_product: (int) the kinds of products
        ### product_weight_list: (list) the product weight list
        ### monte: (int) monte carlo times
        self.graph_dict = graph_dict
        self.seed_cost_dict = seed_cost_dict
        self.product_list = product_list
        self.num_product = len(product_list)
        self.product_weight_list = product_weight_list
        self.monte = 10

    def generateCelfHeap(self):
        # -- calculate expected profit for all combinations of nodes and products --
        ### celf_item: (list) (mg, k_prod, i_node, flag)
        celf_heap = [[] for _ in range(self.num_product)]

        diff_ss = Diffusion(self.graph_dict, self.product_list, self.product_weight_list)
        for i in self.graph_dict:
            s_set = [set() for _ in range(self.num_product)]
            s_set[0].add(i)
            ep = round(sum([diff_ss.getSeedSetProfit(s_set) for _ in range(self.monte)]) / self.monte, 4)

            if ep > 0:
                for k in range(self.num_product):
                    mg = safe_div(ep * self.product_list[k][0], self.product_list[0][0])
                    celf_item = (mg, k, i, 0)
                    heap.heappush_max(celf_heap[k], celf_item)

        return celf_heap

    def solveMultipleChoiceKnapsackProblem(self, bud, s_matrix, c_matrix):
        mep_result = (0.0, [set() for _ in range(self.num_product)])
        ### bud_index: (list) the using budget index for products
        ### bud_bound_index: (list) the bound budget index for products
        bud_index, bud_bound_index = [len(k) - 1 for k in c_matrix], [0 for _ in range(self.num_product)]
        MCKP_list = []

        diff = Diffusion(self.graph_dict, self.product_list, self.product_weight_list)
        while bud_index != bud_bound_index:
            ### bud_pmis: (float) the budget in this pmis execution
            bud_pmis = sum(c_matrix[k][bud_index[k]] for k in range(self.num_product))

            if bud_pmis <= bud:
                seed_set_flag = True
                if MCKP_list:
                    for senpai_item in MCKP_list:
                        compare_list_flag = True
                        for b_index in bud_index:
                            senpai_index = senpai_item[bud_index.index(b_index)]
                            if b_index > senpai_index:
                                compare_list_flag = False
                                break

                        if compare_list_flag:
                            seed_set_flag = False
                            break

                if seed_set_flag:
                    MCKP_list.append(bud_index.copy())

            pointer = self.num_product - 1
            while bud_index[pointer] == bud_bound_index[pointer]:
                bud_index[pointer] = len(c_matrix[pointer]) - 1
                pointer -= 1
            bud_index[pointer] -= 1

        while MCKP_list:
            bud_index = MCKP_list.pop(0)

            s_set = [s_matrix[k][bud_index[k]][k].copy() for k in range(self.num_product)]
            ep = round(sum([diff.getSeedSetProfit(s_set) for _ in range(self.monte)]) / self.monte, 4)

            if ep > mep_result[0]:
                mep_result = (ep, s_set)

        return mep_result[1]


def updateSeedMIOADict(s_set_mioa_dict, k_prod, i_node, s_set, seed_mioa_dict):
    s_total_set = set(s for k in range(len(s_set)) for s in s_set[k])
    del_list = []

    for k in range(len(s_set_mioa_dict)):
        for s in s_set_mioa_dict[k]:
            for i in s_set_mioa_dict[k][s]:
                if i_node in s_set_mioa_dict[k][s][i][1]:
                    del_list.append((k, s, i))

    while del_list:
        k, s, i = del_list.pop()
        del s_set_mioa_dict[k][s][i]

    s_set_mioa_dict[k_prod][i_node] = {i: seed_mioa_dict[i] for i in seed_mioa_dict if not (s_total_set & set(seed_mioa_dict[i][1]))}


def updateSeedDAGDict(s_set_dag_dict, k_prod, i_node):
    del_list = []

    k_list = [k for k in range(len(s_set_dag_dict)) if k != k_prod]
    for k in k_list:
        for s in s_set_dag_dict[k]:
            for i in s_set_dag_dict[k][s]:
                if i_node in s_set_dag_dict[k][s][i][1]:
                    del_list.append((k, s, i))

    while del_list:
        k, s, i = del_list.pop()
        del s_set_dag_dict[k][s][i]


def calculateExpectedInf(seed_exp_mioa_dict):
    exp_inf_dict = [{i: 0.0 for s in seed_exp_mioa_dict[k] for i in seed_exp_mioa_dict[k][s]} for k in range(len(seed_exp_mioa_dict))]
    for k in range(len(seed_exp_mioa_dict)):
        for s in seed_exp_mioa_dict[k]:
            for i in seed_exp_mioa_dict[k][s]:
                exp_inf_dict[k][i] = 1.0 - (1.0 - exp_inf_dict[k][i]) * (1.0 - seed_exp_mioa_dict[k][s][i][0])

    exp_inf = [round(sum(exp_inf_dict[k][i] for i in exp_inf_dict[k]), 4) for k in range(len(exp_inf_dict))]

    return exp_inf


class SeedSelectionMIOA:
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

    def generateDAG1(self, s_set_k):
        node_rank_dict = {}
        source_dict = {s_node: 1.0 for s_node in s_set_k}
        source_heap = [(1.0, s_node) for s_node in s_set_k]

        # -- it will not find a better path than the existing MIP --
        # -- because if this path exists, it should be pop earlier from the heap. --
        while source_heap:
            (i_prob, i_node) = heap.heappop_max(source_heap)

            node_rank_dict[i_node] = i_prob

            if i_node in self.graph_dict:
                for ii_node in self.graph_dict[i_node]:
                    # -- not yet find MIP from source_node to ii_node --
                    if ii_node not in node_rank_dict:
                        ii_prob = round(i_prob * self.graph_dict[i_node][ii_node], 4)

                        if ii_prob >= self.prob_threshold:
                            # -- if ii_node is in heap --
                            if ii_node in source_dict:
                                ii_prob_d = source_dict[ii_node]
                                if ii_prob > ii_prob_d:
                                    source_dict[ii_node] = ii_prob
                                    source_heap.remove((ii_prob_d, ii_node))
                                    source_heap.append((ii_prob, ii_node))
                                    heap.heapify_max(source_heap)
                            # -- if ii_node is not in heap --
                            else:
                                source_dict[ii_node] = ii_prob
                                heap.heappush_max(source_heap, (ii_prob, ii_node))

        dag_dict = {i: {} for i in self.graph_dict}
        i_set = set(i for i in self.graph_dict if i in node_rank_dict)
        for i in i_set:
            j_set = set(j for j in self.graph_dict[i] if j in node_rank_dict and node_rank_dict[i] > node_rank_dict[j])
            for j in j_set:
                dag_dict[i][j] = self.graph_dict[i][j]
            if not dag_dict[i]:
                del dag_dict[i]

        return dag_dict

    def generateDAG2(self, s_set_k, mioa_dict):
        node_rank_dict = {i: 0.0 for i in self.seed_cost_dict[0]}
        for s_node in s_set_k:
            node_rank_dict[s_node] = 1.0
            for i in mioa_dict[s_node]:
                if mioa_dict[s_node][i][0] > node_rank_dict[i]:
                    node_rank_dict[i] = mioa_dict[s_node][i][0]

        dag_dict = {i: {} for i in self.graph_dict}
        for s_node in s_set_k:
            for i in mioa_dict[s_node]:
                i_path = [s_node] + mioa_dict[s_node][i][1]
                for len_path in range(len(i_path) - 1):
                    i_node, ii_node = i_path[len_path], i_path[len_path + 1]
                    if ii_node not in dag_dict[i_node] and node_rank_dict[i_node] > node_rank_dict[ii_node]:
                        dag_dict[i_node][ii_node] = self.graph_dict[i_node][ii_node]

        for i in self.graph_dict:
            if not dag_dict[i]:
                del dag_dict[i]

        return dag_dict

    def generateSeedDAGDict(self, dag_dict, s_set_k):
        sdag_dict = {}

        s_set = set(s for s in s_set_k if s in dag_dict)
        for s_node in s_set:
            sdag_dict[s_node] = {}
            source_dict = {i: (dag_dict[s_node][i], s_node) for i in dag_dict[s_node]}
            source_dict[s_node] = (1.0, s_node)
            source_heap = [(dag_dict[s_node][i], i) for i in dag_dict[s_node]]
            heap.heapify_max(source_heap)

            # -- it will not find a better path than the existing MIP --
            # -- because if this path exists, it should be pop earlier from the heap. --
            while source_heap:
                (i_prob, i_node) = heap.heappop_max(source_heap)
                i_prev = source_dict[i_node][1]

                # -- find MIP from source_node to i_node --
                i_path = [i_node, i_prev]
                while i_prev != s_node:
                    i_prev = source_dict[i_prev][1]
                    i_path.append(i_prev)
                i_path.pop()
                i_path.reverse()

                sdag_dict[s_node][i_node] = (i_prob, i_path)

                if i_node in dag_dict:
                    for ii_node in dag_dict[i_node]:
                        # -- not yet find MIP from source_node to ii_node --
                        if ii_node not in sdag_dict[s_node]:
                            ii_prob = round(i_prob * dag_dict[i_node][ii_node], 4)

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

        return sdag_dict