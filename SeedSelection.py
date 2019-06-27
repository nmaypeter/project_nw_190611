from Diffusion import *
import heap
import random


class SeedSelectionNAPG:
    def __init__(self, graph_dict, seed_cost_dict, product_list, product_weight_list, r_flag, epw_flag):
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
        self.epw_flag = epw_flag

    def generateCelfHeap(self):
        # -- calculate expected profit for all combinations of nodes and products --
        ### celf_item: (list) (mg, k_prod, i_node, flag)
        celf_heap = []

        diffap = DiffusionAccProb(self.graph_dict, self.product_list, self.product_weight_list, self.epw_flag)
        if not self.epw_flag:
            for i in self.graph_dict:
                i_dict = diffap.buildNodeExpectedInfDict({i}, 0, i, 1)
                ei = getExpectedInf(i_dict)

                if ei > 0:
                    for k in range(self.num_product):
                        mg = round(ei * self.product_list[k][0] * self.product_weight_list[k], 4)
                        if self.r_flag:
                            mg = safe_div(mg, self.seed_cost_dict[k][i])
                        celf_item = (mg, k, i, 0)
                        heap.heappush_max(celf_heap, celf_item)
        else:
            for k in range(self.num_product):
                for i in self.graph_dict:
                    i_dict = diffap.buildNodeExpectedInfDict({i}, k, i, 1)
                    ei = getExpectedInf(i_dict)

                    if ei > 0:
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
        self.monte = 100

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
        senpai_list = []

        diff = Diffusion(self.graph_dict, self.product_list, self.product_weight_list)
        while bud_index != bud_bound_index:
            ### bud_pmis: (float) the budget in this pmis execution
            bud_pmis = sum(c_matrix[k][bud_index[k]] for k in range(self.num_product))

            if bud_pmis <= bud:
                seed_set_flag = True
                if senpai_list:
                    for senpai_item in senpai_list:
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
                    senpai_list.append(bud_index.copy())
                    s_set = [s_matrix[k][bud_index[k]][k].copy() for k in range(self.num_product)]
                    ep = round(sum([diff.getSeedSetProfit(s_set) for _ in range(self.monte)]) / self.monte, 4)

                    if ep > mep_result[0]:
                        mep_result = (ep, s_set)

            pointer = self.num_product - 1
            while bud_index[pointer] == bud_bound_index[pointer]:
                bud_index[pointer] = len(c_matrix[pointer]) - 1
                pointer -= 1
            bud_index[pointer] -= 1

        return mep_result[1]


def updateSeedInfluence(s_inf_dict, i_node, mioa_new_seed):
    if i_node in s_inf_dict:
        del s_inf_dict[i_node]

    for i in mioa_new_seed:
        if i in s_inf_dict:
            s_inf_dict[i].append(mioa_new_seed[i])
        else:
            s_inf_dict[i] = [mioa_new_seed[i]]


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
        self.prob_threshold = 0.01

    def generateMIA(self):
        mioa_dict, miia_dict = {}, {}

        for source_node in self.graph_dict:
            ### source_dict[i_node] = (prob, in-neighbor)
            mioa_dict[source_node] = {}
            source_dict = {source_node: 1.0}
            source_heap = []
            for i in self.graph_dict[source_node]:
                source_dict[i] = self.graph_dict[source_node][i]
                heap.heappush_max(source_heap, (self.graph_dict[source_node][i], i))

            # -- it will not find a better path than the existing MIP --
            # -- because if this path exists, it should be pop earlier from the heap. --
            while source_heap:
                (i_prob, i_node) = heap.heappop_max(source_heap)

                mioa_dict[source_node][i_node] = i_prob
                if i_node not in miia_dict:
                    miia_dict[i_node] = {source_node}
                else:
                    miia_dict[i_node].add(source_node)

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
                                        source_dict[ii_node] = ii_prob
                                        source_heap.remove((ii_prob_d, ii_node))
                                        source_heap.append((ii_prob, ii_node))
                                        heap.heapify_max(source_heap)
                                # -- if ii_node is not in heap --
                                else:
                                    source_dict[ii_node] = ii_prob
                                    heap.heappush_max(source_heap, (ii_prob, ii_node))

        return mioa_dict, miia_dict


class SeedSelectionDAG:
    def __init__(self, graph_dict, seed_cost_dict, product_list):
        ### graph_dict: (dict) the graph
        ### seed_cost_dict: (dict) the set of cost for seeds
        ### product_list: (list) the set to record products [k's profit, k's cost, k's price]
        ### num_product: (int) the kinds of products
        self.graph_dict = graph_dict
        self.seed_cost_dict = seed_cost_dict
        self.product_list = product_list
        self.num_product = len(product_list)
        self.prob_threshold = 0.01

    def generateMIOA(self):
        mioa_dict = {}

        for source_node in self.graph_dict:
            ### source_dict[i_node] = (prob, in-neighbor)
            mioa_dict[source_node] = {}
            source_dict = {source_node: 1.0}
            source_heap = []
            for i in self.graph_dict[source_node]:
                source_dict[i] = self.graph_dict[source_node][i]
                heap.heappush_max(source_heap, (self.graph_dict[source_node][i], i))

            # -- it will not find a better path than the existing MIP --
            # -- because if this path exists, it should be pop earlier from the heap. --
            while source_heap:
                (i_prob, i_node) = heap.heappop_max(source_heap)

                mioa_dict[source_node][i_node] = i_prob

                if i_node in self.graph_dict:
                    for ii_node in self.graph_dict[i_node]:
                        # -- not yet find MIP from source_node to ii_node --
                        if ii_node not in mioa_dict[source_node]:
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

        return mioa_dict

    def generateSeedExpectedInfDictUsingDAG1(self, k_prod, s_set):
        node_rank_dict = {}
        source_dict = {s_node: 1.0 for s_node in s_set[k_prod]}
        source_heap = [(1.0, s_node) for s_node in s_set[k_prod]]

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

        s_total_set = set(s for k in range(self.num_product) for s in s_set[k])
        dag_dict = {}
        for i in self.graph_dict:
            if i in node_rank_dict:
                for j in self.graph_dict[i]:
                    if j in node_rank_dict and j not in s_total_set:
                        if node_rank_dict[i] > node_rank_dict[j]:
                            if i not in dag_dict:
                                dag_dict[i] = {}
                            dag_dict[i][j] = self.graph_dict[i][j]

        seed_exp_inf_dict = {}
        for s_node in s_set[k_prod]:
            if s_node in dag_dict:
                ### source_dict[i_node] = (prob, in-neighbor)
                mioa_set = set()
                s_node_dict = {s_node: 1.0}
                source_heap = []
                for i in dag_dict[s_node]:
                    s_node_dict[i] = dag_dict[s_node][i]
                    heap.heappush_max(source_heap, (dag_dict[s_node][i], i))

                # -- it will not find a better path than the existing MIP --
                # -- because if this path exists, it should be pop earlier from the heap. --
                while source_heap:
                    (i_prob, i_node) = heap.heappop_max(source_heap)

                    mioa_set.add(i_node)
                    if i_node in seed_exp_inf_dict:
                        seed_exp_inf_dict[i_node].append(i_prob)
                    else:
                        seed_exp_inf_dict[i_node] = [i_prob]

                    if i_node in dag_dict:
                        for ii_node in dag_dict[i_node]:
                            # -- not yet find MIP from source_node to ii_node --
                            if ii_node not in mioa_set:
                                ii_prob = round(i_prob * dag_dict[i_node][ii_node], 4)

                                if ii_prob >= self.prob_threshold:
                                    # -- if ii_node is in heap --
                                    if ii_node in s_node_dict:
                                        ii_prob_d = s_node_dict[ii_node]
                                        if ii_prob > ii_prob_d:
                                            s_node_dict[ii_node] = ii_prob
                                            source_heap.remove((ii_prob_d, ii_node))
                                            source_heap.append((ii_prob, ii_node))
                                            heap.heapify_max(source_heap)
                                    # -- if ii_node is not in heap --
                                    else:
                                        s_node_dict[ii_node] = ii_prob
                                        heap.heappush_max(source_heap, (ii_prob, ii_node))

        return seed_exp_inf_dict

    def generateSeedExpectedInfDictUsingDAG2(self, k_prod, s_set, mioa_dict):
        node_rank_dict = {i: 0.0 for i in self.seed_cost_dict[0]}
        for s_node in s_set[k_prod]:
            node_rank_dict[s_node] = 1.0
            for i in mioa_dict[s_node]:
                if mioa_dict[s_node][i] > node_rank_dict[i]:
                    node_rank_dict[i] = mioa_dict[s_node][i]

        s_total_set = set(s for k in range(self.num_product) for s in s_set[k])
        seed_exp_inf_dict = {}
        for s_node in s_set[k_prod]:
            mioa_set = set()
            s_node_dict = {s_node: (1.0, s_node)}
            source_heap = []
            for i in self.graph_dict[s_node]:
                s_node_dict[i] = (self.graph_dict[s_node][i], s_node)
                heap.heappush_max(source_heap, (self.graph_dict[s_node][i], i))

            # -- it will not find a better path than the existing MIP --
            # -- because if this path exists, it should be pop earlier from the heap. --
            while source_heap:
                (i_prob, i_node) = heap.heappop_max(source_heap)
                i_prev = s_node_dict[i_node][1]

                if i_node in s_total_set:
                    continue
                if node_rank_dict[i_prev] <= node_rank_dict[i_node]:
                    continue

                mioa_set.add(i_node)
                if i_node in seed_exp_inf_dict:
                    seed_exp_inf_dict[i_node].append(i_prob)
                else:
                    seed_exp_inf_dict[i_node] = [i_prob]

                if i_node in self.graph_dict:
                    for ii_node in self.graph_dict[i_node]:
                        # -- not yet find MIP from source_node to ii_node --
                        if ii_node not in mioa_set:
                            ii_prob = round(i_prob * self.graph_dict[i_node][ii_node], 4)

                            if ii_prob >= self.prob_threshold:
                                # -- if ii_node is in heap --
                                if ii_node in s_node_dict:
                                    ii_prob_d = s_node_dict[ii_node][0]
                                    if ii_prob > ii_prob_d:
                                        s_node_dict[ii_node] = (ii_prob, i_node)
                                        source_heap.remove((ii_prob_d, ii_node))
                                        source_heap.append((ii_prob, ii_node))
                                        heap.heapify_max(source_heap)
                                # -- if ii_node is not in heap --
                                else:
                                    s_node_dict[ii_node] = (ii_prob, i_node)
                                    heap.heappush_max(source_heap, (ii_prob, ii_node))

        return seed_exp_inf_dict