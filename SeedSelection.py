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
        ### temp_bound_index: (list) the bound to exclude the impossible budget combination s.t. the k-budget is smaller than the temp bound
        temp_bound_index = [0 for _ in range(self.num_product)]

        diff = Diffusion(self.graph_dict, self.product_list, self.product_weight_list)
        while bud_index != bud_bound_index:
            ### bud_pmis: (float) the budget in this pmis execution
            bud_pmis = sum(c_matrix[k][bud_index[k]] for k in range(self.num_product))

            if bud_pmis <= bud:
                temp_bound_flag = True
                for k in range(self.num_product):
                    if temp_bound_index[k] > bud_index[k]:
                        temp_bound_flag = False
                        break
                if temp_bound_flag:
                    temp_bound_index = bud_index.copy()

                    s_set = [s_matrix[k][bud_index[k]][k].copy() for k in range(self.num_product)]
                    ep = round(sum([diff.getSeedSetProfit(s_set) for _ in range(self.monte)]) / self.monte, 4)

                    if ep > mep_result[0]:
                        mep_result = (ep, s_set)

            pointer = self.num_product - 1
            while bud_index[pointer] == bud_bound_index[pointer]:
                bud_index[pointer] = len(c_matrix[pointer]) - 1
                pointer -= 1
            bud_index[pointer] -= 1

        return mep_result