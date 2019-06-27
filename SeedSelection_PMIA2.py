import copy
import heap


def findNIn_set(i_node, miia):
    n_in_set = set()
    for i in miia:
        if i_node in miia[i][1] and i_node != miia[i][1][0]:
            # -- i_node_w is i_node_u's in-neighbor --
            i_node_w = miia[i][1][miia[i][1].index(i_node) - 1]
            n_in_set.add(i_node_w)

    return n_in_set


def induceGraph(graph, s_node):
    if s_node in graph:
        del graph[s_node]
    for i in graph:
        if s_node in graph[i]:
            del graph[i][s_node]

    del_list = [i for i in graph if not graph[i]]
    while del_list:
        del_node = del_list.pop()
        del graph[del_node]


class SeedSelectionPMIA:
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

    def generateMIA(self, s_seq):
        ### mioa_dict[i_node_u][i_node_v] = (prob, path): MIP from i_node_u to i_node_v
        ### miia_dict[i_node_v][i_node_u] = (prob, path): MIP from i_node_u to i_node_v
        mioa_dict, miia_dict = {}, {}
        sub_graph = copy.deepcopy(self.graph_dict)
        for s_node in s_seq:
            induceGraph(sub_graph, s_node)

        for source_node in sub_graph:
            ### source_dict[i_node] = (prob, in-neighbor)
            mioa_dict[source_node] = {}
            source_dict = {source_node: (1.0, source_node)}
            source_heap = []
            for i in sub_graph[source_node]:
                source_dict[i] = (sub_graph[source_node][i], source_node)
                heap.heappush_max(source_heap, (sub_graph[source_node][i], i))

            # -- it will not find a better path than the existing MIP --
            # -- because if this path exists, it should be pop earlier from the heap. --
            while source_heap:
                (i_prob, i_node) = heap.heappop_max(source_heap)
                i_prev = source_dict[i_node][1]

                # -- find MIP from source_node to i_node --
                i_path = [i_node, i_prev]
                while i_prev != source_dict[i_prev][1]:
                    i_prev = source_dict[i_prev][1]
                    i_path.append(i_prev)
                i_path.reverse()

                mioa_dict[source_node][i_node] = (i_prob, i_path)
                if i_node not in miia_dict:
                    miia_dict[i_node] = {source_node: (i_prob, i_path)}
                else:
                    miia_dict[i_node][source_node] = (i_prob, i_path)

                if i_node in sub_graph:
                    for ii_node in sub_graph[i_node]:
                        # -- not yet find MIP from source_node to ii_node --
                        if ii_node not in mioa_dict[source_node]:
                            ii_prob = round(i_prob * sub_graph[i_node][ii_node], 4)

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

        return mioa_dict, miia_dict

    def updateAlpha(self, i_node_v, s_seq, miia_v, ap_dict_v):
        alpha_v = {i_node_v: 1.0}
        miia_v_seq = sorted([i_node_u for i_node_u in miia_v], key=lambda i_node_u: len(miia_v[i_node_u][1]))
        for i_node_u in miia_v_seq:
            # -- i_node_w is i_node_u's out-neighbor --
            i_node_w = miia_v[i_node_u][1][miia_v[i_node_u][1].index(i_node_u) + 1]
            if i_node_w in s_seq:
                alpha_v[i_node_u] = 0.0
            else:
                n_in_set_w = findNIn_set(i_node_w, miia_v)
                n_in_set_w.remove(i_node_u)
                acc_prob = 1.0
                if n_in_set_w:
                    for nis_u_prime in n_in_set_w:
                        acc_prob *= (1 - ap_dict_v[nis_u_prime] * self.graph_dict[nis_u_prime][i_node_w])
                alpha_v[i_node_u] = round(alpha_v[i_node_w] * self.graph_dict[i_node_u][i_node_w] * acc_prob, 4)

        return alpha_v

    # @staticmethod
    # def updateIS(is_dict, s_seq, i_node_u, mioa_u, miia_dict):
    #     for i_node_v in mioa_u:
    #         for s_node in s_seq:
    #             if (s_node in miia_dict[i_node_v]) and (s_node not in is_dict[i_node_v]) and (i_node_u in miia_dict[i_node_v][s_node]):
    #                 is_dict[i_node_v].add(s_node)
    #
    # def updatePMIIA(self, s_set, miia_dict, is_dict):
    #     s_list = s_set.copy()
    #
    #     s_node = ''
    #     sub_graph = copy.deepcopy(self.graph_dict)
    #     mioa_dict = {}
    #     while s_list:
    #         if len(s_list) != len(s_set):
    #             induceGraph(sub_graph, s_node)
    #         s_node = s_list.pop(0)
    #
    #         mioa_dict[s_node] = {}
    #         source_dict = {s_node: (1.0, s_node)}
    #         source_heap = []
    #         if s_node in sub_graph:
    #             for i in sub_graph[s_node]:
    #                 source_dict[i] = (sub_graph[s_node][i], s_node)
    #                 heap.heappush_max(source_heap, (sub_graph[s_node][i], i))
    #
    #         while source_heap:
    #             (i_prob, i_node) = heap.heappop_max(source_heap)
    #             i_prev = source_dict[i_node][1]
    #
    #             # -- find MIP from source_node to i_node --
    #             i_path = [i_node, i_prev]
    #             while i_prev != source_dict[i_prev][1]:
    #                 i_prev = source_dict[i_prev][1]
    #                 i_path.append(i_prev)
    #             i_path.reverse()
    #
    #             mioa_dict[s_node][i_node] = (i_prob, i_path)
    #             if s_node not in is_dict[i_node]:
    #                 if i_node not in miia_dict:
    #                     miia_dict[i_node] = {}
    #                 miia_dict[i_node][s_node] = (i_prob, i_path)
    #
    #             if i_node in sub_graph:
    #                 for ii_node in sub_graph[i_node]:
    #                     # -- not yet find MIP from source_node to ii_node --
    #                     if ii_node not in mioa_dict[s_node]:
    #                         ii_prob = round(i_prob * sub_graph[i_node][ii_node], 4)
    #
    #                         if ii_prob >= self.prob_threshold:
    #                             # -- if ii_node is in heap --
    #                             if ii_node in source_dict:
    #                                 ii_prob_d = source_dict[ii_node][0]
    #                                 if ii_prob > ii_prob_d:
    #                                     source_dict[ii_node] = (ii_prob, i_node)
    #                                     source_heap.remove((ii_prob_d, ii_node))
    #                                     source_heap.append((ii_prob, ii_node))
    #                                     heap.heapify_max(source_heap)
    #                             # -- if ii_node is not in heap --
    #                             else:
    #                                 source_dict[ii_node] = (ii_prob, i_node)
    #                                 heap.heappush_max(source_heap, (ii_prob, ii_node))
    #
    #     return miia_dict

    def updatePMIIA(self, s_set, pmiia_dict):
        s_list = s_set.copy()

        s_node = ''
        sub_graph = copy.deepcopy(self.graph_dict)
        mioa_dict = {}
        while s_list:
            if len(s_list) != len(s_set):
                induceGraph(sub_graph, s_node)
            s_node = s_list.pop(0)

            mioa_dict[s_node] = {}
            source_dict = {s_node: (1.0, s_node)}
            source_heap = []
            if s_node in sub_graph:
                for i in sub_graph[s_node]:
                    source_dict[i] = (sub_graph[s_node][i], s_node)
                    heap.heappush_max(source_heap, (sub_graph[s_node][i], i))

            while source_heap:
                (i_prob, i_node) = heap.heappop_max(source_heap)
                i_prev = source_dict[i_node][1]

                # -- find MIP from source_node to i_node --
                i_path = [i_node, i_prev]
                while i_prev != source_dict[i_prev][1]:
                    i_prev = source_dict[i_prev][1]
                    i_path.append(i_prev)
                i_path.reverse()

                mioa_dict[s_node][i_node] = (i_prob, i_path)

                if i_node in sub_graph:
                    for ii_node in sub_graph[i_node]:
                        # -- not yet find MIP from source_node to ii_node --
                        if ii_node not in mioa_dict[s_node]:
                            ii_prob = round(i_prob * sub_graph[i_node][ii_node], 4)

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

        is_dict = {i_node_v: set() for s_node in mioa_dict for i_node_v in mioa_dict[s_node]}
        s_list = s_set.copy()
        s_node = s_list.pop(0)
        while s_list:
            for i_node_v in mioa_dict[s_node]:
                # --- if there is a seed j > i s.t. sj is in MIP_G(Si) (si, v), si will be added into IS(v) ---
                is_condition = {i for i in mioa_dict[s_node][i_node_v][1] if i in s_list}
                if is_condition:
                    is_dict[i_node_v].add(s_node)
            s_node = s_list.pop(0)

        for i_node_v in is_dict:
            s_set_reduce_is_set = set(s_set).difference(is_dict[i_node_v])
            for s_node in s_set_reduce_is_set:
                if i_node_v in mioa_dict[s_node]:
                    if i_node_v not in pmiia_dict:
                        pmiia_dict[i_node_v] = {}
                    pmiia_dict[i_node_v][s_node] = mioa_dict[s_node][i_node_v]

        return pmiia_dict

    def updateAP(self, i_node_v, s_seq, miia_v):
        # ap_v = {}
        ap_v = {i_node_v: 1.0}
        miia_v_seq = sorted([i_node_u for i_node_u in miia_v], reverse=True, key=lambda i_node_u: len(miia_v[i_node_u][1]))
        for i_node_u in miia_v_seq:
            if i_node_u in s_seq:
                ap_v[i_node_u] = 1.0
            else:
                n_in_set_u = findNIn_set(i_node_u, miia_v)
                acc_prob = 1.0
                if n_in_set_u:
                    for nis_w in n_in_set_u:
                        acc_prob *= (1 - ap_v[nis_w] * self.graph_dict[nis_w][i_node_u])
                ap_v[i_node_u] = round(1 - acc_prob, 4)

        return ap_v