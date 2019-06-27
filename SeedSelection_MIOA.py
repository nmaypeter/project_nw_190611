import heap


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