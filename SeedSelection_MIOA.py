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