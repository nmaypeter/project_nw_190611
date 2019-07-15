import heap
import copy


class SeedSelectionMDAG:
    def __init__(self, graph_dict, seed_cost_dict, product_list):
        ### graph_dict: (dict) the graph
        ### product_list: (list) the set to record products [kk's profit, kk's cost, kk's price]
        ### num_node: (int) the number of nodes
        ### num_product: (int) the kinds of products
        ### product_weight_list: (list) the product weight list
        self.graph_dict = graph_dict
        self.seed_cost_dict = seed_cost_dict
        self.product_list = product_list
        self.num_product = len(product_list)
        self.prob_threshold = 0.001

    def generateDAG(self):
        node_rank_dict = {i: {j: 0.0 for j in self.graph_dict[i]} for i in self.graph_dict}

        for s in self.graph_dict:
            print(s)
            i_seq = [(i, self.graph_dict[s][i], [(s, i)]) for i in self.graph_dict[s] if self.graph_dict[s][i] >= self.prob_threshold]
            while i_seq:
                for i_item in i_seq:
                    i_node, i_prod, i_frag = i_item
                    for i_frag_item in i_frag:
                        i, j = i_frag_item
                        node_rank_dict[i][j] += i_prod

                i_seq = [i for i in i_seq if i[0] in self.graph_dict]
                i_seq = [(ii, round(i_item[1] * self.graph_dict[i_item[0]][ii], 4), i_item[2] + [(i_item[0], ii)]) for i_item in i_seq for ii in self.graph_dict[i_item[0]]
                         if round(i_item[1] * self.graph_dict[i_item[0]][ii], 4) >= self.prob_threshold]

        node_rank_list = [(node_rank_dict[i][j], i, j) for i in node_rank_dict for j in node_rank_dict[i]]
        sub_graph = {i: set(self.graph_dict[i]) for i in self.graph_dict}
        reverse_graph_dict = {ii: {i for i in sub_graph if ii in sub_graph[i]} for i in sub_graph for ii in sub_graph[i]}
        topological_list = [i for i in set(self.seed_cost_dict[0]).difference(set(reverse_graph_dict))]
        candidate_set = set(topological_list)
        while reverse_graph_dict:
            reverse_graph_dict = {i: reverse_graph_dict[i].difference(candidate_set) for i in reverse_graph_dict}
            node_rank_list = [i for i in node_rank_list if i[1] not in candidate_set]
            candidate_set = {i for i in reverse_graph_dict if not reverse_graph_dict[i]}
            while not candidate_set:
                min_node_rank = node_rank_list.pop(node_rank_list.index(min(node_rank_list)))
                sub_graph[min_node_rank[1]].remove(min_node_rank[2])
                reverse_graph_dict[min_node_rank[2]].remove(min_node_rank[1])
                candidate_set = {i for i in reverse_graph_dict if not reverse_graph_dict[i]}
            topological_list += list(candidate_set)
            reverse_graph_dict = {i: reverse_graph_dict[i] for i in reverse_graph_dict if reverse_graph_dict[i]}
        sub_graph = {i: set(self.graph_dict[i]) for i in self.graph_dict if sub_graph[i]}

        return sub_graph, topological_list

    # def

