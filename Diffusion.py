import random
import numpy as np
from scipy import stats
from random import choice


def safe_div(x, y):
    if y == 0:
        return 0.0
    return round(x / y, 4)


def getProductWeight(prod_list, wallet_dist_name):
    price_list = [prod[2] for prod in prod_list]
    pw_list = [1.0 for _ in range(len(price_list))]
    if wallet_dist_name in ['m50e25', 'm99e96']:
        mu, sigma = 0, 1
        if wallet_dist_name == 'm50e25':
            mu = np.mean(price_list)
            sigma = (max(price_list) - mu) / 0.6745
        elif wallet_dist_name == 'm99e96':
            mu = sum(price_list)
            sigma = abs(min(price_list) - mu) / 3
        X = np.arange(0, 2, 0.001)
        Y = stats.norm.sf(X, mu, sigma)
        pw_list = [round(float(Y[np.argwhere(X == p)]), 4) for p in price_list]

    return pw_list


class Diffusion:
    def __init__(self, graph_dict, product_list, product_weight_list):
        ### graph_dict: (dict) the graph
        ### product_list: (list) the list to record products [k's profit, k's cost, k's price]
        ### num_product: (int) the kinds of products
        ### product_weight_list: (list) the product weight list
        ### monte: (int) monte carlo times
        self.graph_dict = graph_dict
        self.product_list = product_list
        self.num_product = len(product_list)
        self.product_weight_list = product_weight_list
        self.prob_threshold = 0.001

    def getSeedSetProfit(self, s_set):
        s_total_set = set(s for k in range(self.num_product) for s in s_set[k])
        ep = 0.0
        for k in range(self.num_product):
            a_n_set = s_set[k].copy()
            a_n_sequence, a_n_sequence2 = [(s, 1) for s in s_set[k]], []
            benefit = self.product_list[k][0]
            product_weight = self.product_weight_list[k]

            while a_n_sequence:
                i_node, i_acc_prob = a_n_sequence.pop(choice([i for i in range(len(a_n_sequence))]))

                # -- notice: prevent the node from owing no receiver --
                if i_node not in self.graph_dict:
                    continue

                i_dict = self.graph_dict[i_node]
                for ii_node in i_dict:
                    if random.random() > i_dict[ii_node]:
                        continue

                    if ii_node in s_total_set:
                        continue
                    if ii_node in a_n_set:
                        continue
                    a_n_set.add(ii_node)

                    # -- purchasing --
                    ep += benefit * product_weight

                    ii_acc_prob = round(i_acc_prob * i_dict[ii_node], 4)
                    if ii_acc_prob > self.prob_threshold:
                        a_n_sequence2.append((ii_node, ii_acc_prob))

                if not a_n_sequence:
                    a_n_sequence, a_n_sequence2 = a_n_sequence2, a_n_sequence

        return round(ep, 4)