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


def getExpectedInf(i_dict):
    ei = 0.0
    for item in i_dict:
        acc_prob = 1.0
        for prob in i_dict[item]:
            acc_prob *= (1 - prob)
        ei += (1 - acc_prob)

    return ei


def insertProbAncIntoDict(i_dict, i_node, i_prob, i_anc_set):
    if i_node not in i_dict:
        i_dict[i_node] = [(i_prob, i_anc_set)]
    else:
        i_dict[i_node].append((i_prob, i_anc_set))


def insertProbIntoDict(i_dict, i_node, i_prob):
    if i_node not in i_dict:
        i_dict[i_node] = [i_prob]
    else:
        i_dict[i_node].append(i_prob)


def combineDict(o_dict, n_dict):
    for item in n_dict:
        if item not in o_dict:
            o_dict[item] = n_dict[item]
        else:
            o_dict[item] += n_dict[item]


class DiffusionAccProb:
    def __init__(self, graph_dict, product_list):
        ### graph_dict: (dict) the graph
        ### product_list: (list) the set to record products [kk's profit, kk's cost, kk's price]
        ### num_node: (int) the number of nodes
        ### num_product: (int) the kinds of products
        ### product_weight_list: (list) the product weight list
        self.graph_dict = graph_dict
        self.product_list = product_list
        self.num_product = len(product_list)
        self.prob_threshold = 0.001

    def buildNodeExpectedInfDict(self, s_set, k_prod, i_node, i_acc_prob):
        i_dict = {}

        if i_node in self.graph_dict:
            for ii_node in self.graph_dict[i_node]:
                if ii_node in s_set:
                    continue
                ii_prob = round(float(self.graph_dict[i_node][ii_node]) * i_acc_prob, 4)

                if ii_prob >= self.prob_threshold:
                    insertProbIntoDict(i_dict, ii_node, ii_prob)

                    if ii_node in self.graph_dict:
                        for iii_node in self.graph_dict[ii_node]:
                            if iii_node in s_set:
                                continue
                            iii_prob = round(float(self.graph_dict[ii_node][iii_node]) * ii_prob, 4)

                            if iii_prob >= self.prob_threshold:
                                insertProbIntoDict(i_dict, iii_node, iii_prob)

                                if iii_node in self.graph_dict:
                                    for iv_node in self.graph_dict[iii_node]:
                                        if iv_node in s_set:
                                            continue
                                        iv_prob = round(float(self.graph_dict[iii_node][iv_node]) * iii_prob, 4)

                                        if iv_prob >= self.prob_threshold:
                                            insertProbIntoDict(i_dict, iv_node, iv_prob)

                                            if iv_node in self.graph_dict and iv_prob > self.prob_threshold:
                                                diff_d = DiffusionAccProb(self.graph_dict, self.product_list)
                                                iv_dict = diff_d.buildNodeExpectedInfDict(s_set, k_prod, iv_node, iv_prob)
                                                combineDict(i_dict, iv_dict)

        return i_dict


class DiffusionAccProb2:
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

    def getExpectedInf(self, s_set):
        s_total_set = set(s for k in range(self.num_product) for s in s_set[k])
        i_dict = [{i: 0.0 for i in self.seed_cost_dict[0]} for _ in range(self.num_product)]

        for k in range(self.num_product):
            i_seq = [s for s in s_set[k] if s in self.graph_dict]
            i_seq = [(ii, self.graph_dict[i][ii]) for i in i_seq for ii in self.graph_dict[i] if ii not in s_total_set and self.graph_dict[i][ii] >= self.prob_threshold]
            while i_seq:
                for i_item in i_seq:
                    i_node, i_prod = i_item
                    i_dict[k][i_node] = round(1.0 - (1.0 - i_dict[k][i_node]) * (1.0 - i_prod), 4)

                i_seq = [i for i in i_seq if i[0] in self.graph_dict]
                i_seq = [(ii, round(i[1] * self.graph_dict[i[0]][ii], 4)) for i in i_seq for ii in self.graph_dict[i[0]] if ii not in s_total_set and round(i[1] * self.graph_dict[i[0]][ii], 4) >= self.prob_threshold]

        exp_inf = [round(sum(i_dict[k][i] for i in i_dict[k]), 4) for k in range(self.num_product)]

        return exp_inf