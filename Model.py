from SeedSelection import *
from Evaluation import *
import time
import copy


def generateHeapOrder(heap_o, model_name, dataset_name, product_name, cascade_model, wallet_distribution_type):
    heap_des = sorted(heap_o, reverse=True)
    wd_seq = [wallet_distribution_type] if wallet_distribution_type else ['m50e25', 'm99e96']
    ini = Initialization(dataset_name, product_name)
    seed_cost_dict = ini.constructSeedCostDict()
    product_list = ini.constructProductList()
    num_product = len(product_list)

    order_dict = {(k, i): ('', '') for k in range(num_product) for i in seed_cost_dict[k]}
    now_order, last_mg = 0, -1
    for item in heap_des:
        p, k, i = item[0], item[1], item[2]
        now_order += 1 if p != last_mg else 0
        order_dict[(k, i)] = (p, now_order)
        last_mg = p

    for wd in wd_seq:
        path = 'heap_order/' + model_name + '_' + wd + '_' + dataset_name + '_' + cascade_model + '_' + product_name + '.txt'
        fw = open(path, 'w')
        for k in range(num_product):
            for i in seed_cost_dict[k]:
                index = (k, i)
                mg, order = order_dict[index]
                fw.write(str(index) + '\t' + str(mg) + '\t' + str(order) + '\n')
        fw.close()


class Model:
    def __init__(self, model_name, dataset_name, product_name, cascade_model, wallet_distribution_type=''):
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.product_name = product_name
        self.cascade_model = cascade_model
        self.budget_iteration = [i for i in range(10, 5, -1)]
        self.wallet_distribution_type = wallet_distribution_type
        self.wd_seq = ['m50e25', 'm99e96']
        self.sample_number = 1
        self.monte_carlo = 100

    def model_napg(self, r_flag, sr_flag, epw_flag):
        ini = Initialization(self.dataset_name, self.product_name)
        seed_cost_dict = ini.constructSeedCostDict()
        graph_dict = ini.constructGraphDict(self.cascade_model)
        product_list = ini.constructProductList()
        num_product = len(product_list)
        product_weight_list = getProductWeight(product_list, self.wallet_distribution_type)
        total_cost = sum(seed_cost_dict[k][i] for i in seed_cost_dict[0] for k in range(num_product))

        seed_set_sequence = [[-1 for _ in range(self.sample_number)] for _ in range(len(self.budget_iteration))]
        ss_time_sequence = [[-1 for _ in range(self.sample_number)] for _ in range(len(self.budget_iteration))]
        ssnapg_model = SeedSelectionNAPG(graph_dict, seed_cost_dict, product_list, product_weight_list, r_flag=r_flag, epw_flag=epw_flag)
        for sample_count in range(self.sample_number):
            ss_start_time = time.time()
            bud_iter = self.budget_iteration.copy()
            b_iter = bud_iter.pop(0)
            now_budget, now_profit = 0.0, 0.0
            seed_set = [set() for _ in range(num_product)]
            wd_seq = [self.wallet_distribution_type] if self.wallet_distribution_type else self.wd_seq
            expected_profit_k = [0.0 for _ in range(num_product)]
            celf_heap = ssnapg_model.generateCelfHeap()
            generateHeapOrder(celf_heap, self.model_name, self.dataset_name, self.product_name, self.cascade_model, self.wallet_distribution_type)
            ss_acc_time = round(time.time() - ss_start_time, 4)
            temp_sequence = [[ss_acc_time, now_budget, now_profit, seed_set, expected_profit_k, celf_heap]]
            temp_seed_data = [[]]
            while temp_sequence:
                ss_start_time = time.time()
                bi_index = self.budget_iteration.index(b_iter)
                total_budget = round(total_cost / (2 ** b_iter), 4)
                [ss_acc_time, now_budget, now_profit, seed_set, expected_profit_k, celf_heap] = temp_sequence.pop()
                seed_data = temp_seed_data.pop()
                print('@ ' + self.model_name + ' seed selection @ dataset_name = ' + self.dataset_name + '_' + self.cascade_model + ', product_name = ' + self.product_name +
                      ', bud_iter = ' + str(b_iter) + ', budget = ' + str(total_budget) + ', sample_count = ' + str(sample_count))

                celf_heap_c = []
                while now_budget < total_budget and celf_heap:
                    if round(now_budget + seed_cost_dict[celf_heap[0][1]][celf_heap[0][2]], 4) >= total_budget and bud_iter and not temp_sequence:
                        celf_heap_c = copy.deepcopy(celf_heap)
                    mep_item = heap.heappop_max(celf_heap)
                    mep_mg, mep_k_prod, mep_i_node, mep_flag = mep_item
                    sc = seed_cost_dict[mep_k_prod][mep_i_node]
                    seed_set_length = sum(len(seed_set[k]) for k in range(num_product))

                    if round(now_budget + sc, 4) >= total_budget and bud_iter and not temp_sequence:
                        ss_time = round(time.time() - ss_start_time + ss_acc_time, 4)
                        b_iter = bud_iter.pop(0)
                        temp_sequence.append([ss_time, now_budget, now_profit, copy.deepcopy(seed_set), copy.deepcopy(expected_profit_k), celf_heap_c])
                        temp_seed_data.append(seed_data)

                    if round(now_budget + sc, 4) > total_budget:
                        continue

                    if mep_flag == seed_set_length:
                        seed_set[mep_k_prod].add(mep_i_node)
                        now_budget = round(now_budget + sc, 4)
                        now_profit = round(now_profit + mep_mg * ((now_budget if sr_flag else sc) if r_flag else 1.0), 4)
                        expected_profit_k[mep_k_prod] = round(expected_profit_k[mep_k_prod] + mep_mg * ((now_budget if sr_flag else sc) if r_flag else 1.0), 4)
                        seed_data.append(str(round(time.time() - ss_start_time + ss_acc_time, 4)) + '\t' + str(mep_k_prod) + '\t' + str(mep_i_node) + '\t' +
                                         str(now_budget) + '\t' + str(now_profit) + '\t' + str([len(seed_set[k]) for k in range(num_product)]) + '\n')
                    else:
                        mep_item_sequence = [mep_item]
                        while len(mep_item_sequence) < 20 and celf_heap:
                            if celf_heap[0][3] != seed_set_length:
                                mep_item = heap.heappop_max(celf_heap)
                                mep_ratio, mep_k_prod, mep_i_node, mep_flag = mep_item
                                if round(now_budget + seed_cost_dict[mep_k_prod][mep_i_node], 4) <= total_cost:
                                    mep_item_sequence.append(mep_item)
                        mep_item_sequence_dict = ssnapg_model.updateExpectedInfBatch(seed_set, mep_item_sequence)
                        for midl in range(len(mep_item_sequence_dict)):
                            k_prod_t = mep_item_sequence[midl][1]
                            i_node_t = mep_item_sequence[midl][2]
                            s_dict = mep_item_sequence_dict[midl]
                            expected_inf = getExpectedInf(s_dict)
                            ep_t = round(expected_inf * product_list[mep_k_prod][0] * (1.0 if epw_flag else product_weight_list[mep_k_prod]), 4)
                            mg_t = round(ep_t - expected_profit_k[k_prod_t], 4)
                            if r_flag:
                                mg_t = safe_div(mg_t, now_budget + sc if sr_flag else sc)
                            flag_t = seed_set_length

                            if mg_t > 0:
                                celf_item_t = (mg_t, k_prod_t, i_node_t, flag_t)
                                heap.heappush_max(celf_heap, celf_item_t)

                ss_time = round(time.time() - ss_start_time + ss_acc_time, 4)
                print('ss_time = ' + str(ss_time) + 'sec, cost = ' + str(now_budget) + ', seed_set_length = ' + str([len(s_set_k) for s_set_k in seed_set]))
                seed_set_sequence[bi_index][sample_count] = seed_set
                ss_time_sequence[bi_index][sample_count] = ss_time

                for wd in wd_seq:
                    seed_data_path = 'seed_data/' + self.model_name + '_' + wd + '_' + self.dataset_name + '_' + self.cascade_model + '_' + self.product_name + '_bi' + str(b_iter) + '.txt'
                    seed_data_file = open(seed_data_path, 'w')
                    for sd in seed_data:
                        seed_data_file.write(sd)
                    seed_data_file.close()

        eva_model = EvaluationM(self.model_name, self.dataset_name, self.product_name, self.cascade_model)
        for bi in self.budget_iteration:
            bi_index = self.budget_iteration.index(bi)
            while -1 in seed_set_sequence[bi_index]:
                no_data_index = seed_set_sequence[bi_index].index(-1)
                seed_set_sequence[bi_index][no_data_index] = seed_set_sequence[bi_index - 1][no_data_index]
                ss_time_sequence[bi_index][no_data_index] = ss_time_sequence[bi_index - 1][no_data_index]
            if self.wallet_distribution_type:
                eva_model.evaluate(bi, self.wallet_distribution_type, seed_set_sequence[bi_index], ss_time_sequence[bi_index])
            else:
                for wallet_distribution_type in self.wd_seq:
                    eva_model.evaluate(bi, wallet_distribution_type, seed_set_sequence[bi_index], ss_time_sequence[bi_index])

    def model_ng(self, r_flag, sr_flag):
        ini = Initialization(self.dataset_name, self.product_name)
        seed_cost_dict = ini.constructSeedCostDict()
        graph_dict = ini.constructGraphDict(self.cascade_model)
        product_list = ini.constructProductList()
        num_product = len(product_list)
        product_weight_list = getProductWeight(product_list, self.wallet_distribution_type)
        total_cost = sum(seed_cost_dict[k][i] for i in seed_cost_dict[0] for k in range(num_product))

        seed_set_sequence = [[-1 for _ in range(self.sample_number)] for _ in range(len(self.budget_iteration))]
        ss_time_sequence = [[-1 for _ in range(self.sample_number)] for _ in range(len(self.budget_iteration))]
        ssng_model = SeedSelectionNG(graph_dict, seed_cost_dict, product_list, product_weight_list, r_flag=r_flag)
        diff_model = Diffusion(graph_dict, product_list, product_weight_list)
        for sample_count in range(self.sample_number):
            ss_start_time = time.time()
            bud_iter = self.budget_iteration.copy()
            b_iter = bud_iter.pop(0)
            now_budget, now_profit = 0.0, 0.0
            seed_set = [set() for _ in range(num_product)]
            wd_seq = [self.wallet_distribution_type] if self.wallet_distribution_type else self.wd_seq
            celf_heap = ssng_model.generateCelfHeap()
            generateHeapOrder(celf_heap, self.model_name, self.dataset_name, self.product_name, self.cascade_model, self.wallet_distribution_type)
            ss_acc_time = round(time.time() - ss_start_time, 4)
            temp_sequence = [[ss_acc_time, now_budget, now_profit, seed_set, celf_heap]]
            temp_seed_data = [[]]
            while temp_sequence:
                ss_start_time = time.time()
                bi_index = self.budget_iteration.index(b_iter)
                total_budget = round(total_cost / (2 ** b_iter), 4)
                [ss_acc_time, now_budget, now_profit, seed_set, celf_heap] = temp_sequence.pop()
                seed_data = temp_seed_data.pop()
                print('@ ' + self.model_name + ' seed selection @ dataset_name = ' + self.dataset_name + '_' + self.cascade_model + ', product_name = ' + self.product_name +
                      ', bud_iter = ' + str(b_iter) + ', budget = ' + str(total_budget) + ', sample_count = ' + str(sample_count))

                celf_heap_c = []
                while now_budget < total_budget and celf_heap:
                    if round(now_budget + seed_cost_dict[celf_heap[0][1]][celf_heap[0][2]], 4) >= total_budget and bud_iter and not temp_sequence:
                        celf_heap_c = copy.deepcopy(celf_heap)
                    mep_item = heap.heappop_max(celf_heap)
                    mep_mg, mep_k_prod, mep_i_node, mep_flag = mep_item
                    sc = seed_cost_dict[mep_k_prod][mep_i_node]
                    seed_set_length = sum(len(seed_set[k]) for k in range(num_product))

                    if round(now_budget + sc, 4) >= total_budget and bud_iter and not temp_sequence:
                        ss_time = round(time.time() - ss_start_time + ss_acc_time, 4)
                        b_iter = bud_iter.pop(0)
                        temp_sequence.append([ss_time, now_budget, now_profit, copy.deepcopy(seed_set), celf_heap_c])
                        temp_seed_data.append(seed_data)

                    if round(now_budget + sc, 4) > total_budget:
                        continue

                    if mep_flag == seed_set_length:
                        seed_set[mep_k_prod].add(mep_i_node)
                        now_budget = round(now_budget + sc, 4)
                        now_profit = round(sum([diff_model.getSeedSetProfit(seed_set) for _ in range(self.monte_carlo)]) / self.monte_carlo, 4)
                        seed_data.append(str(round(time.time() - ss_start_time + ss_acc_time, 4)) + '\t' + str(mep_k_prod) + '\t' + str(mep_i_node) + '\t' +
                                         str(now_budget) + '\t' + str(now_profit) + '\t' + str([len(seed_set[k]) for k in range(num_product)]) + '\n')
                    else:
                        seed_set_t = copy.deepcopy(seed_set)
                        seed_set_t[mep_k_prod].add(mep_i_node)
                        ep_t = round(sum([diff_model.getSeedSetProfit(seed_set_t) for _ in range(self.monte_carlo)]) / self.monte_carlo, 4)
                        mg_t = round(ep_t - now_profit, 4)
                        if r_flag:
                            mg_t = safe_div(mg_t, now_budget + sc if sr_flag else sc)
                        flag_t = seed_set_length

                        if mg_t > 0:
                            celf_item_t = (mg_t, mep_k_prod, mep_i_node, flag_t)
                            heap.heappush_max(celf_heap, celf_item_t)

                ss_time = round(time.time() - ss_start_time + ss_acc_time, 4)
                print('ss_time = ' + str(ss_time) + 'sec, cost = ' + str(now_budget) + ', seed_set_length = ' + str([len(s_set_k) for s_set_k in seed_set]))
                seed_set_sequence[bi_index][sample_count] = seed_set
                ss_time_sequence[bi_index][sample_count] = ss_time

                for wd in wd_seq:
                    seed_data_path = 'seed_data/' + self.model_name + '_' + wd + '_' + self.dataset_name + '_' + self.cascade_model + '_' + self.product_name + '_bi' + str(b_iter) + '.txt'
                    seed_data_file = open(seed_data_path, 'w')
                    for sd in seed_data:
                        seed_data_file.write(sd)
                    seed_data_file.close()

        eva_model = EvaluationM(self.model_name, self.dataset_name, self.product_name, self.cascade_model)
        for bi in self.budget_iteration:
            bi_index = self.budget_iteration.index(bi)
            while -1 in seed_set_sequence[bi_index]:
                no_data_index = seed_set_sequence[bi_index].index(-1)
                seed_set_sequence[bi_index][no_data_index] = seed_set_sequence[bi_index - 1][no_data_index]
                ss_time_sequence[bi_index][no_data_index] = ss_time_sequence[bi_index - 1][no_data_index]
            if self.wallet_distribution_type:
                eva_model.evaluate(bi, self.wallet_distribution_type, seed_set_sequence[bi_index], ss_time_sequence[bi_index])
            else:
                for wallet_distribution_type in self.wd_seq:
                    eva_model.evaluate(bi, wallet_distribution_type, seed_set_sequence[bi_index], ss_time_sequence[bi_index])

    def model_hd(self):
        ini = Initialization(self.dataset_name, self.product_name)
        seed_cost_dict = ini.constructSeedCostDict()
        graph_dict = ini.constructGraphDict(self.cascade_model)
        product_list = ini.constructProductList()
        num_product = len(product_list)
        total_cost = sum(seed_cost_dict[k][i] for i in seed_cost_dict[0] for k in range(num_product))

        seed_set_sequence = [[-1 for _ in range(self.sample_number)] for _ in range(len(self.budget_iteration))]
        ss_time_sequence = [[-1 for _ in range(self.sample_number)] for _ in range(len(self.budget_iteration))]
        sshd_model = SeedSelectionHD(graph_dict, product_list)
        for sample_count in range(self.sample_number):
            ss_start_time = time.time()
            bud_iter = self.budget_iteration.copy()
            b_iter = bud_iter.pop(0)
            now_budget = 0.0
            seed_set = [set() for _ in range(num_product)]
            wd_seq = [self.wallet_distribution_type] if self.wallet_distribution_type else self.wd_seq
            degree_heap = sshd_model.generateDegreeHeap()
            generateHeapOrder(degree_heap, self.model_name, self.dataset_name, self.product_name, self.cascade_model, self.wallet_distribution_type)
            ss_acc_time = round(time.time() - ss_start_time, 4)
            temp_sequence = [[ss_acc_time, now_budget, seed_set, degree_heap]]
            temp_seed_data = [[]]
            while temp_sequence:
                ss_start_time = time.time()
                bi_index = self.budget_iteration.index(b_iter)
                total_budget = round(total_cost / (2 ** b_iter), 4)
                [ss_acc_time, now_budget, seed_set, degree_heap] = temp_sequence.pop()
                seed_data = temp_seed_data.pop()
                print('@ ' + self.model_name + ' seed selection @ dataset_name = ' + self.dataset_name + '_' + self.cascade_model + ', product_name = ' + self.product_name +
                      ', bud_iter = ' + str(b_iter) + ', budget = ' + str(total_budget) + ', sample_count = ' + str(sample_count))

                degree_heap_c = []
                while now_budget < total_budget and degree_heap:
                    if round(now_budget + seed_cost_dict[degree_heap[0][1]][degree_heap[0][2]], 4) >= total_budget and bud_iter and not temp_sequence:
                        degree_heap_c = copy.deepcopy(degree_heap)
                    mep_item = heap.heappop_max(degree_heap)
                    mep_deg, mep_k_prod, mep_i_node = mep_item
                    sc = seed_cost_dict[mep_k_prod][mep_i_node]

                    if round(now_budget + sc, 4) >= total_budget and bud_iter and not temp_sequence:
                        ss_time = round(time.time() - ss_start_time + ss_acc_time, 4)
                        b_iter = bud_iter.pop(0)
                        temp_sequence.append([ss_time, now_budget, copy.deepcopy(seed_set), degree_heap_c])
                        temp_seed_data.append(seed_data)

                    if round(now_budget + sc, 4) > total_budget:
                        continue

                    seed_set[mep_k_prod].add(mep_i_node)
                    now_budget = round(now_budget + sc, 4)
                    seed_data.append(str(round(time.time() - ss_start_time + ss_acc_time, 4)) + '\t' + str(mep_k_prod) + '\t' + str(mep_i_node) + '\t' +
                                     str(now_budget) + '\t' + str([len(seed_set[k]) for k in range(num_product)]) + '\n')

                ss_time = round(time.time() - ss_start_time + ss_acc_time, 4)
                print('ss_time = ' + str(ss_time) + 'sec, cost = ' + str(now_budget) + ', seed_set_length = ' + str([len(s_set_k) for s_set_k in seed_set]))
                seed_set_sequence[bi_index][sample_count] = seed_set
                ss_time_sequence[bi_index][sample_count] = ss_time

                for wd in wd_seq:
                    seed_data_path = 'seed_data/' + self.model_name + '_' + wd + '_' + self.dataset_name + '_' + self.cascade_model + '_' + self.product_name + '_bi' + str(b_iter) + '.txt'
                    seed_data_file = open(seed_data_path, 'w')
                    for sd in seed_data:
                        seed_data_file.write(sd)
                    seed_data_file.close()

        eva_model = EvaluationM(self.model_name, self.dataset_name, self.product_name, self.cascade_model)
        for bi in self.budget_iteration:
            bi_index = self.budget_iteration.index(bi)
            while -1 in seed_set_sequence[bi_index]:
                no_data_index = seed_set_sequence[bi_index].index(-1)
                seed_set_sequence[bi_index][no_data_index] = seed_set_sequence[bi_index - 1][no_data_index]
                ss_time_sequence[bi_index][no_data_index] = ss_time_sequence[bi_index - 1][no_data_index]
            for wallet_distribution_type in self.wd_seq:
                eva_model.evaluate(bi, wallet_distribution_type, seed_set_sequence[bi_index], ss_time_sequence[bi_index])

    def model_r(self):
        ini = Initialization(self.dataset_name, self.product_name)
        seed_cost_dict = ini.constructSeedCostDict()
        graph_dict = ini.constructGraphDict(self.cascade_model)
        product_list = ini.constructProductList()
        num_product = len(product_list)
        total_cost = sum(seed_cost_dict[k][i] for i in seed_cost_dict[0] for k in range(num_product))

        seed_set_sequence = [[-1 for _ in range(self.sample_number)] for _ in range(len(self.budget_iteration))]
        ss_time_sequence = [[-1 for _ in range(self.sample_number)] for _ in range(len(self.budget_iteration))]
        ssr_model = SeedSelectionRandom(graph_dict, product_list)
        for sample_count in range(self.sample_number):
            ss_start_time = time.time()
            bud_iter = self.budget_iteration.copy()
            b_iter = bud_iter.pop(0)
            now_budget = 0.0
            seed_set = [set() for _ in range(num_product)]
            random_node_list = ssr_model.generateRandomList()
            ss_acc_time = round(time.time() - ss_start_time, 4)
            temp_sequence = [[ss_acc_time, now_budget, seed_set, random_node_list]]
            while temp_sequence:
                ss_start_time = time.time()
                bi_index = self.budget_iteration.index(b_iter)
                total_budget = round(total_cost / (2 ** b_iter), 4)
                [ss_acc_time, now_budget, seed_set, random_node_list] = temp_sequence.pop()
                print('@ ' + self.model_name + ' seed selection @ dataset_name = ' + self.dataset_name + '_' + self.cascade_model + ', product_name = ' + self.product_name +
                      ', bud_iter = ' + str(b_iter) + ', budget = ' + str(total_budget) + ', sample_count = ' + str(sample_count))

                random_node_list_c = []
                while now_budget < total_budget and random_node_list:
                    if round(now_budget + seed_cost_dict[random_node_list[0][0]][random_node_list[0][1]], 4) >= total_budget and bud_iter and not temp_sequence:
                        random_node_list_c = copy.deepcopy(random_node_list)
                    mep_item = random_node_list.pop(0)
                    mep_k_prod, mep_i_node = mep_item
                    sc = seed_cost_dict[mep_k_prod][mep_i_node]

                    if round(now_budget + sc, 4) >= total_budget and bud_iter and not temp_sequence:
                        ss_time = round(time.time() - ss_start_time + ss_acc_time, 4)
                        b_iter = bud_iter.pop(0)
                        temp_sequence.append([ss_time, now_budget, copy.deepcopy(seed_set), random_node_list_c])

                    if round(now_budget + sc, 4) > total_budget:
                        continue

                    seed_set[mep_k_prod].add(mep_i_node)
                    now_budget = round(now_budget + sc, 4)

                ss_time = round(time.time() - ss_start_time + ss_acc_time, 4)
                print('ss_time = ' + str(ss_time) + 'sec, cost = ' + str(now_budget) + ', seed_set_length = ' + str([len(s_set_k) for s_set_k in seed_set]))
                seed_set_sequence[bi_index][sample_count] = seed_set
                ss_time_sequence[bi_index][sample_count] = ss_time

        eva_model = EvaluationM(self.model_name, self.dataset_name, self.product_name, self.cascade_model)
        for bi in self.budget_iteration:
            bi_index = self.budget_iteration.index(bi)
            while -1 in seed_set_sequence[bi_index]:
                no_data_index = seed_set_sequence[bi_index].index(-1)
                seed_set_sequence[bi_index][no_data_index] = seed_set_sequence[bi_index - 1][no_data_index]
                ss_time_sequence[bi_index][no_data_index] = ss_time_sequence[bi_index - 1][no_data_index]
            for wallet_distribution_type in self.wd_seq:
                eva_model.evaluate(bi, wallet_distribution_type, seed_set_sequence[bi_index], ss_time_sequence[bi_index])

    def model_pmis(self):
        ini = Initialization(self.dataset_name, self.product_name)
        seed_cost_dict = ini.constructSeedCostDict()
        graph_dict = ini.constructGraphDict(self.cascade_model)
        product_list = ini.constructProductList()
        num_product = len(product_list)
        product_weight_list = getProductWeight(product_list, self.wallet_distribution_type)
        total_cost = sum(seed_cost_dict[k][i] for i in seed_cost_dict[0] for k in range(num_product))

        seed_set_sequence = [[-1 for _ in range(self.sample_number)] for _ in range(len(self.budget_iteration))]
        ss_time_sequence = [[-1 for _ in range(self.sample_number)] for _ in range(len(self.budget_iteration))]
        sspmis_model = SeedSelectionPMIS(graph_dict, seed_cost_dict, product_list, product_weight_list)
        diff_model = Diffusion(graph_dict, product_list, product_weight_list)
        for sample_count in range(self.sample_number):
            ss_start_time = time.time()
            celf_heap = sspmis_model.generateCelfHeap()
            s_matrix_sequence, c_matrix_sequence = [[] for _ in range(num_product)], [[] for _ in range(num_product)]
            for k in range(num_product):
                bud_iter = self.budget_iteration.copy()
                b_iter = bud_iter.pop(0)
                now_budget, now_profit = 0.0, 0.0
                seed_set = [set() for _ in range(num_product)]
                s_matrix, c_matrix = [[set() for _ in range(num_product)]], [0.0]
                ss_acc_time = round(time.time() - ss_start_time, 4)
                temp_sequence = [[ss_acc_time, now_budget, now_profit, seed_set, s_matrix, c_matrix, celf_heap[k]]]
                while temp_sequence:
                    ss_start_time = time.time()
                    bi_index = self.budget_iteration.index(b_iter)
                    total_budget = round(total_cost / (2 ** b_iter), 4)
                    [ss_acc_time, now_budget, now_profit, seed_set, s_matrix, c_matrix, celf_heap_k] = temp_sequence.pop()
                    print('@ ' + self.model_name + ' seed selection @ dataset_name = ' + self.dataset_name + '_' + self.cascade_model + ', product_name = ' + self.product_name +
                          ', bud_iter = ' + str(b_iter) + ', budget = ' + str(total_budget) + ', sample_count = ' + str(sample_count) + '_' + str(k))

                    celf_heap_k_c = []
                    while now_budget < total_budget and celf_heap_k:
                        if round(now_budget + seed_cost_dict[celf_heap_k[0][1]][celf_heap_k[0][2]], 4) >= total_budget and bud_iter and not temp_sequence:
                            celf_heap_k_c = copy.deepcopy(celf_heap_k)
                        mep_item = heap.heappop_max(celf_heap_k)
                        mep_mg, mep_k_prod, mep_i_node, mep_flag = mep_item
                        sc = seed_cost_dict[mep_k_prod][mep_i_node]
                        seed_set_length = sum(len(seed_set[k]) for k in range(num_product))

                        if round(now_budget + sc, 4) >= total_budget and bud_iter and not temp_sequence:
                            ss_time = round(time.time() - ss_start_time + ss_acc_time, 4)
                            b_iter = bud_iter.pop(0)
                            temp_sequence.append([ss_time, now_budget, now_profit, copy.deepcopy(seed_set), copy.deepcopy(s_matrix), copy.deepcopy(c_matrix), celf_heap_k_c])

                        if round(now_budget + sc, 4) > total_budget:
                            continue

                        if mep_flag == seed_set_length:
                            seed_set[mep_k_prod].add(mep_i_node)
                            now_budget = round(now_budget + sc, 4)
                            now_profit = round(sum([diff_model.getSeedSetProfit(seed_set) for _ in range(self.monte_carlo)]) / self.monte_carlo, 4)
                            s_matrix.append(copy.deepcopy(seed_set))
                            c_matrix.append(now_budget)
                        else:
                            seed_set_t = copy.deepcopy(seed_set)
                            seed_set_t[mep_k_prod].add(mep_i_node)
                            ep_t = round(sum([diff_model.getSeedSetProfit(seed_set_t) for _ in range(self.monte_carlo)]) / self.monte_carlo, 4)
                            mg_t = round(ep_t - now_profit, 4)
                            flag_t = seed_set_length

                            if mg_t > 0:
                                celf_item_t = (mg_t, mep_k_prod, mep_i_node, flag_t)
                                heap.heappush_max(celf_heap_k, celf_item_t)

                    s_matrix_sequence[k].append(s_matrix)
                    c_matrix_sequence[k].append(c_matrix)
                    ss_time = round(time.time() - ss_start_time + ss_acc_time, 4)
                    print('ss_time = ' + str(ss_time) + 'sec, cost = ' + str(now_budget) + ', seed_set_length = ' + str([len(s_set_k) for s_set_k in seed_set]))
                    ss_time_sequence[bi_index][sample_count] += ss_time

            ss_start_time = time.time()
            for k in range(num_product):
                while len(s_matrix_sequence[k]) < len(self.budget_iteration):
                    s_matrix_sequence[k].append(s_matrix_sequence[k][-1])
                    c_matrix_sequence[k].append(c_matrix_sequence[k][-1])
            for bi in self.budget_iteration:
                bi_index = self.budget_iteration.index(bi)
                total_budget = round(total_cost / (2 ** bi), 4)
                s_matrix_bi, c_matrix_bi = [], []
                for k in range(num_product):
                    s_matrix_bi.append(s_matrix_sequence[k][bi_index])
                    c_matrix_bi.append(c_matrix_sequence[k][bi_index])
                seed_set = sspmis_model.solveMultipleChoiceKnapsackProblem(total_budget, s_matrix_bi, c_matrix_bi)
                ss_time = round(time.time() - ss_start_time, 4)
                ss_time_sequence[bi_index][sample_count] += ss_time
                seed_set_sequence[bi_index][sample_count] = seed_set

        eva_model = EvaluationM(self.model_name, self.dataset_name, self.product_name, self.cascade_model)
        for bi in self.budget_iteration:
            bi_index = self.budget_iteration.index(bi)
            while -1 in seed_set_sequence[bi_index]:
                no_data_index = seed_set_sequence[bi_index].index(-1)
                seed_set_sequence[bi_index][no_data_index] = seed_set_sequence[bi_index - 1][no_data_index]
                ss_time_sequence[bi_index][no_data_index] = ss_time_sequence[bi_index - 1][no_data_index]
            for wallet_distribution_type in self.wd_seq:
                eva_model.evaluate(bi, wallet_distribution_type, seed_set_sequence[bi_index], ss_time_sequence[bi_index])