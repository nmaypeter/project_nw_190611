from Model import *

if __name__ == '__main__':
    dataset_seq = [1]
    prod_seq = [1]
    cm_seq = [1]
    wd_seq = [1]

    for data_setting in dataset_seq:
        dataset_name = 'email' * (data_setting == 1) + 'dnc_email' * (data_setting == 2) + \
                       'email_Eu_core' * (data_setting == 3) + 'NetHEPT' * (data_setting == 4)
        for cm in cm_seq:
            cascade_model = 'ic' * (cm == 1) + 'wc' * (cm == 2)
            for prod_setting in prod_seq:
                product_name = 'item_lphc' * (prod_setting == 1) + 'item_hplc' * (prod_setting == 2)

                Model('mnapg', dataset_name, product_name, cascade_model).model_napg(r_flag=False)
                Model('mnapgr', dataset_name, product_name, cascade_model).model_napg(r_flag=True)
                Model('mng', dataset_name, product_name, cascade_model).model_ng(r_flag=False)
                Model('mngr', dataset_name, product_name, cascade_model).model_ng(r_flag=True)
                Model('mhd', dataset_name, product_name, cascade_model).model_hd()
                Model('mr', dataset_name, product_name, cascade_model).model_r()
                Model('mpmis', dataset_name, product_name, cascade_model).model_pmis()
                Model('mmioa', dataset_name, product_name, cascade_model).model_mioa(r_flag=False)
                Model('mmioar', dataset_name, product_name, cascade_model).model_mioa(r_flag=True)
                Model('mdag1', dataset_name, product_name, cascade_model).model_dag1(r_flag=False)
                Model('mdag1r', dataset_name, product_name, cascade_model).model_dag1(r_flag=True)
                Model('mdag2', dataset_name, product_name, cascade_model).model_dag2(r_flag=False)
                Model('mdag2r', dataset_name, product_name, cascade_model).model_dag2(r_flag=True)

                for wd in wd_seq:
                    wallet_distribution_type = 'm50e25' * (wd == 1) + 'm99e96' * (wd == 2)

                    Model('mnapgpw', dataset_name, product_name, cascade_model, wallet_distribution_type).model_napg(r_flag=False)
                    Model('mnapgrpw', dataset_name, product_name, cascade_model, wallet_distribution_type).model_napg(r_flag=True)
                    Model('mngpw', dataset_name, product_name, cascade_model, wallet_distribution_type).model_ng(r_flag=False)
                    Model('mngrpw', dataset_name, product_name, cascade_model, wallet_distribution_type).model_ng(r_flag=True)
                    Model('mmioapw', dataset_name, product_name, cascade_model, wallet_distribution_type).model_mioa(r_flag=False)
                    Model('mmioarpw', dataset_name, product_name, cascade_model, wallet_distribution_type).model_mioa(r_flag=True)
                    Model('mdag1pw', dataset_name, product_name, cascade_model, wallet_distribution_type).model_dag1(r_flag=False)
                    Model('mdag1rpw', dataset_name, product_name, cascade_model, wallet_distribution_type).model_dag1(r_flag=True)
                    Model('mdag2pw', dataset_name, product_name, cascade_model, wallet_distribution_type).model_dag2(r_flag=False)
                    Model('mdag2rpw', dataset_name, product_name, cascade_model, wallet_distribution_type).model_dag2(r_flag=True)