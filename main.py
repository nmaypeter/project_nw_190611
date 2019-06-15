from Model import *

if __name__ == '__main__':
    dataset_seq = [1]
    prod_seq = [1, 2]
    cm_seq = [1, 2]
    wd_seq = [1, 2]

    for data_setting in dataset_seq:
        dataset_name = 'email' * (data_setting == 1) + 'dnc_email' * (data_setting == 2) + \
                       'email_Eu_core' * (data_setting == 3) + 'NetHEPT' * (data_setting == 4)
        for cm in cm_seq:
            cascade_model = 'ic' * (cm == 1) + 'wc' * (cm == 2)
            for prod_setting in prod_seq:
                product_name = 'item_lphc' * (prod_setting == 1) + 'item_hplc' * (prod_setting == 2)

                Model('mnapg', dataset_name, product_name, cascade_model).model_napg(r_flag=False, sr_flag=False, epw_flag=False)
                Model('mnapgr', dataset_name, product_name, cascade_model).model_napg(r_flag=True, sr_flag=False, epw_flag=False)
                Model('mnapgsr', dataset_name, product_name, cascade_model).model_napg(r_flag=True, sr_flag=True, epw_flag=False)
                Model('mng', dataset_name, product_name, cascade_model).model_ng(r_flag=False, sr_flag=False)
                Model('mngr', dataset_name, product_name, cascade_model).model_ng(r_flag=True, sr_flag=False)
                Model('mngsr', dataset_name, product_name, cascade_model).model_ng(r_flag=True, sr_flag=True)
                Model('mhd', dataset_name, product_name, cascade_model).model_hd()
                Model('mr', dataset_name, product_name, cascade_model).model_r()
                Model('mpmis', dataset_name, product_name, cascade_model).model_pmis()

                for wd in wd_seq:
                    wallet_distribution_type = 'm50e25' * (wd == 1) + 'm99e96' * (wd == 2)

                    Model('mnapgpw', dataset_name, product_name, cascade_model, wallet_distribution_type).model_napg(r_flag=False, sr_flag=False, epw_flag=False)
                    Model('mnapgrpw', dataset_name, product_name, cascade_model, wallet_distribution_type).model_napg(r_flag=True, sr_flag=False, epw_flag=False)
                    Model('mnapgsrpw', dataset_name, product_name, cascade_model, wallet_distribution_type).model_napg(r_flag=True, sr_flag=True, epw_flag=False)
                    Model('mnapgepw', dataset_name, product_name, cascade_model, wallet_distribution_type).model_napg(r_flag=False, sr_flag=False, epw_flag=True)
                    Model('mnapgrepw', dataset_name, product_name, cascade_model, wallet_distribution_type).model_napg(r_flag=True, sr_flag=False, epw_flag=True)
                    Model('mnapgsrepw', dataset_name, product_name, cascade_model, wallet_distribution_type).model_napg(r_flag=True, sr_flag=True, epw_flag=True)
                    Model('mngpw', dataset_name, product_name, cascade_model, wallet_distribution_type).model_ng(r_flag=False, sr_flag=False)
                    Model('mngrpw', dataset_name, product_name, cascade_model, wallet_distribution_type).model_ng(r_flag=True, sr_flag=False)
                    Model('mngsrpw', dataset_name, product_name, cascade_model, wallet_distribution_type).model_ng(r_flag=True, sr_flag=True)