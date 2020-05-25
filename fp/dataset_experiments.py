from fp.experiments import BinaryClassificationExperiment
import pandas as pd
import numpy as np
import fp.data_utils as data_u
from pathlib import Path


class AdultDatasetWhiteMaleExperiment(BinaryClassificationExperiment):

    def __init__(self, fixed_random_seed, train_data_sampler, missing_value_handler, numeric_attribute_scaler, learners, pre_processors, post_processors):

        test_set_ratio = 0.2
        validation_set_ratio = 0.1
        label_name = 'income-per-year'
        positive_label = '>50K'
        numeric_attribute_names = ['capital-gain', 'capital-loss', 'age', 'hours-per-week']
        categorical_attribute_names = ['workclass', 'education', 'marital-status', 'occupation', 'relationship',
                                       'native-country']
        attributes_to_drop_names = ['fnlwgt']

        protected_attribute_names = ['race', 'sex']
        privileged_classes = [['White'], ['Male']]

        privileged_groups = [{'race': 1, 'sex': 1}]
        unprivileged_groups = [{'race': 1, 'sex': 0}, {'sex': 0}]

        dataset_metadata = {
            'label_maps': [{1.0: '>50K', 0.0: '<=50K'}],
            'protected_attribute_maps': [{1.0: 'White', 0.0: 'Non-white'}, {1.0: 'Male', 0.0: 'Female'}]
        }

        super().__init__(fixed_random_seed, test_set_ratio, validation_set_ratio, label_name, positive_label,
                         numeric_attribute_names, categorical_attribute_names, attributes_to_drop_names,
                         train_data_sampler, missing_value_handler, numeric_attribute_scaler, learners, pre_processors,
                         post_processors, protected_attribute_names, privileged_classes, privileged_groups,
                         unprivileged_groups, dataset_metadata, 'adultwhitemale')

    def load_raw_data(self):
        data = pd.read_csv('datasets/raw/adult.csv', na_values='?', sep=',')
        print(data.head())
        return data

class AdultDatasetMaleExperiment(BinaryClassificationExperiment):

    def __init__(self, fixed_random_seed, train_data_sampler, missing_value_handler, numeric_attribute_scaler,  learners, pre_processors, post_processors):

        test_set_ratio = 0.2
        validation_set_ratio = 0.1
        label_name = 'income-per-year'
        positive_label = '>50K'
        numeric_attribute_names = ['capital-gain', 'capital-loss', 'age', 'hours-per-week']
        categorical_attribute_names = ['workclass', 'education', 'marital-status', 'occupation', 'relationship',
                                       'native-country']
        attributes_to_drop_names = ['fnlwgt', 'race']

        protected_attribute_names = ['sex']
        privileged_classes = [['Male']]

        privileged_groups = [{'sex': 1}]
        unprivileged_groups = [{'sex': 0}]

        dataset_metadata = {
            'label_maps': [{1.0: '>50K', 0.0: '<=50K'}],
            'protected_attribute_maps': [{1.0: 'Male', 0.0: 'Female'}]
        }

        super().__init__(fixed_random_seed, test_set_ratio, validation_set_ratio, label_name, positive_label,
                         numeric_attribute_names, categorical_attribute_names, attributes_to_drop_names,
                         train_data_sampler, missing_value_handler, numeric_attribute_scaler, learners, pre_processors,
                         post_processors, protected_attribute_names, privileged_classes, privileged_groups,
                         unprivileged_groups, dataset_metadata, 'adultmale')

    def load_raw_data(self):
        return pd.read_csv('datasets/raw/adult.csv', na_values='?', sep=',')


class AdultDatasetWhiteExperiment(BinaryClassificationExperiment):

    def __init__(self, fixed_random_seed, train_data_sampler, missing_value_handler, numeric_attribute_scaler,
                 learners, pre_processors, post_processors):

        test_set_ratio = 0.2
        validation_set_ratio = 0.1
        label_name = 'income-per-year'
        positive_label = '>50K'
        numeric_attribute_names = ['capital-gain', 'capital-loss', 'age', 'hours-per-week']
        categorical_attribute_names = ['workclass', 'education', 'marital-status', 'occupation', 'relationship',
                                       'native-country']
        attributes_to_drop_names = ['fnlwgt', 'sex']

        protected_attribute_names = ['race']
        privileged_classes = [['White']]

        privileged_groups = [{'race': 1}]
        unprivileged_groups = [{'race': 0}]

        dataset_metadata = {
            'label_maps': [{1.0: '>50K', 0.0: '<=50K'}],
            'protected_attribute_maps': [{1.0: 'White', 0.0: 'Non-white'}]
        }

        super().__init__(fixed_random_seed, test_set_ratio, validation_set_ratio, label_name, positive_label,
                         numeric_attribute_names, categorical_attribute_names, attributes_to_drop_names,
                         train_data_sampler, missing_value_handler, numeric_attribute_scaler, learners, pre_processors,
                         post_processors, protected_attribute_names, privileged_classes, privileged_groups,
                         unprivileged_groups, dataset_metadata, 'adultwhite')

    def load_raw_data(self):
        data = pd.read_csv('datasets/raw/adult.csv', na_values='?', sep=',')
        print(data.head())
        return data
        #return pd.read_csv('datasets/raw/adult.csv', na_values='?', sep=',')



class PropublicaDatasetWhiteExperiment(BinaryClassificationExperiment):

    def __init__(self, fixed_random_seed, train_data_sampler, missing_value_handler, numeric_attribute_scaler,
                 learners, pre_processors, post_processors):
        test_set_ratio = 0.2
        validation_set_ratio = 0.1
        label_name = 'two_year_recid'
        positive_label = 1
        numeric_attribute_names = ['age', 'decile_score', 'priors_count', 'days_b_screening_arrest', 'decile_score',
                                   'is_recid']
        categorical_attribute_names = ['c_charge_degree', 'age_cat', 'score_text']
        attributes_to_drop_names = ['sex', 'c_jail_in', 'c_jail_out']

        protected_attribute_names = ['race']
        privileged_classes = [['Caucasian']]

        privileged_groups = [{'race': 1}]
        unprivileged_groups = [{'race': 0}]

        dataset_metadata = {
            'label_maps': [{1.0: 1, 0.0: 0}],
            'protected_attribute_maps': [{1.0: 'Caucasian', 0.0: 'Non-white'}]
        }

        super().__init__(fixed_random_seed, test_set_ratio, validation_set_ratio, label_name, positive_label,
                         numeric_attribute_names, categorical_attribute_names, attributes_to_drop_names,
                         train_data_sampler, missing_value_handler, numeric_attribute_scaler, learners, pre_processors,
                         post_processors, protected_attribute_names, privileged_classes, privileged_groups,
                         unprivileged_groups, dataset_metadata, 'propublicawhite')

    def load_raw_data(self):
        """The custom pre-processing function is adapted from
        https://github.com/IBM/AIF360/blob/master/aif360/algorithms/preprocessing/optim_preproc_helpers/data_preproc_functions.py
        https://github.com/fair-preprocessing/nips2017/blob/master/compas/code/Generate_Compas_Data.ipynb
        """
        df = pd.read_csv('datasets/raw/propublica-recidivism.csv', na_values='?', sep=',')
        df = df[['age', 'c_charge_degree', 'race', 'age_cat', 'score_text',
                 'sex', 'priors_count', 'days_b_screening_arrest', 'decile_score',
                 'is_recid', 'two_year_recid', 'c_jail_in', 'c_jail_out']]
        ix = df['days_b_screening_arrest'] <= 100
        ix = (df['days_b_screening_arrest'] >= -100) & ix
        ix = (df['is_recid'] != -1) & ix
        ix = (df['c_charge_degree'] != "O") & ix
        ix = (df['score_text'] != 'N/A') & ix
        df = df.loc[ix, :]
        df['length_of_stay'] = (pd.to_datetime(df['c_jail_out']) - pd.to_datetime(df['c_jail_in'])).apply(
            lambda x: x.days)
        return df


class GermanCreditDatasetSexExperiment(BinaryClassificationExperiment):


    def __init__(self, fixed_random_seed, train_data_sampler, missing_value_handler, numeric_attribute_scaler,
                 learners, pre_processors, post_processors):

        test_set_ratio = 0.2
        validation_set_ratio = 0.1
        label_name = 'credit'
        positive_label = 1
        numeric_attribute_names = ['month', 'credit_amount', 'residence_since', 'age', 'number_of_credits',
                                   'people_liable_for']
        categorical_attribute_names = ['credit_history', 'savings', 'employment']
        attributes_to_drop_names = ['personal_status', 'status', 'purpose', 'investment_as_income_percentage',
                                    'other_debtors', 'property', 'installment_plans', 'housing', 'skill_level',
                                    'telephone', 'foreign_worker']

        protected_attribute_names = ['sex']
        privileged_classes = [[1.0]]

        privileged_groups = [{'sex': 1.0}]
        unprivileged_groups = [{'sex': 0.0}]

        dataset_metadata = {
            'label_maps': [{1.0: 1, 0.0: 0}],
            'protected_attribute_maps': [{1.0: 'male', 0.0: 'female'}]
        }

        super().__init__(fixed_random_seed, test_set_ratio, validation_set_ratio, label_name, positive_label,
                         numeric_attribute_names, categorical_attribute_names, attributes_to_drop_names,
                         train_data_sampler, missing_value_handler, numeric_attribute_scaler, learners, pre_processors,
                         post_processors, protected_attribute_names, privileged_classes, privileged_groups,
                         unprivileged_groups, dataset_metadata, 'germancreditsex')

    def load_raw_data(self):
        df = pd.read_csv('datasets/raw/german.csv', na_values='?', sep=',')

        def group_credit_hist(x):
            if x in ['A30', 'A31', 'A32']:
                return 'None/Paid'
            elif x == 'A33':
                return 'Delay'
            elif x == 'A34':
                return 'Other'
            else:
                return 'NA'

        def group_employ(x):
            if x == 'A71':
                return 'Unemployed'
            elif x in ['A72', 'A73']:
                return '1-4 years'
            elif x in ['A74', 'A75']:
                return '4+ years'
            else:
                return 'NA'

        def group_savings(x):
            if x in ['A61', 'A62']:
                return '<500'
            elif x in ['A63', 'A64']:
                return '500+'
            elif x == 'A65':
                return 'Unknown/None'
            else:
                return 'NA'

        def group_status(x):
            if x in ['A11', 'A12']:
                return '<200'
            elif x in ['A13']:
                return '200+'
            elif x == 'A14':
                return 'None'
            else:
                return 'NA'

        status_map = {'A91': 1.0, 'A93': 1.0, 'A94': 1.0, 'A92': 0.0, 'A95': 0.0}
        df['sex'] = df['personal_status'].replace(status_map)
        #         group credit history, savings, and employment
        df['credit_history'] = df['credit_history'].apply(lambda x: group_credit_hist(x))
        df['savings'] = df['savings'].apply(lambda x: group_savings(x))
        df['employment'] = df['employment'].apply(lambda x: group_employ(x))
        df['age'] = df['age'].apply(lambda x: np.float(x >= 25))
        df['status'] = df['status'].apply(lambda x: group_status(x))
        return df
   
class RicciRaceExperiment(BinaryClassificationExperiment):
    '''
    Check for fairness based on race (white vs minority i.e Black and Hispanic) while predicting if a candidate will pass i.e obtain total 
    marks greater than or equal to 70.0  
    '''

    def __init__(self, fixed_random_seed, train_data_sampler, missing_value_handler, numeric_attribute_scaler,
                 learners, pre_processors, post_processors):

        test_set_ratio = 0.2
        validation_set_ratio = 0.1
        
        label_name = 'combine'
        positive_label = 1
        numeric_attribute_names = ['oral', 'written']
        categorical_attribute_names = ['position']
        attributes_to_drop_names = []
                
        protected_attribute_names = ['race']
        privileged_classes = [[1.0]]
        
        privileged_groups = [{'race': 1.0}]
        unprivileged_groups = [{'race': 0.0}]

        dataset_metadata = {
            'label_maps': [{1.0: 1, 0.0: 0}],
            'protected_attribute_maps': [{1.0: 'W', 0.0: 'NW'}]
        }
        
        super().__init__(fixed_random_seed, test_set_ratio, validation_set_ratio, label_name, positive_label,
                         numeric_attribute_names, categorical_attribute_names,attributes_to_drop_names,
                         train_data_sampler, missing_value_handler, numeric_attribute_scaler, learners, pre_processors,
                         post_processors, protected_attribute_names, privileged_classes, privileged_groups,
                         unprivileged_groups, dataset_metadata, 'riccirace')

    def load_raw_data(self):
        df = pd.read_csv('datasets/raw/ricci.txt', na_values='?', sep=',')
        df.columns = map(str.lower, df.columns)

        def group_race_minority(x):
            if x in ['B', 'H', 'B']:
                return 'NW'
            else:
                return 'W'

        post_map = {'Captain': 0.0, 'Lieutenant': 1.0}
        df['position'] = df['position'].replace(post_map)
        
        #group minorities i.e Black and Hispanic are combined to 'NW'(non white)
        df['race'] = df['race'].apply(lambda x: group_race_minority(x))
        df['combine'] = df['combine'].apply(lambda x: int(x >= 70))

        return df

class GiveMeSomeCreditExperiment(BinaryClassificationExperiment):
    '''
    Fairness intervention for the Age attribute (priviledge for age>=25) while predicting if a person will experience 90 days past due delinquency or worse. 
    '''
    def __init__(self, fixed_random_seed, train_data_sampler, missing_value_handler, numeric_attribute_scaler,
                 learners, pre_processors, post_processors):

        test_set_ratio = 0.2
        validation_set_ratio = 0.1
        label_name = 'SeriousDlqin2yrs'
        positive_label = 1

        numeric_attribute_names = ['RevolvingUtilizationOfUnsecuredLines','age','NumberOfTime30-59DaysPastDueNotWorse',
                                  'DebtRatio','MonthlyIncome','NumberOfOpenCreditLinesAndLoans','NumberOfTimes90DaysLate',
                                  'NumberRealEstateLoansOrLines','NumberOfTime60-89DaysPastDueNotWorse','NumberOfDependents']
        categorical_attribute_names = []
        attributes_to_drop_names = []

        protected_attribute_names = ['age']
        privileged_classes = [[1.0]]

        privileged_groups = [{'age': 1}]
        unprivileged_groups = [{'age': 0}]

        dataset_metadata = {
            'label_maps': [{1.0: 1, 0.0: 0}]
        }

        super().__init__(fixed_random_seed, test_set_ratio, validation_set_ratio, label_name, positive_label,
                         numeric_attribute_names, categorical_attribute_names, attributes_to_drop_names,
                         train_data_sampler, missing_value_handler, numeric_attribute_scaler, learners, pre_processors,
                         post_processors, protected_attribute_names, privileged_classes, privileged_groups,
                         unprivileged_groups, dataset_metadata, 'givecredit')

    def load_raw_data(self):
        df = pd.read_csv('datasets/raw/givemesomecredit.csv', na_values='?', sep=',',index_col=False)
        df['age'] = df['age'].apply(lambda x: int(x >= 25))
        return df
        

class PSIDDataset(BinaryClassificationExperiment):

    def __init__(self, fixed_random_seed, train_data_sampler, missing_value_handler, numeric_attribute_scaler, \
         learners, pre_processors, post_processors, \
         year="11", classification_target=("labels1", 5), protected_attributes=[("gender", ["woman"])], protected_attributes_for_classification=["gender", "age"]):



        self.data_file = Path("../dev/PSID_data_household_income/J277040_2011_ANN_paper/dataset_processed_20" + year + "_notOHE.csv")

        test_set_ratio = 0.2
        validation_set_ratio = 0.1
        label_name = classification_target[0]
        attributes_to_drop_names = ['year']


        ### Handle labels' attributes.

        # We define for now the classification task as: classification_target=(label_id, ratio). 
        # The cell is positive when its value is "ratio" times higher than the mean of the attribute.
        # Get average of the attribute.
        data = pd.read_csv(self.data_file, na_values='', sep=',')
        self.threshold_value = data[[classification_target[0]]].mean(axis=0)[classification_target[0]] * classification_target[1] 
        positive_label = '>threshold'
        self.negative_label = "<=threshold"
        if classification_target[0] == "labels1":
            attributes_to_drop_names += ["labels2"]
            attributes_to_drop_names += ["labels3"]
        elif classification_target[0] == "labels2":
            attributes_to_drop_names += ["labels1"]
            attributes_to_drop_names += ["labels3"]
        elif classification_target[0] == "labels3":
            attributes_to_drop_names += ["labels1"]
            attributes_to_drop_names += ["labels2"]

        
        ### Handle protected attributes.
        list_possible_protected_attributes = ["headedu_" + year, "hdrace", "sprace", "gender", "spedu_"+ year, "hage_"+ year, "spage_"+ year]
        self.protected_attributes = protected_attributes

        ## Process protected attriutes for evaluation to binary attributes.
        # We create new attributes just for evaluation.
        protected_attribute_names = []
        privileged_classes = []
        privileged_classes_concat = []
        privileged_groups = [{}]
        unprivileged_groups = []
        for eval_protected_attributes in protected_attributes:
            protected_attribute_names += ["eval_" + eval_protected_attributes[0]]
            list_protected_classes = "_".join(eval_protected_attributes[1])
            #for protected_class in eval_protected_attributes[1]:
            #    list_protected_classes += ("_" + protected_class)
            privileged_classes_concat += [list_protected_classes]
            privileged_classes.append(eval_protected_attributes[1]) # += [list_protected_classes]
            privileged_groups[0]["eval_" + eval_protected_attributes[0]] = 1

        # Create the list of unpriviledged groups.
        nb_protected_attributes = len(protected_attribute_names)
        arr = [None] * nb_protected_attributes
        list_combinations = []
        list_combinations = data_u.generateAllBinaryStrings(nb_protected_attributes, arr, 0, list_combinations)
        list_combinations.remove([1]*nb_protected_attributes)
        for combi in list_combinations:
            dict_to_append = {}
            for protected_att, bit in zip(protected_attribute_names, combi):
                dict_to_append[protected_att] = bit
            unprivileged_groups.append(dict_to_append)

        # Prepare protected attributes for classification: remove the ones that should not be used for training.
        for att in list_possible_protected_attributes:
            if att not in protected_attributes_for_classification:
                attributes_to_drop_names.append(att)

        dataset_metadata = {
            'label_maps': [{1.0: positive_label, 0.0: self.negative_label}],
            'protected_attribute_maps': [{1.0: class_name, 0.0: "Non-" + class_name} for class_name in privileged_classes_concat]
        }


        ### Handle all attributes used for classification.

        numeric_attribute_names = ["intrate_" + year, "housingexp_" + year, "transportexp_" + year, "trpexp_" + year, "fdexp_" + year, "clothexp_" + year, \
                                    "educexp_" + year, "chcare_" + year, "hthcarexp_" + year, "debtot_" + year, "hvyact_" + year, "litact_" + year, "stract_" + year, \
                                    "ciga_" + year, "alcannual_" + year, "hvyactsp_" + year, "litactsp_" + year, "stractsp_" + year, "cigasp_" + year, \
                                    "alcannualsp_" + year, "stock_" + year, "hsequity_" + year, "networth_" + year]
        categorical_attribute_names = ["bigovsub_" + year, "headprob_" + year, "spprob_" + year, "jobstatus_" + year]

        for att in protected_attributes_for_classification:
            if att in ["headedu_" + year, "hdrace", "sprace", "gender", "spedu_"+ year]:
                categorical_attribute_names.append(att)
            else:
                numeric_attribute_names.append(att)

        super().__init__(fixed_random_seed, test_set_ratio, validation_set_ratio, label_name, positive_label,
                         numeric_attribute_names, categorical_attribute_names, attributes_to_drop_names,
                         train_data_sampler, missing_value_handler, numeric_attribute_scaler, learners, pre_processors,
                         post_processors, protected_attribute_names,
                         privileged_classes, privileged_groups, unprivileged_groups, dataset_metadata, 'psid' + "_".join(privileged_classes_concat))


    def load_raw_data(self):
        data = pd.read_csv(self.data_file, na_values='', sep=',', index_col="Unnamed: 0")

        ### Process the data according to what we need.

        # Transform the classification attribute into a binary one.
        data["intermediate"] = data[self.label_name].copy()
        data.loc[data[self.label_name] <= self.threshold_value,"intermediate"] = self.negative_label
        data.loc[data[self.label_name] > self.threshold_value, "intermediate"] = self.positive_label
        data = data.drop(self.label_name, 1)
        data.rename(columns={"intermediate":self.label_name}, inplace=True)

        # Transform the evaluation protected attributes into binary ones.
        for protected_att, protected_class_list in self.protected_attributes:
            # We create a new attribute with the binary values.
            data.loc[data[protected_att].isin(protected_class_list), "eval_" + protected_att] = "_".join(protected_class_list)
            data.loc[~data[protected_att].isin(protected_class_list), "eval_" + protected_att] = "not_in_" +  "_".join(protected_class_list)
            data.loc[data[protected_att] == np.nan, "eval_" + protected_att ] = np.nan
        
        print(data[self.label_name].value_counts(dropna=False))
        return data