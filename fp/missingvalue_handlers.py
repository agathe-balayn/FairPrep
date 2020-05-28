from sklearn.impute import SimpleImputer
import numpy as np
import datawig
import copy
import pandas as pd

class MissingValueMethodDispatcher:
    def __init__(self, method_list, data, handle_protected=False, protected_att=[]):
        
        self.handle_protected = handle_protected
        if self.handle_protected:
            self.protected_att_list = protected_att
            # There might be multiple protected attributes. We need to merge them all to handle only one binary protected / nonprotected scenario.

            self.protected_att = "_".join(protected_att)
            data_to_compute = data.copy(deep=True)
            #print((data_to_compute[protected_att].values.astype(str).tolist()))
            #print("_".join(data_to_compute[protected_att].values.astype(str).tolist()))
            #data_to_compute[self.protected_att] = "_".join(data_to_compute[protected_att].values.astype(str).tolist())
            data_to_compute[self.protected_att] = (data_to_compute[protected_att].values.astype(str).tolist())
            data_to_compute[self.protected_att] = data_to_compute[self.protected_att].apply(lambda row: "_".join(row))

            list_classes = data_to_compute[self.protected_att].unique()
            for class_name in list_classes:
                if ("nan" not in class_name) and ("not_in_" not in class_name):
                    self.protected_class = class_name
                    break

        methods_numerical = [CompleteCaseAnalysis, ModeImputer, MeanImputer, MedianImputer, DummyImputerNumerical, DataWigSimpleImputer]
        methods_categorical = [CompleteCaseAnalysis, ModeImputer, DummyImputerCategorical, DataWigSimpleImputer]
        
        # Either we input an object - when specifying the columns directly, or a class and here we instanciate the columns.

        if (type(method_list[0]) == type) and  (type(method_list[1]) == type):

            
            if method_list[0] not in methods_numerical:
                raise "Error: missing value method not applicable to numerical attributes."
            if method_list[1] not in methods_categorical:
                raise "Error: missing value method not applicable to categorical attributes."

            # Here we need to instanciate the classes. We have to get the columns first.
            num_df = data.select_dtypes(include='number')
            cat_df = data.select_dtypes(exclude='number')

            if self.handle_protected:
                # We need to create a class per protected class.
                self.missing_value_handler_numerical = [method_list[0](list(num_df.columns)), method_list[0](list(num_df.columns))]
                self.missing_value_handler_categorical = [method_list[1](list(cat_df.columns)), method_list[1](list(cat_df.columns))]
            else:
                self.missing_value_handler_numerical = method_list[0](list(num_df.columns))
                self.missing_value_handler_categorical = method_list[1](list(cat_df.columns))
        
        elif isinstance(method_list[0], tuple(methods_numerical)) and  isinstance(method_list[1], tuple(methods_categorical)):   
            if self.handle_protected:
                # We need to create a class per protected class.
                self.missing_value_handler_numerical = [method_list[0], copy.deepcopy(method_list[0])]
                self.missing_value_handler_categorical = [method_list[1], copy.deepcopy(method_list[1])]
            else:        
                self.missing_value_handler_numerical = method_list[0]
                self.missing_value_handler_categorical = method_list[1]

        else:
            raise "Error: missing value method not applicable to given attributes."

    def name(self):
        if self.handle_protected:
            name_ = "numerical_categorical_dispatcher_" + self.missing_value_handler_numerical[0].name() + "_" + self.missing_value_handler_categorical[0].name()
        else:
            name_ = "numerical_categorical_dispatcher_" + self.missing_value_handler_numerical.name() + "_" + self.missing_value_handler_categorical.name()
        return name_

    def fit(self, df):
        if self.handle_protected:
            # We need to separate the dataframe into protected and non protected rows.
            df[self.protected_att] = (df[self.protected_att_list].values.astype(str).tolist())
            df[self.protected_att] = df[self.protected_att].apply(lambda row: "_".join(row))

            protected_df = df.loc[df[self.protected_att] == self.protected_class]
            non_protected_df = df.loc[df[self.protected_att] != self.protected_class]

            
            protected_df.drop(columns=[self.protected_att], inplace=True)
            non_protected_df.drop(columns=[self.protected_att], inplace=True)
            df.drop(columns=[self.protected_att], inplace=True)

            num_protected_df = protected_df.select_dtypes(include='number')
            cat_protected_df = protected_df.select_dtypes(exclude='number')
            num_non_protected_df = non_protected_df.select_dtypes(include='number')
            cat_non_protected_df = non_protected_df.select_dtypes(exclude='number')

            self.missing_value_handler_numerical[0].fit(num_protected_df)
            self.missing_value_handler_numerical[1].fit(num_non_protected_df)
            self.missing_value_handler_categorical[0].fit(cat_protected_df)
            self.missing_value_handler_categorical[1].fit(cat_non_protected_df)

        else:
            num_df = df.select_dtypes(include='number')
            cat_df = df.select_dtypes(exclude='number')
            self.missing_value_handler_numerical.fit(num_df)
            self.missing_value_handler_categorical.fit(cat_df)

    def handle_missing(self, df):
        if self.handle_protected:
            df[self.protected_att] = (df[self.protected_att_list].values.astype(str).tolist())
            df[self.protected_att] = df[self.protected_att].apply(lambda row: "_".join(row))

            # We need to separate the dataframe into protected and non protected rows.
            protected_df = df.loc[df[self.protected_att] == self.protected_class]
            non_protected_df = df.loc[df[self.protected_att] != self.protected_class]

            
            protected_df.drop(columns=[self.protected_att], inplace=True)
            non_protected_df.drop(columns=[self.protected_att], inplace=True)
            df.drop(columns=[self.protected_att], inplace=True)

            num_protected_df = protected_df.select_dtypes(include='number')
            cat_protected_df = protected_df.select_dtypes(exclude='number')
            num_non_protected_df = non_protected_df.select_dtypes(include='number')
            cat_non_protected_df = non_protected_df.select_dtypes(exclude='number')

            imputed_num_protected_df = self.missing_value_handler_numerical[0].handle_missing(num_protected_df)
            inputed_cat_protected_df = self.missing_value_handler_categorical[0].handle_missing(cat_protected_df)
            imputed_num_non_protected_df = self.missing_value_handler_numerical[1].handle_missing(num_non_protected_df)
            imputed_cat_non_protected_df = self.missing_value_handler_categorical[1].handle_missing(cat_non_protected_df)

            # Merge all the imputed dataframes.
            imputed_protected_df = pd.concat([imputed_num_protected_df, inputed_cat_protected_df], axis=1, join='inner')
            imputed_non_protected_df = pd.concat([imputed_num_non_protected_df, imputed_cat_non_protected_df], axis=1, join='inner')
            # Merge the protected and non protected dataframes.
            imputed_df = pd.concat([imputed_protected_df, imputed_non_protected_df], axis=0)
      
            return imputed_df


        else:
            #print("DF shape", df.shape)
            num_df = df.select_dtypes(include='number')
            cat_df = df.select_dtypes(exclude='number')
            #print("num DF shape", num_df.shape)
            #print("cat DF shape", cat_df.shape)
            imputed_num_df = self.missing_value_handler_numerical.handle_missing(num_df)
            imputed_cat_df = self.missing_value_handler_categorical.handle_missing(cat_df)
            #print("imp num DF shape", imputed_num_df.shape)
            #print("imp cat DF shape", imputed_cat_df.shape)
            #concat_result = pd.concat([imputed_num_df, imputed_cat_df], axis=1, join='inner')
            #print(concat_result.shape)

            imputed_df = pd.concat([imputed_num_df, imputed_cat_df], axis=1, join='inner')
           
            return imputed_df


class MissingValueHandler:

    def name(self):
        raise NotImplementedError

    def fit(self, df):
        raise NotImplementedError

    def handle_missing(self, df):
        raise NotImplementedError


class CompleteCaseAnalysis(MissingValueHandler):

    def __init__(self, columns=""):
        super().__init__()

    def name(self):
        return 'complete_case'

    def fit(self, data):
        pass

    def handle_missing(self, df):
        return df.dropna()


class ModeImputer(MissingValueHandler):

    def __init__(self, columns):
        self.columns = columns
        self.imputers = {}
        super().__init__()

    def name(self):
        return 'mode_imputation'

    def fit(self, df):
        for column in self.columns:
            values = np.array(df[column]).reshape(-1, 1)
            self.imputers[column] = SimpleImputer(strategy='most_frequent').fit(values)

    def handle_missing(self, df):
        imputed_df = df.copy(deep=True)
        for column in self.columns:
            values = np.array(df[column]).reshape(-1, 1)
            imputed_df[column] = self.imputers[column].transform(values)
        return imputed_df

class MeanImputer(MissingValueHandler):

    def __init__(self, columns):
        self.columns = columns
        self.imputers = {}
        super().__init__()

    def name(self):
        return 'mean_imputation'

    def fit(self, df):
        for column in self.columns:
            values = np.array(df[column]).reshape(-1, 1)
            self.imputers[column] = SimpleImputer(strategy='mean').fit(values)

    def handle_missing(self, df):
        imputed_df = df.copy(deep=True)
        for column in self.columns:
            values = np.array(df[column]).reshape(-1, 1)
            imputed_df[column] = self.imputers[column].transform(values)
        return imputed_df

class MedianImputer(MissingValueHandler):

    def __init__(self, columns):
        self.columns = columns
        self.imputers = {}
        super().__init__()

    def name(self):
        return 'median_imputation'

    def fit(self, df):
        for column in self.columns:
            values = np.array(df[column]).reshape(-1, 1)
            self.imputers[column] = SimpleImputer(strategy='median').fit(values)

    def handle_missing(self, df):
        imputed_df = df.copy(deep=True)
        for column in self.columns:
            values = np.array(df[column]).reshape(-1, 1)
            imputed_df[column] = self.imputers[column].transform(values)
        return imputed_df


class DummyImputerNumerical(MissingValueHandler):

    def __init__(self, columns):
        self.columns = columns
        self.imputers = {}
        super().__init__()

    def name(self):
        return 'dummy_imputation'

    def fit(self, df):
        for column in self.columns:
            values = np.array(df[column]).reshape(-1, 1)
            self.imputers[column] = SimpleImputer(strategy='constant', fill_value="99999999").fit(values)

    def handle_missing(self, df):
        imputed_df = df.copy(deep=True)
        for column in self.columns:
            values = np.array(df[column]).reshape(-1, 1)
            imputed_df[column] = self.imputers[column].transform(values)
        return imputed_df


class DummyImputerCategorical(MissingValueHandler):

    def __init__(self, columns):
        self.columns = columns
        self.imputers = {}
        super().__init__()

    def name(self):
        return 'dummy_imputation'

    def fit(self, df):
        for column in self.columns:
            values = np.array(df[column]).reshape(-1, 1)
            self.imputers[column] = SimpleImputer(strategy='constant', fill_value="dummy").fit(values)

    def handle_missing(self, df):
        imputed_df = df.copy(deep=True)
        for column in self.columns:
            values = np.array(df[column]).reshape(-1, 1)
            imputed_df[column] = self.imputers[column].transform(values)
        return imputed_df

class DataWigSimpleImputer(MissingValueHandler):

    def __init__(self, columns_to_impute, label_column, out):
        self.columns_to_impute = columns_to_impute
        self.label_column = label_column
        self.out = out
        self.imputers = {}
        super().__init__()

    def name(self):
        return 'datawig_imputation'

    def fit(self, df):
        for column in self.columns_to_impute:
            input_columns = list(set(df.columns) - set([self.label_column, column]))
            self.imputers[column] = datawig.SimpleImputer(input_columns=input_columns,
                                                          output_column=column, output_path=self.out).fit(train_df=df)
            
    def handle_missing(self, df):
        imputed_df = df.copy(deep=True)
        for column in self.columns_to_impute:
            imputed_df[column] = self.imputers[column].predict(df)[column + '_imputed']
        return imputed_df
