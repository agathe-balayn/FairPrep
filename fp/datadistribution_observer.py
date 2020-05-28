import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns; sns.set(style="ticks", color_codes=True)
import pandas as pd



# This can be applied on the raw data before and after sampling.

def missingValuesDistribution(data_to_investigate, protected_attributes, attributes_to_drop_names, privileged_groups, label_name, results_file):
    #print(protected_attributes) 
    #print(attributes_to_drop_names)   
     
    #print(privileged_groups)    
    #print(label_name)
    # Pre-process the data.    
    data = data_to_investigate.copy(deep=True)
    data = data.drop(attributes_to_drop_names, 1) 
    data.to_csv(results_file)


    # Everything can be done if we save the data!

    print("Handle save info.")


def getDataErrorDistributions(data, protected_attributes, attributes_to_drop_names, privileged_groups, label_name, results_file, list_error_types=["missing_values"]):
    print("TODO")
    if "missing_values" in list_error_types:
        missingValuesDistribution(data, protected_attributes, attributes_to_drop_names, privileged_groups, label_name, results_file)

def getNbMissingValues(data):
    print("Total number of missing values:", data.isnull().sum().sum())
    print("Missing values per feature:")
    for col in data.columns:
        pct_missing = np.mean(data[col].isnull())
        print('{} - {}%'.format(col, round(pct_missing*100)))

def plotMissingValueFeature(list_data, list_names=[], fig_size=(10,10)):
    nb_data = len(list_data)
    if nb_data > 1:
        #fig, ax = plt.subplots(figsize=(10,10))         # Sample figsize in inches
        if nb_data % 2 == 0:
            nb_row = int(nb_data/2)
            fig, axs = plt.subplots(nb_row, 2, sharex=True, figsize=fig_size)
        else:
            nb_row = int(nb_data/2+1)
            fig, axs = plt.subplots(nb_row, 2, sharex=True, figsize=fig_size)
        for i in range(nb_data):
            data = list_data[i]
            cols = data.columns
            colours = ['#000099', '#ffff00'] # specify the colours - yellow is missing. blue is not missing.

            if i % 2 == 0:
                if nb_row == 1:
                    _ax = axs[0]
                else:
                    _ax = axs[int(i/2), 0]
                
                sns.heatmap(data[cols].isnull(), cmap=sns.color_palette(colours), ax=_ax)#, ax=axs[int(i/2), 0])
                _ax.legend()
                if list_names != []:
                    _ax.set_title(list_names[i])
            else:
                if nb_row == 1:
                    _ax = axs[1]
                else:
                    _ax = axs[int(i/2), 1]
                sns.heatmap(data[cols].isnull(), cmap=sns.color_palette(colours), ax=_ax)
                if list_names != []:
                    _ax.set_title(list_names[i])
                _ax.legend()
        if nb_data % 2 != 0:
            for l in axs[int(i/2-1),1].get_xaxis().get_majorticklabels():
                l.set_visible(True)
            fig.delaxes(axs[int(i/2), 1])
        plt.show()
    else:
        fig, ax = plt.subplots(figsize=fig_size) 
        
        data = list_data[0]
        cols = data.columns
        colours = ['#000099', '#ffff00'] # specify the colours - yellow is missing. blue is not missing.
        sns.heatmap(data[cols].isnull(), cmap=sns.color_palette(colours), ax=ax)
        plt.show()
        
def plotMissingValueDistribution(list_df, list_names="", fig_size=(10,10)):
    nb_data = len(list_df)
    if nb_data > 1:
        #fig, ax = plt.subplots(figsize=(10,10))         # Sample figsize in inches
        if nb_data % 2 == 0:
            nb_row = int(nb_data/2)
            fig, axs = plt.subplots(nb_row, 2, sharex=True, sharey=True, figsize=fig_size)
        else:
            nb_row = int(nb_data/2+1)
            fig, axs = plt.subplots(nb_row, 2, sharex=True, sharey=True, figsize=fig_size)
        for i in range(nb_data):
            df = list_df[i]
            null_counts = df.isnull().sum()/len(df)
            if i % 2 == 0:
                if nb_row == 1:
                    _ax = axs[0]
                else:
                    _ax = axs[int(i/2), 0]                   
            else:
                if nb_row == 1:
                    _ax = axs[1]
                else:
                    _ax = axs[int(i/2), 1]
            #plt.figure(figsize=(10,8))
            _ax.set_xticks(np.arange(len(null_counts))+0.5)
            _ax.set_xticklabels( list(null_counts.index))
            _ax.set_ylabel('fraction of rows with missing data')
            _ax.bar(np.arange(len(null_counts)),null_counts)
            _ax.legend()
            plt.setp(_ax.get_xticklabels(), rotation='vertical')
            if list_names != []:
                _ax.set_title(list_names[i])
        if nb_data % 2 != 0:
            for l in axs[int(i/2-1),1].get_xaxis().get_majorticklabels():
                l.set_visible(True)
            fig.delaxes(axs[int(i/2), 1])
        plt.show()
        
    else:
        df = list_df[0]
        null_counts = df.isnull().sum()/len(df)
        fig, ax = plt.subplots(figsize=fig_size) 
        ax.set_xticks(np.arange(len(null_counts))+0.5)
        ax.set_xticklabels( list(null_counts.index))
        ax.set_ylabel('fraction of rows with missing data')
        ax.bar(np.arange(len(null_counts)),null_counts)
        ax.legend()
        plt.setp(ax.get_xticklabels(), rotation='vertical')
        plt.show()
        
def plotMissingValueDistributionMergedTables(list_df, list_names, fig_size=(10,10)):
    # Merge the dataframes.
    df_to_concat = [df.isnull().sum()/len(df) for df in list_df]
    df_to_concat = [_series.rename(_name) for _series, _name in zip(df_to_concat, list_names)]
    df = pd.concat(df_to_concat, axis=1)
    fig, ax = plt.subplots(figsize=fig_size) 
    df.plot.bar(ax=ax, width=0.7)
    ax.set_ylabel('fraction of rows with missing data')
    ax.legend()
    plt.setp(ax.get_xticklabels(), rotation='vertical')
    plt.show()
    
    

# Get protected attributes.
def getProtectedAttributes(data):
    protected_attributes = {}
    for col_names in list(data.columns):
        if "eval_" in col_names:
            protected_attributes[col_names] = {"protected":[], "not_protected":[]}
    # Get protected and non-protected values for each attribute.
    for key in protected_attributes:
        # Get unique values.
        unique_values_series = data[key].value_counts(dropna=False)
        attribute_values = (list(unique_values_series.index))
        for attribute_name in attribute_values:
            if "not" in attribute_name:
                protected_attributes[key]["not_protected"].append(attribute_name)
            elif attribute_name == np.nan:
                protected_attributes[key]["not_protected"].append(attribute_name)
            else:
                protected_attributes[key]["protected"].append(attribute_name)
    return protected_attributes

def createBinaryCondition(protected_attributes, data):
    condition_binary = True
    for element in protected_attributes:
        condition_binary = condition_binary & (data[element].isin(protected_attributes[element]["protected"]))
    return condition_binary

            
# Prepare the list of dataframes separated by the different types of attributes.            
def create4_4Conditions(protected_attributes, data):
    list_protected_attributes = list(protected_attributes.keys())
    list_condition = [(data[list_protected_attributes[0]].isin(protected_attributes[list_protected_attributes[0]]["protected"]) & data[list_protected_attributes[1]].isin(protected_attributes[list_protected_attributes[1]]["protected"])), \
                     (data[list_protected_attributes[0]].isin(protected_attributes[list_protected_attributes[0]]["protected"]) & data[list_protected_attributes[1]].isin(protected_attributes[list_protected_attributes[1]]["not_protected"])),\
                     (data[list_protected_attributes[0]].isin(protected_attributes[list_protected_attributes[0]]["not_protected"]) & data[list_protected_attributes[1]].isin(protected_attributes[list_protected_attributes[1]]["protected"])),\
                     (data[list_protected_attributes[0]].isin(protected_attributes[list_protected_attributes[0]]["not_protected"]) & data[list_protected_attributes[1]].isin(protected_attributes[list_protected_attributes[1]]["not_protected"]))]
    condition_names = [list_protected_attributes[0] + protected_attributes[list_protected_attributes[0]]["protected"][0] + list_protected_attributes[1] + protected_attributes[list_protected_attributes[1]]["protected"][0],\
                      list_protected_attributes[0] + protected_attributes[list_protected_attributes[0]]["protected"][0] + list_protected_attributes[1] + protected_attributes[list_protected_attributes[1]]["not_protected"][0],\
                      list_protected_attributes[0] + protected_attributes[list_protected_attributes[0]]["not_protected"][0] + list_protected_attributes[1] + protected_attributes[list_protected_attributes[1]]["protected"][0],\
                      list_protected_attributes[0] + protected_attributes[list_protected_attributes[0]]["not_protected"][0] + list_protected_attributes[1] + protected_attributes[list_protected_attributes[1]]["not_protected"][0]]
    return list_condition, condition_names    
