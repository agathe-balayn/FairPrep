import numpy as np
import json

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