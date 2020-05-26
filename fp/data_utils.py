# Get all binary strings for n bits. 
# This function is used to create all possible combinations of protected attributes.

def printTheArray(arr, n, saved_array):  
    to_save = []
    for i in range(0, n):  
        #print(arr[i], end = " ")  
        to_save += [arr[i]]
    saved_array += [to_save]
    #print() 
    return saved_array

def generateAllBinaryStrings(n, arr, i, saved_array):  
  
    if i == n: 
        printTheArray(arr, n, saved_array) 
        return saved_array
      
    # First assign "0" at ith position  
    # and try for all other permutations  
    # for remaining positions  
    arr[i] = 0
    generateAllBinaryStrings(n, arr, i + 1, saved_array)  
  
    # And then assign "1" at ith position  
    # and try for all other permutations  
    # for remaining positions  
    arr[i] = 1
    generateAllBinaryStrings(n, arr, i + 1, saved_array)  
    
    return saved_array