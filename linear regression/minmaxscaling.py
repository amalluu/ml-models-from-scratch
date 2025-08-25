
X = [1000, 1500, 2000]#SIZES
Y=  [195, 305, 390]#PRICES



def minmaxscaler(Data):
    scaled_results =[]

    min_val=min(Data)
    max_val =max(Data)
    for value in Data:
        scaled_val = (value-min_val)/(max_val- min_val)
        scaled_results.append(scaled_val)
        print(f"Scaling {value} → {scaled_val}") 
    print("Final scaled_results:", scaled_results)  # Print before returning

    return scaled_results
minmaxscaler(X)
minmaxscaler(Y)

