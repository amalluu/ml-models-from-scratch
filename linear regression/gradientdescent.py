
X = [1000, 1500, 2000]#SIZES
Y=  [195, 305, 390]#PRICES




def minmaxscaler(Data):
    scaled_results =[]

    min_val=min(Data)
    max_val =max(Data)
    for value in Data:
        scaled_val = (value-min_val)/(max_val- min_val)
        scaled_results.append(scaled_val)
    

    return scaled_results
scaled_x=minmaxscaler(X)
scaled_y=minmaxscaler(Y)


m = 1.0#slope
b = -0.0#y intercept
learning_rate= 0.01

for iteration in range(10):
   total_error=0
   total_sum=0
   total_sum_m=0
   for i in range(len(scaled_x)):
       x= scaled_x[i]# Current house size
       y= scaled_y[i]
       predicted_price= m*x+b
       error= y- predicted_price
       square= error**2
       total_error += square
       total_sum += error
       total_sum_m += x*error
   deri_m= -(1/(len(scaled_x)))*total_sum_m
   deri_b= -(1/(len(scaled_x)))*total_sum

   print("total error:", total_error)
   '''  print("deri m", deri_m)
        print("deri b", deri_b)
    '''
   m= m- learning_rate*deri_m
   b= b- learning_rate*deri_b
       
   

final_m= m
final_b= b


new_size= 0.78 # Some scaled size between 0 and 1
new_prediction= final_m*new_size + final_b

print("Price predicted by MY MODEL IS",new_prediction)