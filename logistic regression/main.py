import math

def sigmoid_converter(score):
    
    formula = 1/(1+ math.exp(-score))
    return formula

'''DATASET CREATION'''

# Features: [num_exclamation_marks, num_caps_words, email_length]
emails = [
    [0, 1, 50],   # Normal email
    [5, 8, 200],  # Likely spam
    [1, 2, 100],  # Normal email  
    [10, 15, 300] # Definitely spam
]

# Extract features
exc_mark=[]
caps=[]
length=[]
for sublist in emails:
    exc_mark.append(sublist[0])
    caps.append(sublist[1])
    length.append(sublist[2])
'''print("exc",exc_mark)
print("caps", caps)
print("length", length)
'''
labels = [0, 1, 0, 1]  # 0=not spam, 1=spam  #  y train

# Min-max scaling function
def minmaxscaler(Data):
    scaled_results =[]
    min_val=min(Data)
    max_val =max(Data)
    for value in Data:
        scaled_val = (value-min_val)/(max_val- min_val)
        scaled_results.append(scaled_val)
    return scaled_results

# Scale each feature separately
scaled_excmark=minmaxscaler(exc_mark)#[0.0, 0.5, 0.1, 1.0]
scaled_caps=minmaxscaler(caps)#[0.0, 0.5, 0.07142857142857142, 1.0]
scaled_len=minmaxscaler(length)#[0.0, 0.6, 0.2, 1.0]

# Reconstruct scaled dataset
newlist=[] #X train   [[0.0, 0.0, 0.0]-mail1, [0.5, 0.5, 0.6]-mail2 , [0.1, 0.07142857142857142, 0.2], [1.0, 1.0, 1.0]]
for i in range(4):
    newlist.append([scaled_excmark[i] ,scaled_caps[i], scaled_len[i]])

print("Training Data:")
print("X_train:", newlist)
print("y_train:", labels)
print()

#choose your weights
w1 = 0.8  # Weight for exclamation marks
w2 = 1.2  # Weight for caps words  
w3 = 0.5  # Weight for email length
bias = -2.0 #assume emails are NOT spam unless the features convince me otherwise(negative bias)
learning_rate = 0.1

print(f"Initial weights: w1={w1}, w2={w2}, w3={w3}, bias={bias}")
print(f"Learning rate: {learning_rate}")
print()

# TRAINING LOOP
for mainloop in range(50):

    # Predictions and cost
    cost=[]
    gradient_w1 = []
    gradient_w2 = []
    gradient_w3 = []
    gradient_bias = []
    for i in range(len(newlist)):
        mailfeatures= newlist[i]

        # Calculate prediction
        score_mail=w1*mailfeatures[0]+ w2*mailfeatures[1]+w3*mailfeatures[2] + bias
        email_prob =sigmoid_converter(score_mail)#prediction    

        #Calculate cost
        if labels[i] == 1:#spam
            cost.append( -math.log(email_prob))
            #print(f"email{i+1} cost:{cost}")
        elif labels[i]== 0:#not spam
            cost.append( -math.log(1 - email_prob))
            #print(f"email{i+1} cost:{cost}")

        # Calculate gradients- How much to change each weight to reduce cost
        #email_prob- labels[i] = error
        gradient_w1.append((email_prob- labels[i])*newlist[i][0])
        gradient_w2.append((email_prob- labels[i])*newlist[i][1])
        gradient_w3.append((email_prob- labels[i])*newlist[i][2])
        gradient_bias.append((email_prob - labels[i]))
        
        
     # Calculate average cost and gradients

    average=sum(cost)/len(cost)
    #print("average: ", average)#Average cost = 0.467 -   how "wrong" your randomly chosen weights are on average across all emails
    averagew1=sum(gradient_w1)/len(gradient_w1)
    #print("averagew1: ", averagew1)
    averagew2=sum(gradient_w2)/len(gradient_w2)
    #print("averagew2: ", averagew2)
    averagew3=sum(gradient_w3)/len(gradient_w3)
    #print("averagew3: ", averagew3)
    averagebias=sum(gradient_bias)/len(gradient_bias)
    #print("averagebias: ", averagebias)


    '''
        Average w1 gradient = -0.174 → INCREASE w1 (exclamation marks weight)
        Average w2 gradient = -0.175 → INCREASE w2 (caps words weight)
        Average w3 gradient = -0.187 → INCREASE w3 (email length weight)
        Average bias gradient = -0.194 → INCREASE bias
        '''
    
    # Update weights
    w1 = w1 - learning_rate * averagew1
    w2 = w2 - learning_rate * averagew2  
    w3 = w3 - learning_rate * averagew3
    bias = bias - learning_rate * averagebias

    # Print progress
    if mainloop % 10 == 0:#If the mainloop number is divisible by 10
        print(f"Loop {mainloop:2d}: Cost = {average:.4f}")

print()


# TEST THE IMPROVED MODEL
print("\n--- TESTING IMPROVED MODEL ---")
print(f"Updated weights: w1={w1:.3f}, w2={w2:.3f}, w3={w3:.3f}, bias={bias:.3f}")

print("TESTING TRAINED MODEL:")
print("-" * 20)
for i in range(len(newlist)):
    features = newlist[i]
    score = w1*features[0] + w2*features[1] + w3*features[2] + bias
    prediction = sigmoid_converter(score)
    actual = labels[i]
    
    print(f"Email {i+1}: Features {features}")
    print(f"  Predicted: {prediction:.1%} spam")
    print(f"  Actual: {'Spam' if actual == 1 else 'Not Spam'}")
    print(f"  Correct: {'✅' if (prediction > 0.5) == actual else '❌'}")
    print()

print("CONGRATULATIONS! You built logistic regression from scratch!")