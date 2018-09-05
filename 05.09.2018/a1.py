import numpy as np
import random
import matplotlib.pyplot as plt
data = open ("iris.data.txt" , "r").readlines();
for i in range(len(data)):
    data[i] = data[i].strip()
    data[i] = data[i].split(",")
input_data=[]
print len(data)
print data[0]

print data[150]
data.remove(data[150])

labels=['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
r=[]

for j in data:
    #print j[4]
    
    r.append([float(j[4]==labels[0]), float(j[4]==labels[1]) , float(j[4]==labels[2])]) 

new_data=[]
new_data.append(1.0)
print new_data
for d in data:
    for i in range(len(d)):
        if i<=3:
            new_data.append(float(d[i])) 
              
    input_data.append(new_data)
    new_data=[]
    new_data.append(1.0)
hidden=7
input=4
output_dim=3
w=[]
b=[]
print input_data[4]
print input_data[5]
# weights for input features
for j in range(5):
    for h in range(hidden+1):
        first_weight=random.uniform(-0.01,0.01)
        b.append(first_weight) 
    w.append(b)
    b=[]
w=np.matrix(w)

a=[]
c=[]

#weights for hidden layer
for h in range(hidden+1):
    for i in range(output_dim):
        second_weight=random.uniform(-0.01,0.01)
        c.append(second_weight) 
    a.append(c)
    c=[]  
a=np.matrix(a)
input_data=np.matrix(input_data)

output=[]

CMissClassificationList=[]
iterList=[]
iterations= 100 #number of iterations for MLP
for iter in range(iterations):
    print w.shape
    print input_data.shape
    z = np.matrix.dot(input_data,w)
    z = 1.0/(1.0 + np.exp(-z))
    for i in range(150):
        z[j,0]=1.0
    iterList.append(iter+1)
    #print z
    y = np.matrix.dot(z,a)
    output = 1.0/(1.0 + np.exp(-y))

    CMissClassification=0
    index=0
    RIndex=[]
    for i in r:
        for d in range(3):
            if i[d]==1.0:
                RIndex.append(d)
    for j in output:
        if j.argmax()!=RIndex[index]:
            CMissClassification+=1  
        index+=1  
    print CMissClassification         
    print "the number of errors is" + str(CMissClassification)
    CMissClassificationList.append(CMissClassification)
    V = np.zeros((hidden+1,output_dim))
    etha=0.006
    for h in range(hidden+1):
        TotalV = np.zeros((1,output_dim))
        for i in range(len(input_data)):
            TotalV += ((r[i]-output[i])*z[i,h])
            V[h] = etha*TotalV
 
    WHj = np.zeros((5,hidden+1))
    for j in range(5):
        for h in range(hidden+1):
            WTotal = 0
            for t in range(len(input_data)):
                c = np.multiply((r[t]-output[t]) , a[h]).sum()
                d = c * z[t,h] * (1 - z[t,h]) * input_data[t,j]
                WTotal += d
            WHj[j,h] = etha*WTotal
    w+=WHj  
    a+=V        
plt.plot(iterList,CMissClassificationList)
plt.xlabel('iteration')
plt.ylabel('number of missclassifications')    
plt.show()
