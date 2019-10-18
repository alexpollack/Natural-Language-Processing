
# coding: utf-8

# In[1]:

import numpy as np
from numpy.linalg import norm as norm
from numpy import log, exp, matmul as mm, transpose as Tr, dot
import pandas as pd
import csv
from scipy.sparse import csc_matrix,csr_matrix,lil_matrix
import collections


# In[13]:

#open('data2-2018/pnp-train.txt', 'r', encoding='iso-8859-1')


# In[464]:

train = pd.read_fwf('data2-2018/pnp-train.txt',header=None,encoding='iso-8859-1') #latin1
validate = pd.read_fwf('data2-2018/pnp-validate.txt',header=None,encoding='iso-8859-1')
test = pd.read_fwf('data2-2018/pnp-test.txt',header=None,encoding='iso-8859-1')


# In[465]:

train.drop(2,axis=1, inplace=True)
validate.drop(2,axis=1, inplace=True)


# In[466]:

train.rename(columns={0: 'Label',1: 'Phrase'},inplace=True)
validate.rename(columns={0: 'Label',1: 'Phrase'},inplace=True)
test.rename(columns={0: 'Phrase'},inplace=True)


# In[5]:

train.head(5)


# In[6]:

encoded_classes = dict({'drug': 0, 'pers': 1, 'plac': 2, 'movi': 3, 'comp': 4})


# In[7]:

classes = list(train['Label'].unique())
print(classes)


# # Training Set

# ## Objective function

# ### Unigram features f(x)

# In[8]:

# ### Get features as unigrams f('Xylexx') = [..., "X" : 1.0, "y" : 1.0, "l" : 1.0, "e" : 1.0, "x" : 2.0, ...]
# ### f(x) is 256x1 for the extended ASCII
# def f(s):
#     fx = [0] * 256
# #     fx = csc_matrix((256, 1), dtype=np.int8).toarray()
#     for i in range(len(s)):
#         if i < len(s)-1 :
# #             if fx[ord(s[i])] > 0:
#             if ord(s[i]) == 9:
#                 fx[ord(s[i])] = fx[ord(s[i])] + 10
#             elif ord(s[i]) == 10:
#                 fx[ord(s[i])] = fx[ord(s[i])] + 10
#             else:
#                 fx[ord(s[i])] = fx[ord(s[i])] + 1
#     fx = np.asarray(fx).reshape(len(fx),1)

# #     fx = np.asarray(fx,dtype=np.float128).reshape(len(fx),1)
#     return fx


# ### Bigram features f(x)

# In[509]:

0-31 
# 127
# 157 - 190
N = 65*65 + 1 # plus lenth of phrase


# In[407]:

#### workin block
def f(s):
    #### add in number of words, number of capitals,
    ### ' -' and '- ' in movies vs 'char-' and '-char' in drug 
    n = len(s)
    data = np.zeros(n-1)
    row = np.zeros(n-1)
    col = np.zeros(n-1)
    for i in range(n):
        if i < len(s)-1:
            row[i] = ord(s[i])*n + ord(s[i+1])
            if ord(s[i]) >= 48 and ord(s[i]) <= 57:
                data[i] = 15
            elif ord(s[i]) >= 65 and ord(s[i]) <= 90:
                if ord(s[i+1]) >= 65 and ord(s[i]) <= 90:
                    data[i] = 15
            elif ord(s[i]) == 45 and ord(s[i+1]) != 32: #'-char' in drug
                data[i] = 15
            elif ord(s[i]) != 32 and ord(s[i+1]) == 45: #'char-' in drug
                data[i] = 15
            
            
            elif ord(s[i]) >= 33 and ord(s[i]) <= 46: #!#$%&?
                data[i] = 15
            elif ord(s[i]) == 121 and ord(s[i+1]) == 9: ## ny\t in comp
                data[i] = 15
            elif ord(s[i]) == 110 and ord(s[i+1]) == 9: ###n\t in pers
                data[i] = 15
            elif ord(s[i]) == 32: # space, most places are one word
                data[i] = 10
            elif ord(s[i]) == 58:
                data[i] = 15
#             elif ord(s[i]) == 45:
#                 data[i] = 5
            else:
                data[i] = 1
    fx = csr_matrix((data, (row, col)), shape=(256*256, 1),dtype=np.float128)#.toarray()
    return fx


# In[249]:

# #### workin block
# def f(s):
    
#     n = len(s)
#     data = np.zeros(n-1)#np.zeros(256*2256)
#     row = np.zeros(n-1)
#     col = np.zeros(n-1)
#     for i in range(n):
#         if i < len(s)-1:
#             row[i] = ord(s[i])*n + ord(s[i+1])
#     #         col[i] = ord(s[i+1])
#             if ord(s[i]) == 9: #'\t'
#                 data[i] = 10
#             elif ord(s[i]) == 121 and ord(s[i+1]) == 9: ## ny\t in comp
#                 data[i] = 15
#             elif ord(s[i]) == 110 and ord(s[i+1]) == 9: ###n\t in pers
#                 data[i] = 15
#             elif ord(s[i]) == 10:
#                 data[i] = 10
#             elif ord(s[i]) == 58:
#                 data[i] = 15
#             else:
#                 data[i] = 1
#     fx = csr_matrix((data, (row, col)), shape=(256*256, 1),dtype=np.float128)#.toarray()
#     return fx


# ### Compute the gradient

# In[474]:

def grad(w,train):
    encoded_classes = dict({'drug': 0, 'pers': 1, 'plac': 2, 'movi': 3, 'comp': 4})
    
    lmbda = 5
    Lw = np.zeros((1,5))
    # take away sparse 
    pfx = csr_matrix(np.zeros((256*256,5)), shape=(256*256,5))#np.zeros((256*256,5))
    f_sum = csr_matrix(np.zeros((256*256,5)), shape=(256*256,5),dtype=np.float128)
    for I in range(len(train)):
        x = train.values[I][1]
        y = train.values[I][0]
        fx = f(x)

        num = np.zeros((1,5))
#         pfx = csr_matrix(np.zeros((256*256,5)), shape=(256*256,5))#np.zeros((256*256,5))
#         f_sum = csr_matrix(np.zeros((256*256,5)), shape=(256*256,5),dtype=np.float128)
        yi = encoded_classes.get(y)

        denom = 0
        ### remove loop and add just dot(w.T,fx) ??
        for i in range(w.shape[1]):
            wi = w[:,i]
            denom = denom + exp(dot(wi.T,fx).toarray()[0][0])
        for j in range(w.shape[1]):
            num[0,j] = exp(dot(w[:,j].T,fx).toarray()[0][0])
            pfx[:,j] = pfx[:,j] + (num[0,j]/denom)*fx
        
#         Lw[0,yi] = Lw[0,yi] + log(num[0,yi]/denom)
        f_sum[:,yi] = f_sum[:,yi] + fx
    dLdw = f_sum - pfx - 2*lmbda*w
    return dLdw#, Lw


# # Bigram  training

# In[469]:

train = train.sample(frac=0.062).reset_index(drop=True)
len(train)


# In[510]:

### Bigram block###
wtm1 = csr_matrix(np.random.rand(256*256,5), shape=(256*256,5))
t = 1#t = 0 # Step 0     
resid = 1
alpha = 0.05
eps = 0.0001
Lw_t = np.zeros((15,5))
t_t = []
y = train.Label
while (resid > eps): # (t == 0) or
    #dLdw, Lw = grad(wtm1,train)
    dLdw = grad(wtm1,train)
    Lw_t[(t-1),:]  = Lw
    t_t.append(t)
    wt = wtm1.copy() + alpha*dLdw
    resid = norm(np.float64((wt - wtm1).toarray()),2) # Compute residual
#     print('t: %s, resid: %s, alpha: %s'%(t,norm(np.float64((wt - wtm1).toarray()),2),alpha))
    wtm1 = wt.copy()
    t = t + 1 # Next step
    if t < 10:
        alpha = alpha/np.sqrt(t) # Adjust step size


# In[ ]:

import matplotlib as plt
plt.(t_t,Lw_t)


# # Validate

# In[227]:

# validate = validate.sample(frac=1).reset_index(drop=True)
len(validate)


# In[477]:

#### Working bloc
### drug: 0-687
y_pred = 'none'
m = len(validate)

 
accuracy = 0
drug_accuracy = 0
pers_accuracy = 0
plac_accuracy = 0
mov_accuracy = 0
comp_accuracy = 0

for I in range(m):
    Pyx = -1
    ### Set current x y training pair
    x_val = validate.values[I][1]
    y_val = validate.values[I][0]
    tot = []
    [tot.append(x.isdigit()) for x in s.split(' ')]
    sm = sum(tot)
    ### Compute the feature vector f(x)
    val_fx = f(x_val) #np.array(f(x)).reshape(len(f(x)),1)
    prob = dot(wtm1.T,val_fx).toarray()
    
    for i in range(5):
        num_words = len(x_val.split(' '))
        if (i == 2) and num_words == 1:#i == 0 or
            prob[i,0] = abs(prob[i,0])*8
#             print(num_words,prob[i,0])
        elif (i == 0) and num_words == 1:#i == 0 or
            prob[i,0] = abs(prob[i,0])*6
            if '-' in x_val:#i == 0 or
                prob[i,0] = abs(prob[i,0])*6
#                 print(num_words,prob[i,0])
        elif '-' in x_val:
            prob[i,0] = abs(prob[i,0])*2
        elif i == 3 and num_words > 3:
            prob[i,0] = abs(prob[i,0])*2
#             print(num_words,prob[i,0])
        if prob[i,0] > Pyx:
            Pyx = prob[i,0]
            y_pred = classes[i]
#     print(x_val,y_val,y_pred,'\n',prob,'\n')
    
#     print('True: %s\nPred: %s\n'%(y_val,y_pred))
    if y_pred == y_val:
        accuracy = accuracy + 1
        if y_pred == 'drug':
            drug_accuracy = drug_accuracy + 1
        elif y_pred == 'pers':
            pers_accuracy = pers_accuracy + 1
        elif y_pred == 'plac':
            plac_accuracy = plac_accuracy + 1
        elif y_pred == 'movi':
            mov_accuracy = mov_accuracy + 1
        elif y_pred == 'comp':
            comp_accuracy = comp_accuracy + 1
accuracy = accuracy/m
print(accuracy)

    


# In[478]:

print(accuracy)
print('Drug: ',drug_accuracy/len(validate[validate.Label == 'drug']))
print('Pers: ',pers_accuracy/len(validate[validate.Label == 'pers']))
print('Plac: ',plac_accuracy/len(validate[validate.Label == 'plac']))
print('Movi: ',mov_accuracy/len(validate[validate.Label == 'movi']))
print('Comp: ',comp_accuracy/len(validate[validate.Label == 'comp']))
# overall accuracy: 0.6206896551724138
### best accuracy 0.6206896551724138 with
### alpha = 0.05, lmbda = 5, on 1% of trainig (~230) and 10% of validate (~230)



# 0.5924645696508815
# Drug:  0.47674418604651164
# Pers:  0.8880597014925373
# Plac:  0.31673306772908366
# Movi:  0.5916167664670658
# Comp:  0.7740963855421686



# 0.5966125129623229
# Drug:  0.39825581395348836
# Pers:  0.8899253731343284
# Plac:  0.4342629482071713
# Movi:  0.5988023952095808
# Comp:  0.7740963855421686

# 0.5737988247493951
# Drug:  0.4433139534883721
# Pers:  0.8675373134328358
# Plac:  0.4860557768924303
# Movi:  0.5077844311377245
# Comp:  0.6686746987951807


# # Test

# In[480]:

y_test_pred = 'none'
m = len(test)

test_preds = []
accuracy = 0
drug_accuracy = 0
pers_accuracy = 0
plac_accuracy = 0
mov_accuracy = 0
comp_accuracy = 0

for I in range(m):
    Pyx = -1
    ### Set current x y training pair
    x_test = test.values[I][0]
    tot = []
    [tot.append(x.isdigit()) for x in s.split(' ')]
    sm = sum(tot)
    ### Compute the feature vector f(x)
    test_fx = f(x_test) #np.array(f(x)).reshape(len(f(x)),1)
    prob = dot(wtm1.T,test_fx).toarray()
    
    for i in range(5):
        num_words = len(x_test.split(' '))
        if (i == 2) and num_words == 1:#i == 0 or
            prob[i,0] = abs(prob[i,0])*8
#             print(num_words,prob[i,0])
        elif (i == 0) and num_words == 1:#i == 0 or
            prob[i,0] = abs(prob[i,0])*6
            if '-' in x_val:#i == 0 or
                prob[i,0] = abs(prob[i,0])*6
#                 print(num_words,prob[i,0])
        elif '-' in x_val:
            prob[i,0] = abs(prob[i,0])*2
        elif i == 3 and num_words > 3:
            prob[i,0] = abs(prob[i,0])*2
#             print(num_words,prob[i,0])
        if prob[i,0] > Pyx:
            Pyx = prob[i,0]
            y_test_pred = classes[i]
            test_preds.append(y_test_pred)
#     print(x_test,y_test_pred,'\n',prob,'\n')


# In[481]:

with open('preds.txt', 'w') as file:
    for item in test_preds:
        file.write("%s\n" % item)

