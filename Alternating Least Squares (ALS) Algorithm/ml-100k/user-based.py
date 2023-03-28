import time
import numpy as np
from statistics import mean

def dicti():
    return {}

def open_file():
    return open('data.data','r')

def get_indi(str,indi):
    return str[indi]

def return0():
    return 0    

def square_root(val):
    return np.sqrt(val)

def get_dickey(dict):
    return list(dict.keys())  

def multiply(a,b):
    return a*b

def changetoint(val):
    return int(val)

def get_one():
    return 1

def assign_num(num2):
    return num2

def is_npzero(mat):
    return np.zeros((mat.shape[0],mat.shape[0]))

def get_logging(num):
    return np.log(1+num)

def random_shuffle(dataset):
    return np.random.shuffle(dataset)

def fold1(num):
    print("E1 : ",num)

def fold2(num):
    print("E2 : ",num)

def fold3(num):
    print("E3 : ",num)

def fold4(num):
    print("E4 : ",num)

def fold5(num):
    print("E5 : ",num)

def print_k(num):
    print("K = ",num)

def get_sum(num):
    return np.sum(num)

def get_logical(dataset):
    return np.logical_and(dataset[i,:] > 0, dataset[j,:] > 0)

average_rating = dicti()

def file(filename):
    filename=open(filename,'r')
    file_list=[]
    for line in filename:
        split_line=line.split("\t")
        arr0=get_indi(split_line,0)
        arr1=get_indi(split_line,1)
        arr2=get_indi(split_line,2)
        file_list.append([int(arr0),int(arr1),float(arr2)])
    dataset=np.asarray(file_list)
    return dataset


similarity_dict = dict()
def similarity(matrix):     
    users_index=get_dickey(matrix)    
    for i, user1 in enumerate(users_index):
        similarity_dict[user1] = dict()
        for j, user2 in enumerate(users_index):
            if(j>=i):
                similarity_dict[user1][user2]=cos_distance(user1, user2, matrix)
            else:
                similarity_dict[user1][user2]=similarity_dict[user2][user1]
    print(similarity_dict[1])
    return similarity_dict

def cos_distance(user1, user2, matrix): 
    cos_similarity=0
    dist1=0
    dist2=0
    for item in matrix[user1]: 
        if item in matrix[user2]:
            cos_similarity+=multiply(matrix[user1][item], matrix[user2][item])
            dist1 += multiply(matrix[user1][item] , matrix[user1][item])
            dist2 += multiply(matrix[user2][item] , matrix[user2][item])

    ele1=square_root(dist1)
    ele2=np.sqrt(dist2)
    if (multiply(ele1,ele2))==0:
        similarity=return0()
    else:
        similarity=cos_similarity/(ele1*ele2)
    return similarity


def get_matrix(data):
    global average_rating
    average_rating={}
    count=dicti()
    user_vectors={}
    for i in data:
        zero=get_indi(i, 0)
        one=get_indi(i, 1)
        user=int(zero)
        item=int(one)
        
        if user not in user_vectors:
            user_vectors[user]=dicti()
            average_rating[user]=0
            count[user]=0
        
        user_vectors[user][item]=get_indi(i, 2)
        average_rating[user]+=get_indi(i, 2)
        
        count[user]+=get_one()
    
    for key in average_rating:
        average_rating[key]=average_rating[key]/count[key]
    
    return user_vectors


def model(check,train,n):
    
    matr=get_matrix(train)
    sim=similarity(matr)
    MAE=get_mean_error(check,matr,sim,n)
    return MAE

def predict(user, movie, similarity, matrix, n_count ): 
    global average_rating
    rating = return0()
    k = assign_num(n_count)
    neighbours = []
    
    for key in similarity[user]:
        neighbours.append( (similarity[user][key], key) )
    neighbours.sort(reverse = True)
    neighbours = neighbours[:k]

    sum_sim = return0()
    for val1 in neighbours:
        val=get_indi(val1, 1)
        if movie in matrix[val]:
            rating+=multiply(similarity[user][val], matrix[val][movie] - average_rating[val] )
            sum_sim += similarity[user][val]

    if sum_sim == return0():
        return get_indi(average_rating, user)

    else:
        rating = rating/sum_sim
        rating += get_indi(average_rating, user)
        return rating


def significance_matrix(dataset):
    significance_matrix=is_npzero(dataset)
    for i in range(dataset.shape[0]):
        for j in range(dataset.shape[0]):
            if i != j:
                common_items = get_logical(dataset)
                num_common_items = get_sum(common_items)
                significance_matrix[i,j]=get_logging(num_common_items)
                
            else:
                significance_matrix[i, j] = 1
                



def get_mean_error(test, matrix, similarity, n_count):
    two=assign_num(2)
    one=assign_num(1)
    zero=assign_num(0)
    errors = [abs(values[two] - predict(values[zero], values[one], similarity, matrix, n_count)) for values in test]
    return mean(errors)

x=assign_num(1)


def k_fold(u1_t,u1_b,u2_t,u2_b,u3_t,u3_b,u4_t,u4_b,u5_t,u5_b, n_count):
    error = assign_num(0.0)
    test=u1_t
    train=u1_b
    temp = model(test, train, n_count)
    error = error+temp
   # print("E1: ",temp)
    fold1(temp)
    two=assign_num(2)
    test=u2_t
    train=u2_b
    temp = model(test, train, n_count)
    error += temp
    fold2(temp)
    three=assign_num(3)
    test=u3_t
    train=u3_b
    temp = model(test, train, n_count)
    error += temp
   # print("E3: ",temp)
    fold3(temp)
    four=assign_num(4)
    test=u4_t
    train=u4_b
    temp = model(test, train, n_count)
    error += temp
    # print(temp)
   # print("E4: ",temp)
    fold4(temp)
    test=u5_t
    train=u5_b
    temp = model(test, train, n_count)
    error += temp
    # print(temp)
    #print("E5: ",temp)
    fold5(temp)

    five=assign_num(5)
    print(error/5)
    # x=x+1
    return error/5


dataset = file('ml-100k/u.data')
u1_test,u2_test,u3_test,u4_test,u5_test=file("ml-100k/u1.test"),file("ml-100k/u2.test"),file("ml-100k/u3.test"),file("ml-100k/u4.test"),file("ml-100k/u5.test")
u1_base,u2_base,u3_base,u4_base,u5_base=file("ml-100k/u1.base"),file("ml-100k/u2.base"),file("ml-100k/u3.base"),file("ml-100k/u4.base"),file("ml-100k/u5.base")
error = []
n_count =[]
n_count.append(10)
n_count.append(20)
n_count.append(30)
n_count.append(40)
n_count.append(50)
for n in n_count:
    print_k(n)
   # error.append(k_fold(dataset, n))
    error.append(k_fold(u1_test,u1_base,u2_test,u2_base,u3_test,u3_base,u4_test,u4_base,u5_test,u5_base, n))

print("Mean average error for all folds : ",error)

