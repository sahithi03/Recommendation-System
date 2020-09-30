import sys
from pyspark import SparkContext
import time
import math
import itertools

sc = SparkContext('local[*]',"task2.1")
sc.setLogLevel("ERROR")

start = time.time()

train_file_name = sys.argv[1]
val_file_name = sys.argv[2]
output_file_name = sys.argv[3]

def minhash(x,m,a):
    #formula => ((ax+b) % m)
    return min(((a*x + 3) % m) for x in x[1])

def gen_candidates(x,no_of_bands,rows_per_band):
    #no_of_bands = (len(x[1]) // rows_per_band)
    res = []
    for b in range(no_of_bands):
        res.append(((b,tuple(x[1][b*rows_per_band:(b+1)*rows_per_band])),[x[0]]))
    return res

def get_pairs(x):
    candidates = []
    for t in itertools.combinations(x[1],2):
        pair = list(t)
        pair.sort()
        pair = tuple(pair)
        candidates.append((pair,1))
    return candidates

def LSH():
    users_rdd = train_rdd.map(lambda x: x[0]).distinct()
    business_rdd = train_rdd.map(lambda x: x[1]).distinct()

    users = users_rdd.collect()
    business = business_rdd.collect()

    no_of_rows = len(users)

    users_dict = dict()
    for idx, elem in enumerate(users):
        users_dict[elem] = idx

    business_dict = dict()
    for idx, elem in enumerate(business):
        business_dict[elem] = idx

    char_matrix = train_rdd.map(lambda x: (x[1], [users_dict[x[0]]])).reduceByKey(lambda x, y: x + y)


    # create signature matrix
    # ************************ Minhash Signatures ***********************************

    hashes = [647, 727, 227, 787, 617, 151, 211, 701, 661, 577, 997, 41, 11, 641, 277, 239, 503, 199, 37, 109, 101, 599,
              367, 281, 419, 293, 271, 97, 887, 673]
    no_of_bands = 15
    rows_per_band = len(hashes) // no_of_bands
    m = no_of_rows
    signature_matrix = char_matrix.map(lambda x: (x[0], [minhash(x,m, hash) for hash in hashes]))


    # ******************* Locality Sensitive Hashing ************************************

    bands = signature_matrix.flatMap(lambda x: gen_candidates(x,no_of_bands,rows_per_band))
    candidates = bands.reduceByKey(lambda x, y: x + y).filter(lambda x: len(x[1]) > 1). \
        flatMap(get_pairs).map(lambda x: x[0]).distinct()

    return candidates

def pearsonCorrelation(business1, business2):
    # get corated users

    user_list1 = set(business_user_map[business1]) #active movie
    user_list2 = set(business_user_map[business2])

    corrated = user_list1.intersection(user_list2)

    if len(corrated) == 0 :
        return -2.0

    r_a = []
    r_b = []
    for user in corrated:
        r_a.append(ratings_map[(user,business1)])
        r_b.append(ratings_map[(user,business2)])
    a_mean = sum(r_a) / len(r_a)
    b_mean = sum(r_b) / len(r_b)

    # a_mean = business_avg[business1]
    # b_mean = business_avg[business2]
    numerator = 0
    den1 = 0
    den2 = 0
    for i in range(len(corrated)):
        sum1 = r_a[i] - a_mean
        sum2 = r_b[i] - b_mean
        numerator += sum1 * sum2
        den1 += (sum1 * sum1)
        den2 += (sum2 * sum2)
    denominator = math.sqrt(den1) * math.sqrt(den2)
    if numerator == 0 or denominator == 0:
        return -2.0
    else:
        return numerator / denominator

def getSimilarBusinesses(user,business):
    topSimilarBusinesses = []
    if business not in similarPairs1 and business not in similarPairs2:
        topSimilarBusinesses.append((1,business))
        return topSimilarBusinesses

    if business not in business_user_map:
        #business not rated by any user
        topSimilarBusinesses.append((1,business))
        return topSimilarBusinesses

    similarBusinesses = []
    if business in similarPairs1:
        similarBusinesses = similarBusinesses + similarPairs1[business]
    if business in similarPairs2:
        similarBusinesses = similarBusinesses + similarPairs2[business]

    similarBusinesses = list(set(similarBusinesses))


    for business2 in similarBusinesses:
        if business != business2:
            similarity = pearsonCorrelation(business,business2)
            #print(similarity)
            if similarity != -2.0:
                topSimilarBusinesses.append((abs(similarity),business2))
            else:
                topSimilarBusinesses.append((1,business))
    top_list = sorted(topSimilarBusinesses,reverse=True)
    N = 8
    if len(top_list) < 8:
        return top_list
    return top_list[:8]



def predictRatings(user,business,topSimilarBusinesses):
    #user not present in training dataset return avg of all businesses
    if user not in user_business_map and business not in business_user_map:
        return 2.5                  #return 3.0
    elif user not in user_business_map:
        return business_avg[business]   #user dint rate before then return avg rating of item
    elif business not in business_user_map:
        return users_avg[user]           #business not rated return avg rating of that user
    elif len(topSimilarBusinesses) == 0 or len(topSimilarBusinesses) == 1:
        return business_avg[business]

    else:
        numerator = 0
        denominator = 0
        for topSimilarBusiness in topSimilarBusinesses:
            business2 = topSimilarBusiness[1]
            key = (user,business2)
            if key in ratings_map:
                rating = ratings_map[key]
                numerator += topSimilarBusiness[0] * rating
                denominator += abs(topSimilarBusiness[0])
        if numerator == 0 :
            return business_avg[business]

        prediction = numerator / denominator
        if prediction < 0:
            return -1 * prediction
        return prediction

# ************** driver script ***************************
#read training data
train_data = sc.textFile(train_file_name)
header = train_data.first()
train_rdd = train_data.filter(lambda x: x != header).map(lambda x:x.split(","))
#ratings dictionary
ratings_map = train_rdd.map(lambda x:((x[0],x[1]),float(x[2]))).collectAsMap()

#read validation data
validation_data = sc.textFile(val_file_name)
header2 = validation_data.first()
validation_rdd = validation_data.filter(lambda x:x != header2).map(lambda x:x.split(","))
validation_results = validation_rdd.map(lambda x:((x[0],x[1]),float(x[2])))


train_rdd = train_rdd.map(lambda x: (x[0], x[1], x[2]))
#create required dictionaries
business_user_map = train_rdd.map(lambda x: (x[1],[x[0]])).reduceByKey(lambda x,y:x+y).collectAsMap()
user_business_rdd = train_rdd.map(lambda x:(x[0],[x[1]])).reduceByKey(lambda x,y:x+y)
user_business_map = user_business_rdd.collectAsMap()

business_avg_rdd = train_rdd.map(lambda x:(x[1],[float(x[2])])).reduceByKey(lambda x,y:x+y)
business_avg = business_avg_rdd.mapValues(lambda x: sum(x)/len(x)).collectAsMap()

users_avg_rdd = train_rdd.map(lambda x:(x[0],[float(x[2])])).reduceByKey(lambda x,y:x+y)
users_avg = users_avg_rdd.mapValues(lambda x: sum(x)/len(x)).collectAsMap()

#get candidates for finding similarity
candidates = LSH()

similarPairs1 = candidates.map(lambda x:(x[0],[x[1]])).reduceByKey(lambda x,y:x+y).collectAsMap()
similarPairs2 = candidates.map(lambda x:(x[1],[x[0]])).reduceByKey(lambda x,y: x+y).collectAsMap()
topSimilarBusinesses = validation_rdd.map(lambda x: (x[0],x[1],getSimilarBusinesses(x[0],x[1])))
predictions = topSimilarBusinesses.map(lambda x:(x[0],x[1],predictRatings(x[0],x[1],x[2])))

prediction_list = predictions.collect()


results = predictions.map(lambda x:((x[0],x[1]),float(x[2]))).join(validation_results)
differences = results.map(lambda x:abs(x[1][0]-x[1][1]))
rmse  = math.sqrt(differences.map(lambda x:x**2).mean())
print(rmse)

f = open(output_file_name,"w")
f.write("user_id, business_id, prediction")
f.write("\n")
for  i in prediction_list:
    f.write(i[0]+','+i[1]+','+str(i[2]))
    f.write("\n")
f.close()

print("execution timee", time.time() - start)