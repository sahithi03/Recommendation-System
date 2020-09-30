import sys
from pyspark import SparkContext
import time
import itertools

sc = SparkContext('local[*]','task1')
sc.setLogLevel("ERROR")

start = time.time()

input_file_name = sys.argv[1]
output_file_name = sys.argv[2]

hashes = [647, 727, 227, 787, 617, 151, 211, 701, 661, 577, 997, 41, 11, 641, 277, 239, 503, 199, 37, 109, 101, 599,
              367, 281, 419, 293, 271, 97, 887, 673]
no_of_bands = 15
rows_per_band = len(hashes) // no_of_bands

# ************************** functions **********************************
def minhash(x):

    m = no_of_users
    res = []
    temp = []
    for hash in hashes:
        for j in x[1]:
            temp.append((hash*j + 3) % m )
        res.append(min(temp))
        temp = []
    return res
    #formula => ((ax+b) % m)
    #return min(((a*x + 3) % m) for x in x[1])

def gen_candidates(x):

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
        candidates.append(pair)
    return candidates

def get_jaccard_similarity(x):
    business1 = x[0]
    business2 = x[1]

    users1 = set(business_user_map[business1])
    users2 = set(business_user_map[business2])

    jaccard_sim = float(len(users1.intersection(users2)) / len(users1.union(users2)))

    return (business1, business2, jaccard_sim)

# ******************** driver script ****************************
input_data = sc.textFile(input_file_name)
header = input_data.first()
input_rdd = input_data.filter(lambda x:x != header).map(lambda x:x.split(',')).persist()

users_rdd = input_rdd.map(lambda x:x[0]).distinct()
business_rdd = input_rdd.map(lambda x:x[1]).distinct()

users = users_rdd.collect()
business = business_rdd.collect()

business_user_map = input_rdd.map(lambda x:(x[1],x[0])).groupByKey().mapValues(list).collectAsMap()

no_of_users = len(users)

users_dict = dict()
for idx,elem in enumerate(users):
    users_dict[elem]  = idx


#build character matrix
#business -> users
char_matrix = input_rdd.map(lambda x:(x[1],[users_dict[x[0]]])).reduceByKey(lambda x,y: x+y)

# ************************ Minhash Signatures ***********************************
signature_matrix = char_matrix.map(lambda x:(x[0],minhash(x)))

# ******************* Locality Sensitive Hashing ************************************

bands = signature_matrix.flatMap(lambda x:gen_candidates(x))

candidates = bands.reduceByKey(lambda x,y:x+y).filter(lambda x:len(x[1]) > 1).\
    flatMap(get_pairs).map(lambda x: (1,x)).groupByKey().flatMapValues(set).values()

#print(candidates.take(5))

result = candidates.map(get_jaccard_similarity).filter(lambda x: x[2] >= 0.5).sortBy(lambda x: x[1]).sortBy(lambda x: x[0])

final_result = result.collect()

f = open(output_file_name,'w')

f.write("business_id_1, business_id_2, similarity")
for i in final_result:
    f.write("\n")
    f.write(i[0]+","+i[1]+","+str(i[2]))
f.close()

ground = sc.textFile("/Users/sahithi/Documents/DM_Assignments/Assignment3/pure_jaccard_similarity.csv") \
        .map(lambda x: x.split(",")).filter(lambda x:x[0] != "business_id_1").map(lambda x: (x[0], x[1]))

pair = result.map(lambda x: (x[0], x[1]))
my_results = list(pair.collect())

ground_results= list(ground.collect())
print(len(ground_results))
tp = len(ground_results)
fn = 0
fp = 0
for i in ground_results:
    if i not in my_results:
        fn += 1
print("false negatives",fn)
for i in my_results:
    if i not in ground_results:
        fp += 1
print("false positives",fp)

precision = tp / (fp+tp)
recall = tp / (fn+tp)

print("precision:")
print(precision)
print("recall:")
print(recall)

print("execution time")
print(time.time()-start)






