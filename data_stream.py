import pickle
from os import path
from skmultiflow.data import AGRAWALGenerator
from skmultiflow.meta import AdaptiveRandomForestClassifier

DATASET_SIZE = 100000

agrawalGenerator = AGRAWALGenerator(random_state=2023)

batch = agrawalGenerator.next_sample(DATASET_SIZE)

agrawal_file_name = 'AGRAWALGenerator.pkl'
with open(path.join('data', agrawal_file_name), 'wb') as file:
    pickle.dump(batch, file)
    print(f'AGRAWAL Dataset saved to "{agrawal_file_name}"')