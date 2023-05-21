import pickle
from os import path
from skmultiflow.data import AGRAWALGenerator
from skmultiflow.data import SEAGenerator
from skmultiflow.meta import AdaptiveRandomForestClassifier
from skmultiflow.meta import StreamingRandomPatchesClassifier
from skmultiflow.lazy import SAMKNNClassifier
from sklearn.metrics import accuracy_score

DATASET_SIZE = 100000
agrawal_file_name = 'AGRAWALGenerator.pkl'
sea_file_name = 'SEADataset.pkl'

agrawalGenerator = AGRAWALGenerator(random_state=2023)
agrawalBatch = agrawalGenerator.next_sample(DATASET_SIZE)

with open(path.join('data', agrawal_file_name), 'wb') as file:
    pickle.dump(agrawalBatch, file)
    print(f'AGRAWAL Dataset saved to "{agrawal_file_name}"')

seaGenerator = SEAGenerator(random_state=2023)
seaBatch = seaGenerator.next_sample(DATASET_SIZE)

with open(path.join('data', sea_file_name), 'wb') as file:
    pickle.dump(seaBatch, file)
    print(f'SEA Dataset saved to "{sea_file_name}"')

with open(path.join('data', agrawal_file_name), 'rb') as file:
    agrawalBatch = pickle.load(file)

with open(path.join('data', sea_file_name), 'rb') as file:
    seaBatch = pickle.load(file)

print(agrawalBatch[0].shape)
print(seaBatch[0].shape)

seaData = seaBatch[0]
seaLabels = seaBatch[1]

'''
arf = AdaptiveRandomForestClassifier()
for i in range(20):
    X, y = seaData[i * (DATASET_SIZE // 20):(i + 1) * (DATASET_SIZE // 20), :], seaLabels[i * (DATASET_SIZE // 20):(i + 1) * (DATASET_SIZE // 20)]
    y_pred = arf.predict(X)
    acc = accuracy_score(y, y_pred)
    print(f'Accuracy of Batch {i + 1}: {acc}')
    arf.partial_fit(X, y)'''


sam = SAMKNNClassifier()
for i in range(20):
    X, y = seaData[i * (DATASET_SIZE // 20):(i + 1) * (DATASET_SIZE // 20), :], seaLabels[i * (DATASET_SIZE // 20):(i + 1) * (DATASET_SIZE // 20)]
    y_pred = sam.predict(X)
    acc = accuracy_score(y, y_pred)
    print(f'Accuracy of Batch {i + 1}: {acc}')
    sam.partial_fit(X, y)
