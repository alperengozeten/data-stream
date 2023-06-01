import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from os import path
from skmultiflow.data import AGRAWALGenerator
from skmultiflow.data import SEAGenerator
from skmultiflow.meta import AdaptiveRandomForestClassifier
from skmultiflow.meta import StreamingRandomPatchesClassifier
from skmultiflow.meta import DynamicWeightedMajorityClassifier
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

# Get the dataset and labels
seaData = seaBatch[0]
seaLabels = seaBatch[1]
agrawalData = agrawalBatch[0]
agrawalLabels = agrawalBatch[1]

# load the spam dataset
spamDataset = pd.read_csv(path.join('data', 'spam.csv'))
spamDataset = spamDataset.to_numpy()
spamData = spamDataset[:, :-1]
spamLabels = spamDataset[:, -1]
SPAM_DATASET_SIZE = len(spamData)
print(spamData.shape)
print(spamLabels.shape)

# load the electricity dataset
elecDataset = pd.read_csv(path.join('data', 'elec.csv'))
elecDataset = elecDataset.to_numpy()
elecData = elecDataset[:, :-1]
elecLabels = elecDataset[:, -1]
ELEC_DATASET_SIZE = len(elecData)
print(elecData.shape)
print(elecLabels.shape)

batchList = [k for k in range(1, 21)]
def plot_training(hist, title='Train Classification Accuracy vs The Window Number'):
    plt.figure(figsize=(18, 12))
    plt.xlabel('Window Number')
    plt.ylabel('Training Accuracy')
    plt.xticks(batchList)
    plt.plot(batchList, hist, label='Train Classification Accuracy')
    plt.legend()
    plt.title(title)
    plt.savefig(path.join('plot', title + '.jpg'))
    plt.show()

'''
AdaptiveRandomForestClassifier
'''
'''
arf_sea_hist = []
arf_sea_correct = 0
arf_agrawal_hist = []
arf_agrawal_correct = 0
arf_spam_hist = []
arf_spam_correct = 0
arf_elec_hist = []
arf_elec_correct = 0
arf_file = open(path.join('plot', 'arf.txt'), 'w')

arf = AdaptiveRandomForestClassifier(random_state=2023)
for i in range(20):
    X, y = seaData[i * (DATASET_SIZE // 20):(i + 1) * (DATASET_SIZE // 20), :], seaLabels[i * (DATASET_SIZE // 20):(i + 1) * (DATASET_SIZE // 20)]
    y_pred = arf.predict(X)
    acc = accuracy_score(y, y_pred)
    arf_sea_correct += np.sum(y == y_pred)
    print(f'Accuracy of SEA Dataset Batch {i + 1}: {acc}')
    arf_sea_hist.append(acc)
    arf.partial_fit(X, y)
print(f'Overall Accuracy For The SEA Dataset: {arf_sea_correct / DATASET_SIZE}')
arf_file.writelines(f'Overall Accuracy For The SEA Dataset: {arf_sea_correct / DATASET_SIZE}')
plot_training(arf_sea_hist, title='ARF Classifier And SEA Dataset')

arf = AdaptiveRandomForestClassifier(random_state=2023)
for i in range(20):
    X, y = agrawalData[i * (DATASET_SIZE // 20):(i + 1) * (DATASET_SIZE // 20), :], agrawalLabels[i * (DATASET_SIZE // 20):(i + 1) * (DATASET_SIZE // 20)]
    y_pred = arf.predict(X)
    acc = accuracy_score(y, y_pred)
    arf_agrawal_correct += np.sum(y == y_pred)
    print(f'Accuracy of AGRAWAL Dataset Batch {i + 1}: {acc}')
    arf_agrawal_hist.append(acc)
    arf.partial_fit(X, y)
print(f'Overall Accuracy For The AGRAWAL Dataset: {arf_agrawal_correct / DATASET_SIZE}')
arf_file.writelines(f'Overall Accuracy For The AGRAWAL Dataset: {arf_agrawal_correct / DATASET_SIZE}\n')
plot_training(arf_agrawal_hist, title='ARF Classifier And AGRAWAL Dataset')

arf = AdaptiveRandomForestClassifier(random_state=2023)
for i in range(20):
    start_index = i * (ELEC_DATASET_SIZE // 20)
    end_index = (i + 1) * (ELEC_DATASET_SIZE // 20) if i < 19 else ELEC_DATASET_SIZE
    X, y = elecData[start_index : end_index, :], elecLabels[start_index : end_index]
    y_pred = arf.predict(X)
    acc = accuracy_score(y, y_pred)
    arf_elec_correct += np.sum(y == y_pred)
    print(f'Accuracy of Electricity Dataset Batch {i + 1}: {acc}')
    arf_elec_hist.append(acc)
    arf.partial_fit(X, y)
print(f'Overall Accuracy For The Electricity Dataset: {arf_elec_correct / ELEC_DATASET_SIZE}')
arf_file.writelines(f'Overall Accuracy For The Electricity Dataset: {arf_elec_correct / ELEC_DATASET_SIZE}')
plot_training(arf_elec_hist, title='ARF Classifier And Electricity Dataset')

arf = AdaptiveRandomForestClassifier(random_state=2023)
for i in range(20):
    start_index = i * (SPAM_DATASET_SIZE // 20)
    end_index = (i + 1) * (SPAM_DATASET_SIZE // 20) if i < 19 else SPAM_DATASET_SIZE
    X, y = spamData[start_index : end_index, :], spamLabels[start_index : end_index]
    y_pred = arf.predict(X)
    acc = accuracy_score(y, y_pred)
    arf_spam_correct += np.sum(y == y_pred)
    print(f'Accuracy of Spam Dataset Batch {i + 1}: {acc}')
    arf_spam_hist.append(acc)
    arf.partial_fit(X, y)
print(f'Overall Accuracy For The Spam Dataset: {arf_spam_correct / SPAM_DATASET_SIZE}')
arf_file.writelines(f'Overall Accuracy For The Spam Dataset: {arf_spam_correct / SPAM_DATASET_SIZE}\n')
plot_training(arf_spam_hist, title='ARF Classifier And Spam Dataset')'''

'''
SAMKNNClassifier
'''
'''
sam_sea_hist = []
sam_sea_correct = 0
sam_agrawal_hist = []
sam_agrawal_correct = 0
sam_spam_hist = []
sam_spam_correct = 0
sam_elec_hist = []
sam_elec_correct = 0
sam_file = open(path.join('plot', 'sam.txt'), 'w')

sam = SAMKNNClassifier()
for i in range(20):
    X, y = seaData[i * (DATASET_SIZE // 20):(i + 1) * (DATASET_SIZE // 20), :], seaLabels[i * (DATASET_SIZE // 20):(i + 1) * (DATASET_SIZE // 20)]
    y_pred = sam.predict(X)
    acc = accuracy_score(y, y_pred)
    sam_sea_correct += np.sum(y == y_pred)
    print(f'Accuracy of SEA Dataset Batch {i + 1}: {acc}')
    sam_sea_hist.append(acc)
    sam.partial_fit(X, y)
print(f'Overall Accuracy For The SEA Dataset: {sam_sea_correct / DATASET_SIZE}')
sam_file.writelines(f'Overall Accuracy For The SEA Dataset: {sam_sea_correct / DATASET_SIZE}')
plot_training(sam_sea_hist, title='SAMKNN Classifier And SEA Dataset')

sam = SAMKNNClassifier()
for i in range(20):
    X, y = agrawalData[i * (DATASET_SIZE // 20):(i + 1) * (DATASET_SIZE // 20), :], agrawalLabels[i * (DATASET_SIZE // 20):(i + 1) * (DATASET_SIZE // 20)]
    y_pred = sam.predict(X)
    acc = accuracy_score(y, y_pred)
    sam_agrawal_correct += np.sum(y == y_pred)
    print(f'Accuracy of AGRAWAL Dataset Batch {i + 1}: {acc}')
    sam_agrawal_hist.append(acc)
    sam.partial_fit(X, y)
print(f'Overall Accuracy For The AGRAWAL Dataset: {sam_agrawal_correct / DATASET_SIZE}')
sam_file.writelines(f'Overall Accuracy For The AGRAWAL Dataset: {sam_agrawal_correct / DATASET_SIZE}\n')
plot_training(sam_agrawal_hist, title='SAMKNN Classifier And AGRAWAL Dataset')

sam = SAMKNNClassifier()
for i in range(20):
    start_index = i * (ELEC_DATASET_SIZE // 20)
    end_index = (i + 1) * (ELEC_DATASET_SIZE // 20) if i < 19 else ELEC_DATASET_SIZE
    X, y = elecData[start_index : end_index, :], elecLabels[start_index : end_index]
    y_pred = sam.predict(X)
    acc = accuracy_score(y, y_pred)
    sam_elec_correct += np.sum(y == y_pred)
    print(f'Accuracy of Electricity Dataset Batch {i + 1}: {acc}')
    sam_elec_hist.append(acc)
    sam.partial_fit(X, y)
print(f'Overall Accuracy For The Electricity Dataset: {sam_elec_correct / ELEC_DATASET_SIZE}')
sam_file.writelines(f'Overall Accuracy For The Electricity Dataset: {sam_elec_correct / ELEC_DATASET_SIZE}')
plot_training(sam_elec_hist, title='SAMKNN Classifier And Electricity Dataset')

sam = SAMKNNClassifier()
for i in range(20):
    start_index = i * (SPAM_DATASET_SIZE // 20)
    end_index = (i + 1) * (SPAM_DATASET_SIZE // 20) if i < 19 else SPAM_DATASET_SIZE
    X, y = spamData[start_index : end_index, :], spamLabels[start_index : end_index]
    y_pred = sam.predict(X)
    acc = accuracy_score(y, y_pred)
    sam_spam_correct += np.sum(y == y_pred)
    print(f'Accuracy of Spam Dataset Batch {i + 1}: {acc}')
    sam_spam_hist.append(acc)
    sam.partial_fit(X, y)
print(f'Overall Accuracy For The Spam Dataset: {sam_spam_correct / SPAM_DATASET_SIZE}')
sam_file.writelines(f'Overall Accuracy For The Spam Dataset: {sam_spam_correct / SPAM_DATASET_SIZE}\n')
plot_training(sam_spam_hist, title='SAMKNN Classifier And Spam Dataset')'''

'''
StreamingRandomPatchesClassifier
'''
srp_sea_hist = []
srp_sea_correct = 0
srp_agrawal_hist = []
srp_agrawal_correct = 0
srp_spam_hist = []
srp_spam_correct = 0
srp_elec_hist = []
srp_elec_correct = 0
srp_file = open(path.join('plot', 'srp.txt'), 'w')

srp = StreamingRandomPatchesClassifier(random_state=2023)
for i in range(20):
    X, y = seaData[i * (DATASET_SIZE // 20):(i + 1) * (DATASET_SIZE // 20), :], seaLabels[i * (DATASET_SIZE // 20):(i + 1) * (DATASET_SIZE // 20)]
    y_pred = srp.predict(X)
    acc = accuracy_score(y, y_pred)
    srp_sea_correct += np.sum(y == y_pred)
    print(f'Accuracy of SEA Dataset Batch {i + 1}: {acc}')
    srp_sea_hist.append(acc)
    srp.partial_fit(X, y)
print(f'Overall Accuracy For The SEA Dataset: {srp_sea_correct / DATASET_SIZE}')
srp_file.writelines(f'Overall Accuracy For The SEA Dataset: {srp_sea_correct / DATASET_SIZE}')
plot_training(srp_sea_hist, title='StreamingRandomPatches Classifier And SEA Dataset')

srp = StreamingRandomPatchesClassifier(random_state=2023)
for i in range(20):
    X, y = agrawalData[i * (DATASET_SIZE // 20):(i + 1) * (DATASET_SIZE // 20), :], agrawalLabels[i * (DATASET_SIZE // 20):(i + 1) * (DATASET_SIZE // 20)]
    y_pred = srp.predict(X)
    acc = accuracy_score(y, y_pred)
    srp_agrawal_correct += np.sum(y == y_pred)
    print(f'Accuracy of AGRAWAL Dataset Batch {i + 1}: {acc}')
    srp_agrawal_hist.append(acc)
    srp.partial_fit(X, y)
print(f'Overall Accuracy For The AGRAWAL Dataset: {srp_agrawal_correct / DATASET_SIZE}')
srp_file.writelines(f'Overall Accuracy For The AGRAWAL Dataset: {srp_agrawal_correct / DATASET_SIZE}\n')
plot_training(srp_agrawal_hist, title='StreamingRandomPatches Classifier And AGRAWAL Dataset')

srp = StreamingRandomPatchesClassifier(random_state=2023)
for i in range(20):
    start_index = i * (ELEC_DATASET_SIZE // 20)
    end_index = (i + 1) * (ELEC_DATASET_SIZE // 20) if i < 19 else ELEC_DATASET_SIZE
    X, y = elecData[start_index : end_index, :], elecLabels[start_index : end_index]
    y_pred = srp.predict(X)
    acc = accuracy_score(y, y_pred)
    srp_elec_correct += np.sum(y == y_pred)
    print(f'Accuracy of Electricity Dataset Batch {i + 1}: {acc}')
    srp_elec_hist.append(acc)
    srp.partial_fit(X, y)
print(f'Overall Accuracy For The Electricity Dataset: {srp_elec_correct / ELEC_DATASET_SIZE}')
srp_file.writelines(f'Overall Accuracy For The Electricity Dataset: {srp_elec_correct / ELEC_DATASET_SIZE}')
plot_training(srp_elec_hist, title='StreamingRandomPatches Classifier And Electricity Dataset')

srp = StreamingRandomPatchesClassifier(random_state=2023)
for i in range(20):
    start_index = i * (SPAM_DATASET_SIZE // 20)
    end_index = (i + 1) * (SPAM_DATASET_SIZE // 20) if i < 19 else SPAM_DATASET_SIZE
    X, y = spamData[start_index : end_index, :], spamLabels[start_index : end_index]
    y_pred = srp.predict(X)
    acc = accuracy_score(y, y_pred)
    srp_spam_correct += np.sum(y == y_pred)
    print(f'Accuracy of Spam Dataset Batch {i + 1}: {acc}')
    srp_spam_hist.append(acc)
    srp.partial_fit(X, y)
print(f'Overall Accuracy For The Spam Dataset: {srp_spam_correct / SPAM_DATASET_SIZE}')
srp_file.writelines(f'Overall Accuracy For The Spam Dataset: {srp_spam_correct / SPAM_DATASET_SIZE}\n')
plot_training(srp_spam_hist, title='StreamingRandomPatches Classifier And Spam Dataset')

'''
DynamicWeightedMajorityClassifier
'''
'''
dwm_sea_hist = []
dwm_sea_correct = 0
dwm_agrawal_hist = []
dwm_agrawal_correct = 0
dwm_spam_hist = []
dwm_spam_correct = 0
dwm_elec_hist = []
dwm_elec_correct = 0

dwm = DynamicWeightedMajorityClassifier()
for i in range(20):
    X, y = seaData[i * (DATASET_SIZE // 20):(i + 1) * (DATASET_SIZE // 20), :], seaLabels[i * (DATASET_SIZE // 20):(i + 1) * (DATASET_SIZE // 20)]
    y_pred = dwm.predict(X)
    acc = accuracy_score(y, y_pred)
    dwm_sea_correct += np.sum(y == y_pred)
    print(f'Accuracy of SEA Dataset Batch {i + 1}: {acc}')
    dwm_sea_hist.append(acc)
    dwm.partial_fit(X, y)
print(f'Overall Accuracy For The SEA Dataset: {dwm_sea_correct / DATASET_SIZE}')
plot_training(dwm_sea_hist, title='DWM Classifier And SEA Dataset')

dwm = DynamicWeightedMajorityClassifier()
for i in range(20):
    X, y = agrawalData[i * (DATASET_SIZE // 20):(i + 1) * (DATASET_SIZE // 20), :], agrawalLabels[i * (DATASET_SIZE // 20):(i + 1) * (DATASET_SIZE // 20)]
    y_pred = dwm.predict(X)
    acc = accuracy_score(y, y_pred)
    dwm_agrawal_correct += np.sum(y == y_pred)
    print(f'Accuracy of AGRAWAL Dataset Batch {i + 1}: {acc}')
    dwm_agrawal_hist.append(acc)
    dwm.partial_fit(X, y)
print(f'Overall Accuracy For The AGRAWAL Dataset: {dwm_agrawal_correct / DATASET_SIZE}')
plot_training(dwm_agrawal_hist, title='DWM Classifier And AGRAWAL Dataset')

dwm = DynamicWeightedMajorityClassifier()
for i in range(20):
    X, y = elecData[i * (DATASET_SIZE // 20):(i + 1) * (DATASET_SIZE // 20), :], elecLabels[i * (DATASET_SIZE // 20):(i + 1) * (DATASET_SIZE // 20)]
    y_pred = dwm.predict(X)
    acc = accuracy_score(y, y_pred)
    dwm_elec_correct += np.sum(y == y_pred)
    print(f'Accuracy of Electricity Dataset Batch {i + 1}: {acc}')
    dwm_elec_hist.append(acc)
    dwm.partial_fit(X, y)
print(f'Overall Accuracy For The Electricity Dataset: {dwm_elec_correct / DATASET_SIZE}')
plot_training(dwm_elec_hist, title='DWM Classifier And Electricity Dataset')

dwm = DynamicWeightedMajorityClassifier()
for i in range(20):
    X, y = spamData[i * (DATASET_SIZE // 20):(i + 1) * (DATASET_SIZE // 20), :], spamLabels[i * (DATASET_SIZE // 20):(i + 1) * (DATASET_SIZE // 20)]
    y_pred = dwm.predict(X)
    acc = accuracy_score(y, y_pred)
    dwm_spam_correct += np.sum(y == y_pred)
    print(f'Accuracy of Spam Dataset Batch {i + 1}: {acc}')
    dwm_spam_hist.append(acc)
    dwm.partial_fit(X, y)
print(f'Overall Accuracy For The Spam Dataset: {dwm_spam_correct / DATASET_SIZE}')
plot_training(dwm_spam_hist, title='DWM Classifier And Spam Dataset')'''
