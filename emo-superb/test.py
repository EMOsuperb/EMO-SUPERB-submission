import pickle as pk

wav_mean = 0
wav_std = 1
data = {"wav_mean": wav_mean, "wav_std": wav_std}

with open("stats.pkl", "wb") as f:
    pk.dump(data, f)
    
with open("stats.pkl", 'rb') as fp:
    dataset = pk.load(fp)

print(dataset)
print(data)