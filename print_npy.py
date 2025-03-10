import numpy as np

# Replace 'filename.npy' with your actual file path
data = np.load('myeongpum_test_2048.npy')
print(data[:10])
print(data.size)
print(data.shape)