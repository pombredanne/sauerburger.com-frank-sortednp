import numpy as np
import sortednp
import time

x = np.arange(50*1000*1000).astype('float')
np.random.shuffle(x)

p = int(len(x) * 0.7)
a = sorted(x[:p])
b = sorted(x[p:])
# print(a)
# print(b)
print(time.time())
sortednp.merge(a, b)
print(time.time())
