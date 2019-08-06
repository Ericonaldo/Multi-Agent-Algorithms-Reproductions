import numpy as np
import sys
import time
from collections import namedtuple
Transition = namedtuple("Transition", "state, action, next_state, reward, done")

num = int(sys.argv[1])#10000
n = 2

a = [[[1,2,3,0,1] for _ in range(num)] for __ in range(n)]
c = [[11*_ for _ in range(num)] for __ in range(n)]


t_start1=time.time()
for i in range(n):
    for j in range(len(c[i])):
        a[i][j] = Transition(a[i][j][0], a[i][j][1], a[i][j][2], c[i][j], a[i][j][4])
print("time1: ", time.time()-t_start1)
# print(a)


a = [[[1,2,3,0,1] for _ in range(num)] for __ in range(n)]
c = [[11*_ for _ in range(num)] for __ in range(n)]
t_start2 = time.time()
for i in range(n):
    tmp = zip(*zip(*a[i]))
    a[i] = list(map(lambda x,y:Transition(x[0], x[1], x[2], y, x[4]), tmp, c[i]))
print("time2: ", time.time()-t_start2)
# print(a)
