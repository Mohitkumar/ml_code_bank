arr = [1,0,0,1,0,1,0,0,1,0,1]

temp = 0
ls = 0
rs = 0
for i in range(len(arr)):
    if arr[i] == 1:
        ls += i -temp
        temp +=1
temp =0
for i in reversed(range(len(arr))):
    if arr[i] == 1:
        rs += len(arr) -1- i -temp
        temp +=1
print ls, rs
print min(ls,rs)