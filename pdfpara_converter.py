import os

with open("tmp.txt", 'r') as f:
    x = f.readlines()

# print(x)
f = lambda x: x.replace("\n", " ").replace("  ", " ")
re = list(map(f, x))
print("".join(re))