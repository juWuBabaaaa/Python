import os, json
from tqdm import tqdm

def load(path, name):
    fp = os.path.join(path, f"{name}")
    with open(fp) as f:
        ff = json.load(f)
    return ff


def find_pieces(arr):
    """
    arr is something like [111001110011], this function returns all "1111" pieces' indices.
    """
    re = []
    n = len(arr)
    e = 0
    for i in tqdm(range(n)):
        if i < e:
            continue
        if arr[i]:
            s = i
            e = i
            while e<n and arr[e]:
                e += 1
            print((s, e))

            re.append([s, e])

    return re
