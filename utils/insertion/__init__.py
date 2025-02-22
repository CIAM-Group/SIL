from typing import Union
try:
    from . import insertion
except:
    import insertion
import numpy as np
import torch


def _to_numpy(arr):
    if isinstance(arr, torch.Tensor):
        return arr.detach().cpu().numpy()
    else:
        return arr


def random_insertion(cities, order=None):
    assert len(cities.shape) == 2 and cities.shape[1] == 2
    citycount = cities.shape[0]
    if order is None:
        order = np.arange(citycount, dtype=np.uint32)
    else:
        assert len(order.shape) == 1 and order.shape[0] == cities.shape[0]
        order = _to_numpy(order).astype(np.uint32)

    if cities.dtype is not np.float32:
        cities = _to_numpy(cities).astype(np.float32)

    result, cost = insertion.random(cities, order, True)

    return result, cost

def random_insertion_non_euclidean(distmap, order=None):
    assert len(distmap.shape) == 2 and distmap.shape[1] == distmap.shape[0]
    citycount = distmap.shape[0]
    if order is None:
        order = np.arange(citycount, dtype=np.uint32)
    else:
        assert len(order.shape) == 1 and order.shape[0] == citycount
        order = _to_numpy(order).astype(np.uint32)

    if distmap.dtype is not np.float32:
        distmap = _to_numpy(distmap).astype(np.float32)

    result, cost = insertion.random(distmap, order, False)

    return result, cost

def cvrp_random_insertion(customerpos, depotpos, demands, capacity, order = None, exploration = 1.0):
    assert len(customerpos.shape) == 2 and customerpos.shape[1] == 2
    assert isinstance(capacity, int)

    if isinstance(depotpos, tuple):
        assert len(depotpos)==2
        depotx, depoty = depotpos
    else:
        assert len(depotpos.shape)==1 and depotpos.shape[0]==2
        depotx, depoty = depotpos[0].item(), depotpos[1].item()
    depotx, depoty = float(depotx), float(depoty)

    ccount = customerpos.shape[0]
    if order is None:
        # generate order
        dx, dy = (customerpos - np.array([[depotx, depoty]])).T
        phi = np.arctan2(dy, dx)
        order = np.argsort(phi).astype(np.uint32)
    else:
        assert len(order.shape) == 1 and order.shape[0] == ccount
        order = _to_numpy(order).astype(np.uint32)

    customerpos = _to_numpy(customerpos)
    if customerpos.dtype is not np.float32:
        customerpos = customerpos.astype(np.float32)
    demands = _to_numpy(demands)
    if demands.dtype is not np.uint32:
        demands = demands.astype(np.uint32)
    
    outorder, sep = insertion.cvrp_random(customerpos, depotx, depoty, demands, capacity, order, exploration)
    routes = [outorder[i:j] for i,j in zip(sep, sep[1:])]
    return routes

def cvrplib_random_insertion(positions, demands, capacity, order = None, exploration = 1.0):
    customerpos = positions[1:]
    depotpos = positions[0]
    demands = demands[1:]
    if order is not None:
        order = np.delete(order, order==0) - 1
    routes = cvrp_random_insertion(customerpos, depotpos, demands, capacity, order, exploration)
    for r in routes:
        r += 1
    return routes


if __name__=="__main__":
    from datetime import datetime, date
    np.random.seed(1)
    CPACITY = {
        1000: 250,
        5000: 400,
        10000: 500,
        30000: 1000,
        50000: 1500,
        100000: 2000
    }
    data_types = [1000,5000,10000,30000,50000,100000]
    dataset_sizes = [128,16,16,16,16,16]

    for k in range(len(data_types)):
        import time

        time_start = time.time()  # 记录开始时间
        # function() 执行的程序

        all_length = []
        n = data_types[k]
        dataset_size = dataset_sizes[k]
        for _ in range(dataset_size):
            pos = np.random.rand(n, 2)
            # depotpos = pos.mean(axis=0)
            depotpos = np.random.rand(2)
            # print(depotpos.shape)
            # assert False
            demands = np.random.randint(1, 10, size = n)
            capacity = CPACITY[n]
            route = cvrp_random_insertion(pos, depotpos, demands, capacity)
            solution = []
            for i in range(len(route)):
                sub_tour = (route[i] + 1).tolist()
                solution +=[0]
                solution+=sub_tour
                solution+=[0]
            solution = np.array(solution)
            locations = np.concatenate((depotpos.reshape(1,-1),pos),axis=0)
            length =0
            for i in range(1,len(solution)):
                previous_node_index = solution[i-1]
                current_node_index = solution[i]
                previous_node = locations[previous_node_index]
                current_node = locations[current_node_index]
                distance = np.sqrt(((previous_node-current_node)**2).sum())
                length+=distance
            all_length.append(length)
        time_end = time.time()  # 记录结束时间
        time_sum = time_end - time_start  # 计算的时间差为程序的执行时间，单位为秒/s

        print('problem_size:',n, 'mean_length:',np.mean(all_length), 'time_sum',time_sum)
    #print(*cvrp_random_insertion(pos, depotpos, demands, capacity), sep='\n')