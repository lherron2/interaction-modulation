import numpy as np

def expit(x, beta, thresh):
    return 1/(1 + np.exp(-1*beta*(abs(x) - thresh)))

def norm_expit(expit_series, beta, thresh):
    expit0 = expit(0, beta=beta, thresh=thresh)
    expit1 = expit(1, beta=beta, thresh=thresh)
    return (expit_series - expit0)/(expit1 - expit0)

def g(m, beta=10, thresh=0.8, reg=0.01):
    expit_series = expit(m, beta=beta, thresh=thresh)
    garr = norm_expit(expit_series, beta=beta, thresh=thresh)
    gsum = np.sum(garr, axis=0)
    gscores = garr/(gsum + reg)
    return gscores

def make_one_hot(array, condition):
    "makes one-hot on condition"
    array[array!=condition] = 1
    array[array==condition] = 0
    return array

def burst(arr, ignore_zero=False):
    "useful for time series of transition states"
    elems = list(set(arr))
    if ignore_zero:
        try:
            elems.remove(0)
        except:
            None
    for elem in elems:
        new_arr = np.ones_like(arr)*-1
        new_arr[arr == elem] = elem
        yield new_arr

def search_sequence(arr, seq):
    arr = np.array(arr)
    seq = np.array(seq)
    Na, Nseq = arr.size, seq.size

    r_seq = np.arange(Nseq)

    M = (arr[np.arange(Na-Nseq+1)[:,None] + r_seq] == seq).all(1)
    if M.any() >0:
        return np.where(np.convolve(M,np.ones((Nseq),dtype=int))>0)[0]
    else:
        return np.array([])

def element_prod(arrays):
    return [np.prod(array) for array in zip(*arrays)]

def check_near_zero(array, condition=False, tolerance=1e-1):
    "checks if arrays are near zero"
    near_zero = np.isclose(array, 0, atol=tolerance)
    (idxs,) = np.where(near_zero==condition)
    ser = np.zeros(len(near_zero))
    ser[idxs]=1
    return ser

def check_surpass(arr1, arr2, threshold=1e-2):
    "finds when array1 surpasses array2"
    total_surpass = np.zeros_like(arr1, dtype=bool)
    surpass = arr1 - arr2
    surpass[surpass<threshold] = 0
    surpass[surpass>threshold] = 1
    total_surpass += surpass.astype(bool)
    return total_surpass.astype(int)

def check_monotonicity(array, condition):
    checked = np.zeros_like(array)
    if condition == "increasing":
        (idxs,) = np.where(array[1:] > array[:-1])
        checked[idxs + 1] = 1
    elif condition == "decreasing":
        (idxs,) = np.where(array[1:] < array[:-1])
        checked[idxs + 1] = 1
    return checked

def near_others(arrays, condition=False, tolerance=1e-2):
    close_dict = {}
    for i, array in enumerate(arrays):
        close_arr = np.zeros_like(arrays[0])
        for j, other in enumerate(arrays):
            if i != j:
                close = np.isclose(other, array, atol=tolerance)
                close_arr += close
        (idxs,) = np.where(close_arr==condition)
        ser = np.zeros(len(close))
        ser[idxs] = 1
        close_dict[f"{i}"] = ser
    return close_dict

def check_current_pattern(garr):
    current_patterns = [np.argmax(lis) for lis in garr.T]
    cps = list(burst(current_patterns))

    undetected_patterns = set(range(len(garr))) - set(current_patterns)
    for pattern_idx in undetected_patterns:
        cps.insert(pattern_idx, np.ones_like(current_patterns)*-1)
    for cp in cps:
        cp  = make_one_hot(cp, condition=-1)
    return cps

def get_continuous(array):
    array = np.array(array)
    bursteds = list(burst(array, ignore_zero=True))
    for bursted in bursteds:
        bursted = make_one_hot(bursted, condition=-1)
        lb = search_sequence(bursted, [0,1,1])
        if lb.size:
            lb = lb[1::3]
        else:
            lb = np.array([])
        ub = search_sequence(bursted, [1,1,0])
        if ub.size:
            ub = ub[1::3]
        else:
            ub = np.array([])
        single = search_sequence(bursted, [0,1,0])
        if single.size:
            single = single[1::3]
        else:
            single = np.array([])
        yield (lb, ub, single)

def get_intervals(garr):
    cps = check_current_pattern(garr)
    close_dict = near_others(garr, tolerance=1e-5)
    possible_transitions, intervals = {}, {}

    for key, values in close_dict.items():
        possible = element_prod([cps[int(key)], values])
        possible_transitions[key] = possible
        ranges = list(get_continuous(possible))
        if ranges == []:
            lbs = np.array([])
            ubs = np.array([])
        else:
            [(lbs, ubs, singles)] = ranges
            
        # trimming the time series
        if min(lbs, default=np.nan) > min(ubs, default=np.nan):
            ubs = np.delete(ubs, 0)
        if max(lbs, default=np.nan) > max(ubs, default=np.nan):
            lbs = np.delete(lbs, -1)

        intervals[key] = list(zip(lbs, ubs))
    return intervals

def get_scores(intervals, garr):
    score_arr = []
    for mu, pairs in intervals.items():
        for pair in pairs:
            (lb, ub) = pair
            score = np.trapz(garr[int(mu), lb:ub])/(ub - lb)
            score_arr.append((mu, score.round(4)))
    return score_arr

def get_off_diag(arr, offset):
    (Nstates,_) = np.asarray(arr).shape
    upper_diag_elems = np.diagonal(arr, offset)
    lower_diag_elems = np.diagonal(arr, -(Nstates - offset))
    diag_elems = np.concatenate((upper_diag_elems, lower_diag_elems), axis=0)
    return diag_elems
