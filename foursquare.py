import torch
import numpy as np
from torch.autograd import Variable
import pickle
from collections import deque, Counter





def generate_input_history(data_neural, mode, mode2=None, candidate=None):
    data_train = {}
    train_idx = {}
    if candidate is None:
        candidate = data_neural.keys()
    for u in candidate:
        sessions = data_neural[u]['sessions']
        train_id = data_neural[u][mode]
        data_train[u] = {}
        for c, i in enumerate(train_id):
            if mode == 'train' and c == 0:
                continue
            session = sessions[i]
            trace = {}
            loc_np = np.reshape(np.array([s[0] for s in session[:-1]]), (len(session[:-1]), 1))
            tim_np = np.reshape(np.array([s[1] for s in session[:-1]]), (len(session[:-1]), 1))
            # voc_np = np.reshape(np.array([s[2] for s in session[:-1]]), (len(session[:-1]), 27))
            target = np.array([s[0] for s in session[1:]])
            trace['loc'] = Variable(torch.LongTensor(loc_np))
            trace['target'] = Variable(torch.LongTensor(target))
            trace['tim'] = Variable(torch.LongTensor(tim_np))
            # trace['voc'] = Variable(torch.LongTensor(voc_np))

            history = []
            if mode == 'test':
                test_id = data_neural[u]['train']
                for tt in test_id:
                    history.extend([(s[0], s[1]) for s in sessions[tt]])
            for j in range(c):
                history.extend([(s[0], s[1]) for s in sessions[train_id[j]]])
            history = sorted(history, key=lambda x: x[1], reverse=False)

            # merge traces with same time stamp
            if mode2 == 'max':
                history_tmp = {}
                for tr in history:
                    if tr[1] not in history_tmp:
                        history_tmp[tr[1]] = [tr[0]]
                    else:
                        history_tmp[tr[1]].append(tr[0])
                history_filter = []
                for t in history_tmp:
                    if len(history_tmp[t]) == 1:
                        history_filter.append((history_tmp[t][0], t))
                    else:
                        tmp = Counter(history_tmp[t]).most_common()
                        if tmp[0][1] > 1:
                            history_filter.append((history_tmp[t][0], t))
                        else:
                            ti = np.random.randint(len(tmp))
                            history_filter.append((tmp[ti][0], t))
                history = history_filter
                history = sorted(history, key=lambda x: x[1], reverse=False)
            elif mode2 == 'avg':
                history_tim = [t[1] for t in history]
                history_count = [1]
                last_t = history_tim[0]
                count = 1
                for t in history_tim[1:]:
                    if t == last_t:
                        count += 1
                    else:
                        history_count[-1] = count
                        history_count.append(1)
                        last_t = t
                        count = 1
            ################

            history_loc = np.reshape(np.array([s[0] for s in history]), (len(history), 1))
            history_tim = np.reshape(np.array([s[1] for s in history]), (len(history), 1))
            trace['history_loc'] = Variable(torch.LongTensor(history_loc))
            trace['history_tim'] = Variable(torch.LongTensor(history_tim))
            if mode2 == 'avg':
                trace['history_count'] = history_count

            data_train[u][i] = trace
        train_idx[u] = train_id
    return data_train, train_idx

with open(r"archive/foursquare.pk", "rb") as input_file:
    a = pickle.load(input_file, encoding="latin1")

all_data = generate_input_history(a["data_neural"], "train")


train_idx = all_data[1]
mode = 'random'
mode2 = 'train'
user = train_idx.keys()
train_queue = deque()
if mode == 'random':
    initial_queue = {}
    for u in user:
        if mode2 == 'train':
            initial_queue[u] = deque(train_idx[u][1:])
        else:
            initial_queue[u] = deque(train_idx[u])
    queue_left = 1
    user_list = list(user)
    while queue_left > 0:
        print(len(train_queue))
        np.random.shuffle(user_list)
        for j, u in enumerate(user_list):
            if len(initial_queue[u]) > 0:
                train_queue.append((u, initial_queue[u].popleft()))
            if j >= int(0.01 * len(user)):
                break
        queue_left = sum([1 for x in initial_queue if len(initial_queue[x]) > 0])
elif mode == 'normal':
    for u in user:
        for i in train_idx[u]:
            train_queue.append((u, i))


run_queue = train_queue
data = all_data[0]

users_acc = {}
u, i = run_queue.popleft()
if u not in users_acc:
    users_acc[u] = [0, 0]
loc = data[u][i]['loc']
tim = data[u][i]['tim']
target = data[u][i]['target']
uid = Variable(torch.LongTensor([u]))