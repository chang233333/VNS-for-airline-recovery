import parameters
import json
from datetime import datetime
import time
import random
import pandas as pd
import xarray as xr
import numpy as np
import copy
import multiprocessing as mp
from itertools import combinations
import matplotlib.pyplot as plt

def get_initial_solution(datanumber): # Flight 要按时间顺序来
    # get R
    f = pd.read_csv(f"./cost_{datanumber}.csv",header = 0)
    route = f.groupby("AC").agg({"Flight":list}).reset_index()
    index_R = route.apply(lambda x :x.AC, axis =1)
    R = route.set_index(keys = index_R)["Flight"].to_dict()
    for i in R.keys():
        head = R[i][0]
        tail = R[i][-1]
        head_s = f[f["Flight"] == head].DEP.to_list()
        tail_s = f[f["Flight"] == tail].ARR.to_list()
        R[i] = head_s + R[i][:] +tail_s
        for _i,j in enumerate(R[i]):
            R[i][_i] = j
    return R

def ben_route_ac(R):#conculate the benefit of route and crew
    ben_r = {}
    for i,r in R.items():
        legs = r[1:-1]
        ben_r[i] = sum(min(param.Cap_i[str(i)],param.DEMj[j-1]) * param.f_j[str(j)] for j in legs)-\
                        sum(param.cij[str(i)+"_"+str(j)] for j in legs)-\
                        sum(param.cij[str(i)+"_"+str(j)]//360 * param.IDj[str(j)] for j in legs)-\
                        sum(param.c1pj[str(i)+"_"+str(j)] for j in legs)
    return ben_r

def ben_route_cp(RC):
    cost_r = {}
    for p,r in RC.items():
        legs = r[1:-1]
        cost_r[p] = -sum(param.c1pj[str(p)+"_"+str(j)] for j in legs )
    return cost_r

def ben_route_a(R):#conculate the benefit of aircraft
    ben_r = {}
    for i,r in R.items():
        legs = r[1:-1]
        ben_r[i] = sum(min(param.Cap_i[str(i)],param.DEMj[j-1]) * param.f_j[str(j)] for j in legs)-\
                        sum(param.cij[str(i)+"_"+str(j)] for j in legs)-\
                        sum(param.cij[str(i)+"_"+str(j)]//360 * param.IDj[str(j)] for j in legs)
    return ben_r

def ben_route_afc(R):
    ben_r = {}
    for i,r in R.items():
        legs = r[1:-1]
        ben_r[i] = sum(min(param.Cap_i[str(i)],param.DEMj[j-1]) * param.f_j[str(j)] for j in legs)-\
                        sum(param.cij[str(i)+"_"+str(j)] for j in legs)-\
                        sum(param.cij[str(i)+"_"+str(j)]//360 * param.IDj[str(j)] for j in legs)-\
                        sum(param.c1pj[str(FC[j])+"_"+str(j)] for j in legs)
    return ben_r

def getNei_exc(route,ben_r,ben):
    non_visited = copy.deepcopy(param.F)
    #route_copy = route #怎么会动态拷贝哦
    global obj
    det_exc = True
    deter = True
    #num = 0
    while deter == True:
        if len(non_visited) == 1:
            print("can't exchange")
            det_exc = False
            return det_exc
        #sort_r = sort_route(R)
        #i = sort_r.aircraft[num]  #数据更新或者未更新的问题 
        #num = num + 1
        #i = int(i)
        i = sorted(random.sample(non_visited,1)) # 随机选择一架飞机==>由权重值选择一架飞机
        i = i[0]
        non_visited.remove(i)
        pairs = {}
        r1 = route[i]
        dictr1 = {x: y for y, x in enumerate(r1[1:-1])}
        for j in r1[1:-1]:
             for k in r1[1:-1]:
                if int(j) <= int(k):
                    pairs[j] = k
        '''if len(pairs) == 0: # 是否无法交换？
            continue'''
        for _i in non_visited:# 找到合适的可更换区间  注意找出问题后改为non_visited
            r2 = route[_i]
            dictr2 =  {x: y for y, x in enumerate(r2[1:-1])}
            for j,k in pairs.items():# 但这个交换总是从找到的第一个区间开始找？ 先给区间进行寻找还是先找第二条再历遍区间会好些????
                index_bef = dictr1[j]+1
                index_aft = dictr1[k]+1
                new1 ={i: -100000}
                new2 ={_i: -100000}#初始化所选飞机的benefit
                for _j in  r2[1:-1]:
                    for _k in r2[1:-1]:
                        if int(_j) <= int(_k):
                            if param.Oj[str(_j)] == param.Oj[str(j)] and param.Dj[str(_k)] == param.Dj[str(k)]:#判断前站后站是否相同
                                index_before = dictr2[_j]+1
                                index_after = dictr2[_k]+1
                                # 获取前后航班段
                                leg_bef = str(r1[int(index_bef - 1)]) if index_bef - 1 == 0 else r1[int(index_bef - 1)]
                                leg_aft = str(r1[int(index_aft + 1)]) if index_aft + 2 == len(r1) else r1[int(index_aft + 1)]
                                leg_before = str(r2[int(index_before-1)]) if index_before - 1 == 0 else r2[int(index_before-1)]
                                leg_after = str(r2[int(index_after+1)])  if index_after + 2 == len(r2) else r2[int(index_after+1)]
                                
                                if CON.loc[leg_bef, _j] == 0: # 判断插入段最前的航班段能否与原航线里的断开处形成连接，即时间上是否合理
                                    continue
                                if CON.loc[_k, leg_aft] == 0: # 插入段最后段与断开处后端
                                    continue
                                if CON.loc[leg_before, j] == 0:
                                    continue
                                if CON.loc[k,leg_after] == 0:
                                    continue 
                                route_new1 = r1[:index_bef] + r2[index_before:index_after+1] + r1[index_aft+1:]
                                route_new2 = r2[:index_before] + r1[index_bef:index_aft+1] + r2[index_after+1:]
                                A = {i:route_new1}
                                B = {_i:route_new2}   
                                new1 = ben(A)#本交换只涉及两条航路，故这样就够了
                                new2 = ben(B)
                                if new1[i] + new2[_i] > ben_r[i] + ben_r[_i] + 1 : # obj 没有更新，看什么时候更新
                                    print("------------exc")
                                    '''print(A)
                                    print(B)
                                    print(ben_r[i])
                                    print(new1[i])
                                    print(ben_r[_i])
                                    print(new2[_i])'''
                                    route[i] = route_new1
                                    route[_i] = route_new2
                                    ben_r[i] = new1[i]
                                    ben_r[_i] = new2[_i]
                                    return det_exc
                                elif PROBA <= random.random() and new1[i] + new2[_i] > ben_r[i] + ben_r[_i] - MINBOUND:
                                    route[i] = route_new1
                                    route[_i] = route_new2
                                    ben_r[i] = new1[i]
                                    ben_r[_i] = new2[_i]
                                    return det_exc

def getNei_rep(route,ben_r,ben):#0.034 0.083 0.238
    non_visited = copy.deepcopy(param.F)
    global obj
    det_rep =True
    while True:
        if len(non_visited) == 1:
            print("can't replace")
            det_rep = False
            return det_rep
        i = sorted(random.sample(non_visited,1)) # 随机选择一架飞机==>由权重值选择一架飞机
        i = i[0]
        non_visited.remove(i)
        r1 = route[i]
        for _i in non_visited:
            r2 = route[_i]
            for idx_j,j in enumerate(r1[1:-1]):
                for idx__j,_j in enumerate(r2[1:-1]):
                    if param.Oj[str(_j)] == param.Oj[str(j)] and param.Dj[str(_j)] == param.Dj[str(j)]:
                        index1 = idx_j+1
                        index2 = idx__j+1
                        legbef1 = str(r1[index1-1]) if index1-1 == 0 else r1[index1-1]
                        legaft1 = str(r1[index1+1]) if index1+2 == len(r1) else r1[index1+1]
                        legbef2 = str(r2[index2-1]) if index2-1 == 0 else r2[index2-1]
                        legaft2 = str(r2[index2+1]) if index2+2 == len(r2) else r2[index2+1]
                        if CON.loc[legbef1,_j] == 0:
                            continue
                        if CON.loc[legbef2,j] == 0:
                            continue
                        if CON.loc[_j,legaft1] == 0:
                            continue
                        if CON.loc[j,legaft2] == 0:
                            continue
                        route_new1 = r1[:index1] + [_j] + r1[index1+1:]
                        route_new2 = r2[:index2] + [j] + r2[index2+1:]
                        A = {i:route_new1}
                        B = {_i:route_new2}   
                        new1 = ben(A)
                        new2 = ben(B)
                        if new1[i] + new2[_i] > ben_r[i] + ben_r[_i] + 1 : # obj 没有更新，看什么时候更新
                            # print("-------------------------rep")
                            route[i] = route_new1
                            route[_i] = route_new2
                            ben_r[i] = new1[i]
                            ben_r[_i] = new2[_i]
                            return det_rep
                        elif PROBA <= random.random() and new1[i] + new2[_i] > ben_r[i] + ben_r[_i] - MINBOUND:
                            route[i] = route_new1
                            route[_i] = route_new2
                            ben_r[i] = new1[i]
                            ben_r[_i] = new2[_i]
                            return det_rep

def getNei_ins(route,ben_r,ben):#0.031 0.079 0.225
    non_visited = copy.deepcopy(param.F)
    det_ins =True
    while True:
        if len(non_visited) == 1:
            print("can't insert")
            det_ins = False
            return det_ins
        i = sorted(random.sample(non_visited,1)) # 随机选择一架飞机==>由权重值选择一架飞机
        i = i[0]
        non_visited.remove(i)
        pairs = {}
        r1 = route[i]
        dictr1 = {x: y for y, x in enumerate(r1[1:-1])}
        for j in r1[1:-1]:
             for k in r1[1:-1]:
                if int(j) < int(k):
                    if param.Oj[str(j)] == param.Dj[str(k)]:
                        pairs[j] = k
        if len(pairs)==0:
            continue
        for _i in param.F:
            r2 = route[_i]
            for idx__j,_j in enumerate(r2[1:-1]):
                for j,k in pairs.items():
                    if param.Dj[str(_j)] == param.Oj[str(j)]:
                        index_j = idx__j+1
                        legaft = str(r2[index_j+1]) if index_j+2==len(r2) else r2[index_j+1]
                        if CON.loc[_j,j] == 0 or CON.loc[k,legaft] == 0:
                            continue
                        route_new1 = r1[:dictr1[j]+1] + r1[dictr1[k]+2:]
                        route_new2 = r2[:index_j+1] + r1[dictr1[j]+1:dictr1[k]+2] + r2[index_j+1:]
                        A = {i:route_new1}
                        B = {_i:route_new2}   
                        new1 = ben(A)
                        new2 = ben(B)
                        if new1[i] + new2[_i] > ben_r[i] + ben_r[_i] +1 : # obj 没有更新，看什么时候更新
                            print("-------------------ins")
                            route[i] = route_new1
                            route[_i] = route_new2
                            ben_r[i] = new1[i]
                            ben_r[_i] = new2[_i]
                            return det_ins
                        elif PROBA <= random.random() and new1[i] + new2[_i] > ben_r[i] + ben_r[_i] - MINBOUND:
                            route[i] = route_new1
                            route[_i] = route_new2
                            ben_r[i] = new1[i]
                            ben_r[_i] = new2[_i]
                            return det_ins
                    elif param.Oj[str(_j)] == param.Oj[str(j)] and idx__j == 0: #完好性保证，都可以插入
                        if CON.loc[k,_j] == 0:
                            continue
                        route_new1 = r1[:dictr1[j]+1] + r1[dictr1[k]+2:]
                        route_new2 = [r2[0]] + r1[dictr1[j]+1:dictr1[k]+2] + r2[1:]
                        A = {i:route_new1}
                        B = {_i:route_new2}   
                        new1 = ben(A)
                        new2 = ben(B)
                        if new1[i] + new2[_i] > ben_r[i] + ben_r[_i] +1 : # obj 没有更新，看什么时候更新
                            print("-------------------ins")
                            route[i] = route_new1
                            route[_i] = route_new2
                            ben_r[i] = new1[i]
                            ben_r[_i] = new2[_i]
                            return det_ins
                        elif PROBA <= random.random() and new1[i] + new2[_i] > ben_r[i] + ben_r[_i] - MINBOUND:
                            route[i] = route_new1
                            route[_i] = route_new2
                            ben_r[i] = new1[i]
                            ben_r[_i] = new2[_i]
                            return det_ins

def getNei_cro(route,ben_r,ben):
    non_visited = copy.deepcopy(param.F)
    det_cro =True
    while True:
        if len(non_visited) == 1:
            print("can't cross")
            det_cro = False
            return det_cro
        i = sorted(random.sample(non_visited,1)) 
        i = i[0]
        non_visited.remove(i)
        r1 = route[i]
        for _i in non_visited:
            r2 = route[_i]
            if r2[0] != r1[0]:
                continue
            for idx_j,j in enumerate(r1[1:-1]):
                for idx__j,_j in enumerate(r2[1:-1]):
                    if param.Dj[str(j)] == param.Oj[str(_j)]:
                        legaft1 = str(r1[idx_j+2]) if idx_j + 3 == len(r1) else r1[idx_j+2]
                        legbef2 = str(r2[idx__j]) if idx__j == 0 else r2[idx__j]
                        if CON.loc[j,_j] == 0 or CON.loc[legbef2,legaft1] == 0:
                            continue
                        route_new1 = r1[:idx_j+2] + r2[idx__j+1:]
                        route_new2 = r2[:idx__j+1] + r1[idx_j+2:]
                        A = {i:route_new1}
                        B = {_i:route_new2}   
                        new1 = ben(A)
                        new2 = ben(B)
                        if new1[i] + new2[_i] > ben_r[i] + ben_r[_i] + 1: # obj 没有更新，看什么时候更新
                            print("-------------------cro")
                            route[i] = route_new1
                            route[_i] = route_new2
                            ben_r[i] = new1[i]
                            ben_r[_i] = new2[_i]
                            return det_cro
                        elif PROBA <= random.random() and new1[i] + new2[_i] > ben_r[i] + ben_r[_i] -MINBOUND:
                            print("-------------------cro")
                            route[i] = route_new1
                            route[_i] = route_new2
                            ben_r[i] = new1[i]
                            ben_r[_i] = new2[_i]
                            return det_cro
                    elif param.Oj[str(j)] == param.Oj[str(_j)] and idx_j == 0:
                        if CON.loc[r2[idx__j],j] == 0:
                            continue
                        route_new1 = [r1[0]] + r2[idx__j+1:]
                        route_new2 = r2[:idx__j+1] + r1[1:]
                        A = {i:route_new1}
                        B = {_i:route_new2}   
                        new1 = ben(A)
                        new2 = ben(B)
                        if new1[i] + new2[_i] > ben_r[i] + ben_r[_i] + 1 : # obj 没有更新，看什么时候更新
                            print("-------------------cro2")
                            route[i] = route_new1
                            route[_i] = route_new2
                            ben_r[i] = new1[i]
                            ben_r[_i] = new2[_i]
                            return det_cro
                    elif param.Dj[str(j)] == param.Dj[str(_j)] and _j == r2[-2]:
                        if CON.loc[_j,r1[idx_j+2]] == 0:
                            continue
                        route_new1 = r2[:-1] + r1[idx_j+2:]
                        route_new2 = r1[:idx_j+2] + [r2[-1]]
                        A = {i:route_new1}
                        B = {_i:route_new2}   
                        new1 = ben(A)
                        new2 = ben(B)
                        if new1[i] + new2[_i] > ben_r[i] + ben_r[_i] + 1 : # obj 没有更新，看什么时候更新
                            print("-------------------cro3")
                            route[i] = route_new1
                            route[_i] = route_new2
                            ben_r[i] = new1[i]
                            ben_r[_i] = new2[_i]
                            return det_cro

def getNei_del(route,ben_r,ben):#0.002 0.003 0.005
    non_visited = copy.deepcopy(param.F)
    det_del = True
    deter = True
    while deter == True:
        if len(non_visited) == 1:
            print("can't delete")
            det_del = False
            return det_del
        i = sorted(random.sample(non_visited,1)) # 随机选择一架飞机==>由权重值选择一架飞机
        i = i[0]
        non_visited.remove(i)
        
        pairs = [] # 在2中此部分进行了更变,加入了LM不删机制和最小环删除机制
        index_v = 1
        r = route[i]
        num_v = len(r[1:-1])
        for index_v in range(num_v)[1:]:
            for v in range(num_v+1)[1:]:
                j = r[v]
                k = r[v+index_v]
                if param.Oj[str(j)] == param.Dj[str(k)]:
                    det = True
                    if det ==  True:
                        pairs.append([j,k])
                if v + index_v == num_v:
                    break

        for j,k in pairs:
            index_s = r.index(j)
            index_e = r.index(k)
            route_new = r[:index_s] + r[index_e+1:]  #  +1?
            C = {i:route_new}
            new = ben(C)
            if new[i] > ben_r[i]:
                print("-----------del")
                del_route = r[index_s:index_e+1]
                SR[param.Oj[str(j)]].append(del_route)
                route[i] = route_new
                ben_r[i] = new[i]
                return det_del

def getNei_add(route,ben_r,ben):
    non_visited = copy.deepcopy(param.F)
    det_add = True
    while True:
        if len(non_visited) == 1:
            print("can't add")
            det_add = False
            return det_add
        i = sorted(random.sample(non_visited,1)) # 随机选择一架飞机==>由权重值选择一架飞机
        i = i[0]
        non_visited.remove(i)
        r = route[i]
        for idx_j,j in enumerate(r[1:-1]):
            if idx_j != 0:
                for leg in SR[param.Dj[str(j)]]:
                    if CON.loc[j,leg[0]] == 0 or CON.loc[leg[-1],r[idx_j+2]] == 0:
                        continue
                    route_new = r[:idx_j+2] + leg + r[idx_j+2:]
                    C = {i:route_new}
                    new = ben(C)
                    if new[i] > ben_r[i] + 1:
                        print("-----------add")
                        SR[param.Dj[str(j)]].remove(leg)
                        route[i] = route_new
                        ben_r[i] = new[i]
                        return det_add
            elif idx_j == 0:
                for leg in SR[param.Oj[str(j)]]:
                    if CON.loc[leg[-1],j] == 0:
                        continue
                    route_new = r[:idx_j+1] + leg + r[idx_j+1:]
                    C = {i:route_new}
                    new = ben(C)
                    if new[i] > ben_r[i] + 1:
                        print("-----------add")
                        SR[param.Oj[str(j)]].remove(leg)
                        route[i] = route_new
                        ben_r[i] = new[i]
                        return det_add

def getNei_del_mix(route,ben_r,ben):
    non_visited = copy.deepcopy(param.F)
    det_del = True
    while True:
        if len(non_visited) == 1:
            print("can't delete")
            det_del = False
            return det_del
        i = sorted(random.sample(non_visited,1)) # 随机选择一架飞机==>由权重值选择一架飞机
        i = i[0]
        non_visited.remove(i)
        
        pairs = [] # 在2中此部分进行了更变,加入了LM不删机制和最小环删除机制
        index_v = 1
        r = route[i]
        num_v = len(r[1:-1])
        for index_v in range(num_v)[1:]:
            for v in range(num_v+1)[1:]:
                j = r[v]
                k = r[v+index_v]
                if param.Oj[str(j)] == param.Dj[str(k)]:
                    det = True
                    if det ==  True:
                        pairs.append([j,k])
                if v + index_v == num_v:
                    break

        for j,k in pairs:
            index_s = r.index(j)
            index_e = r.index(k)
            route_new = r[:index_s] + r[index_e+1:]  #  +1?
            C = {i:route_new}
            new = ben(C)
            if new[i] > ben_r[i]:
                print("-----------del")
                del_route = r[index_s:index_e+1]
                SR[param.Oj[str(j)]].append(del_route)
                route[i] = route_new
                ben_r[i] = new[i]
                return det_del

def getNei_exc_cp(route):
    non_visited = copy.deepcopy(param.P1)
    #route_copy = route #怎么会动态拷贝哦
    global obj
    det_exc = True
    deter = True
    #num = 0
    while deter == True:
        if len(non_visited) == 1:
            print("can't exchange")
            det_exc = False
            return det_exc
        #sort_r = sort_route(R)
        #i = sort_r.aircraft[num]  #数据更新或者未更新的问题 
        #num = num + 1
        #i = int(i)
        i = sorted(random.sample(non_visited,1)) # 随机选择一架飞机==>由权重值选择一架飞机
        i = i[0]
        non_visited.remove(i)
        pairs = {}
        r1 = route[i]
        dictr1 = {x: y for y, x in enumerate(r1[1:-1])}
        for j in r1[1:-1]:
             for k in r1[1:-1]:
                if int(j) <= int(k):
                    pairs[j] = k
        '''if len(pairs) == 0: # 是否无法交换？
            continue'''
        for _i in non_visited:# 找到合适的可更换区间  注意找出问题后改为non_visited
            r2 = route[_i]
            dictr2 =  {x: y for y, x in enumerate(r2[1:-1])}
            for j,k in pairs.items():# 但这个交换总是从找到的第一个区间开始找？ 先给区间进行寻找还是先找第二条再历遍区间会好些????
                index_bef = dictr1[j]+1
                index_aft = dictr1[k]+1
                new1 ={i: -100000}
                new2 ={_i: -100000}#初始化所选飞机的benefit
                for _j in  r2[1:-1]:
                    for _k in r2[1:-1]:
                        if int(_j) <= int(_k):
                            if param.Oj[str(_j)] == param.Oj[str(j)] and param.Dj[str(_k)] == param.Dj[str(k)]:#判断前站后站是否相同
                                index_before = dictr2[_j]+1
                                index_after = dictr2[_k]+1
                                # 获取前后航班段
                                leg_bef = str(r1[int(index_bef - 1)]) if index_bef - 1 == 0 else r1[int(index_bef - 1)]
                                leg_aft = str(r1[int(index_aft + 1)]) if index_aft + 2 == len(r1) else r1[int(index_aft + 1)]
                                leg_before = str(r2[int(index_before-1)]) if index_before - 1 == 0 else r2[int(index_before-1)]
                                leg_after = str(r2[int(index_after+1)])  if index_after + 2 == len(r2) else r2[int(index_after+1)]
                                
                                if CON.loc[leg_bef, _j] == 0: # 判断插入段最前的航班段能否与原航线里的断开处形成连接，即时间上是否合理
                                    continue
                                if CON.loc[_k, leg_aft] == 0: # 插入段最后段与断开处后端
                                    continue
                                if CON.loc[leg_before, j] == 0:
                                    continue
                                if CON.loc[k,leg_after] == 0:
                                    continue 
                                route_new1 = r1[:index_bef] + r2[index_before:index_after+1] + r1[index_aft+1:]
                                route_new2 = r2[:index_before] + r1[index_bef:index_aft+1] + r2[index_after+1:]
                                A = {i:route_new1}
                                B = {_i:route_new2}   
                                new1 = ben_route_cp(A)#本交换只涉及两条航路，故这样就够了
                                new2 = ben_route_cp(B)
                        if new1[i] + new2[_i] > cost_r[i] + cost_r[_i] + 1 : # obj 没有更新，看什么时候更新
                            '''print("------------exc")
                            print(A)
                            print(B)
                            print(cost_r[i])
                            print(new1[i])
                            print(cost_r[_i])
                            print(new2[_i])'''
                            route[i] = route_new1
                            route[_i] = route_new2
                            cost_r[i] = new1[i]
                            cost_r[_i] = new2[_i]
                            return det_exc

def plot_benefits_time(A, B, title, save_path=None):
    """
    绘制以时间B为横坐标，数值A为纵坐标的图表，自动调整坐标范围
    
    参数:
        A (list): 数值列表
        B (list): 时间戳列表，由time.time()生成
        title (str): 图表标题，默认为"收益-时间关系图"
        save_path (str): 保存图表的路径，默认为None（不保存）
    
    返回:
        plt.Figure: matplotlib图表对象
    """
    # 确保输入列表长度一致
    if len(A) != len(B):
        raise ValueError("数值列表A和时间戳列表B长度必须一致")
    
    if len(A) == 0:
        raise ValueError("输入列表不能为空")
    
    # 创建图表
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 计算相对时间（秒）
    relative_times = [t - B[0] for t in B]
    
    # 绘制折线图
    ax.plot(relative_times, A, marker='o', color='#1f77b4', linestyle='-', linewidth=2)
    
    # 设置标题和标签
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Time (s)", fontsize=12)
    ax.set_ylabel("Benefits", fontsize=12)
    
    # 自适应坐标轴
    # 为X轴添加一些边距
    x_min, x_max = min(relative_times), max(relative_times)
    x_margin = (x_max - x_min) * 0.05 if len(relative_times) > 1 else 0.5
    ax.set_xlim([x_min - x_margin, x_max + x_margin])
    
    # 为Y轴添加一些边距
    y_min, y_max = min(A), max(A)
    y_margin = (y_max - y_min) * 0.1 if y_min != y_max else 0.5
    ax.set_ylim([y_min - y_margin, y_max + y_margin])
    
    # 添加网格线
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # 如果数据点小于10个，则在每个点上标注数值
    if len(A) < 10:
        for i, (x, y) in enumerate(zip(relative_times, A)):
            ax.annotate(f"{y:.2f}", (x, y), textcoords="offset points", 
                        xytext=(0, 10), ha='center')
    
    # 美化图表
    plt.tight_layout()
    
    # 如果提供了保存路径，则保存图表
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

def VND_nondel(R,benr,ben):
    for _ in range(10000):
        det_rep = getNei_rep(R,benr,ben)
        T.append(time.time())
        BENI.append(sum(ben_r.values()) )
        det_ins = getNei_ins(R,benr,ben)
        T.append(time.time())
        BENI.append(sum(ben_r.values()) )
        det_cro = getNei_cro(R,benr,ben)
        T.append(time.time())
        BENI.append(sum(ben_r.values()) )
        det_exc = getNei_exc(R,benr,ben)
        T.append(time.time())
        BENI.append(sum(ben_r.values()) )
        if det_rep == False and det_ins == False and det_cro == False and det_exc == False:
            if _ == 0:
                return R,False
            elif _ > 0:
                return R,True

def VND(ben, cooldown_lens, iterations=10000):
    """
    R              -- 当前解结构
    ben_r          -- 存放各部件效益的 dict，用于 sum(ben_r.values())
    T              -- 时间戳列表（传入空列表即可）
    BENI           -- 目标函数值列表（传入空列表即可）
    cooldown_lens  -- 各算子对应的冷却时间字典，例如 {'rep':5, 'ins':3, ...}
    iterations     -- 最大迭代次数（默认 10000）
    """

    # 按顺序排列所有算子
    ops = [
        ('rep', getNei_rep),
        ('ins', getNei_ins),
        ('del', getNei_del),
        ('cro', getNei_cro),
        ('exc', getNei_exc),
    ]
    # 初始化所有冷却计数器为 0
    cooldown = {name: 0 for name, _ in ops}

    for it in range(iterations):
        iteration_success = False  # 本次迭代是否有任一算子成功
        for name, func in ops:
            # 冷却期内跳过本算子并倒计时
            if cooldown[name] > 0:
                cooldown[name] -= 1
                continue

            # 尝试调用算子
            success = func(R,ben_r,ben)
            T.append(time.time())
            BENI.append(sum(ben_r.values()))

            if success:
                iteration_success = True
            else:
                # 失败就设置对应算子的冷却期
                cooldown[name] = cooldown_lens.get(name, 0)

        # 如果本次迭代里没有任何算子成功，则提前退出
        if not iteration_success:
            print(f"第 {it+1} 次迭代中所有算子均失败，提前终止。")
            break

def escape(max_time,manround,ben):
    mt = 0
    for _ in range(manround):
        det_rep = getNei_rep(R,ben_r,ben)
        T.append(time.time())
        BENI.append(sum(ben_r.values()) )
        det_ins = getNei_ins(R,ben_r,ben)
        T.append(time.time())
        BENI.append(sum(ben_r.values()) )
        det_cro = getNei_cro(R,ben_r,ben)
        T.append(time.time())
        BENI.append(sum(ben_r.values()) )
        det_exc = getNei_exc(R,ben_r,ben)
        T.append(time.time())
        BENI.append(sum(ben_r.values()) )
        if det_rep == False and det_ins == False and det_cro == False and det_exc == False:
            mt += 1
            if mt == max_time:
                break

def VND_mix(R,RC):
    for _ in range(10000):
        det_rep = getNei_rep(R,ben_r,ben_route_afc)
        det_ins = getNei_ins(R,ben_r,ben_route_afc)
        det_cro = getNei_cro(R,ben_r,ben_route_afc)
        det_exc = getNei_exc(R,ben_r,ben_route_afc)
        FC = {
        flight: ac
        for ac, path in RC.items()
        for flight in path[1:-1]
        }
        det_repc = getNei_rep(RC,cost_r,ben_route_cp)
        det_insc = getNei_ins(RC,cost_r,ben_route_cp)
        det_croc = getNei_cro(RC,cost_r,ben_route_cp)
        det_excc = getNei_exc(RC,cost_r,ben_route_cp)
        if det_rep == False and det_ins == False and det_cro == False and det_exc == False and det_repc == False and det_insc == False and det_croc == False and det_excc == False:
            if _ == 0:
                return R,RC,False
            elif _ > 0:
                return R,RC,True
    

if __name__ == '__main__':
    num = 93
    
    
    data = json.load(open(f'./data_and_param_{num}.json'))
    param = parameters.Parameters(data)
    param_M = xr.open_dataset(f"./param_matrices_{num}.nc")
    #获取CON
    fs = param.L + list(map(str, param.S))
    CON = pd.DataFrame(
        param_M.CON.values, 
        index=fs,
        columns=fs
    )

    
    R = get_initial_solution(num)
    t = 1
    T = []
    BENI = []
    #测试两行的计算速度
    '''R = {'1': ['SCOK', '3', '39', 'SCOK'], '2': ['SDFW', '68', '87', 'SDFW'], '3': ['SDFW', '24', '52', '74', '88', 'SDFW'], '4': ['SDFW', '57', '71', 'SDFW'], '5': ['SAGA', '47', '58', 'SAGA'], '6': ['SAMS', '18', '40', '62', '77', 'SAMS'], '7': ['SDOH', '14', '44', 'SDOH'], '8': ['SMCT', '21', '49', '67', '80', 'SMCT'], '10': ['SBJV', '60', '76', 'SBJV'], '11': ['SFRA', '13', '31', '46', '66', 'SFRA'], '13': ['SAMS', '28', '56', '75', '86', 'SAMS'], '14': ['SEWR', '25', '36', '38', '51', '63', '78', 'SEWR'], '15': ['SDOH', '4', '41', 'SDOH'], '16': ['SDFW', '9', '30', '84', '90', 'SDFW'], '17': ['SDOH', '61', '79', '91', 'SDOH'], '18': ['SAGA', '8', '34', '54', '92', 'SAGA'], '19': ['SEDI', '11', '15', '50', '55', '59', '65', 'SEDI'], '20': ['SFRA', '16', '26', '43', '64', '70', '85', 'SFRA'], '21': ['SDOH', '1', '22', 'SDOH']}
    RC = {'1': ['SCOK', '3', '39', 'SCOK'], '2': ['SDFW', '68', '87', 'SDFW'], '3': ['SDFW', '9', '34', '54', '70', '85', '90', 'SDFW'], '5': ['SAGA', '8', '30', '57', '71', '84', '92', 'SAGA'], '6': ['SAMS', '28', '51', '63', '86', 'SAMS'], '8': ['SMCT', '21', '80', 'SMCT'], '9': ['SRUH', '22', '61', '79', 'SRUH'], '10': ['SBJV', '60', '76', 'SBJV'], '11': ['SFRA', '13', '31', '56', '75', 'SFRA'], '13': ['SAMS', '18', '40', '62', '77', 'SAMS'], '14': ['SEWR', '25', '36', '38', '43', '64', '78', 'SEWR'], '15': ['SDOH', '4', '49', '67', '91', 'SDOH'], '16': ['SDFW', '24', '52', '74', '88', 'SDFW'], '17': ['SDOH', '14', '44', 'SDOH'], '18': ['SAGA', '47', '58', 'SAGA'], '19': ['SEDI', '11', '15', '50', '55', '59', '65', 'SEDI'], '20': ['SFRA', '16', '26', '46', '66', 'SFRA'], '21': ['SDOH', '1', '41', 'SDOH']}
    R = {k: [int(i) if i.isdigit() else i for i in v] for k, v in R.items()}
    RC = {k: [int(i) if i.isdigit() else i for i in v] for k, v in RC.items()}'''
     
    ben_r = ben_route_ac(R) 
    BEN_R = sum(ben_r.values())
    BEN = BEN_R 
    t1 = time.time()
    '''for _ in range(10):
        getNei_cro(R)'''
    
    SR = {key: [] for key in param.Oj.values()}

    # Initial VND ##########################################
    PROBA = 1
    MINBOUND = 50000
    R,deter = VND_nondel(R,ben_r,ben_route_ac)
    cooldown_lens = {
        'rep': 1,   # 替换操作失败后跳过
        'ins': 2,   # 插入操作失败后跳过
        'del': 1,   # 删除操作失败后跳过
        'cro': 1,   # 交叉操作失败后跳过
        'exc': 2,   # 交换操作失败后跳过
    }
    VND(ben_route_ac,cooldown_lens, iterations=10000)
    print(sum(ben_r.values()))
    print(time.time()-t1)
    # escape and VND ################################
    PROBA = 0.99
    for _ in range(5):
        MINBOUND = 50000
        maxtime = 3
        escape(maxtime,10,ben_route_ac)
        MINBOUND = 10000
        VND(ben_route_ac, cooldown_lens, iterations=10000)
        deter = True
        while deter == True:
            deter = getNei_add(R,ben_r,ben_route_ac)
    
    # recovery ####################################
    PROBA = 1
    VND(ben_route_ac, cooldown_lens, iterations=10000)

    # crew exc ######################################
    RC= copy.deepcopy(R) 
    ben_r = ben_route_a(R) 
    cost_r = ben_route_cp(RC)   
    BEN_R = sum(ben_r.values())
    det_exc_c = True

    '''while det_exc_c == True:
        det_exc_c = getNei_exc_cp(RC)
        T.append(time.time())
        BENI.append(BEN_R+sum(cost_r.values()))'''
    FC = {
    flight: ac
    for ac, path in RC.items()
    for flight in path[1:-1]
    }

    # crew & aircraft ################################
    R,RC,_ = VND_mix(R,RC)

    t2 = time.time()
    BEN = sum(ben_r.values()) + sum(cost_r.values())
    print(t2-t1)
    fig = plot_benefits_time(BENI, T, title=f"Ben-Time_VNS_{num}")
    plt.show()
    print(BEN)
    print(sum(ben_r.values()))
    print(sum(cost_r.values()))
    # print(R)
