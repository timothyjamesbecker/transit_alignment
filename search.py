import argparse
import os
import copy
import sys
import time
import datetime
import re
import glob
import gzip
import pickle
import subprocess
from sklearn.manifold import MDS
import itertools as it
import multiprocessing as mp
import numpy as np
# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import transit_utils as tu
import read_utils as ru

def mem_size(obj,unit='G'):
    if unit =='T':
        return round(sys.getsizeof(obj)/(1024*1024*1024*1024.0),2) #TB
    if unit =='G':
        return round(sys.getsizeof(obj)/(1024*1024*1024.0),2) #GB
    if unit =='M':
        return round(sys.getsizeof(obj)/(1024*1024.0),2) #MB
    if unit =='K':
        return round(sys.getsizeof(obj)/(1024.0),2) #KB

def read_person_trip_list(path,delim=',',quoting='"'): #more open since may want to play around with demand files : IE dynamic...
    header,data = [],[]
    with open(path,'r') as f:
        raw = [re.sub(quoting+'+',quoting,row.replace('\r','').replace('\n','')) for row in f.readlines()]
    header = [f for f in raw[0].rsplit(delim)]# field names=stop_id,stop_name,stop_lat,stop_lon,zone_id
    c_idx = {header[i]:i for i in range(len(header))}
    data,j = [],0
    for row in raw[1:]:
        quotes = row.count(quoting)//2
        if quotes<1: data += [row.split(',')]
        else:        #some variable amount of quoting
            sect,t,sub = [],[],row
            for i in range(quotes):
                start = sub.find(quoting)
                end   = sub[start+1:].find(quoting)+1
                if start>0: sect += [sub[:start-1],sub[start:start+end+1]]
                else:       sect += [sub[start:start+end+1]]
                sub = sub[start+end+2:]
            sect += [sub]
            for s in sect:
                if not s.startswith(quoting): t += s.split(delim)
                else:                         t += [s]
            if len(t)!=len(header): print('issue with data row j=%s'%j)
            data += [t]
        j += 1
    persons,j = {},0
    for i in range(len(data)):
        next_day  = [False,False]
        prior_day = [False,False]
        pid    = int(data[i][c_idx['personid']])
        date   = data[i][c_idx['traveldate']]
        o_taz  = int(data[i][c_idx['origin_bg_geoid_linked']])
        o_add  = data[i][c_idx['o_address_recode_2_linked']].replace('"','')
        d_taz  = int(data[i][c_idx['destination_bg_geoid_recode_linked']])
        d_add  = data[i][c_idx['d_address_recode_linked']].replace('"','')
        o_time = data[i][c_idx['departure_time_hhmm_linked']]
        d_time = data[i][c_idx['arrival_time_hhmm_linked']]
        if o_time.find(' (next day)')>=0: o_time = o_time.split(' (next day)')[0]; next_day[0] = True
        if d_time.find(' (next day)')>=0: d_time = d_time.split(' (next day)')[0]; next_day[1] = True
        o_time = o_time.split(' ')[0]
        d_time = d_time.split(' ')[0]
        try:
            st_time = [datetime.datetime.strptime(date+' '+o_time,'%m/%d/%Y %H:%M'),
                       datetime.datetime.strptime(date+' '+d_time,'%m/%d/%Y %H:%M')]
            if next_day[0]: st_time[0]+=datetime.timedelta(days=1)
            if next_day[1]: st_time[1]+=datetime.timedelta(days=1)
            if pid in persons: persons[pid] += [[o_taz,o_add,st_time[0],d_taz,d_add,st_time[1]]]
            else:              persons[pid]  = [[o_taz,o_add,st_time[0],d_taz,d_add,st_time[1]]]; j+=1
        except Exception as E:
            print('row i=%s was not well formed, possible junk data:%s\n%s'%(i,E,data[i]))
            pass
    ps,i = {},0
    for pid in persons:
        ps[i] = persons[pid]
        i += 1
    return ps

#will load cached data or if not present, generate it
def load_network_data(n_base,walk=0.5,search_time=[0,115200],search_date=None,target_cpus=None):
    if os.path.exists(n_base+'/network.pickle.gz'):
        print('trying to load the network data...')
        with gzip.GzipFile(glob.glob(n_base+'/network*pickle.gz')[0],'rb') as f:
            D = pickle.load(f)
    else:
        print('preprocessing the network..')
        command = ['python3',os.path.dirname(__file__)+'/preprocess_network.py','--in_path',n_base,
                   '--out_dir',n_base,'--walk %s'%walk,
                   '--time %s,%s'%(search_time[0],search_time[1])]
        if search_date is not None:  command += ['--date',search_date]
        if target_cpus is not None: command += ['--cpus',str(target_cpus)]
        print('starting network preprocessing, please stand by for params:%s'%command)
        try:
            t_start = time.time()
            out = ''
            out = subprocess.check_output(' '.join(command),shell=True)
            if out!='': print(out.decode(encoding='utf-8'))
            t_stop = time.time()
            print('preprocessing in %s sec'%round(t_stop-t_start,2))
            with gzip.GzipFile(glob.glob(n_base+'/network.pickle.gz')[0],'rb') as f:
                D = pickle.load(f)
        except Exception as E:
            print(E)
            pass
    return D

def get_datetime(person_trip):
    person_date = person_trip[2].date()
    st_time = [person_trip[2].time(),person_trip[5].time()]
    person_time = [datetime.timedelta(hours=st_time[0].hour,minutes=st_time[0].minute,seconds=st_time[0].second),
                   datetime.timedelta(hours=st_time[1].hour,minutes=st_time[1].minute,seconds=st_time[1].second)]
    person_time = [int(person_time[0].total_seconds()),int(person_time[1].total_seconds())]
    return [person_date.strftime('%m/%d/%Y'),person_time]

def get_processed_service_ids(D):
    match = 'service_id_'
    ks = []
    for k in list(sorted(D.keys())):
        if k.startswith(match):
            ks += [k.split('service_id_')[-1]]
    return ks

#base time is in seconds,
def get_search_time(base_time,buff_time=10,symetric=True):
    if symetric: search_time = [max(0,base_time-buff_time//2),base_time+buff_time//2]
    else:        search_time = [base_time,base_time+buff_time]
    return search_time

#buff_time is in minutes, walking speed is in mi/hour, start with forwar search only...
def start_od_search(person_trip,w_dist,p_dist,s_dist,v_dist,buff_time=10.0,max_time=90.0,
                    walk_speed=3.0,bus_speed=12.0,drive_speed=30.0,min_target=1000,heading_limit=True):
    final_candidates = None
    w_secs = (60.0*60.0)/walk_speed  #time walking in seconds => seconds/mi
    p_secs = (60.0*60.0)/drive_speed #time driving in seconds => seconds/mi
    dt = get_datetime(person_trip)
    search_time = get_search_time(dt[1][0],buff_time=buff_time*60)
    o_taz,d_taz = person_trip[0],person_trip[3]
    service_id  = ru.get_service_id(calendar,dt[0])
    if (o_taz in w_dist or d_taz in w_dist) and service_id in service_ids:
        print('o_taz=%s, d_taz=%s and service_id=%s were found for person trip'%(o_taz,d_taz,service_id))
        sid = 'service_id_%s'%service_id
        seqs,graph,l_dist,l_idx = D[sid]['seqs'],D[sid]['graph'],D[sid]['l_dist'],D[sid]['l_idx']

        #enumerate the posible o,d stop pairs and rank by minimum wdist[otaz][o]+sdist[o,d]+wdist[dtaz][d]
        wods,pods = {'o':{},'d':{}},{'o':{},'d':{}}
        if o_taz in w_dist:
            for o in w_dist[o_taz]:
                if o in seqs: wods['o'][o] = w_dist[o_taz][o]
        if d_taz in w_dist:
            for d in w_dist[d_taz]:
                if d in seqs: wods['d'][d] = w_dist[d_taz][d]
        if o_taz in p_dist:
            for o in p_dist[o_taz]:
                if o in seqs: pods['o'][o] = p_dist[o_taz][o]
        if d_taz in p_dist:
            for d in p_dist[d_taz]:
                if d in seqs: pods['d'][d] = p_dist[d_taz][d]
        od_search = []
        for o in wods['o']:
            for d in wods['d']:
                sw_dist = s_dist[o,d]+wods['o'][o]+wods['d'][d]
                #::: TIME UPPER BOUND :::-------------------------------------------------
                if sw_dist<3.0:
                    time_upper = dt[1][0]+int(round(60.0*(60.0/walk_speed)*(sw_dist)))
                elif sw_dist>=3.0 and sw_dist<6.0:
                    time_upper = dt[1][0]+int(round(60.0*(60.0/walk_speed)*(wods['o'][o]+wods['d'][d])))+\
                        int(round(60.0*(60.0/bus_speed)*(2.0*s_dist[o,d])))
                else:
                    time_upper = dt[1][0]+max_time*60 #90 minute upper bound
                #::: TIME UPPER BOUND :::-------------------------------------------------
                od_search += [[(o,d),sw_dist,time_upper]]
        od_search = sorted(od_search,key=lambda x: x[2])

        #now we can start search for trips------------------------------------------------------------------------------------
        candidate_trips,c_i = [],0
        for c in od_search:
            o,d,e,time_max = c[0][0],c[0][1],c[1],c[2] #origin stop, destination stop and distance estimate
            if o in wods['o'] and d in wods['d']:
                owt = int(round(wods['o'][o]*w_secs+0.5)) #walking time to get to the stop in seconds
                dwt = int(round(wods['d'][d]*w_secs+0.5)) #walking time to get to the dest in seconds
                s_time = [search_time[0]+owt,search_time[1]+owt]
                if dwt<=buff_time*60.0 and owt<=buff_time*60.0 and o in graph:
                    for i in range(len(graph[o]['out'])):
                        if graph[o]['out'][i][1]>=s_time[0] and graph[o]['out'][i][1]<=s_time[1]:
                            tid = graph[o]['out'][i][2:]
                            candidate_trip = [(-2,o,owt),graph[o]['out'][i][1]]+list(tid)
                            candidate_trips  += [candidate_trip+[sdist_trip_trend(tid[0],tid[1],d,seqs,s_dist,time_max),(d,-2,dwt),time_max]]
            c_i += 1
        candidate_trips = sorted(candidate_trips,key=lambda x: (x[1],-1*x[4],x[6]))
        l_can = len(candidate_trips)
        #drive-transit-walk or walk-transit-drive searching...
        if len(candidate_trips)<1:
            print('empty walking candidates for person trip:%s'%[o_taz,d_taz,service_id])
            #check o_taz->(p_dist) d_taz->(p_dist)
            od_search = []
            for o in wods['o']:
                for d in pods['d']:
                    sw_dist = s_dist[o,d]+wods['o'][o]
                    #::: TIME UPPER BOUND :::-------------------------------------------------
                    if sw_dist<3.0:
                        time_upper = dt[1][0]+int(round(60.0*(60.0/walk_speed)*(sw_dist)))
                    elif sw_dist>=3.0 and sw_dist<6.0:
                        time_upper = dt[1][0]+int(round(60.0*(60.0/walk_speed)*(wods['o'][o])))+\
                            int(round(60.0*(60.0/bus_speed)*(2.0*s_dist[o,d])))
                    else:
                        time_upper = dt[1][0]+max_time*60 #90 minute upper bound
                    #::: TIME UPPER BOUND :::-------------------------------------------------
                    od_search += [[(o,d),sw_dist,time_upper]]
            for o in pods['o']:
                for d in wods['d']:
                    sw_dist = s_dist[o,d]+wods['d'][d]
                    #::: TIME UPPER BOUND :::-------------------------------------------------
                    if sw_dist<3.0:
                        time_upper = dt[1][0]+int(round(60.0*(60.0/walk_speed)*(sw_dist)))
                    elif sw_dist>=3.0 and sw_dist<6.0:
                        time_upper = dt[1][0]+int(round(60.0*(60.0/walk_speed)*(wods['d'][d])))+\
                            int(round(60.0*(60.0/bus_speed)*(2.0*s_dist[o,d])))
                    else:
                        time_upper = dt[1][0]+max_time*60 #90 minute upper bound
                    #::: TIME UPPER BOUND :::-------------------------------------------------
                    od_search += [[(o,d),sw_dist,time_upper]]
            od_search = sorted(od_search,key=lambda x: x[2])
            candidate_trips,c_i = [],0
            for c in od_search:
                o,d,e,time_max = c[0][0],c[0][1],c[1],c[2] #origin stop, destination stop and distance estimate
                if o in wods['o'] and d in pods['d']:
                    owt = int(round(wods['o'][o]*w_secs+0.5)) #walking time to get the stop in seconds
                    dpt = int(round(pods['d'][d]*p_secs+0.5)) #walking time to get the stop in seconds
                    s_time = [search_time[0]+owt,search_time[1]+owt]
                    if owt<buff_time*60.0 and o in graph:
                        for i in range(len(graph[o]['out'])):
                            if graph[o]['out'][i][1]>=s_time[0] and graph[o]['out'][i][1]<=s_time[1]:
                                tid = graph[o]['out'][i][2:]
                                candidate_trip = [(-2,o,owt),graph[o]['out'][i][1]]+list(tid)
                                candidate_trips  += [candidate_trip+[sdist_trip_trend(tid[0],tid[1],d,seqs,s_dist,time_max),(d,-3,dpt),time_max]]
                elif o in pods['o'] and d in wods['d']:
                    opt = int(round(pods['o'][o]*p_secs+0.5))
                    dwt = int(round(wods['d'][d]*w_secs+0.5))
                    s_time = [search_time[0]+opt,search_time[1]+opt]
                    if dwt<buff_time*60.0 and o in graph:
                        for i in range(len(graph[o]['out'])):
                            if graph[o]['out'][i][1]>=s_time[0] and graph[o]['out'][i][1]<=s_time[1]:
                                tid = graph[o]['out'][i][2:]
                                candidate_trip = [(-3,o,opt),graph[o]['out'][i][1]]+list(tid)
                                candidate_trips  += [candidate_trip+[sdist_trip_trend(tid[0],tid[1],d,seqs,s_dist,time_max),(d,-2,dwt),time_max]]
                c_i += 1
            candidate_trips = sorted(candidate_trips,key=lambda x: (x[1],-1*x[4],x[6]))
            l_can = len(candidate_trips)
        if len(candidate_trips)<1:
            print('empty walking candidates for person trip:%s'%[o_taz,d_taz,service_id])
        else:
            print('o_taz=%s, d_taz=%s and service_id=%s for person trip generated %s candidate pairs'%\
                  (o_taz,d_taz,service_id,len(candidate_trips)))

        if heading_limit: #filter those trips that are in the opposite direction at walking speed or faster!
            for i in range(len(candidate_trips)):
                if candidate_trips[i][4]<-1*walk_speed: break
            j = i
            if i<min_target: j = min(len(candidate_trips),min_target)
            some_trips = candidate_trips[:i]
            can_idx = np.arange(i,len(candidate_trips),1) #left over trips to get sampled
            can_pr  = can_idx-i+1.0
            can_pr  /= np.sum(can_pr)
            can_pr  = can_pr[::-1]
            if j-i>0:
                ran_idx = sorted(np.random.choice(can_idx,j-i,replace=False,p=can_pr))
                ran_trips = []
                for x in ran_idx: ran_trips += [candidate_trips[x]]
                some_trips += ran_trips
            candidate_trips = sorted(some_trips,key=lambda x: -1*x[4])

        print('%s candidates were filtered, %s candidates remain'%\
              (l_can-len(candidate_trips),len(candidate_trips)))
        final_candidates = {'service_id_%s'%service_id:candidate_trips}
    else: #no single driving trips (drive-tranist-walk or walk-transit->drive)
        if service_id is None: print('service_id was not found for date=%s'%dt[0])
        else:                  print('empty park+walk candidates for person trip:%s'%[o_taz,d_taz,service_id])
    return final_candidates

#miles per hours twords your destination using the minimum (of all stops)
#stop to stop distance and time bounded by destination time-destination walk time
def sdist_trip_trend(tid,tdx,sid,seqs,s_dist,time_max=None):
    if time_max is not None: x = np.where(seqs[tid][tdx:][:,1]<=time_max)[0]
    else: x = [len(seqs[tid][tdx])-1]
    a = tdx+(x[len(x)-1] if len(x)>0 else 0)
    if a>tdx:
        idx        = np.argmin([s_dist[s[0],sid] for s in seqs[tid][tdx:a]])
        min_idx    = seqs[tid][tdx:a][idx]
        delta_dist = s_dist[seqs[tid][tdx-1][0],sid]-s_dist[min_idx[0],sid]
        delta_time = min_idx[1]-seqs[tid][tdx-1][1]
    else:
        delta_dist = s_dist[seqs[tid][tdx-1][0],sid]-s_dist[seqs[tid][tdx][0],sid]
        delta_time = seqs[tid][tdx][1]-seqs[tid][tdx-1][1]
    return ((60*60)*delta_dist/delta_time if delta_time>0 else 0.0) #miles per hour twords destination

#cluster together trips that are similiar/small distance away from eachother
def ldist_cluster(l_dist,thresh=0.5):
    dist,C = np.zeros((l_dist.shape[0],l_dist.shape[1],1),dtype=float),{}
    for i in range(l_dist.shape[0]):
        for j in range(l_dist.shape[1]):
            dist[i][j] = 1.0-(l_dist[i][j][0]/l_dist[i][j][1]+l_dist[i][j][0]/l_dist[i][j][2])/2.0
            if i!=j and dist[i][j]<=thresh:
                if i in C: C[i] += [j]
                else:      C[i]  = [j]

#pick the best path out a bunch that must have a d_stop in it
def path_time_score(trip,c_time):
    if len(trip)>0:
        return sum([trip[i][4] for i in range(len(trip))])

def reduce_trans(trans,pw={-1:10*60,-2:20*60,-3:10*60}):
    T = {}
    for t in trans:
        T[t] = {}
        for k in trans[t]:
            if k[0] not in T[t]:
                T[t][k[0]] = k[1:4]
            elif pw[T[t][k[0]][1]]+T[t][k[0]][2]>pw[k[2]]+k[3]:
                T[t][k[0]] = k[1:4]
    T = {t:set([(k,)+tuple(T[t][k]) for k in T[t]]) for t in T}
    return T

def sub_seq_leg(seqs,tid,tdx,sid,pw):
    i,L = -1,[]
    for a in range(tdx,len(seqs[tid]),1):
        if seqs[tid][a][0]==sid: i = a; break
    if i>=0:
        L = [[tid,j,seqs[tid][j][0],seqs[tid][j][1],pw['offset'][0]] for j in range(tdx,i+1,1)]
    return L

#randomly sample the branches encountered in T
def sample_branches(T,seqs,s,p=1.0,max_time=None,s_dist=None,sid=None,rate=None):
    A,B,S,V,ps = {},set([]),{},{},[]
    for a in range(s[1]+1,len(seqs[s[0]]),1): #can't do a transfer from a transfer
        if (s[0],a) in T:
            for t in T[(s[0],a)]:
                if t[0:3] not in A or t[3]<A[t[0:3]][0]:
                    A[t[0:3]] = [t[3],a]
                    if t[0] in S: S[t[0]] = (S[t[0]] if S[t[0]]<t[1] else t[1])
                    else:         S[t[0]] = t[1]
    if max_time is not None and rate is not None and sid is not None and s_dist is not None:
        for tid in S:
            tdx = S[tid]
            if seqs[tid][tdx][1]<=max_time:
                mph = sdist_trip_trend(tid,tdx,sid,seqs,s_dist)
                if mph>rate: V[tid] = mph
        for k in A:
            if k[0] in V:
                B.add((A[k][1],k+(A[k][0],)))
        B = sorted(list(B),key=lambda x: V[x[1][0]])[::-1]
        ps = np.array([i+1 for i in range(len(B))][::-1])/(len(B)*(len(B)+1)/2.0)
        if len(B)>1:
            b_idx = sorted(np.random.choice(range(len(B)),max(1,int(len(B)*p)),p=ps,replace=False))
            B = sorted([B[b] for b in b_idx],key=lambda x: V[x[1][0]])[::-1]
    else:
        for k in A: B.add((A[k][1],k+(A[k][0],)))
        B = list(B)
        if len(B)>1:
            b_idx = sorted(np.random.choice(range(len(B)),max(1,int(len(B)*p)),replace=False))
            B = sorted([B[b] for b in b_idx])
    return B

#given L: a batch of paths, update K LCS dissimiliar paths
def update_paths(K,D,S,L,s_dist,k=5):
    tm,m = (L[len(L)-1][3]-L[0][3])+np.sum(L[:,4]),-1 #m truggers recalculations if best time, or max diss is reached
    if len(K)<k:
        K += [[tm,L]]
        m = len(K)-1
    else:
        if tm<K[0][0]:
            K = [[tm,L]]+K[:-1] #pop off the last in the sorted list
            m = 0
        else:
            C,sp = [],0.0
            for i in range(k):
                l = tu.lcs(L[:,2:],K[i][1][:,2:],s_dist)
                if l[1]>0 and l[2]>0: C += [1.0-2.0*((l[0]/l[1])*(l[0]/l[2]))/((l[0]/l[1])+(l[0]/l[2]))]
                sp += C[i]
            for i in range(1,len(S),1):
                if sp>S[i]: m = i
            if m>0: K = K[:m]+[[tm,L]]+K[m+1:]
    K = sorted(K,key=lambda x: x[0])
    if m>=0: #some change----------------------------------------------------------------------------------------
        D = {}
        for i,j in it.combinations(range(len(K)),2):
            if i not in D: D[i] = {}
            if j not in D: D[j] = {}
            l = tu.lcs(K[i][1][:,2:],K[j][1][:,2:],s_dist)
            if l[1]>0 and l[2]>0: D[i][j] = D[j][i] = 1.0-2.0*((l[0]/l[1])*(l[0]/l[2]))/((l[0]/l[1])+(l[0]/l[2]))
            else:                 D[i][j] = D[j][i] = 1.0
        for k in D: S[k] = sum([D[k][x] for x in D[k]])
    return K,D,S

#compute penalties by seconds and mode: w*secs
def penalty(secs,mode,pw={'coeff': {0:1.0,-1:1.1,  -2:1.5,  -3:1.1},
                          'offset':{0:0,  -1:10*60,-2:20*60,-3:20*60}}):
    return pw['coeff'][mode]*secs + pw['offset'][mode]

#can add more nuanced penalties: pw{0:lambda x: x
def RTS_FULL(C,T,F,seqs,pw={'coeff': {0:1.0,-1:2.9,  -2:2.4,  -3:2},
                            'offset':{0:0,  -1:20*60,-2:20*60,-3:0}},
             min_paths=5,max_trans=5,trans_p=[1.0,0.75,0.5,0.25,0.125],
             min_rate=-3.0,add_od=True,verbose=True):
    print('applying pw=%s'%pw)
    t_start = time.time()
    X,z = {i:{} for i in range(max_trans+1)},0
    for i in range(len(C)):
        s0,sid,max_time = (C[i][0],C[i][1]),C[i][2],C[i][3] #add the origin section here and when finished the destination
        origin,dest,init_time = C[i][-1][0],C[i][-1][2],C[i][-1][1]
        max_time -= dest[2] #correct for the walk time to get to the destinantion
        if s0[0:2] in F and sid in F[s0[0:2]]:
            L = sub_seq_leg(seqs,s0[0],s0[1],sid,pw)
            if add_od: L = [[origin[0],0,origin[1],init_time-origin[2],origin[2]]]+L+[[dest[1],0,dest[0],L[-1][3]+dest[2],dest[2]]]
            Y  = np.array(L,dtype=np.int32)
            v  = (Y[-1][3]-Y[0][3])+np.sum(Y[:,4])
            if v in X[0]: X[0][v][tuple(set(Y[:,2]))] = Y
            else:         X[0][v] = {tuple(set(Y[:,2])):Y}
        elif len(X[0])<min_paths and max_trans>0: #will get up to 1-transfer more than optimal
            z += 1
            for a1,s1 in sample_branches(T,seqs,s0,trans_p[0],max_time,s_dist,sid,min_rate):
                if s1[0:2] in F and sid in F[s1[0:2]]:
                    L  = sub_seq_leg(seqs,s0[0],s0[1],seqs[s0[0]][a1][0],pw)                #before transfer-1
                    L += [[s1[2], 0, seqs[s1[0]][s1[1]][0],seqs[s1[0]][s1[1]][1],penalty(seqs[s1[0]][s1[1]][1]-L[-1][3],s1[2],pw)]]  #transfer-1
                    L += sub_seq_leg(seqs,s1[0],s1[1],sid,pw)                               #before destination
                    if add_od: L = [[origin[0],0,origin[1],init_time-origin[2],origin[2]]]+L+[[dest[1],0,dest[0],L[-1][3]+dest[2],dest[2]]]
                    Y  = np.array(L,dtype=np.int32)                                         #np array
                    v  = (Y[-1][3]-Y[0][3])+np.sum(Y[:,4])
                    if v in X[1]: X[1][v][tuple(set(Y[:,2]))] = Y
                    else:         X[1][v] = {tuple(set(Y[:,2])):Y}
                elif len(X[1])<min_paths and max_trans>1:
                    z += 1
                    for a2,s2 in sample_branches(T,seqs,s1,trans_p[1],max_time,s_dist,sid,min_rate+1.0):
                        if s2[0:2] in F and sid in F[s2[0:2]]:
                            L  = sub_seq_leg(seqs,s0[0],s0[1],seqs[s0[0]][a1][0],pw)                #before transfer-1
                            L += [[s1[2],0,seqs[s1[0]][s1[1]][0],seqs[s1[0]][s1[1]][1],penalty(seqs[s1[0]][s1[1]][1]-L[-1][3],s1[2],pw)]]  #transfer-1
                            L += sub_seq_leg(seqs,s1[0],s1[1],seqs[s1[0]][a2][0],pw)                #before transfer-2
                            L += [[s2[2],0,seqs[s2[0]][s2[1]][0],seqs[s2[0]][s2[1]][1],penalty(seqs[s2[0]][s2[1]][1]-L[-1][3],s2[2],pw)]]  #transfer-2
                            L += sub_seq_leg(seqs,s2[0],s2[1],sid,pw)                               #before destination
                            if add_od: L = [[origin[0],0,origin[1],init_time-origin[2],origin[2]]]+L+[[dest[1],0,dest[0],L[-1][3]+dest[2],dest[2]]]
                            Y  = np.array(L,dtype=np.int32)                                         #np array
                            v  = (Y[-1][3]-Y[0][3])+np.sum(Y[:,4])
                            if v in X[2]: X[2][v][tuple(set(Y[:,2]))] = Y
                            else:         X[2][v] = {tuple(set(Y[:,2])):Y}
                        elif len(X[2])<min_paths and max_trans>2: #if len(X[1])<1
                            z += 1
                            for a3,s3 in sample_branches(T,seqs,s2,trans_p[2],max_time,s_dist,sid,min_rate+1.0):
                                if s3[0:2] in F and sid in F[s3[0:2]]:
                                    L  = sub_seq_leg(seqs,s0[0],s0[1],seqs[s0[0]][a1][0],pw)                #before transfer-1
                                    L += [[s1[2],0,seqs[s1[0]][s1[1]][0],seqs[s1[0]][s1[1]][1],penalty(seqs[s1[0]][s1[1]][1]-L[-1][3],s1[2],pw)]]  #transfer-1
                                    L += sub_seq_leg(seqs,s1[0],s1[1],seqs[s1[0]][a2][0],pw)                #before transfer-2
                                    L += [[s2[2],0,seqs[s2[0]][s2[1]][0],seqs[s2[0]][s2[1]][1],penalty(seqs[s2[0]][s2[1]][1]-L[-1][3],s2[2],pw)]]  #transfer-2
                                    L += sub_seq_leg(seqs,s2[0],s2[1],seqs[s2[0]][a3][0],pw)                #before transfer-3
                                    L += [[s3[2],0,seqs[s3[0]][s3[1]][0],seqs[s3[0]][s3[1]][1],penalty(seqs[s3[0]][s3[1]][1]-L[-1][3],s3[2],pw)]]  #transfer-3
                                    L += sub_seq_leg(seqs,s3[0],s3[1],sid,pw)                               #before destination
                                    if add_od: L = [[origin[0],0,origin[1],init_time-origin[2],origin[2]]]+L+[[dest[1],0,dest[0],L[-1][3]+dest[2],dest[2]]]
                                    Y  = np.array(L,dtype=np.int32)                                         #np array
                                    v  = (Y[-1][3]-Y[0][3])+np.sum(Y[:,4])
                                    if v in X[3]: X[3][v][tuple(set(Y[:,2]))] = Y
                                    else:         X[3][v] = {tuple(set(Y[:,2])):Y}
                                elif len(X[3])<min_paths and max_trans>3: #if len(X[2])<1
                                    z += 1
                                    for a4,s4 in sample_branches(T,seqs,s3,trans_p[3],max_time,s_dist,sid,min_rate+1.0):
                                        if s4[0:2] in F and sid in F[s4[0:2]]:
                                            L  = sub_seq_leg(seqs,s0[0],s0[1],seqs[s0[0]][a1][0],pw)                #before transfer-1
                                            L += [[s1[2],0,seqs[s1[0]][s1[1]][0],seqs[s1[0]][s1[1]][1],penalty(seqs[s1[0]][s1[1]][1]-L[-1][3],s1[2],pw)]]  #transfer-1
                                            L += sub_seq_leg(seqs,s1[0],s1[1],seqs[s1[0]][a2][0],pw)                #before transfer-2
                                            L += [[s2[2],0,seqs[s2[0]][s2[1]][0],seqs[s2[0]][s2[1]][1],penalty(seqs[s2[0]][s2[1]][1]-L[-1][3],s2[2],pw)]]  #transfer-2
                                            L += sub_seq_leg(seqs,s2[0],s2[1],seqs[s2[0]][a3][0],pw)                #before transfer-3
                                            L += [[s3[2],0,seqs[s3[0]][s3[1]][0],seqs[s3[0]][s3[1]][1],penalty(seqs[s3[0]][s3[1]][1]-L[-1][3],s3[2],pw)]]  #transfer-3
                                            L += sub_seq_leg(seqs,s3[0],s3[1],seqs[s3[0]][a4][0],pw)                #before transfer-4
                                            L += [[s4[2],0,seqs[s4[0]][s4[1]][0],seqs[s4[0]][s4[1]][1],penalty(seqs[s4[0]][s4[1]][1]-L[-1][3],s4[2],pw)]]  #transfer-4
                                            L += sub_seq_leg(seqs,s4[0],s4[1],sid,pw)                               #before destination
                                            if add_od: L = [[origin[0],0,origin[1],init_time-origin[2],origin[2]]]+L+[[dest[1],0,dest[0],L[-1][3]+dest[2],dest[2]]]
                                            Y  = np.array(L,dtype=np.int32)                                         #np array
                                            v  = (Y[-1][3]-Y[0][3])+np.sum(Y[:,4])
                                            if v in X[4]: X[4][v][tuple(Y[:,2])] = Y
                                            else:         X[4][v] = {tuple(Y[:,2]):Y}
                                        elif len(X[4])<min_paths and max_trans>4: #if len(X[2])<1
                                            z += 1
                                            for a5,s5 in sample_branches(T,seqs,s4,trans_p[4],max_time,s_dist,sid,min_rate+1.0):
                                                if s5[0:2] in F and sid in F[s5[0:2]]:
                                                    L  = sub_seq_leg(seqs,s0[0],s0[1],seqs[s0[0]][a1][0],pw)                #before transfer-1
                                                    L += [[s1[2],0,seqs[s1[0]][s1[1]][0],seqs[s1[0]][s1[1]][1],penalty(seqs[s1[0]][s1[1]][1]-L[-1][3],s1[2],pw)]]  #transfer-1
                                                    L += sub_seq_leg(seqs,s1[0],s1[1],seqs[s1[0]][a2][0],pw)                #before transfer-2
                                                    L += [[s2[2],0,seqs[s2[0]][s2[1]][0],seqs[s2[0]][s2[1]][1],penalty(seqs[s2[0]][s2[1]][1]-L[-1][3],s2[2],pw)]]  #transfer-2
                                                    L += sub_seq_leg(seqs,s2[0],s2[1],seqs[s2[0]][a3][0],pw)                #before transfer-3
                                                    L += [[s3[2],0,seqs[s3[0]][s3[1]][0],seqs[s3[0]][s3[1]][1],penalty(seqs[s3[0]][s3[1]][1]-L[-1][3],s3[2],pw)]]  #transfer-3
                                                    L += sub_seq_leg(seqs,s3[0],s3[1],seqs[s3[0]][a4][0],pw)                #before transfer-4
                                                    L += [[s4[2],0,seqs[s4[0]][s4[1]][0],seqs[s4[0]][s4[1]][1],penalty(seqs[s4[0]][s4[1]][1]-L[-1][3],s4[2],pw)]]  #transfer-4
                                                    L += sub_seq_leg(seqs,s4[0],s4[1],seqs[s4[0]][a5][0],pw)                #before transfer-5
                                                    L += [[s5[2],0,seqs[s5[0]][s5[1]][0],seqs[s5[0]][s5[1]][1],penalty(seqs[s5[0]][s5[1]][1]-L[-1][3],s5[2],pw)]]  #transfer-5
                                                    L += sub_seq_leg(seqs,s5[0],s5[1],sid,pw)                               #before destination
                                                    if add_od: L = [[origin[0],0,origin[1],init_time-origin[2],origin[2]]]+L+[[dest[1],0,dest[0],L[-1][3]+dest[2],dest[2]]]
                                                    Y  = np.array(L,dtype=np.int32)                                         #np array
                                                    v  = (Y[-1][3]-Y[0][3])+np.sum(Y[:,4])
                                                    if v in X[5]: X[5][v][tuple(Y[:,2])] = Y
                                                    else:         X[5][v] = {tuple(Y[:,2]):Y}
                                                z += 1
    t_stop = time.time()
    if verbose: print('tree searched %s edges with %s paths in %s sec'%(z,[len(X[x]) for x in X],round(t_stop-t_start)))
    return X

def multi_core_lcs(P,S,s_dist):
    R = []
    for p in P:
        t,i,j = p[0],p[1],p[2]
        l = tu.lcs(S[t][i][0][:,2:],S[t][j][0][:,2:],s_dist)
        R += [[t,i,j,l]]
    return R

def multi_core_jaccard(P,S):
    R = []
    for p in P:
        t,i,j = p[0],p[1],p[2]
        A,B = set(S[t][i][0][:,2]),set(S[t][j][0][:,2])
        l = [len(A.intersection(B)),len(A.union(B))]
        R += [[t,i,j,l]]
    return R

#get lowest cost
def get_path(X,person,trip,method='low-cost'):
    if method=='low-cost':
        path = [np.iinfo(np.int32).max,[]]
        for i in range(len(X[person][trip])):
            if len(X[person][trip][i])>0:
                ks = sorted(X[person][trip][i])
                xp = X[person][trip][i][ks[0]]
                if ks[0] < path[0]: path = [ks[0],xp[sorted(xp)[0]]]
    elif method=='high-cost':
        path = [np.iinfo(np.int32).min,[]]
        for i in range(len(X[person][trip])):
            if len(X[person][trip][i])>0:
                ks = sorted(X[person][trip][i])
                xp = X[person][trip][i][ks[-1]]
                if ks[-1] > path[0]: path = [ks[-1],xp[sorted(xp)[0]]]
    if method=='low-transfer':
        path = [None,[]]
        for i in range(len(X[person][trip])):
            if len(X[person][trip][i])>0:
                ks = sorted(X[person][trip][i])
                xp = X[person][trip][i][ks[0]]
                path = [ks[0],xp[sorted(xp)[0]]]
                break
    if method=='high-transfer':
        path = [None, []]
        for i in range(len(X[person][trip])-1,0,-1):
            if len(X[person][trip][i]) > 0:
                ks = sorted(X[person][trip][i])
                xp = X[person][trip][i][ks[0]]
                path = [ks[0],xp[sorted(xp)[0]]]
                break
    return path

result_list = []
def collect_results(result):
    result_list.append(result)

def get_seq_paths(out_path,C,max_trans=5,trans_p=[1.0,0.75,0.5,0.25,0.125],min_rate=-3.0):
    X = {'error':[]}
    for i in range(len(C)):
        try:
            si = C[i][2]
            seqs,trans = D[si]['seqs'],D[si]['trans'] #doesn't work for multiple service ids...
            T,F = reduce_trans(trans),{} #applies the penalties to select the faster option tid_a=>tid_b
            for (tid,tdx) in T:
                for l in T[(tid,tdx)]:
                    if (l[0],l[1]) not in F: #tid,tdx
                        F[(l[0],l[1])] = set(seqs[l[0]][l[1]:,0])
            R = RTS_FULL(C[i][3],T,F,seqs,max_trans=max_trans,trans_p=trans_p,min_rate=min_rate)
            Y = {'person':C[i][0],'trip':C[i][1],'service_id':C[i][2],'paths':R}
            with gzip.GzipFile(out_path+'person_%s.trip_%s.pickle.gz'%(C[i][0],C[i][1]), 'wb') as f: pickle.dump(Y,f)
        except Exception as E: X['error'] += [str(E)]
    return X

def k_dis_paths(X,s_dist,k=5,verbose=False):
    #unpack the cost keys to seq tuples
    S,K = {},{}
    for t in X:
        S[t] = []
        for v in X[t]:
            for s in X[t][v]: S[t] += [[X[t][v][s],v,0.0,0.0]]
        S[t] = sorted(S[t],key=lambda x: x[1])
        if len(S[t])>1:
            off = min([S[t][i][1] for i in range(len(S[t]))])
            dif = max([S[t][i][1] for i in range(len(S[t]))])-off
            if dif>0.0:
                for i in range(len(S[t])): S[t][i][2] = (S[t][i][1]-off)/dif
    for t in S:
        if verbose: print('t=%s'%t)
        SP,K[t],D = [],[],np.zeros((len(S[t]),len(S[t])),dtype=np.float32) #LCSWT harmonic mean distance from 0.0 to 1.0
        if len(S[t])>k: #k=1 is just lowest cost path
            x = 1
            for i in range(len(S[t])):         #calculate upper triangle
                if verbose and i%10==0: print('t=%s i=%s'%(t,i))
                for j in range(x,len(S[t]),1): #and copy to lower triangle
                    l = tu.lcs(S[t][i][0][:,2:],S[t][j][0][:,2:],s_dist)
                    D[i][j] = D[j][i] = 1.0-2.0*((l[0]/l[1])*(l[0]/l[2]))/((l[0]/l[1])+(l[0]/l[2]))
                x += 1
            if verbose: print('finished all pairs LCSWT for t=%s'%t)
            for i in range(len(S[t])): S[t][i][3] = np.sum(D[i])
            for i in range(1,k+1,1): #S_term(cost)+D_term(distance)
                S[t]  = sorted(S[t],key=lambda y: y[2]*(1.0-(i-1.0)/(k-1.0))+(1.0-y[3])*(i-1.0)/(k-1.0))
                K[t] += [S[t][0]]
                S[t]  = S[t][1:]
    return K
        #now we have the ldist matrix for the viable trips and the cost

#for each person trip run through all select_k_paths types
def cost_diss_full_analysis(S,k):
    return True

#returns a mix coeffcient starting at 1.0 and ending at 0.0 using k steps
def select_k_paths(S,k,type='xy'):
    P = copy.deepcopy(S)
    ks,KS = set([]),[]
    if type=='xy':
        ws = []
        for i in range(1,k+1,1): ws += [1.0-(i-1.0)/(k-1.0)]
    if type=='y_2':
        ws = []
        for i in range(k): ws += [i**2.0]
        ws = ws[::-1]
        a,b = np.min(ws),np.max(ws)
        ws -= a
        ws /= b-a
    if type=='x_2':
        ws = []
        for i in range(k): ws += [i**2.0]
        ws = ws[::-1]
        a,b = np.min(ws),np.max(ws)
        ws -= a
        ws /= b-a
        ws = [1.0-x for x in ws[::-1]]
    if type=='y_3':
        ws = []
        for i in range(k): ws += [i**3.0]
        ws = ws[::-1]
        a,b = np.min(ws),np.max(ws)
        ws -= a
        ws /= b-a
    if type=='x_3':
        ws = []
        for i in range(k): ws += [i**3.0]
        ws = ws[::-1]
        a,b = np.min(ws),np.max(ws)
        ws -= a
        ws /= b-a
        ws = [1.0-x for x in ws[::-1]]
    if type=='x_log_2':
        ws = [1.0]
        if k>1:
            for i in range(k-1):
                ws += [ws[-1]/2]
        a,b = np.min(ws),np.max(ws)
        ws -= a
        ws /= b-a
        ws = [1.0-x for x in ws[::-1]]
    if type=='y_log_2':
        ws = [1.0]
        if k>1:
            for i in range(k-1):
                ws += [ws[-1]/2]
        a,b = np.min(ws),np.max(ws)
        ws -= a
        ws /= b-a
    if type=='x_log_10':
        ws = [1.0]
        if k>1:
            for i in range(k-1):
                ws += [ws[-1]/10]
        a,b = np.min(ws),np.max(ws)
        ws -= a
        ws /= b-a
        ws = [1.0-x for x in ws[::-1]]
    if type=='y_log_10':
        ws = [1.0]
        if k>1:
            for i in range(k-1):
                ws += [ws[-1]/10]
        a,b = np.min(ws),np.max(ws)
        ws -= a
        ws /= b-a
    if type=='log':
        ws = []
        for i in range(k):
            ws += [np.log(k-i)/np.log(k)]
    for i in range(k_value): #S_term(cost)+D_term(distance)
        pw    = [ws[i]*y[2] + (1.0-ws[i])*y[3] for y in P] #apply the k=weighting to mix cost-diss
        s_idx = list(np.argsort(pw))                       #order the results acsending
        x = 0
        while s_idx[x] in ks: x += 1
        ks.add(s_idx[x])
        KS += [P[s_idx[x]]+[s_idx[x]]] #add the element from S[t], but keep track of its orginal place...
    return KS

def get_time_string(t):
    hours   = t//(60*60)
    minutes = (t%(60*60))//60
    seconds = (t%(60*60))%60
    return '%s:%s:%s'%(str(hours).zfill(2),str(minutes).zfill(2),str(seconds).zfill(2))

def abbreviated_path(path):
    ps = [list(x) for x in path[0:2]]
    for i in range(2,len(path)-2,1):
        if path[i][0]==-1 or path[i][0]==-2:
            ps += [list(path[i-1])]
            ps += [list(path[i])]
            ps += [list(path[i+1])]
            i  += 1
    ps += [list(x) for x in path[-2:]]
    ps = np.array(ps,dtype=np.int32)
    return ps

def human_short_path(X,person,trip,persons,trips,s_names,verbose=True):
    if verbose: print('detailing low cost human form for person=%s, trip=%s'%(person,trip))
    cost,path = get_path(X,person,trip)
    #path = [[tid,t_idx,sid,time,penalty]], tid=-1 is waitng, tid=-2 is walking
    #trips = [trip_id,trip_sign,route_id,service_id,direction_id,shape_id]
    H = []
    if len(path)>0:
        start = persons[person][trip][2]
        start = start.hour*60*60+start.minute*60+start.second
        start = ['walk',0,persons[person][trip][1],get_time_string(start),0]
        end   = persons[person][trip][5]
        end   = end.hour*60*60+end.minute*60+end.second
        end   = ['walk',0,persons[person][trip][4],get_time_string(end),0]
        H    += [start]
        idx_s = {s_idx[s]:s for s in s_idx}
        for i in range(len(path)): # trip headsign, t_idx
            tid = path[i][0]
            if tid==-1:   tname = 'wait'
            elif tid==-2: tname = 'walk'
            else:         tname = trips[path[i][0]][1]
            if idx_s[path[i][2]] in s_names: sname = s_names[idx_s[path[i][2]]]
            else:                            sname = 'NA'
            element = [tname,path[i][1],sname,get_time_string(path[i][3]),path[i][4]]
            H += [element]
        H += [end]
    return H

#convert a path to human readible path
def path_to_human(person,trip,k,path,s_names):
    H = []
    for row in path:
        if row[0]==-1:   tid = 'waiting'
        elif row[0]==-2: tid = 'walking'
        elif row[0]==-3: tid = 'driving'
        else:            tid = trips[row[0]][1]
        stop_name = s_names[stops[row[2]][0]]
        h,m,s  = row[3]//(60*60),(row[3]%(60*60))//60,((row[3]%(60*60))%60)%60
        s_time = '%s:%s:%s'%(str(h).zfill(2),str(m).zfill(2),str(s).zfill(2))
        h,m,s  = row[4]//(60*60),(row[4]%(60*60))//60,((row[4]%(60*60))%60)%60
        p_time = '%s:%s:%s'%(str(h).zfill(2),str(m).zfill(2),str(s).zfill(2))
        H += [[person,trip,k,tid,row[1],stop_name,s_time,p_time]]
    return H

def get_human_paths(X,persons,s_names,abbreviate=False): #k=0 => best path
    H = {}
    for i in X:
        H[i] = {}
        for j in X[i]: #all paths are index 0, k path are index 1
            H[i][j] = []
            person_k_paths,k_paths = X[i][j],[]
            for t in sorted(person_k_paths): #transfer number
                if len(person_k_paths[t])>0: #0 is the path, 1 is the cost, 2 is the normalized cost => 0.0 is the lowest, 3 is the sum of pairs
                    k_paths = person_k_paths[t]
                    break
            K = [x[0] for x in k_paths]
            for k in range(len(K)):
                if abbreviate: K[k] = abbreviated_path(K[k])
                inner_path = path_to_human(i,j,k,K[k],s_names)
                origin = inner_path[0]
                dest   = inner_path[-1]
                start  = persons[i][j][1]
                end    = persons[i][j][4]
                origin[3] = 'origin-'+origin[3]
                origin[5] = start
                dest[3]   = 'destination-'+dest[3]
                dest[5]   = end
                H[i][j] += inner_path
    return H

def write_human_paths_tsv(path,H):
    s = '\t'.join(['person','trip','k','trip_sign','t_idx','stop_name','stop_time','penalty_time'])+'\n'
    for i in sorted(H):
        for j in sorted(H[i]):
            for n in range(len(H[i][j])):
                s += '\t'.join([str(i),str(j),str(H[i][j][n][2]),'"%s"'%H[i][j][n][3],str(H[i][j][n][4]),
                                '"%s"'%H[i][j][n][5],H[i][j][n][6],str(H[i][j][n][7])])+'\n'
    with open(path,'w') as f:
        f.write(s)
        return True
    return False

if __name__ == '__main__':
    des = """K-cost-dissimiliar Paths Random Tree Trip Search (KCD-RTS), Copyright (C) 2020-2021 Timothy James Becker
    Random importance sampling of paths with Longest Common Subsequence With Transfer (LCSWT) or Jaccard (J) Metric"""
    parser = argparse.ArgumentParser(description=des,formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--net_dir',type=str,help='network directory base path\t[None]')
    parser.add_argument('--demand_path',type=str,help='demand file path\t[None]')
    parser.add_argument('--out_dir',type=str,help='output directory\t[None]')
    #filtering and limiting of search-----------------------------------------------------------------------------------------
    parser.add_argument('--time_range',type=str,help='search time range in hours\t[0:00-32:00]')
    parser.add_argument('--buff_time',type=float,help='maximum time in minutes to wait or walk at any inner leg of trip\t[10.0]')
    parser.add_argument('--od_walk_time',type=float,help='maximum time in minutes to walk at the origin or destination\t[10.0]')
    parser.add_argument('--max_time',type=float,help='maximum time in minutes for the total trip\t[90.0]')
    parser.add_argument('--max_trans',type=int,help='maximum number of transfers to search for (can go up to 5)\t[3]')
    parser.add_argument('--trans_prop',type=str,help='sample proportion for transfer paths\t[1.0,0.5,0.25,0.125]')
    parser.add_argument('--heading_limit',type=int,help='cutoff value in mph for initial od access/egress candidates\t[-3.0]')
    # initial conditions for search-------------------------------------------------------------------------------------------
    parser.add_argument('--walk_speed',type=float,help='average walking speed in mph\t[3.0]')
    parser.add_argument('--bus_speed',type=float,help='average bus speed in mph for calculate heading vector estimate\t[15.0]')
    parser.add_argument('--drive_speed',type=float,help='average driving speed in mph to get to park and ride or home\t[30.0]')
    parser.add_argument('--k_lim',type=int,help='the maximum number of discrete paths per transfer before sampling occurs in the k-path calculation\t[500]')
    parser.add_argument('--k_value',type=int,help='the maximum number of paths to calculate per transfer\t[5]')
    parser.add_argument('--jaccard',action='store_true',help='use the simple jaccard stop set metric instead of LCSWT\t[False]')
    parser.add_argument('--abbreviate',action='store_true',help='abbreviate the final paths so only include tranfer points\t[False]')
    parser.add_argument('--cpus',type=int,help='number of cpus to use\t[1]')
    args = parser.parse_args()

    if args.net_dir is not None:
        n_base = args.net_dir
    else: raise IOError
    if args.demand_path is not None:
        d_path = args.demand_path
        d_base = '/'.join(d_path.rsplit('/')[:-2])+'/'
    else: raise IOError
    if args.out_dir is not None:
        out_dir = args.out_dir
    else: raise IOError
    if args.time_range is not None:
        search_time = args.time_range.rsplit(',')
    else:
        search_time = ['0:00','32:00']
    if args.buff_time is not None:
        buff_time = args.buff_time
    else:
        buff_time = 10.0
    if args.od_walk_time is not None:
        od_walk_time = args.od_walk_time
    else:
        od_walk_time = 10.0
    if args.max_time is not None:
        max_time = args.buff_time
    else:
        max_time = 90.0
    if args.walk_speed is not None:
        walk_speed = args.walk_speed
    else:
        walk_speed = 3.0
    if args.bus_speed is not None:
        bus_speed = args.bus_speed
    else:
        bus_speed = 15.0
    if args.drive_speed is not None:
        drive_speed = args.drive_speed
    else:
        drive_speed = 30.0
    if args.max_trans is not None:
        max_trans = args.max_trans
    else:
        max_trans = 3
    if args.trans_prop is not None:
        trans_prop = [float(x) for x in args.tras_prop.rsplit(',')]
    else:
        trans_prop = [1.0,0.5,0.25,0.125,0.0625] #divid by 2 per transfer leave in the search tree
    if args.heading_limit is not None:
        heading_limit = args.heading_limit
    else:
        heading_limit = -3.0
    if args.k_lim is not None:
        k_lim = args.k_lim
    else:
        k_lim = 500
    if args.k_value is not None:
        k_value = args.k_value
    else:
        k_value = 5
    if args.cpus is not None:
        cpus = args.cpus
    else:
        cpus = 1
    if args.jaccard: jaccard = True
    else:            jaccard = False

    D = load_network_data(n_base,walk=((walk_speed/60.0)*buff_time),search_time=search_time,target_cpus=cpus) #can run preproccess_network.py

    persons = read_person_trip_list(d_path)
    stops,s_idx,s_names,s_dist,w_dist,p_dist = D['stops'],D['stop_idx'],D['s_names'],D['s_dist'],D['w_dist'],D['p_dist']
    trips,trip_idx,v_dist,calendar    = D['trips'],D['trip_idx'],D['v_dist'],D['calendar']
    service_ids = get_processed_service_ids(D)

    #[1] generate candidates and then run [2] || RST
    if len(glob.glob(out_dir+'/person*trip*.pickle.gz'))<=0:
        #person,i= 144 trip,j=0
        X,P,C = {},[],{}
        for i in sorted(persons):
            C[i] = {}
            for j in range(len(persons[i])):
                C[i][j] = can = start_od_search(persons[i][j],w_dist,p_dist,s_dist,v_dist,
                                                buff_time=buff_time,max_time=max_time,walk_speed=walk_speed,
                                                bus_speed=bus_speed,drive_speed=drive_speed)
                if can is not None and len(can[sorted(can)[0]])>0:
                    print('person=%s,trip=%s was valid on %s, will run RST...'%(i,j,persons[i][j][2].strftime('%m/%d/%Y')))
                    si = list(can.keys())[0]
                    candidates = can[si]
                    K,drv,wlk,wtg = [],0,0,0
                    for c in candidates: #leave the trip direction filterin to the main algorithm
                        K += [(c[2],c[3],c[5][0],c[6],c[4],[c[0],c[1],c[5]])]  #
                        if c[0][0]==-3 or c[5][1]==-3:   drv += 1
                        elif c[0][0]==-2 or c[5][1]==-2: wlk += 1
                        elif c[0][0]==-1 or c[5][1]==-1: wtg += 1
                    if drv>0:
                        print('<<<<<<<<<<  %s driving candidates were found for person=%s,trip=%s  >>>>>>>>>>'%(drv,i,j))
                        print('<<<<<<<<<<  %s walking candidates were found for person=%s,trip=%s  >>>>>>>>>>'%(wlk,i,j))
                        print('<<<<<<<<<<  %s waiting candidates were found for person=%s,trip=%s  >>>>>>>>>>'%(wtg,i,j))
                    P += [[i,j,si,sorted(K,key=lambda x: x[4])[::-1]]]
                else:
                    print('person=%s,trip=%s was not valid on %s...'%(i,j,persons[i][j][2].strftime('%m/%d/%Y')))

        #partitioning by sorted candidate number (pack by largest first and interleve over cpus...)
        cpus = min(cpus,mp.cpu_count())
        partitions = [[] for x in range(cpus)]
        P = sorted(P,key=lambda x: len(x[3]))[::-1] #sort by the number of candidates to search
        for i in range(len(P)):
            partitions[i%cpus] += [P[i]]
        if not os.path.exists(out_dir): os.mkdir(out_dir)
        print('starting || cython random tree search (RTS) computation')
        t_start = time.time()
        p1 = mp.Pool(processes=cpus)
        for i in range(len(partitions)):
            p1.apply_async(get_seq_paths,
                           args=(out_dir,partitions[i],max_trans,trans_prop,heading_limit),
                           callback=collect_results)
        p1.close()
        p1.join()
        t_stop = time.time()
        print('completed in %s minutes'%round((t_stop-t_start)/60,2))
        result_list = []
    metric = ('jaccard' if jaccard else 'lcswt')
    #[3] now we run for each person/trip the LCSWT in || to maximize cpu utilization
    if len(glob.glob(out_dir + '/%s_k%s*person*trip*.pickle.gz'%(metric,k_value)))<=0:
        P,verbose = [],False
        for path in sorted(glob.glob(out_dir+'/person*trip*.pickle.gz'))[0:1]:
            base_dir = '/'.join(path.rsplit('/')[:-1])+'/'
            person   = int(path.rsplit('/')[-1].rsplit('.')[0].rsplit('person_')[-1])
            trip     = int(path.rsplit('/')[-1].rsplit('.')[1].rsplit('trip_')[-1])
            print('working on person=%s, trip=%s'%(person,trip))
            with gzip.GzipFile(path,'rb') as fr:
                D = pickle.load(fr)
                person     = D['person']
                trip       = D['trip']
                service_id = D['service_id']
                X          = D['paths']
            print('loaded prexisting RST search data for person=%s, trip=%s and service_id=%s'%(person,trip,service_id))
            S = {}
            for t in X:
                S[t] = []
                for v in X[t]:
                    for s in X[t][v]: S[t] += [[X[t][v][s],v,0.0,0.0]]
                S[t] = sorted(S[t],key=lambda x: x[1])
                if len(S[t])>1:
                    off = min([S[t][i][1] for i in range(len(S[t]))])
                    dif = max([S[t][i][1] for i in range(len(S[t]))])-off
                    if dif>0.0:
                        for i in range(len(S[t])): S[t][i][2] = (S[t][i][1]-off)/dif
            #use klim to downsample paths proportional to cost-------------------------------------
            for t in S:
                SL = []
                if len(S[t])>k_lim:
                    print('using k_lim=%s on trans=%s to reduce large numbers of high cost paths'%(k_lim,t))
                    s_prop = np.arange(1.0,len(S[t]),1)[::-1]
                    s_prop /= np.sum(s_prop)
                    s_idx = [0]+sorted(list(np.random.choice(range(1,len(S[t])),k_lim-1,replace=False,p=s_prop)))
                    for idx in s_idx: SL += [S[t][idx]]
                    S[t] = SL
            L = []
            for t in S:
                x = 1
                for i in range(len(S[t])):
                    for j in range(x,len(S[t]),1):
                        L += [[t,i,j,len(S[t][i][0][:,2:-1])*len(S[t][j][0][:,2:-1])]]
                    x += 1
            L = sorted(L,key=lambda x: x[3])[::-1] # sort by highest edit distance computation cost

            cpus = min(cpus,mp.cpu_count())
            partitions = [[] for x in range(cpus)]
            for i in range(len(L)): partitions[i%cpus] += [L[i]]

            result_list = []
            if jaccard:
                print('computing %s jaccard pairs using %s cpu partitions for %s'%(len(L),cpus,path))
                t_start = time.time()
                p2 = mp.Pool(processes=cpus)
                for i in range(len(partitions)):
                    p2.apply_async(multi_core_jaccard,
                                   args=(partitions[i],S),
                                   callback=collect_results)
                p2.close()
                p2.join()
                t_stop = time.time()
                print('|| jaccard for %s completed in %s minutes'%(path,round((t_stop-t_start)/60,2)))
                print('performance: %s jaccard pairs/sec'%(int(len(L)/(t_stop-t_start))))
            else:
                print('computing %s LCSWT pairs using %s cpu partitions for %s'%(len(L),cpus,path))
                t_start = time.time()
                p2 = mp.Pool(processes=cpus)
                for i in range(len(partitions)):
                    p2.apply_async(multi_core_lcs,
                                   args=(partitions[i],S,s_dist),
                                   callback=collect_results)
                p2.close()
                p2.join()
                t_stop = time.time()
                print('|| LCSWT for %s completed in %s minutes'%(path,round((t_stop-t_start)/60,2)))
                print('performance: %s LCSWT pairs/sec'%(int(len(L)/(t_stop-t_start))))
            #unpack the per cpu result list---------------------------------------
            L = {}
            for result in result_list:
                for row in result:
                    t,i,j,l = row
                    if t in L:
                        if i in L[t]: L[t][i][j] = l
                        else:         L[t][i] = {j:l}
                    else:             L[t] = {i:{j:l}}
            #----------------------------------------------------------------------
            K = {}
            for t in S:
                SP,K[t],D = [],[],np.zeros((len(S[t]),len(S[t])),dtype=np.float32) #LCSWT harmonic mean distance from 0.0 to 1.0
                if len(S[t])>k_value: #k=1 is just lowest cost path
                    if jaccard:
                        #-------------------------------------------------------------------------------------
                        x = 1
                        for i in range(len(S[t])):         #calculate upper triangle
                            for j in range(x,len(S[t]),1): #and copy to lower triangle
                                D[i][j] = D[j][i] = 1.0-L[t][i][j][0]/L[t][i][j][1]
                            x += 1
                        #--------------------------------------------------------------------------------------
                    else:
                        #-------------------------------------------------------------------------------------
                        x = 1
                        for i in range(len(S[t])):         #calculate upper triangle
                            for j in range(x,len(S[t]),1): #and copy to lower triangle
                                D[i][j] = D[j][i] = 1.0-2.0*L[t][i][j][0]/(L[t][i][j][1]+L[t][i][j][2])
                            x += 1
                        #--------------------------------------------------------------------------------------
                    for i in range(len(S[t])): S[t][i][3] = np.sum(D[i])/len(S[t])
                    K[t] = select_k_paths(S[t],k_value,type='exp')

                    embedding = MDS(n_components=2,n_jobs=cpus,dissimilarity='precomputed')
                    d_trans   = embedding.fit_transform(D)
                    plt.tick_params(left=False,bottom=False,labelleft=False,labelbottom=False)
                    plt.scatter(d_trans[:,0],d_trans[:,1],color=(0.0,0.0,0.0,0.1),label='all')
                    for k in range(k_value):
                        plt.scatter(d_trans[K[t][k][-1],0],d_trans[K[t][k][-1],1],
                                    s=100,marker='D',color=(1.0-0.5*k/k_value,k/k_value,0.0),
                                    zorder=3.0+2*(1.0-k/k_value),alpha=0.8,label='k_%s'%k)
                    plt.title('person=%s,trip=%s,trans=%s'%(person,trip,t))
                    plt.legend()
                    plt.show()

                    d_plot = np.array([x[2:4] for x in S[t]])
                    plt.scatter(d_plot[:,0],d_plot[:,1],color=(0.0,0.0,0.0,0.1),label='all')
                    for k in range(k_value):
                        plt.scatter(d_plot[K[t][k][-1],0],d_plot[K[t][k][-1],1],
                                    s=100,marker='D',color=(1.0-0.5*k/k_value,k/k_value,0.0),
                                    zorder=3.0+2*(1.0-k/k_value),alpha=0.8,label='k_%s'%k)
                    plt.title('person=%s,trip=%s,trans=%s'%(person,trip,t))
                    plt.xlabel('low to high cost')
                    plt.ylabel('low to high simularity')
                    plt.legend()
                    plt.show()

            with gzip.GzipFile(out_dir+'/%s_k%s_person%s_trip%s.pickle.gz'%(metric,k_value,person,trip),'wb') as fk:
                pickle.dump({'K':K,'D':D},fk)
    X = {}
    for k_path in sorted(glob.glob(out_dir+'/%s_k%s*person*trip*.pickle.gz'%(metric,k_value))):
        with gzip.GzipFile(k_path,'rb') as fk:
            metric_check = False
            try:
                metric_check = metric==k_path.rsplit('/')[-1].rsplit('_')[0]
            except: pass
            k_check = False
            try:
                k_check = k_value==int(k_path.rsplit('/')[-1].rsplit('k')[1].rsplit('_')[0])
            except: pass
            person = -1 #not in a set of enumerations since they start at 0
            try:
                person = int(k_path.rsplit('/')[-1].rsplit('_')[2].rsplit('person')[1])
            except: pass
            trip = -1 #not in a set of enumerations since they start at 0
            try:
                trip = int(k_path.rsplit('/')[-1].rsplit('.')[0].rsplit('trip')[1])
            except: pass
            if metric_check and person in persons and trip in range(len(persons[person])):
                print('located final k-dis data for metric=%s, person=%s, trip=%s'%(metric,person,trip))
                if person not in X: X[person] = {}
                if trip not in X[person]: X[person][trip] = {}
                X[person][trip] = pickle.load(fk)
    H = get_human_paths(X,persons,s_names,abbreviate=args.abbreviate)
    if args.abbreviate:
        print('converted all k-paths, writing file to disk:%s'%(out_dir+'/%s_k%s_human_results_abbr.tsv'%(metric,k_value)))
        write_human_paths_tsv(out_dir+'/%s_k%s_human_results_abbr.tsv'%(metric,k_value),H)
    else:
        print('converted all k-paths, writing file to disk:%s'%(out_dir+'/%s_k%s_human_results.tsv'%(metric,k_value)))
        write_human_paths_tsv(out_dir+'/%s_k%s_human_results.tsv'%(metric,k_value),H)
