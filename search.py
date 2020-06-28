import os
import sys
import time
import datetime
import glob
import gzip
import pickle
import subprocess
import numpy as np
import itertools as it
import multiprocessing as mp
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
        raw = [row.replace('\r','').replace('\n','') for row in f.readlines()]
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
def load_network_data(n_base,walk=0.5,search_date=None,search_time=[0,115200]):
    if os.path.exists(n_base+'/network.pickle.gz'):
        with gzip.GzipFile(glob.glob(n_base+'/network*pickle.gz')[0],'rb') as f:
            D = pickle.load(f)
    else:
        print('preprocessing the network..')
        command = ['python3','preprocess_network.py','--in_path',n_base,
                   '--out_dir',n_base,'--walk %s'%walk,
                   '--time %s,%s'%(search_time[0],search_time[1])]
        if search_date is not None:  command += ['--date',search_date]
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
def start_od_search(person_trip,w_dist,s_dist,v_dist,buff_time=10,max_time=90,walk_speed=3,bus_speed=12):
    w_secs = (60.0*60.0)/walk_speed #seconds/mi
    dt = get_datetime(person_trip)
    search_time = get_search_time(dt[1][0],buff_time=buff_time*60)
    o_taz,d_taz = person_trip[0],person_trip[3]
    service_id  = ru.get_service_id(calendar,dt[0])
    if o_taz in w_dist and d_taz in w_dist and service_id in service_ids:
        print('o_taz=%s, d_taz=%s and service_id=%s were found for person trip'%(o_taz,d_taz,service_id))
        sid = 'service_id_%s'%service_id
        seqs,graph,l_dist,l_idx = D[sid]['seqs'],D[sid]['graph'],D[sid]['l_dist'],D[sid]['l_idx']

        #enumerate the posible o,d stop pairs and rank by minimum wdist[otaz][o]+sdist[o,d]+wdist[dtaz][d]
        wods = {'o':{},'d':{}}
        for o in w_dist[o_taz]:
            if o in seqs: wods['o'][o] = w_dist[o_taz][o]
        for d in w_dist[d_taz]:
            if d in seqs: wods['d'][d] = w_dist[d_taz][d]
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
        candidate_trips = []
        for c in od_search:
            o,d,e,time_max = c[0][0],c[0][1],c[1],c[2] #origin stop, destination stop and distance estimate
            owt = int(round(wods['o'][o]*w_secs+0.5)) #walking time to get the stop in seconds
            s_time = [search_time[0]+owt,search_time[1]+owt]
            for i in range(len(graph[o]['out'])):
                if graph[o]['out'][i][1]>=s_time[0] and graph[o]['out'][i][1]<=s_time[1]:
                    tid = graph[o]['out'][i][2:]
                    candidate_trip = [(-1,o),graph[o]['out'][i][1]]+list(tid)
                    candidate_trips  += [candidate_trip+[sdist_trip_trend(tid[0],tid[1],d,seqs,s_dist,time_max),(d,-1),time_max]]
        candidate_trips = sorted(candidate_trips,key=lambda x: (x[1],-1*x[4],x[6]))
        return {'service_id_%s'%service_id:candidate_trips}
    else:
        if service_id is None: print('service_id was not found for date=%s'%dt[0])
        else:                  print('empty candidates for person trip:%s'%[o_taz,d_taz,service_id])
        return None

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

#V is a valid trip buffer that will collect the valid possible trips to score and expand uppon
#c_sid,c_time,c_tid,tdx is the current stop and current time on the currect trip with index tdx
#d_stops d_time are the destination stops and max time, any of which are a goal => stop condidtion
#stops has each stop ids NN <= walk buffer=0.5=>10 minutes or 600 seconds
#seqs is the individual trig sequences that are used for building the trips
#graph is the time-based stop graph that is used for calculating waiting and walking transfers
#buff_time is used to calculate waiting tranfers and 10min=600 seconds
#L is the legs at that point in the search, which works like a stack: [tid,tdx,sid,stime,penalty]
def DFS(c_tid,c_tdx,d_stops,d_time,stops,seqs,graph,s_dist,l_dist,
        trans=3,buff_time=10,walk_speed=3,pw=[0,10,20],verbose=False):
    c_stop,c_time,out_sdx,in_sdx = seqs[c_tid][c_tdx-1] #will be graph indexes for in and out links
    D,B,scores = [],[],[] #L is for prepending links, F is for managing returning branches that are collected
    if seqs[c_tid][c_tdx][1]>=d_time or trans<=0: #this path has failed=> exeeded time or number of transfers
        D = [[c_tid,c_tdx,seqs[c_tid][c_tdx][0],seqs[c_tid][c_tdx][1],(seqs[c_tid][c_tdx][1]-c_time)+pw[0]*60]]
        return D
    else: #search using [(1)] direct [(2)] stoping [(3)] walking, followed by optimization [(4)]
        if seqs[c_tid][c_tdx][0] in d_stops:
            print('------- found stop=%s -------'%seqs[c_tid][c_tdx][0])
            D =  [[c_tid,c_tdx,seqs[c_tid][c_tdx][0],seqs[c_tid][c_tdx][1],(seqs[c_tid][c_tdx][1]-c_time)+pw[0]*60]]
            return D
        elif c_tdx<len(seqs[c_tid])-1: #if c_tdx==len(seqs[c_tid] then it is the last stop...
            D  = [[c_tid,c_tdx,seqs[c_tid][c_tdx][0],seqs[c_tid][c_tdx][1],(seqs[c_tid][c_tdx][1]-c_time)+pw[0]*60]]
            B += [D+DFS(c_tid,c_tdx+1,d_stops,d_time,stops,seqs,graph,s_dist,l_dist,trans)]#directs => depth first
        G,time_a,time_b = graph[c_stop]['out'],c_time,c_time+buff_time*60
        gidx = ([] if len(G)<1 else list(np.where(np.logical_and(G[:,1]>=time_a,G[:,1]<time_b))[0]))
        for i in gidx:
            ws = G[i] #wait at the stop = ws
            if ws[2]!=c_tid and ws[3]+1<len(seqs[ws[2]])-1 and seqs[ws[2]][ws[3]][1]<d_time: #link end time-------------
                D  = [[-1,0,c_stop,seqs[ws[2]][ws[3]][1],(seqs[ws[2]][ws[3]][1]-c_time)+pw[1]*60]] #waiting time + pw[1]
                B += [D+DFS(ws[2],ws[3]+1,d_stops,d_time,stops,seqs,graph,s_dist,l_dist,trans-1)] #recurse waiting
        w_secs = int(round((60*60)/walk_speed))
        for j in range(len(stops[c_stop][2])):
            wt_stop = stops[c_stop][2][j][0]                        #found a walking stop
            time_a  = int(c_time+stops[c_stop][2][j][1]*w_secs+0.5) #time once you get to the stop
            time_b  = int(min(d_time,time_a+buff_time*60))          #waiting time after you get to the stop
            if time_a<d_time and wt_stop in graph: #past max time from the walk time period
                G     = graph[wt_stop]['out']
                gidx = ([] if len(G)<1 else list(np.where(np.logical_and(G[:,1]>=time_a,G[:,1]<time_b))[0]))
                for i in gidx:
                    wt = G[i] #don't want people getting on the same trip after walking...
                    if wt[2]!=c_tid and wt[3]+1<len(seqs[wt[2]]) and seqs[wt[2]][wt[3]][1]<d_time:
                        if wt[2]!=c_tid and seqs[wt[2]][wt[3]][1]<d_time: #trip end times are within the boundry--------
                            D  = [[-2,0,wt_stop,time_a,(time_a-c_time)+pw[2]*60]] # walking leg
                            D += [[-1,0,wt_stop,wt[1],(wt[1]-time_a)+pw[1]*60]]   # waiting leg
                            B += [D+DFS(wt[2],wt[3]+1,d_stops,d_time,stops,seqs,graph,s_dist,l_dist,trans-1)]
    if len(B)>0: #[(4)] optimize/filter the returned paths (some were pruned via time/transfer limits:::::::::::::::::::
        for i in range(len(B)):
            if len(B[i])>0 and B[i][len(B[i])-1][2] in d_stops: #need to have a trip that ends at one of the d_stops
                scores += [[sum([B[i][j][4] for j in range(len(B[i]))]),i]]
        if len(scores)>0: #found a valid trip
            return B[sorted(scores)[0][1]]
        else: return []
    else: return []

#pick the best path out a bunch that must have a d_stop in it
def path_time_score(trip,c_time):
    if len(trip)>0:
        return sum([trip[i][4] for i in range(len(trip))])

def reduce_trans(trans,pw={-1:10*60,-2:20*60}):
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
        L = [[tid,j,seqs[tid][j][0],seqs[tid][j][1],pw[0]] for j in range(tdx,i+1,1)]
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

# random tree search algorithm: RTS for K dissimiliar short paths (RKDSP)
# tid,tdx is the trip at a stop, sid is the destination
# T+F,seqs,pw are the tree,fwd_stops,seq and penalties
def RTS_KD(C,T,F,seqs,pw={0:0,-1:10*60,-2:20*60},t_max=3,k_dis=5,t_p=[1.0,0.5,0.25]):
    start = time.time()
    K,D,S = {i:[] for i in range(t_max+1)},{i:{} for i in range(t_max+1)},{i:{} for i in range(t_max+1)}
    for c in C:
        s0,sid = (c[2],c[3]),c[5][0]
        if sid in F[s0[0:2]]:
            L = sub_seq_leg(seqs,s0[0],s0[1],sid,pw)
            L  = np.array(L,dtype=np.int32)
            K[0],D[0],S[0] = update_paths(K[0],D[0],S[0],L,s_dist,k=k_dis)
        elif t_max>0: #gather transfer branches from tid,tdx
            for a1,s1 in sample_branches(T,seqs,s0,p=t_p[0]): #a1 is the last stop before transfer-1
                if sid in F[s1[0:2]]:
                    L  = sub_seq_leg(seqs,s0[0],s0[1],seqs[s0[0]][a1][0],pw)                #before transfer-1
                    L += [[s1[2],0,seqs[s1[0]][s1[1]][0],seqs[s1[0]][s1[1]][1],pw[s1[2]]]]  #transfer-1
                    L += sub_seq_leg(seqs,s1[0],s1[1],sid,pw)                               #before destination
                    L  = np.array(L,dtype=np.int32)                                         #np array
                    K[1],D[1],S[1] = update_paths(K[1],D[1],S[1],L,s_dist,k=k_dis)   #filter these paths
                elif t_max>1:
                    for a2,s2 in sample_branches(T,seqs,s1,p=t_p[1]): #a2 is the last stop before transfer-2
                        if sid in F[s2[0:2]]:
                            L  = sub_seq_leg(seqs,s0[0],s0[1],seqs[s0[0]][a1][0],pw)                #before transfer-1
                            L += [[s1[2],0,seqs[s1[0]][s1[1]][0],seqs[s1[0]][s1[1]][1],pw[s1[2]]]]  #transfer-1
                            L += sub_seq_leg(seqs,s1[0],s1[1],seqs[s1[0]][a2][0],pw)                #before transfer-2
                            L += [[s2[2],0,seqs[s2[0]][s2[1]][0],seqs[s2[0]][s2[1]][1],pw[s2[2]]]]  #transfer-2
                            L += sub_seq_leg(seqs,s2[0],s2[1],sid,pw)                               #before destination
                            L  = np.array(L,dtype=np.int32)                                         #np array
                            K[2],D[2],S[2] = update_paths(K[2],D[2],S[2],L,s_dist,k=k_dis)   #filter these paths
                        elif t_max>2:
                            for a3,s3 in sample_branches(T,seqs,s2,p=t_p[2]):
                                if sid in F[s3[0:2]]:
                                    L  = sub_seq_leg(seqs,s0[0],s0[1],seqs[s0[0]][a1][0],pw)                #before transfer-1
                                    L += [[s1[2],0,seqs[s1[0]][s1[1]][0],seqs[s1[0]][s1[1]][1],pw[s1[2]]]]  #transfer-1
                                    L += sub_seq_leg(seqs,s1[0],s1[1],seqs[s1[0]][a2][0],pw)                #before transfer-2
                                    L += [[s2[2],0,seqs[s2[0]][s2[1]][0],seqs[s2[0]][s2[1]][1],pw[s2[2]]]]  #transfer-2
                                    L += sub_seq_leg(seqs,s2[0],s2[1],seqs[s2[0]][a3][0],pw)                #before transfer-3
                                    L += [[s3[2],0,seqs[s3[0]][s3[1]][0],seqs[s3[0]][s3[1]][1],pw[s3[2]]]]  #transfer-3
                                    L += sub_seq_leg(seqs,s3[0],s3[1],sid,pw)                               #before destination
                                    L  = np.array(L,dtype=np.int32)                                         #np array
                                    K[3],D[3],S[3] = update_paths(K[3],D[3],S[3],L,s_dist,k=k_dis)
                                elif t_max>3:
                                    for a4,s4 in sample_branches(T,seqs,s3,p=t_p[3]):
                                        if sid in F[s4[0:2]]:
                                            L  = sub_seq_leg(seqs,s0[0],s0[1],seqs[s0[0]][a1][0],pw)                #before transfer-1
                                            L += [[s1[2],0,seqs[s1[0]][s1[1]][0],seqs[s1[0]][s1[1]][1],pw[s1[2]]]]  #transfer-1
                                            L += sub_seq_leg(seqs,s1[0],s1[1],seqs[s1[0]][a2][0],pw)                #before transfer-2
                                            L += [[s2[2],0,seqs[s2[0]][s2[1]][0],seqs[s2[0]][s2[1]][1],pw[s2[2]]]]  #transfer-2
                                            L += sub_seq_leg(seqs,s2[0],s2[1],seqs[s2[0]][a3][0],pw)                #before transfer-3
                                            L += [[s3[2],0,seqs[s3[0]][s3[1]][0],seqs[s3[0]][s3[1]][1],pw[s3[2]]]]  #transfer-3
                                            L += sub_seq_leg(seqs,s3[0],s3[1],seqs[s3[0]][a4][0],pw)                #before transfer-4
                                            L += [[s4[2],0,seqs[s4[0]][s4[1]][0],seqs[s4[0]][s4[1]][1],pw[s4[2]]]]  #transfer-4
                                            L += sub_seq_leg(seqs,s4[0],s4[1],sid,pw)                               #before destination
                                            L  = np.array(L,dtype=np.int32)                                         #np array
                                            K[4],D[4],S[4] = update_paths(K[4],D[4],S[4],L,s_dist,k=k_dis)
    stop = time.time()
    print('searched for max_trans=%s in %s sec'%(t_max,round(stop-start,2)))
    return K,D,S

def RTS_FULL(C,T,F,seqs,pw={0:0,-1:10*60,-2:20*60},min_paths=1,max_trans=4,trans_p=[1.0,1.0,0.5,0.25],min_rate=-3.0,verbose=True):
    t_start = time.time()
    X,z = {i:{} for i in range(max_trans+1)},0
    for i in range(len(C)):
        s0,sid,max_time = (C[i][0],C[i][1]),C[i][2],C[i][3]
        if s0[0:2] in F and sid in F[s0[0:2]]:
            L = sub_seq_leg(seqs,s0[0],s0[1],sid,pw)
            Y  = np.array(L,dtype=np.int32)
            v  = (Y[-1][3]-Y[0][3])+np.sum(Y[:,4])
            if v in X[0]: X[0][v][tuple(Y[:,2])] = Y
            else:         X[0][v] = {tuple(Y[:,2]):Y}
        elif len(X[0])<min_paths and max_trans>0: #will get up to 1-transfer more than optimal
            z += 1
            for a1,s1 in sample_branches(T,seqs,s0,trans_p[0],max_time,s_dist,sid,min_rate):
                if s1[0:2] in F and sid in F[s1[0:2]]:
                    L  = sub_seq_leg(seqs,s0[0],s0[1],seqs[s0[0]][a1][0],pw)                #before transfer-1
                    L += [[s1[2],0,seqs[s1[0]][s1[1]][0],seqs[s1[0]][s1[1]][1],pw[s1[2]]]]  #transfer-1
                    L += sub_seq_leg(seqs,s1[0],s1[1],sid,pw)                               #before destination
                    Y  = np.array(L,dtype=np.int32)                                         #np array
                    v  = (Y[-1][3]-Y[0][3])+np.sum(Y[:,4])
                    if v in X[1]: X[1][v][tuple(Y[:,2])] = Y
                    else:         X[1][v] = {tuple(Y[:,2]):Y}
                elif len(X[1])<min_paths and max_trans>1:
                    z += 1
                    for a2,s2 in sample_branches(T,seqs,s1,trans_p[1],max_time,s_dist,sid,min_rate+1.0):
                        if s2[0:2] in F and sid in F[s2[0:2]]:
                            L  = sub_seq_leg(seqs,s0[0],s0[1],seqs[s0[0]][a1][0],pw)                #before transfer-1
                            L += [[s1[2],0,seqs[s1[0]][s1[1]][0],seqs[s1[0]][s1[1]][1],pw[s1[2]]]]  #transfer-1
                            L += sub_seq_leg(seqs,s1[0],s1[1],seqs[s1[0]][a2][0],pw)                #before transfer-2
                            L += [[s2[2],0,seqs[s2[0]][s2[1]][0],seqs[s2[0]][s2[1]][1],pw[s2[2]]]]  #transfer-2
                            L += sub_seq_leg(seqs,s2[0],s2[1],sid,pw)                               #before destination
                            Y  = np.array(L,dtype=np.int32)                                         #np array
                            v  = (Y[-1][3]-Y[0][3])+np.sum(Y[:,4])
                            if v in X[2]: X[2][v][tuple(Y[:,2])] = Y
                            else:         X[2][v] = {tuple(Y[:,2]):Y}
                        elif len(X[2])<min_paths and max_trans>2: #if len(X[1])<1
                            z += 1
                            for a3,s3 in sample_branches(T,seqs,s2,trans_p[2],max_time,s_dist,sid,min_rate+1.0):
                                if s3[0:2] in F and sid in F[s3[0:2]]:
                                    L  = sub_seq_leg(seqs,s0[0],s0[1],seqs[s0[0]][a1][0],pw)                #before transfer-1
                                    L += [[s1[2],0,seqs[s1[0]][s1[1]][0],seqs[s1[0]][s1[1]][1],pw[s1[2]]]]  #transfer-1
                                    L += sub_seq_leg(seqs,s1[0],s1[1],seqs[s1[0]][a2][0],pw)                #before transfer-2
                                    L += [[s2[2],0,seqs[s2[0]][s2[1]][0],seqs[s2[0]][s2[1]][1],pw[s2[2]]]]  #transfer-2
                                    L += sub_seq_leg(seqs,s2[0],s2[1],seqs[s2[0]][a3][0],pw)                #before transfer-3
                                    L += [[s3[2],0,seqs[s3[0]][s3[1]][0],seqs[s3[0]][s3[1]][1],pw[s3[2]]]]  #transfer-3
                                    L += sub_seq_leg(seqs,s3[0],s3[1],sid,pw)                               #before destination
                                    Y  = np.array(L,dtype=np.int32)                                         #np array
                                    v  = (Y[-1][3]-Y[0][3])+np.sum(Y[:,4])
                                    if v in X[3]: X[3][v][tuple(Y[:,2])] = Y
                                    else:         X[3][v] = {tuple(Y[:,2]):Y}
                                elif len(X[3])<min_paths and max_trans>3: #if len(X[2])<1
                                    z += 1
                                    for a4,s4 in sample_branches(T,seqs,s3,trans_p[3],max_time,s_dist,sid,min_rate+1.0):
                                        if s4[0:2] in F and sid in F[s4[0:2]]:
                                            L  = sub_seq_leg(seqs,s0[0],s0[1],seqs[s0[0]][a1][0],pw)                #before transfer-1
                                            L += [[s1[2],0,seqs[s1[0]][s1[1]][0],seqs[s1[0]][s1[1]][1],pw[s1[2]]]]  #transfer-1
                                            L += sub_seq_leg(seqs,s1[0],s1[1],seqs[s1[0]][a2][0],pw)                #before transfer-2
                                            L += [[s2[2],0,seqs[s2[0]][s2[1]][0],seqs[s2[0]][s2[1]][1],pw[s2[2]]]]  #transfer-2
                                            L += sub_seq_leg(seqs,s2[0],s2[1],seqs[s2[0]][a3][0],pw)                #before transfer-3
                                            L += [[s3[2],0,seqs[s3[0]][s3[1]][0],seqs[s3[0]][s3[1]][1],pw[s3[2]]]]  #transfer-3
                                            L += sub_seq_leg(seqs,s3[0],s3[1],seqs[s3[0]][a4][0],pw)                #before transfer-4
                                            L += [[s4[2],0,seqs[s4[0]][s4[1]][0],seqs[s4[0]][s4[1]][1],pw[s4[2]]]]  #transfer-4
                                            L += sub_seq_leg(seqs,s4[0],s4[1],sid,pw)                               #before destination
                                            Y  = np.array(L,dtype=np.int32)                                         #np array
                                            v  = (Y[-1][3]-Y[0][3])+np.sum(Y[:,4])
                                            if v in X[4]: X[4][v][tuple(Y[:,2])] = Y
                                            else:         X[4][v] = {tuple(Y[:,2]):Y}
                                        z += 1
    t_stop = time.time()
    if verbose: print('tree searched %s edges with %s paths in %s sec'%(z,[len(X[x]) for x in X],round(t_stop-t_start)))
    return X

result_list = []
def collect_results(result):
    result_list.append(result)

def get_seq_paths(C,seqs,trans,max_trans=4,trans_p=[1.0,1.0,0.25,0.125],min_rate=-3.0): #can write recursively too...
    T,F = reduce_trans(trans),{} #applies the penalties to select the faster option tid_a=>tid_b
    for (tid,tdx) in T:
        for l in T[(tid,tdx)]:
            if (l[0],l[1]) not in F: #tid,tdx
                F[(l[0],l[1])] = set(seqs[l[0]][l[1]:,0])
    X    = RTS_FULL(C,T,F,seqs,max_trans=max_trans,trans_p=trans_p,min_rate=min_rate)
    return X

def k_best_paths(X,s_dist,k_dis=5):
    D,S = {},{}
    for t in X:
        D,S = {i:np.zeros((len(X[t]),len(X[t])),dtype=np.float32) for i in range(len(X))},{i:[] for i in range(len(X))}
        if len(X[t])>1:
            ks = sorted(X[t])
            for i in range(len(ks)): D[i,i] = 0.0
            for i,j in it.combinations(range(len(ks)),2):
                l = tu.lcs(X[t][ks[i]][:,2:],X[t][ks[j]][:,2:],s_dist)
                D[t][i,j] = D[t][j,i] = 1.0-2.0*((l[0]/l[1])*(l[0]/l[2]))/((l[0]/l[1])+(l[0]/l[2]))
            for i in range(len(ks)):
                S[t] += [[i,(X[t][ks[i]][-1][3]-X[t][ks[i]][0][3])+np.sum(X[t][ks[i]][:,4]),np.sum(D[t][i,:])]]
            idx = [i for i in range(len(ks))]
            for j in range(k_dis):
                S[t] = sorted(S[t])




n_base,d_base = 'ha_network/','ha_demand/'
search_time = ['7:00','10:00']
D = load_network_data(n_base,search_time=search_time) #will run preproccess_network if it was not already
persons = read_person_trip_list(d_base+'csts.txt')
stops,s_idx,s_names,s_dist,w_dist = D['stops'],D['stop_idx'],D['s_names'],D['s_dist'],D['w_dist']
trips,trip_idx,v_dist,calendar    = D['trips'],D['trip_idx'],D['v_dist'],D['calendar']
service_ids = get_processed_service_ids(D)

#person,i= 144 trip,j=0
C,X,each_person = {},{},True
for i in sorted(persons):
    for j in range(len(persons[i])):
    can = start_od_search(persons[i][j],w_dist,s_dist,v_dist)
    if can is not None and len(can[sorted(can)[0]]):
        print('person=%s,trip=%s was valid on %s, running RST...'%(i,j,persons[i][j][2].strftime('%m/%d/%Y')))
        si = list(can.keys())[0]
        candidates = can[si]
        K = []
        for c in candidates: #leave the trip direction filterin to the main algorithm
            if c[4]>-3.0: K += [(c[2],c[3],c[5][0],c[6],c[4])]
        K = sorted(K,key=lambda x: x[4])[::-1]
        if not each_person:
            for k in K:
                if si in C:
                    if k in C[si]: C[si][k] += [(i,j)]
                    else:          C[si][k]  = [(i,j)]
                else:              C[si] =  {k:[(i,j)]}
        else:
            seqs,graph,l_dist,l_idx,trans = D[si]['seqs'],D[si]['graph'],D[si]['l_dist'],D[si]['l_idx'],D[si]['trans']
            if i in X: X[i][j] = get_seq_paths(K,seqs,trans)
            else:      X[i]= {j:get_seq_paths(K,seqs,trans)}
    print('%s unique service_ids to search'%len(C))

    if not each_person: #pools the unique search possibilities....
        for si in sorted(C):
            print('processing %s'%si)
            seqs,graph,l_dist,l_idx,trans = D[si]['seqs'],D[si]['graph'],D[si]['l_dist'],D[si]['l_idx'],D[si]['trans']
            cpus,ks = mp.cpu_count(),sorted(C[si])
            partitions,n = [],len(ks)//cpus
            for i in range(cpus): partitions     += [ks[i*n:(i+1)*n]]
            if len(ks)%cpus>0:   partitions[-1] += ks[-1*(len(ks)%cpus):]
            print('starting || cython random tree search (RTS) computation')
            t_start = time.time()
            p1 = mp.Pool(processes=cpus)
            for i in range(len(partitions)):
                p1.apply_async(get_seq_paths,args=(partitions[i],seqs,trans),callback=collect_results)
            p1.close()
            p1.join()
            t_stop = time.time()
            X = {}
            for result in result_list:
                for i in result:
                    if i in X:
                        for j in result[i]: X[i][j] = result[i][j]
                    else:
                        X[i] = {}
                        for j in result[i]: X[i][j] = result[i][j]
        #now you have to dig out each persons search to match up results
