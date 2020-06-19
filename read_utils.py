import os
import sys
import copy
import datetime
import time
import ctypes
import gzip
import pickle
import itertools as it
import numpy as np
import transit_utils as tu

#read and enumerate all the stop ids..
#n_path is the path to the GTFS folder
# max_miles is the maximum miles is used for stright-line pairwise distance
# max_nn is the distance that will be used for shape distance correction
# NN is the index of the NN list inside the stops list
def read_gtfs_stops(n_path,max_miles=0.5,NN=2):
    header,data = [],[]
    with open(n_path+'stops.txt','r') as f:
        raw = [row.replace('\r','').replace('\n','') for row in f.readlines()]
    header = [f for f in raw[0].rsplit(',')]# field names=stop_id,stop_name,stop_lat,stop_lon,zone_id
    c_idx = {header[i]:i for i in range(len(header))}
    data = [row.rsplit(',') for row in raw[1:]]
    #type conversion---------------------------
    stops,s_names = [],{}
    for i in range(len(data)): #[id, x=lon,y=lat]
        sid = int(data[i][c_idx['stop_id']].split('_merged_')[0]) #these are ints...
        stops += [[sid,[float(data[i][c_idx['stop_lon']]),float(data[i][c_idx['stop_lat']])],{}]]
        s_names[sid] = data[i][c_idx['stop_name']]
    stops = sorted(stops,key=lambda x: (x[1][0],x[1][1])) #spatial sort by lon,lat
    s_idx = {stops[i][0]:i for i in range(len(stops))}
    start = time.time()
    dist  = tu.pairwise_rectangular(stops) #efficient straight line estimate sufficient for upper bound
    stop = time.time()

    #334-340 Capitol Ave, Hartford, CT 06115 to 1250-1256 Farmington Ave, Hartford, CT 06105
    #print('rectangular->dist[(s_idx[2],s_idx[8203])]=%s mi, in %s sec'%(round(dist[(s_idx[2],s_idx[8203])],2),round(stop-start,2)))
    print('%s stop-pairs calculated in %s sec'%(len(dist)**2,round(stop-start,2)))
    for i in range(dist.shape[0]):
        for j in range(dist.shape[1]):
            if i!=j:
                if dist[i,j]<=max_miles: stops[i][NN][j] = dist[i,j]
    return stops,s_idx,s_names,dist

#read the shape_dist_traveled calculations for each stop to stop link
def gtfs_stop_time_shape_dist(n_path,s_idx):
    start = time.time()
    header, data = [], []
    with open(n_path+'stop_times.txt', 'r') as f:
        raw = [row.replace('\r', '').replace('\n', '') for row in f.readlines()]
    header = [f for f in raw[0].rsplit(',')]  # field names=stop_id,stop_name,stop_lat,stop_lon,zone_id
    c_idx = {header[i]:i for i in range(len(header))}
    data = [row.rsplit(',') for row in raw[1:]]
    trips = {}
    for i in range(len(data)):
        stop_id = int(data[i][c_idx['stop_id']].split('_merged_')[0])      #stop_id
        seq_id  = int(data[i][c_idx['stop_sequence']])                     #seq_pos
        trip_id = int(data[i][c_idx['trip_id']])                           #trip
        if data[i][c_idx['shape_dist_traveled']]=='': d = 0.0              #sid_a to sid_a = 0.0
        else: d = float(data[i][c_idx['shape_dist_traveled']])             #sid_a to sid_b
        if trip_id in trips: trips[trip_id] += [[s_idx[stop_id],seq_id,d]] #pack it up
        else:                trips[trip_id]  = [[s_idx[stop_id],seq_id,d]] #for dist calcs
    for t in trips: trips[t] = sorted(trips[t],key=lambda x: x[1])
    E = {} #exact distance uses minimum sid_a to sid_b shape_distance traveled
    for t in trips:
        if len(trips[t])>1:
            for i in range(len(trips[t])):
                for j in range(i+1,len(trips[t]),1):
                    k = (min(trips[t][i][0],trips[t][j][0]),max(trips[t][i][0],trips[t][j][0]))
                    if k in E: E[k] += [abs(trips[t][i][2]-trips[t][j][2])]
                    else:      E[k]  = [abs(trips[t][i][2]-trips[t][j][2])]
    for k in E: E[k] = np.float32(np.min(E[k]))
    stop = time.time()
    print('calculated %s exact shape distance pairs in %s sec'%(len(E),round(stop-start,2)))
    return E

def read_gtfs_calendar(n_base):
    header,data = [],[]
    with open(n_base+'calendar.txt','r') as f:
        raw = [row.replace('\r','').replace('\n','') for row in f.readlines()]
    header = [f for f in raw[0].rsplit(',')]# field names=stop_id,stop_name,stop_lat,stop_lon,zone_id
    c_idx = {header[i]:i for i in range(len(header))}
    data = [row.rsplit(',') for row in raw[1:]]
    for i in range(len(data)):
        data[i][0] = data[i][c_idx['service_id']].split('_merged_')[0]
        data[i][1] = datetime.datetime.strptime(data[i][c_idx['start_date']],'%Y%m%d')
        data[i][2] = datetime.datetime.strptime(data[i][c_idx['end_date']],'%Y%m%d')
        data[i] = [data[i][0]]+[[data[i][1],data[i][2]]]+[[bool(int(x)) for x in data[i][c_idx['monday']:]]]
    C = {}
    for i in range(len(data)):
        if any(data[i][2]):
            if data[i][0] in C:
                C[data[i][0]][1][0] = min(C[data[i][0]][1][0],data[i][1][0])
                C[data[i][0]][1][1] = max(C[data[i][0]][1][1],data[i][1][1])
            else: C[data[i][0]] = data[i]
    return C

def read_gtfs_trips(n_base):
    header,data = [],[]
    with open(n_base+'trips.txt', 'r') as f:
        raw = [row.replace('\r', '').replace('\n', '') for row in f.readlines()]
    header = [f for f in raw[0].rsplit(',')]  # field names=stop_id,stop_name,stop_lat,stop_lon,zone_id
    c_idx = {header[i]:i for i in range(len(header))}
    data = [row.rsplit(',') for row in raw[1:]]
    trips = []
    for i in range(len(data)):
        trip_id      = data[i][c_idx['trip_id']].split('_merged_')[0]
        service_id   = data[i][c_idx['service_id']].split('_merged_')[0]
        route_id     = data[i][c_idx['route_id']].split('_merged_')[0]
        direction_id = int(data[i][c_idx['direction_id']])
        trip_sign    = data[i][c_idx['trip_headsign']]
        trips += [[trip_id,trip_sign,route_id,service_id,direction_id]]
    t_idx = {trips[i][0]:i for i in range(len(trips))}
    return trips,t_idx

#search date string '5/26/2016' to service if using the calendar
def get_service_id(calendar,search_date):
    for t in ['%m/%d/%Y','%m-%d-%Y','%Y/%m/%d','%Y-%m-%d']:
        try: search_date = datetime.datetime.strptime(search_date,t)
        except Exception as E: pass
    s_id = None
    for srv_id in calendar: #find the first matching service_id for the search_date
        if calendar[srv_id][1][0]<=search_date and \
                calendar[srv_id][1][1]>=search_date and \
                calendar[srv_id][2][search_date.weekday()]:
            s_id = srv_id
    return s_id

#has time conversions inside and can aply calendar
#needs to have calendar.txt, trips.txt and select the day Mo,Tu,We,Th,Fr,Sa,Su
#stops = [[sid,sname,lon,lat,NN=[]],...[]]
def read_gtfs_seqs(n_base,s_idx,trips,t_idx,calendar,service_id=None,search_date=None,search_time=[0,115200]):
    if type(search_date) is str: s_id = get_service_id(calendar,search_date)
    elif service_id is not None: s_id = service_id
    if s_id is not None: #can only select one service id (weekday,sat,sun)
        header,data = [],[]
        with open(n_base+'stop_times.txt', 'r') as f:
            raw = [row.replace('\r', '').replace('\n', '') for row in f.readlines()]
        header = [f for f in raw[0].rsplit(',')]  # field names=stop_id,stop_name,stop_lat,stop_lon,zone_id
        c_idx = {header[i]:i for i in range(len(header))}
        data = [row.rsplit(',') for row in raw[1:]]
        seqs = {}
        for i in range(len(data)):
            trip_id = data[i][c_idx['trip_id']]
            trip = trips[t_idx[trip_id]]  #trips=[trip_id,trip_name,route_id,service_id,direction]
            if trip[3]==s_id: # filter trips that are not in the service_id from calendar
                raw_time = data[i][c_idx['arrival_time']].rsplit(':')
                if int(raw_time[0])>23:
                    raw_time = ':'.join([str(int(raw_time[0])-24).zfill(2)]+raw_time[1:])
                    st_time = time.strptime(raw_time,'%H:%M:%S')
                    st_time = datetime.timedelta(days=st_time.tm_mday,hours=st_time.tm_hour,
                                                 minutes=st_time.tm_min,seconds=st_time.tm_sec) #:::TIME:::
                else:
                    st_time = time.strptime(data[i][c_idx['arrival_time']],'%H:%M:%S')
                    st_time = datetime.timedelta(days=st_time.tm_mday,hours=st_time.tm_hour,
                                                 minutes=st_time.tm_min,seconds=st_time.tm_sec) #:::TIME:::
                    st_time -= datetime.timedelta(days=1)
                # w,r = st_time.total_seconds()//60,st_time.total_seconds()%60
                # st_time = int(w if r<=30.0 else w+1)
                st_time = int(st_time.total_seconds())
                stop_id = int(data[i][c_idx['stop_id']].split('_merged_')[0]) #stop_id
                if t_idx[trip_id] in seqs: seqs[t_idx[trip_id]] += [[s_idx[stop_id],st_time,-1,-1]]
                else:                      seqs[t_idx[trip_id]]  = [[s_idx[stop_id],st_time,-1,-1]]
        for t in seqs: seqs[t] = sorted(seqs[t],key=lambda x: (x[1],x[0]))
        seqs,G = filter_seqs(seqs,time_window=search_time) #up to 32 hours => 8am the next day = 115200
    return seqs,G

#given seqs and time window in elapsed seconds, filter out some seqs and reindex the result
def filter_seqs(seqs,time_window=[0,115200]):
    S = copy.deepcopy(seqs)
    if type(time_window[0]) is str:
        if len(time_window[0].split(':'))==3:   t = '%H:%M:%S'
        elif len(time_window[0].split(':'))==2: t = '%H:%M'
        raw_time = [time_window[0].split(':'),time_window[1].split(':')]
        next_day = [datetime.timedelta(days=0),datetime.timedelta(days=0)]
        if int(raw_time[0][0])>23:
            time_window[0] = ':'.join([str(int(raw_time[0][0])-24)]+raw_time[0][1:])
            next_day[0]  = datetime.timedelta(days=1)
        if int(raw_time[1][0])>23:
            time_window[1] = ':'.join([str(int(raw_time[1][0])-24)]+raw_time[1][1:])
            next_day[1]  = datetime.timedelta(days=1)
        st_time = time.strptime(time_window[0],t)
        time_window[0] = datetime.timedelta(hours=st_time.tm_hour,minutes=st_time.tm_min,seconds=st_time.tm_sec)
        time_window[0] += next_day[0]
        time_window[0] = np.int32(time_window[0].total_seconds())
        st_time = time.strptime(time_window[1],t)
        time_window[1] = datetime.timedelta(hours=st_time.tm_hour,minutes=st_time.tm_min,seconds=st_time.tm_sec)
        time_window[1] += next_day[1]
        time_window[1] = np.int32(time_window[1].total_seconds())
        time_window = sorted(time_window)
    for t in S:
        teqs = []
        if type(S[t]) is not list: S[t] = S[t].tolist()
        for i in range(len(S[t])):
            if S[t][i][1]>=time_window[0] and S[t][i][1]<time_window[1]:
                teqs += [S[t][i]]
        S[t] = teqs
    sk = list(S.keys())
    for k in sk:
        if len(S[k])<=1: S.pop(k)
    #stop_graph--------------------------------------------------------------------------------------
    G = {}
    for tid in S: #trip S have to have at least two stops
        if len(S[tid])==2:
            if S[tid][1][0] in G:     G[S[tid][1][0]]['in']  += [[S[tid][0][0],S[tid][1][1],tid,0]]
            else:                     G[S[tid][1][0]] =   {'in':[[S[tid][0][0],S[tid][1][1],tid,0]],'out':[]}
            if S[tid][0][0] in G:     G[S[tid][0][0]]['out'] += [[S[tid][1][0],S[tid][0][1],tid,1]]
            else:                     G[S[tid][0][0]] =  {'out':[[S[tid][1][0],S[tid][0][1],tid,1]],'in':[]}
        elif len(S[tid])>2:
            if S[tid][0][0] in G:     G[S[tid][0][0]]['out'] += [[S[tid][1][0],S[tid][0][1],tid,1]]
            else:                     G[S[tid][0][0]] =  {'out':[[S[tid][1][0],S[tid][0][1],tid,1]],'in':[]}
            for i in range(1,len(S[tid])-1,1):
                if S[tid][i][0] in G: G[S[tid][i][0]]['in']  += [[S[tid][i-1][0],S[tid][i][1],tid,i-1]]
                else:                 G[S[tid][i][0]] =   {'in':[[S[tid][i-1][0],S[tid][i][1],tid,i-1]],'out':[]}
                G[S[tid][i][0]]['out'] +=                       [[S[tid][i+1][0],S[tid][i][1],tid,i+1]]
            if S[tid][i+1][0] in G:   G[S[tid][i+1][0]]['in']+= [[S[tid][i][0],S[tid][i+1][1],tid,i]]
            else:                     G[S[tid][i+1][0]] = {'in':[[S[tid][i][0],S[tid][i+1][1],tid,i]],'out':[]}
    for sid in G:
        for d in ['in','out']:
            if len(G[sid][d])>0:
                G[sid][d] = np.array(sorted(G[sid][d],key=lambda x: (x[1],x[0])),dtype=np.int32)
            else: G[sid][d] = np.array([],dtype=np.int32)
        for i in range(len(G[sid]['in'])):
            tid,idx = G[sid]['in'][i][2],G[sid]['in'][i][3]
            S[tid][idx][2] = i
        for i in range(len(G[sid]['out'])):
            tid,idx = G[sid]['out'][i][2],G[sid]['out'][i][3]
            S[tid][idx][3] = i
    for t in S: S[t] = np.array(S[t],dtype=np.int32)
    #data check----------------------------------------------------------------------------------------------------------
    ends,clipped = [],0 #check to see how may trips have been trimmed/clipped by the time_window
    for t in S:
        if list(S[t][:,3]).count(-1)<=1 and list(S[t][:,3]).count(-1)<=1: #zero or one in/out link missing (start/end)
            if not (S[t][-1][2]==-1 and S[t][0][3]==-1):
                clipped += 1
            ends += [True]
        else:
            ends += [False]
    print('%s seqs have consistant-links=%s with %s clipped terminals'%(len(S),all(ends),clipped))
    return S,G

def seqs_to_ndarray(seqs):
    sk = sorted(list(seqs.keys()))
    idx = {sk[i]:i for i in range(len(sk))}
    max_len = max([len(seqs[k]) for k in seqs])
    D = np.zeros((len(seqs),max_len,4),dtype=np.int32)
    L = np.zeros((len(seqs),),dtype=np.int32)
    D = D.reshape(len(seqs),max_len,4)
    for i in range(len(sk)):
        n = len(seqs[sk[i]])
        for j in range(n): D[i,:n,:] = seqs[sk[i]][:]
        L[i] = n
    return D,L,idx

#from fast trips data analysis---------------------------------------------------

#will give taz to stop_id associated distances in m (walk_access_ft dist is in km)
#walk buff is in meters => 800 ~0.5 miles
def read_walk_access(n_path,s_idx,walk_buff=0.5,walk_conv=1.0):
    header, data = [], []
    with open(n_path+'walk_access.txt', 'r') as f:
        raw = [row.replace('\r', '').replace('\n', '') for row in f.readlines()]
    header = [f for f in raw[0].rsplit(',')]  # field names=stop_id,stop_name,stop_lat,stop_lon,zone_id
    data = [row.rsplit(',') for row in raw[1:]]
    c_idx = {header[i]:i for i in range(len(header))}

    walk,short,filt = {},0,0
    for i in range(len(data)):
        taz,stop_id,dir = int(data[i][0]),int(data[i][1].split('_merged_')[0]),(1 if data[i][2]=='access' else 0)
        dist = float(data[i][3])*walk_conv; #miles
        if dist<=walk_buff: #will trim long walks...
            if taz in walk:
                if stop_id in walk[taz]: walk[taz][stop_id][dir] = dist
                else:                    walk[taz][stop_id] = {dir:dist}
            else:                        walk[taz] = {stop_id:{dir:dist}}
            short+=1
        else: filt+=1
    print('%s out of %s taz-stop pairs were beyond the %s mi walk buffer and were filtered'%(filt,filt+short,walk_buff))
    D = {}
    for taz in walk:
        for sid in walk[taz]:
            if taz in D:        D[taz][s_idx[sid]] = np.float32(walk[taz][sid][1])
            else:               D[taz] = {s_idx[sid]:np.float32(walk[taz][sid][1])} #access=1
            if s_idx[sid] in D: D[s_idx[sid]][taz] = np.float32(walk[taz][sid][0])
            else:               D[s_idx[sid]] = {taz:np.float32(walk[taz][sid][0])} #egress=0
    return D

def trip_set_analysis(seqs,stops,idx):
    t_start = time.time()
    F,S,E,D = {},{},set([]),[]
    for t in seqs:
        S[t],L = set([]),[s for s in seqs[t][:,0]]
        for l in L:
            for j in stops[l][2]: S[t].add(j[0])
    for i,j in it.combinations(sorted(list(S.keys())),2):
        if i>j: i,j = j,i
        if (i,j) not in F: F[(i,j)] = [len(S[i].intersection(S[j])),len(S[j].union(S[i]))]
    for k in F:
        if F[k][0]<1: E.add(k)
    E = sorted(list(E))
    for e in E:
        D += [sorted([idx[e[0]],idx[e[1]]])]
    t_stop = time.time()
    print('found %s trip pairs to exclude'%(len(E)))
    return E