import glob
import os
import sys
import time
import copy
import datetime
import time
import itertools as it
import numpy as np
import matplotlib.pyplot as plt
import transit_utils as tu

#--------------------------------------

def mem_size(obj,unit='G'):
    if unit =='T':
        return round(sys.getsizeof(obj)/(1024*1024*1024*1024.0),2) #TB
    if unit =='G':
        return round(sys.getsizeof(obj)/(1024*1024*1024.0),2) #GB
    if unit =='M':
        return round(sys.getsizeof(obj)/(1024*1024.0),2) #MB
    if unit =='K':
        return round(sys.getsizeof(obj)/(1024.0),2) #KB

# compute the edit graph between two strings using
# the Wagner-Fisher Algorithm and the edit values:
#M = Match, I = Insertion, D = Delete, S = Substitution
def edit_graph(s1,s2):
    u,v = len(s1),len(s2)
    d = [[0 for col in range(v+1)] for row in range(u+1)]
    b = [['_' for col in range(v+1)] for row in range(u+1)]
    for i in range(0,u+1): d[i][0] = i
    for j in range(0,v+1): d[0][j] = j
    for j in range(1,v+1):
        for i in range(1,u+1):
            if s1[i-1] == s2[j-1]:
                d[i][j],b[i][j] = d[i-1][j-1],  'M'
            elif d[i-1][j] <= d[i][j-1] and d[i-1][j] <= d[i-1][j-1]:
                d[i][j],b[i][j] = d[i-1][j]+1,  'D'
            elif d[i][j-1] <= d[i-1][j-1]:
                d[i][j],b[i][j] = d[i][j-1]+1,  'I'
            else:
                d[i][j],b[i][j] = d[i-1][j-1]+1,'S'
    return b, d[u][v]

# Wagner-Fisher Edit Distance Algorithm
# from two strings s1,s2 computes the
# min # of moves to make s1 into s2
def edit_dist(s1,s2,w):
   u,v = len(s1),len(s2)
   d = [[0 for col in range(v+1)] for row in range(u+1)]
   for i in range(0,u+1): d[i][0] = i
   for j in range(0,v+1): d[0][j] = j
   for j in range(1,v+1):
      for i in range(1,u+1):
         if s1[i-1] == s2[j-1]:d[i][j] = d[i-1][j-1]
         else:
             d[i][j] = min(d[i-1][j]+w[0],d[i][j-1]+w[1],d[i-1][j-1]+w[2])
   return d[u][v]

# Minimum Sum of Pairs Score with string index i
# uses an upper bound for the worste minSum as
# number of seq * longest sequence -> n*m
def min_sp(strings,m,n):
    minSum,i = n*m,0
    for j in range(0,len(strings)):
        currSum = sum(edit_dist(strings[j],k) for k in strings)
        if currSum < minSum:
            minSum,i = currSum,j
    return minSum,i

# align s1 to s2
def edit(s1,s2):
   accum = 0
   for i in range(0,len(s1)):
      if s1[i] == '-':
         s2 = s2[0:i]+'-'+s2[i+1:]
         accum+=1
   m,n,t     = len(s1), len(s2),''
   path,cost = edit_graph(s1,s2)
   while (m > 0) or (n > 0):
      if path[m][n] == 'M':
         m,n,t = m-1,n-1,s1[m-1]+t
      elif path[m][n] == 'S':
         m,n,t = m-1,n-1,s1[n-1]+t
      elif path[m][n] == 'I':
         m,n,t = m,n-1, '-'+t
      else:
         m,n = m-1,n
   return t,accum+cost

def star(s,c):
    accum = 0
    for i in range(0,len(s)):
        if i != c:
            if len(s[c]) >= len(s[i]):
               s[i],cost = edit(s[i],s[c])
               accum+=cost
            else:
                s[c],cost = edit(s[c],s[i])
                accum+=cost
            for j in range(0,i+1):
                if i != j:
                    if len(s[c]) >= len(s[j]):
                        s[j],cost = edit(s[j],s[c])
                        accum+=cost
                    else:
                        s[c],cost = edit(s[j],s[c])
                        accum+=cost
    return s,accum

#-------------------------------------

#sequence distances-----------------------
def affine_sim(c1,c2,w=[0,1,1,0.9,0.05]):#match,miss,gap,space,scale
    u,v = len(c1),len(c2)
    if u<v: u,v,c1,c2 = v,u,c2,c1    #u and c1 are longer
    D = np.zeros((u+1,),dtype=float)
    C = np.zeros((u+1,),dtype=float)
    P = np.zeros((u+1,),dtype=float)
    C[0] = 0
    for j in range(1,u+1):
        C[j] = w[2]+w[3]*j
        D[j] = np.iinfo(int).max
    for i in range(1,v+1):
        y2 = c2[i-1]
        for x in range(0,u+1): P[x] = C[x]
        C[0] = w[2]+w[3]*i
        I = np.iinfo(int).max
        for j in range(1,u+1):
            y1 = c1[j-1]
            if j <= v:   I = min(I,C[j-1]+w[2])+w[3]
            else:        I = min(I,C[j-1]+w[2]*w[4])+w[3]*w[4]
            D[j] = min(D[j],P[j]+w[2])+w[3]
            if y2 == y1: M = P[j-1]+w[0]
            else:        M = P[j-1]+w[1]
            C[j] = min(I,D[j],M)
    return [u-C[u],u]

def edit_sim(c1,c2,w=[0,1,1]): #[m,d,i,s]
    u,v,k = len(c1),len(c2),2
    if u<v: u,v,c1,c2 = v,u,c2,c1
    D = np.zeros((u+1,k),dtype=int)
    A = []
    for i in range(u+1):   D[i][0] = i
    for j in range(v%k+1): D[0][j] = j
    for j in range(1,v+1):
        for i in range(1,u+1):
            if c1[i-1] == c2[j-1]:  #match
                D[i][j%k] = D[i-1][(j-1)%k]+w[0]
            else:
                x,y,z = D[i-1][j%k]+w[1],D[i][(j-1)%k]+w[2],D[i-1][(j-1)%k]+w[3]
                if x<=y and x<=z:   #delete
                    D[i][j%k] = x
                elif y<=x and y<=z: #insert
                    D[i][j%k] = y
                else:
                    D[i][j%k] = z   #substitute
    return [u-D[u][v%k],u]

def lcs(c1,c2,w=[1,0,0]):
    u,v,k, = len(c1),len(c2),2
    if u<v: u,v,c1,c2 = v,u,c2,c1
    D = np.zeros((k,v+1),dtype=int)
    for i in range(1,u+1):      #rows--------------
        for j in range(1,v+1):  #columns-----------
            if c1[i-1]==c2[j-1]:                  #matching
                D[i%k][j] = D[(i-1)%k][j-1]+w[0]  #diagnal
            elif D[i%k][j-1] >= D[(i-1)%k][j]:    #extend
                D[i%k][j] = D[i%k][j-1]+w[1]      #left
            else:                                 #delete
                D[i%k][j] = D[(i-1)%k][j]+w[2]    #top
    return [D[u%k][v],u]

def pairwise_lcs_dict(S,dist,lim=0.5,w=[1.0,0.0,0.0],pos=0): #w=[match,extend,delete]
    idx = list(sorted(S.keys()))
    n,l,z,k = len(S),max([len(S[i]) for i in S]),1,2
    L = np.zeros((n,n,2),dtype=np.float32)
    D = np.zeros((k,l+1),dtype=np.float32)   #two of the longest seq
    for x in range(n):
        u = len(S[idx[x]])
        for y in range(z,n,1):
            v = len(S[idx[y]])
            D[:] = 0.0
            for i in range(1,u+1):      #rows--------------
                d1 = S[idx[x]][i-1,pos]
                for j in range(1,v+1):  #columns-----------
                    d2 = S[idx[y]][j-1,pos]
                    if dist[d1][d2]<=lim:              D[i%k][j] = D[(i-1)%k][j-1]+w[0]*(1.0-dist[d1][d2]/lim)
                    elif D[i%k][j-1] >= D[(i-1)%k][j]: D[i%k][j] = D[i%k][j-1]+w[1]
                    else:                              D[i%k][j] = D[(i-1)%k][j]+w[2]
            L[x][y] = L[x][y] = [D[u%k][v],u*w[0]]
        L[x][x] = [u*w[0],u*w[0]]
        z += 1
    return L

def jaccard_sim(c1,c2):
    C1,C2 = set(c1),set(c2)
    return [len(C1.intersection(C2)),len(C1.union(C2))]

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
        sid = int(data[i][c_idx['stop_id']].split('_merged_')[0])
        stops += [[sid,[float(data[i][c_idx['stop_lon']]),float(data[i][c_idx['stop_lat']])],[]]]
        s_names[sid] = data[i][c_idx['stop_name']]
    stops = sorted(stops,key=lambda x: (x[1][0],x[1][1])) #spatial sort by lon,lat
    s_idx = {stops[i][0]:i for i in range(len(stops))}
    start = time.time()
    dist = tu.pairwise_rectangular(stops) #efficient straight line estimate sufficient for upper bound
    stop = time.time()

    #334-340 Capitol Ave, Hartford, CT 06115 to 1250-1256 Farmington Ave, Hartford, CT 06105
    #print('rectangular->dist[(s_idx[2],s_idx[8203])]=%s mi, in %s sec'%(round(dist[(s_idx[2],s_idx[8203])],2),round(stop-start,2)))
    print('%s stop-pairs calculated in %s sec'%(len(dist)**2,round(stop-start,2)))
    for i in range(dist.shape[0]):
        for j in range(dist.shape[1]):
            if i!=j:
                if dist[i,j]<=max_miles:
                    stops[i][NN] += [[j,dist[i,j]]]
    for i in range(len(stops)): #will return NN < max_dist in sorted order
        stops[i][NN] = sorted(stops[i][NN],key=lambda x: x[1])
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
        data[i][0] = int(data[i][c_idx['service_id']].split('_merged_')[0])
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
        trip_id      = int(data[i][c_idx['trip_id']].split('_merged_')[0])
        service_id   = int(data[i][c_idx['service_id']].split('_merged_')[0])
        route_id     = int(data[i][c_idx['route_id']].split('_merged_')[0])
        direction_id = int(data[i][c_idx['direction_id']])
        trip_sign    = data[i][c_idx['trip_headsign']]
        trips += [[trip_id,trip_sign,route_id,service_id,direction_id]]
    t_idx = {trips[i][0]:i for i in range(len(trips))}
    return trips,t_idx

#has time conversions inside and can aply calendar
#needs to have calendar.txt, trips.txt and select the day Mo,Tu,We,Th,Fr,Sa,Su
#stops = [[sid,sname,lon,lat,NN=[]],...[]]
def read_gtfs_seqs(n_base,s_idx,trips,t_idx,calendar,search_date,search_time=[0,115200]):
    if type(search_date) is str: #search_date.weekday() = 0 => monday
        for t in ['%m/%d/%Y','%m-%d-%Y','%Y/%m/%d','%Y-%m-%d']:
            try: search_date = datetime.datetime.strptime(search_date,t)
            except Exception as E: pass
    s_id = None
    for service_id in calendar: #find the first matching service_id for the search_date
        if calendar[service_id][1][0]<=search_date and \
                calendar[service_id][1][1]>=search_date and \
                calendar[service_id][2][search_date.weekday()]:
            s_id = service_id
    if s_id is not None: #can only select one service id (weekday,sat,sun)
        header,data = [],[]
        with open(n_base+'stop_times.txt', 'r') as f:
            raw = [row.replace('\r', '').replace('\n', '') for row in f.readlines()]
        header = [f for f in raw[0].rsplit(',')]  # field names=stop_id,stop_name,stop_lat,stop_lon,zone_id
        c_idx = {header[i]:i for i in range(len(header))}
        data = [row.rsplit(',') for row in raw[1:]]
        seqs = {}
        for i in range(len(data)):
            trip_id = int(data[i][c_idx['trip_id']])
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
#:::TO DO: can add spatial filters via circles and rectangular bounding boxes :::
def filter_seqs(seqs,time_window=[0,115200]):
    S = copy.deepcopy(seqs)
    if type(time_window[0]) is str:
        for t in ['%H:%M:%S','%H:%M','%H']: #try hours:minutes:seconds, then hours:minutes, then hours
            passed = False
            try:
                st_time = time.strptime(time_window[0],t)
                time_window[0] = datetime.timedelta(hours=st_time.tm_hour,minutes=st_time.tm_min,seconds=st_time.tm_sec)
                time_window[0] = np.int32(time_window[0].total_seconds())
                st_time = time.strptime(time_window[1],t)
                time_window[1] = datetime.timedelta(hours=st_time.tm_hour,minutes=st_time.tm_min,seconds=st_time.tm_sec)
                time_window[1] = np.int32(time_window[1].total_seconds())
                time_window = sorted(time_window)
                passed = True
            except Exception as E: pass
            if passed: break
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
    D,L = np.zeros((len(seqs),max_len,4),dtype=np.int32),[]
    for i in range(len(sk)):
        n = len(seqs[sk[i]])
        for j in range(n): D[i,:n,:] = seqs[sk[i]][:]
        L += [n]
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
        o_add  = data[i][c_idx['o_address_recode_2_linked']]
        d_taz  = int(data[i][c_idx['destination_bg_geoid_recode_linked']])
        d_add  = data[i][c_idx['d_address_recode_linked']]
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
            if pid in persons: persons[pid] += [[o_taz,st_time[0],d_taz,st_time[1]]]
            else:              persons[pid]  = [[o_taz,st_time[0],d_taz,st_time[1]]]; j+=1
        except Exception as E:
            print('row i=%s was not well formed, possible junk data:%s\n%s'%(i,E,data[i]))
            pass
    return persons

#---------------------------------------------------------------------------------------------
def filter_trips(trips,dep_time='08:00:00',dep_window='00:10:00'):
    raw_time = dep_time.rsplit(':')
    if int(raw_time[0]) > 23:
        d_time = ':'.join([str(int(raw_time[0]) - 24).zfill(2)] + raw_time[1:])
        d_time = time.strptime(raw_time, '%H:%M:%s')
    else:
        d_time = time.strptime(dep_time, '%H:%M:%S')
    d_time = datetime.timedelta(hours=d_time.tm_hour+24, minutes=d_time.tm_min, seconds=d_time.tm_sec)

    raw_time = dep_window.rsplit(':')
    if int(raw_time[0]) > 23:
        w_time = ':'.join([str(int(raw_time[0])-24).zfill(2)] + raw_time[1:])
        w_time = time.strptime(raw_time, '%H:%M:%s')
    else:
        w_time = time.strptime(dep_window, '%H:%M:%S')
    w_time = datetime.timedelta(hours=w_time.tm_hour, minutes=w_time.tm_min, seconds=w_time.tm_sec)

    S = {}
    for t in trips:
        for i in range(len(trips[t])):
            if trips[t][i][1] >= (d_time - w_time) and trips[t][i][1] <= (d_time + w_time):
                S[t] = copy.deepcopy(trips[t])
                break
    return S

def trip_set_analysis(trips,cutoff=1):
    T,M = {},{}
    for t in trips:
        T[t] = set([x[0] for x in trips[t]])
    for i,j in it.combinations(trips.keys(),2): #set feature magnitudes-----------------------
        F = [len(T[i].intersection(T[j])),len(T[j].union(T[i]))] #iIj,iUj => I/J = jaccard sim
        if F[0]>=cutoff:
            if i in M: M[i] += [F+[j]]
            else:      M[i]  = [F+[j]]
            if j in M: M[j] += [F+[i]]
            else:      M[j]  = [F+[i]]
    return M

#read passenger data from demand file-------------------------
#read passenger data from demand file-------------------------

#---------------------------------------------------------------------------------------------

w_buff = 0.5 #walk distance =>straightline distance for stop-stop, and taz-stop
n_base,d_base,search_date,search_time = 'ha_network/','ha_demand/','5/24/2016',['8:00','8:30']
stops,s_idx,s_names,s_dist = read_gtfs_stops(n_base,max_miles=w_buff) #{enum_stop_id:[stop_id,stop_name,x,y,[NN<=10.0]], ... }
v_dist      = gtfs_stop_time_shape_dist(n_base,s_idx) #in vehicle distances
trips,t_idx = read_gtfs_trips(n_base) #trips=[trip_id,trip_name,route_id,service_id,direction]
calendar    = read_gtfs_calendar(n_base) #{service_id,[start,end],[mon,tue,wed,thu,fri,sat,sun])
seqs,graph  = read_gtfs_seqs(n_base,s_idx,trips,t_idx,calendar,search_date)
print('%s total trips for date=%s'%(len(seqs),search_date))
w_dist      = read_walk_access(n_base,s_idx,walk_buff=w_buff)
persons     = read_person_trip_list(d_base+'csts.txt')
seqs,graph = filter_seqs(seqs,time_window=search_time)
print('%s total trips left after filtering using time_window=%s'%(len(seqs),search_time))
p_start = time.time()
L1 = pairwise_lcs_dict(seqs,s_dist)
p_stop      = time.time()
print('%s LCSWT pairs calculated in %s sec'%(len(seqs)**2,round(p_stop-p_start,2)))

ss,ls,idx = seqs_to_ndarray(seqs)
p_start = time.time()
L2 = tu.pairwise_lcs(ss,ls,s_dist)
p_stop      = time.time()
print('%s LCSWT pairs calculated in %s sec'%(len(seqs)**2,round(p_stop-p_start,2)))

# f_start = time.time()
# viable_start_trips = filter_trips(n_trips,dep_time='08:00:00',dep_window='00:10:00') #all the trips that are within a pickup
# clusters = trip_set_analysis(viable_start_trips,cutoff=1) #at least one stop the same
# f_stop = time.time()
#
#
# S = {}
# for t in viable_start_trips:
#     for i in range(len(viable_start_trips[t])):
#         sid,td = viable_start_trips[t][i][0],viable_start_trips[t][i][1]
#         if sid in S: S[sid] += [[t,td]]
#         else:        S[sid]  = [[t,td]]
# for sid in S: S[sid] = sorted(S[sid],key=lambda x: x[1])

#
# print('finished processing stops/trips prior to filtering in %s sec'%round(p_stop-p_start,2))
# print('finished time filtering and set analysis in %s sec'%round(f_stop-f_start,2))
# #take a look at clusters----------------------------------------
# C = np.asarray([len(clusters[x])for x in clusters])
# # plt.hist(C,bins='auto')
# # plt.show()
