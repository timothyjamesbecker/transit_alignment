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

def edit_sim(c1,c2,w=[0,1,1,1]): #[m,d,i,s]
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
    for k in E: E[k] = np.min(E[k])
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
def read_gtfs_stop_times(n_base,s_idx,trips,t_idx,calendar,search_date):
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
        times = {}
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
                if t_idx[trip_id] in times: times[t_idx[trip_id]] += [[s_idx[stop_id],st_time]]
                else:                       times[t_idx[trip_id]]  = [[s_idx[stop_id],st_time]]
        #stop_graph--------------------------------------------------------------------------------------
        G = {}
        for tid in times: #trip times have to have at least two stops
            times[tid] = sorted(times[tid],key=lambda x: x[1])
            if len(times[tid])==2:
                if times[tid][1][0] in G: G[times[tid][1][0]]['in']  += [[times[tid][0][0],times[tid][1][1],tid,0]]
                else:                     G[times[tid][1][0]] = {'in':[[times[tid][0][0],times[tid][1][1],tid,0]],'out':[]}
                if times[tid][0][0] in G: G[times[tid][0][0]]['out'] += [[times[tid][1][0],times[tid][0][1],tid,1]]
                else:                     G[times[tid][0][0]] = {'out':[[times[tid][1][0],times[tid][0][1]],tid,1],'in':[]}
            elif len(times[tid])>2:
                if times[tid][0][0] in G: G[times[tid][0][0]]['out'] += [[times[tid][1][0],times[tid][0][1],tid,1]]
                else:                     G[times[tid][0][0]] = {'out':[[times[tid][1][0],times[tid][0][1],tid,1]],'in':[]}
                for i in range(1,len(times[tid])-1,1):
                    if times[tid][i][0] in G: G[times[tid][i][0]]['in']  += [[times[tid][i-1][0],times[tid][i][1],tid,i-1]]
                    else:                     G[times[tid][i][0]] = {'in':[[times[tid][i-1][0],times[tid][i][1],tid,i-1]],'out':[]}
                    G[times[tid][i][0]]['out'] += [[times[tid][i+1][0],times[tid][i][1],tid,i+1]]
                if times[tid][i+1][0] in G: G[times[tid][i+1][0]]['in']  += [[times[tid][i][0],times[tid][i+1][1],tid,i]]
                else:                       G[times[tid][i+1][0]] = {'in':[[times[tid][i][0],times[tid][i+1][1],tid,i]],'out':[]}
        #now use a dict to bin each hour (24hours*60minutes*60seconds)
        for sid in G:
            for d in ['in','out']:
                if len(G[sid][d])>0:
                    G[sid][d] = sorted([[y[0],y[1],y[2],y[3]] for y in set([(x[0],x[1],x[2],x[3]) for x in G[sid][d]])],key=lambda f: (f[1],f[0]))
                    hm = {}
                    for i in range(32):
                        hm[i] = {}
                        for j in range(60): hm[i][j] = []
                    for i in range(len(G[sid][d])):
                        hr,mn = G[sid][d][i][1]//(3600),G[sid][d][i][1]%(3600)//60
                        hm[hr][mn] += [G[sid][d][i]]
                    G[sid][d] = hm
        for sid in G:
            for d in ['in','out']:
                for hr in G[sid][d]:
                    for mn in G[sid][d][hr]:
                        G[sid][d][hr][mn] = sorted(G[sid][d][hr][mn],key=lambda x: (x[1],x[0]))
        #stop graph-------------------------------------------------------------------------------------------------------
        for t in times:
            times[t] = np.asarray(sorted(times[t],key=lambda x: x[1])) #time sorted
    return times,G

#from fast trips data analysis---------------------------------------------------

#will give taz to stop_id associated distances in m (walk_access_ft dist is in km)
#walk buff is in meters => 800 ~0.5 miles
def read_walk_access(path,walk_buff=800,val_sym=True,flat_sym=True):
    header, data = [], []
    with open(path, 'r') as f:
        raw = [row.replace('\r', '').replace('\n', '') for row in f.readlines()]
    header = [f for f in raw[0].rsplit(',')]  # field names=stop_id,stop_name,stop_lat,stop_lon,zone_id
    data = [row.rsplit(',') for row in raw[1:]]

    walk,short,filt = {},0,0
    for i in range(len(data)):
        taz,stop_id,dir = int(data[i][0]),int(data[i][1]),(1 if data[i][2]=='access' else 0) #1 for on 0 for off
        dist = float(data[i][3])*1000; #km to m
        if dist<=walk_buff: #will trim long walks...
            if taz in walk:
                if stop_id in walk[taz]: walk[taz][stop_id][dir] = dist
                else:                    walk[taz][stop_id] = {dir:dist}
            else:                        walk[taz] = {stop_id:{dir:dist}}
            short+=1
        else: filt+=1
    print('%s out of %s taz-stop pairs were beyond the %sm walk buffer and were filtered'%(filt,filt+short,walk_buff))
    if val_sym: #will check to see that all stops have access/egress the same
        s,d = 0,0
        for k in walk:
            for s in walk[k]:
                if len(walk[k][s])<2: d+=1
                elif walk[k][s][0]!=walk[k][s][1]: d+=1
                else: s+=1
        if d<=0: print('all %s remaining stop_id(s) are symetric for walk access'%s)
        else:    print('%s stops out of %s remaining stop_id(s) are asymetric for walk access'%(d,d+s))
        if d<=0 and flat_sym:
            flat = {}
            for k in walk:
                flat[k] = {}
                for s in walk[k]:
                    flat[k][s] = walk[k][s][0]
            walk = flat
    return walk

def read_person_trip_list(path):
    path = d_base+'trip_list.txt'
    header, data = [], []
    with open(path, 'r') as f:
        raw = [row.replace('\r', '').replace('\n', '') for row in f.readlines()]
    header = [f for f in raw[0].rsplit(',')]  # field names=stop_id,stop_name,stop_lat,stop_lon,zone_id
    data = [row.rsplit(',') for row in raw[1:]]

    persons = {}
    for i in range(len(data)):
        p_id,o_taz,d_taz,target,vot = data[i][0],int(data[i][3]),int(data[i][4]),\
                                      (1 if data[i][9]=='departure' else 0),float(data[i][10])
        mode,purpose,raw_time,st_time = data[i][5],data[i][6],[data[i][7].rsplit(':'),data[i][8].rsplit(':')],[None,None]
        for j in range(len(raw_time)):
            if int(raw_time[j][0]) > 23:
                raw_time[j] = ':'.join([str(int(raw_time[j][0])-24).zfill(2)]+raw_time[j][1:])
                st_time[j]  = time.strptime('2 '+raw_time[j], '%d %H:%M:%S')  #:::TIME:::
            else:
                raw_time[j] = ':'.join(raw_time[j])
                st_time[j]  = time.strptime('1 '+raw_time[j], '%d %H:%M:%S')  #:::TIME:::
            st_time[j] = datetime.timedelta(days=st_time[j].tm_mday,hours=st_time[j].tm_hour,
                                            minutes=st_time[j].tm_min,seconds=st_time[j].tm_sec)  #:::TIME:::
            persons[p_id] = [(o_taz,st_time[0]),(d_taz,st_time[1]),mode,target,purpose,vot]

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

p_start = time.time()
n_base,d_base,search_date = 'ha_network/','ha_demand/','5/24/2016'
stops,s_idx,s_names,dist = read_gtfs_stops(n_base) #{enum_stop_id:[stop_id,stop_name,x,y,[NN<=10.0]], ... }
v_dist = gtfs_stop_time_shape_dist(n_base,s_idx) #in vehicle distances
trips,t_idx = read_gtfs_trips(n_base) #trips=[trip_id,trip_name,route_id,service_id,direction]
calendar = read_gtfs_calendar(n_base) #{service_id,[start,end],[mon,tue,wed,thu,fri,sat,sun])
times = read_gtfs_stop_times(n_base,s_idx,trips,t_idx,calendar,search_date)
p_stop = time.time()

#stop_graph--------------------------------------------------------------------------------------
S = {sid:{'in':[],'out':[]} for sid in range(len(stops))}
for tid in times: #trip times have to have at least two stops
    if times[tid].shape[0]==2:
        S[times[tid][1][0]]['in']  += [[times[tid][0][0],times[tid][1][1]]]
        S[times[tid][0][0]]['out'] += [[times[tid][1][0],times[tid][0][1]]]
    elif times[tid].shape[0]>2:
        S[times[tid][0][0]]['out'] += [[times[tid][1][0],times[tid][0][1]]]         #first stop
        for i in range(1,times[tid].shape[0]-1,1):                                  #general case
            S[times[tid][i][0]]['in']  += [[times[tid][i-1][0],times[tid][i][1]]]   #have in and
            S[times[tid][i][0]]['out'] += [[times[tid][i+1][0],times[tid][i][[1]]]] #out edges
        S[times[tid][i+1][0]]['in']  += [[times[tid][i][0],times[tid][i+1][1]]]     #last stop




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
