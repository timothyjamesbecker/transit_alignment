import glob
import os
import time
import copy
import numpy as np
import datetime
import time
import itertools as it
import matplotlib.pyplot as plt

#--------------------------------------

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

#spatial distances-------------
def haversine(p1,p2):  # returns meters
    lat1,lat2,lon1,lon2 = p1[1],p2[1],p1[0],p2[0]
    x = pow(np.sin((lat2-lat1)/2),2)+np.cos(lat1)*np.cos(lat2)*pow(np.sin((lon2-lon1)/2),2)
    return 12.742e3*np.arctan2(np.sqrt(x),np.sqrt(1-x))

def lawofcosines(p1,p2):  # returns meters
    lat1,lat2,lon1,lon2 = p1[1],p2[1],p1[0],p2[0]
    return 6.371e3*np.arccos(np.sin(lat1)*np.sin(lat2)+np.cos(lat1)*np.cos(lat2)*np.cos(lon2-lon1))

def equirectangular(p1, p2):
    lat1,lat2,lon1,lon2 = p1[1],p2[1],p1[0],p2[0]
    return 6.371e3*np.sqrt((pow((lat2-lat1)*np.cos((lat1+lat2)/2),2)+pow((lon2-lon1),2)))

def log_L2_norm(p1, p2):  # returns log L2 norm on lat/lon
    lat1,lat2,lon1,lon2 = p1[1],p2[1],p1[0],p2[0]
    return np.log(np.sqrt(pow(lat2-lat1, 2)+pow(lon2-lon1,2)))

#read and enumerate all the stop ids...
def read_stops(path,max_dist=1E1):
    header,data = [],[]
    with open(path,'r') as f:
        raw = [row.replace('\r','').replace('\n','') for row in f.readlines()]
    header = [f for f in raw[0].rsplit(',')]# field names=stop_id,stop_name,stop_lat,stop_lon,zone_id
    data = [row.rsplit(',') for row in raw[1:]]
    #type conversion---------------------------
    stops = {}
    for i in range(len(data)): #id, x=lon,y=lat
        stops[int(data[i][0])] = [i,float(data[i][3]),float(data[i][2]),[]]
    dist = {}
    for i,j in it.combinations(stops.keys(),2): #pairwise distance
        d = haversine(stops[i][1:],stops[j][1:])
        if d <= max_dist: dist[(i,j)] = d
    for (i,j) in dist:
        stops[i][3] += [[j,dist[(i,j)]]]
        stops[j][3] += [[i,dist[(i,j)]]]
    for i in stops: #will return NN < max_dist in sorted order
        stops[i][3] = sorted(stops[i][3],key=lambda x: x[1])
    return stops

#has time conversions inside----
def read_stop_times(path,stops):
    header, data = [], []
    with open(path, 'r') as f:
        raw = [row.replace('\r', '').replace('\n', '') for row in f.readlines()]
    header = [f for f in raw[0].rsplit(',')]  # field names=stop_id,stop_name,stop_lat,stop_lon,zone_id
    data = [row.rsplit(',') for row in raw[1:]]

    trips = {}
    for i in range(len(data)):
        trip_id = int(data[i][0])                      #trip id
        raw_time = data[i][1].rsplit(':')
        if int(raw_time[0])>23:
            raw_time = ':'.join([str(int(raw_time[0])-24).zfill(2)]+raw_time[1:])
            st_time = time.strptime('2 '+raw_time,'%d %H:%M:%S')  #:::TIME:::
        else:
            st_time = time.strptime('1 '+data[i][1], '%H:%M:%S')  #:::TIME:::
        st_time = datetime.timedelta(days=st_time.tm_mday,hours=st_time.tm_hour,
                                     minutes=st_time.tm_min,seconds=st_time.tm_sec) #:::TIME:::
        stop_id = int(data[i][3])                      #stop_id
        seq_id  = int(data[i][4])                      #seq_pos
        if trip_id in trips: trips[trip_id] += [[seq_id,stops[stop_id][0],st_time]]
        else:                trips[trip_id]  = [[seq_id,stops[stop_id][0],st_time]]

    for t in trips: #ensures you are sorted by sequence field-------------------
        trips[t] = sorted(trips[t],key=lambda x: x[0])
    for t in trips: #reform final sequence datastructure: stop_id, time
        for i in range(len(trips[t])):
            trips[t][i] = [trips[t][i][1],trips[t][i][2]]
    return trips

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





def filter_trips(trips,dep_time='08:00:00',dep_window='00:05:00'):
    raw_time = dep_time.rsplit(':')
    if int(raw_time[0]) > 23:
        d_time = ':'.join([str(int(raw_time[0]) - 24).zfill(2)] + raw_time[1:])
        d_time = time.strptime(raw_time, '%H:%M:%s')
    else:
        d_time = time.strptime(dep_time, '%H:%M:%S')
    d_time = datetime.timedelta(hours=d_time.tm_hour, minutes=d_time.tm_min, seconds=d_time.tm_sec)

    raw_time = dep_window.rsplit(':')
    if int(raw_time[0]) > 23:
        w_time = ':'.join([str(int(raw_time[0]) - 24).zfill(2)] + raw_time[1:])
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
n_base = '/media/data/SF_CHAMP_ToyNetwork/network_draft1.14_fare/'
d_base = '/media/data/SF_CHAMP_ToyNetwork/fasttrips_demand_v0.5/'
n_stops = read_stops(n_base+'stops.txt',10.0)                     #{stop_id:[enum_stop_id,x,y,[NN<=10.0]], ... }
n_trips = read_stop_times(n_base+'stop_times.txt',n_stops)        #{trip_id:[enum_stop_id,timedelta], ... }

p_stop = time.time()

f_start = time.time()
viable_trips = filter_trips(n_trips,dep_time='09:00:00')
clusters = trip_set_analysis(viable_trips,cutoff=1)
f_stop = time.time()

print('finished processing stops/trips prior to filtering in %s sec'%round(p_stop-p_start,2))
print('finished time filtering and set analysis in %s sec'%round(f_stop-f_start,2))
#take a look at clusters----------------------------------------
C = np.asarray([len(clusters[x])for x in clusters])
# plt.hist(C,bins='auto')
# plt.show()
