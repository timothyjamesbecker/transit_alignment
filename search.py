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
        except Exception as E: pass
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
def start_od_search(person_trip,w_dist,s_dist,v_dist,buff_time=10,walk_speed=3,bus_speed=12):
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
                    time_upper = dt[1][0]+75*60
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
                    candidate_trips  += [candidate_trip+[sdist_trip_trend(d,tid,seqs,s_dist,time_max),(d,-1),time_max]]
        candidate_trips = sorted(candidate_trips,key=lambda x: (x[1],-1*x[4],x[6]))
        return {'service_id_%s'%service_id:candidate_trips}
    else:
        if service_id is None: print('service_id was not found for date=%s'%dt[0])
        else:                  print('empty candidates for person trip:%s'%[o_taz,d_taz,service_id])
        return None

#miles per hours twords your destination using the minimum (of all stops)
#stop to stop distance and time bounded by destination time-destination walk time
def sdist_trip_trend(d,tid,seqs,s_dist,time_max):
    a = tid[1]
    x = np.where(seqs[tid[0]][a:][:,1]<=time_max)[0]
    b = a+(x[-1] if len(x)>0 else 0)
    if b>a:
        idx        = np.argmin([s_dist[s[0],d] for s in seqs[tid[0]][a:b]])
        min_idx    = seqs[tid[0]][a:b][idx]
        delta_dist = s_dist[seqs[tid[0]][a-1][0],d]-s_dist[min_idx[0],d]
        delta_time = min_idx[1]-seqs[tid[0]][a-1][1]
    else:
        delta_dist = s_dist[seqs[tid[0]][a-1][0],d]-s_dist[seqs[tid[0]][a][0],d]
        delta_time = seqs[tid[0]][a][1]-seqs[tid[0]][a-1][1]
    return (60*60)*delta_dist/delta_time #miles per hour twords destination

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
        counts[-3]+=1
        D = [[c_tid,c_tdx,seqs[c_tid][c_tdx][0],seqs[c_tid][c_tdx][1],(seqs[c_tid][c_tdx][1]-c_time)+pw[0]*60]]
        return D
    else: #search using [(1)] direct [(2)] stoping [(3)] walking, followed by optimization [(4)]
        if seqs[c_tid][c_tdx][0] in d_stops:
            print('------- found stop=%s -------'%seqs[c_tid][c_tdx][0])
            counts[1]+=1
            D =  [[c_tid,c_tdx,seqs[c_tid][c_tdx][0],seqs[c_tid][c_tdx][1],(seqs[c_tid][c_tdx][1]-c_time)+pw[0]*60]]
            return D
        elif c_tdx<len(seqs[c_tid])-1: #if c_tdx==len(seqs[c_tid] then it is the last stop...
            counts[0]+=1
            D  = [[c_tid,c_tdx,seqs[c_tid][c_tdx][0],seqs[c_tid][c_tdx][1],(seqs[c_tid][c_tdx][1]-c_time)+pw[0]*60]]
            B += [D+DFS(c_tid,c_tdx+1,d_stops,d_time,stops,seqs,graph,s_dist,l_dist,trans)]#directs => depth first
        G,time_a,time_b = graph[c_stop]['out'],c_time,c_time+buff_time*60
        gidx = ([] if len(G)<1 else list(np.where(np.logical_and(G[:,1]>=time_a,G[:,1]<time_b))[0]))
        for i in gidx:
            ws = G[i] #wait at the stop = ws
            if ws[2]!=c_tid and ws[3]+1<len(seqs[ws[2]])-1 and seqs[ws[2]][ws[3]][1]<d_time: #link end time-------------
                counts[-1]+=1
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
                            counts[-2]+=1
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

n_base,d_base = 'ha_network/','ha_demand/'
search_time = ['7:00','8:30']
D = load_network_data(n_base,search_time=search_time) #will run preproccess_network if it was not already
persons = read_person_trip_list(d_base+'csts.txt')
stops,s_idx,s_names,s_dist,w_dist = D['stops'],D['stop_idx'],D['s_names'],D['s_dist'],D['w_dist']
trips,trip_idx,v_dist,calendar    = D['trips'],D['trip_idx'],D['v_dist'],D['calendar']
service_ids = get_processed_service_ids(D)

i,j = 6,0
person_trip = persons[6][1]
C = start_od_search(persons[i][j],w_dist,s_dist,v_dist)
si = list(C.keys())[0]
candidates = C[si]
c_tid,c_tdx,d_stop,d_time = candidates[0][2],candidates[0][3],candidates[0][5][0],candidates[0][6]
d_stops = set([])
for c in candidates: d_stops.add(c[5][0])
seqs,graph,l_dist,l_idx,trans = D[si]['seqs'],D[si]['graph'],D[si]['l_dist'],D[si]['l_idx'],D[si]['trans']
c_stop,c_time = seqs[c_tid][c_tdx-1][0:2]
#
# t_start = time.time()
# buff_time,walk_speed,trans = 10,3,1
#
# counts = {-3:0,-2:0,-1:0,0:0,1:0}
# F = DFS(c_tid,c_tdx,d_stops,d_time,stops,seqs,graph,s_dist,l_dist,trans)
# t_stop = time.time()
# print('starting_stop=%s, starting_time=%s'%(c_stop,c_time))
# print('destination stops=%s, time_limit=%s'%(d_stops,d_time))
# print('python search completed in %s secs'%round(t_stop-t_start,2))
