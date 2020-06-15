#!/usr/bin/env python

import argparse
import sys
import time
import ctypes
import gzip
import pickle
import itertools as it
import numpy as np
import multiprocessing as mp
import read_utils as ru
import transit_utils as tu

des = """Preproccess Network Tool, Copyright (C) 2020 Timothy James Becker"""
parser = argparse.ArgumentParser(description=des,formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument('--in_path',type=str,help='GTFS input directory\t[None]')
parser.add_argument('--out_dir',type=str,help='output directory\t[None]')
parser.add_argument('--date',type=str,help='date to preprocess with, leave out to iterate on all\t[all from network]')
parser.add_argument('--walk',type=float,help='walking distance in miles to use as upper bound for access and transfers\t[0.5]')
parser.add_argument('--time',type=str,help='comma seperated time range. Leave out to use all\t[0:00,32:00]')
parser.add_argument('--cpus',type=int,help='number of parallel cores for pairwise LCSWT\t[all]')
# parser.add_argument('--test',action='store_true',help='')
args = parser.parse_args()
#--------------------------------------
if args.in_path is not None:
    n_base = args.in_path
else: raise IOError
if args.out_dir is not None:
    out_dir = args.out_dir
else: out_dir = n_base
if args.date is not None:
    search_date = args.date
else: search_date = None
if args.time is not None:
    search_time = args.time.split(',')
else: search_time = [0,32*60*60]
if args.walk is not None:
    walk_buffer = args.walk
else: walk_buffer = 0.5
if args.cpus is not None:
    cpus = args.cpus
else: cpus = mp.cpu_count()

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
def affine_sim(c1,c2,w=[0,1,1,0.9,0.05],s_pos=0):#match,miss,gap,space,scale
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
        y2 = c2[i-1,s_pos]
        for x in range(0,u+1): P[x] = C[x]
        C[0] = w[2]+w[3]*i
        I = np.iinfo(int).max
        for j in range(1,u+1):
            y1 = c1[j-1,s_pos]
            if j <= v:   I = min(I,C[j-1]+w[2])+w[3]
            else:        I = min(I,C[j-1]+w[2]*w[4])+w[3]*w[4]
            D[j] = min(D[j],P[j]+w[2])+w[3]
            if y2 == y1: M = P[j-1]+w[0] #match
            else:        M = P[j-1]+w[1] #miss
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

def lcs(c1,c2,w=[1,0,0],s_pos=0):
    u,v,k, = len(c1),len(c2),2
    if u<v: u,v,c1,c2 = v,u,c2,c1
    D = np.zeros((k,v+1),dtype=int)
    for i in range(1,u+1):      #rows--------------
        for j in range(1,v+1):  #columns-----------
            if c1[i-1,s_pos]==c2[j-1,s_pos]:   D[i%k][j] = D[(i-1)%k][j-1]+w[0]  #diagnal
            elif D[i%k][j-1] >= D[(i-1)%k][j]: D[i%k][j] = D[i%k][j-1]+w[1]      #left
            else:                              D[i%k][j] = D[(i-1)%k][j]+w[2]    #top
    return [D[u%k][v],v,u]

def shared_lcs(xys,ss,ls,s_dist,l_dist,lim=0.5,w=[1.0,0.0,0.0],pos=0):
    k,l = 2,ss.shape[1] #longest sequence in ss is the actual dimension
    D = np.zeros([k,l+1],dtype=np.float32) #reuse a circular DP table
    for a in range(xys.shape[0]):
        x,y = xys[a,:]
        u,v = ls[x],ls[y]
        l_dist[x][x][0] = l_dist[x][x][1] = l_dist[x][x][2] = u*w[0]
        l_dist[y][y][0] = l_dist[y][y][1] = l_dist[y][y][2] = v*w[0]
        D[:,:] = 0.0
        if u<v: u,v,x,y = v,u,y,x
        for i in range(1,u+1):
            d1 = ss[x,i-1,pos]
            for j in range(1,v+1):
                d2 = ss[y,j-1,pos]
                if s_dist[d1][d2]<=lim:            D[i%k][j] = D[(i-1)%k][j-1]+w[0]*(1.0-s_dist[d1][d2]/lim)
                elif D[i%k][j-1] >= D[(i-1)%k][j]: D[i%k][j] = D[i%k][j-1]+w[1]
                else:                              D[i%k][j] = D[(i-1)%k][j]+w[2]
        l_dist[x][y][0] = l_dist[y][x][0] = D[(u+1)%k][v]
        l_dist[x][y][1] = l_dist[y][x][1] = v*w[0]
        l_dist[x][y][2] = l_dist[y][x][2] = u*w[0]

#part is the series of [i,j] index pairs into ss
def parallel_lcs(xys):
    tu.shared_lcs(xys,ss,ls,s_dist,l_dist)
    return True

def jaccard_sim(c1,c2):
    C1,C2 = set(c1),set(c2)
    return [len(C1.intersection(C2)),len(C1.union(C2))]

#---------------------------------------------------------------------------------------------

if __name__ == '__main__':
    print('starting to preprocess GTFS data folder = %s'%n_base)
    stops,stop_idx,s_names,s_dist = ru.read_gtfs_stops(n_base,max_miles=walk_buffer) #{enum_stop_id:[stop_id,stop_name,x,y,[NN<=10.0]], ... }
    v_dist         = ru.gtfs_stop_time_shape_dist(n_base,stop_idx) #in vehicle distances
    trips,trip_idx = ru.read_gtfs_trips(n_base) #trips=[trip_id,trip_name,route_id,service_id,direction]
    w_dist         = ru.read_walk_access(n_base,stop_idx,walk_buff=walk_buffer)
    calendar       = ru.read_gtfs_calendar(n_base) #{service_id,[start,end],[mon,tue,wed,thu,fri,sat,sun])
    DATA = {'stops':stops,'s_names':s_names,'stop_idx':stop_idx,'s_dist':s_dist,'w_dist':w_dist,
            'trips':trips,'trip_idx':trip_idx,'v_dist':v_dist,'calendar':calendar}
    if search_date is not None and type(search_date) is str:
        service_id  = ru.get_service_id(calendar,search_date)
        seqs,graph  = ru.read_gtfs_seqs(n_base,stop_idx,trips,trip_idx,calendar,service_id=service_id)
        print('%s total trips for date=%s'%(len(seqs),search_date))
        seqs,graph  = ru.filter_seqs(seqs,time_window=search_time)
        print('%s total trips left after filtering using time_window=%s'%(len(seqs),search_time))
        #|| lcs in cython--------------------------------------------------------------------------
        ss,ls,seq_idx = ru.seqs_to_ndarray(seqs)
        #globally bound mp.ctypes arrays--------------------------------------------------------------------------
        ss_array  = np.ctypeslib.as_array(mp.Array(ctypes.c_int,(ss.shape[0]*ss.shape[1]*ss.shape[2]),lock=False))
        ss_array  = ss_array.reshape(ss.shape[0],ss.shape[1],ss.shape[2])
        ss_array[:,:,:] = ss[:,:,:]
        ss = ss_array
        ls_array  = np.ctypeslib.as_array(mp.Array(ctypes.c_int,(ls.shape[0]),lock=False))
        ls_array  = ls_array.reshape(ls.shape[0])
        ls_array[:] = ls[:]
        ls = ls_array
        s_array  = np.ctypeslib.as_array(mp.Array(ctypes.c_float,(s_dist.shape[0]*s_dist.shape[1]),lock=False))
        s_array  = s_array.reshape(s_dist.shape[0],s_dist.shape[1])
        s_array[:,:] = s_dist[:,:]
        s_dist = s_array
        l_dist    = np.ctypeslib.as_array(mp.Array(ctypes.c_float,(len(ls)**2)*3,lock=False))
        l_dist    = l_dist.reshape(len(ls),len(ls),3)
        #globally bound mp.ctypes arrays--------------------------------------------------------------------------

        cpus = mp.cpu_count()
        xys = sorted([[x,y] for x,y in it.combinations(range(len(ls)),2)])
        partitions,n = [],len(xys)//cpus
        for i in range(cpus): partitions     += [xys[i*n:(i+1)*n]]
        if len(xys)%cpus>0:   partitions[-1] += xys[-1*(len(xys)%cpus):]
        for i in range(len(partitions)):
            temp = np.zeros((len(partitions[i]),2),dtype=np.int32)
            temp[:] = [k for k in partitions[i]]
            partitions[i] = temp
        #fire it up---------------------------------------------------
        t_start = time.time()
        print('starting || computation')
        p1 = mp.Pool(processes=cpus)
        for i in range(len(partitions)):
            p1.apply_async(parallel_lcs,args=(partitions[i],))
        p1.close()
        p1.join()
        t_stop = time.time()
        print('ending || computation in %s sec'%round(t_stop-t_start,2))
        # || lcs in cython--------------------------------------------------------------------------
        DATA['service_id_%s'%service_id] = {'seqs':seqs,'graph':graph,'l_dist':l_dist,'l_idx':seq_idx}
    else:
        for service_id in sorted(list(calendar.keys())):
            seqs,graph  = ru.read_gtfs_seqs(n_base,stop_idx,trips,trip_idx,calendar,service_id=service_id)
            print('%s total trips for service_id=%s %s'%(len(seqs),service_id,calendar[service_id]))
            seqs,graph  = ru.filter_seqs(seqs,time_window=search_time)
            print('%s total trips left after filtering using time_window=%s'%(len(seqs),search_time))
            #|| lcs in cython--------------------------------------------------------------------------
            ss,ls,seq_idx = ru.seqs_to_ndarray(seqs)
            #globally bound mp.ctypes arrays--------------------------------------------------------------------------
            ss_array  = np.ctypeslib.as_array(mp.Array(ctypes.c_int,(ss.shape[0]*ss.shape[1]*ss.shape[2]),lock=False))
            ss_array  = ss_array.reshape(ss.shape[0],ss.shape[1],ss.shape[2])
            ss_array[:,:,:] = ss[:,:,:]
            ss = ss_array
            ls_array  = np.ctypeslib.as_array(mp.Array(ctypes.c_int,(ls.shape[0]),lock=False))
            ls_array  = ls_array.reshape(ls.shape[0])
            ls_array[:] = ls[:]
            ls = ls_array
            s_array  = np.ctypeslib.as_array(mp.Array(ctypes.c_float,(s_dist.shape[0]*s_dist.shape[1]),lock=False))
            s_array  = s_array.reshape(s_dist.shape[0],s_dist.shape[1])
            s_array[:,:] = s_dist[:,:]
            s_dist = s_array
            l_dist    = np.ctypeslib.as_array(mp.Array(ctypes.c_float,(len(ls)**2)*3,lock=False))
            l_dist    = l_dist.reshape(len(ls),len(ls),3)
            #globally bound mp.ctypes arrays--------------------------------------------------------------------------
            cpus = mp.cpu_count()
            xys = sorted([[x,y] for x,y in it.combinations(range(len(ls)),2)])
            if len(xys)>cpus:
                partitions,n = [],len(xys)//cpus
                for i in range(cpus): partitions     += [xys[i*n:(i+1)*n]]
                if len(xys)%cpus>0:   partitions[-1] += xys[-1*(len(xys)%cpus):]
                for i in range(len(partitions)):
                    temp = np.zeros([len(partitions[i]),2],dtype=np.int32)
                    temp[:] = [k for k in partitions[i]]
                    partitions[i] = temp
            else:
                partitions = [np.zeros([len(xys),2],dtype=np.int32)]
                for i in range(len(xys)):
                    partitions[0][i][:] = xys[i]
            #fire it up---------------------------------------------------
            t_start = time.time()
            print('starting || computation')
            p1 = mp.Pool(processes=cpus)
            for i in range(len(partitions)):
                p1.apply_async(parallel_lcs,args=(partitions[i],))
            p1.close()
            p1.join()
            t_stop = time.time()
            print('ending || computation in %s sec'%round(t_stop-t_start,2))
            # || lcs in cython--------------------------------------------------------------------------
            DATA['service_id_%s'%service_id] = {'seqs':seqs,'graph':graph,'l_dist':l_dist,'l_idx':seq_idx}
    print('saving network data structures to disk (distance matrices, sequences and graph)')
    with gzip.GzipFile(out_dir+'/network.pickle.gz','wb') as f:
        pickle.dump(DATA,f)
        print('finished writing network data structures, exiting...')