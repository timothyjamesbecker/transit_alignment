import math
import itertools as it
from libc.math cimport sqrt
from libc.math cimport sin
from libc.math cimport cos
import numpy as np
cimport numpy as np
cimport cython

@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.wraparound(False) #w=[match,extend,delete]
def lcs(int[:,:] c1, int[:,:] c2, float[:,:] dist, float lim=0.5,list w=[1.0,0.0,0.0], int pos=0):
    cdef int i,j,k,n,u,v
    k,u,v = 2,len(c1),len(c2)
    if u<v: u,v,c1,c2 = v,u,c2,c1
    cdef np.ndarray D = np.zeros([k,v+1],dtype=np.float32)
    for i in range(1,u+1):
        for j in range(1,v+1):
            if dist[c1[i-1,pos]][c2[j-1,pos]]<=lim: D[i%k][j] = D[(i-1)%k][j-1]+w[0]*(1.0-dist[c1[i-1,pos]][c2[j-1,pos]]/lim)
            elif D[i%k][j-1] >= D[(i-1)%k][j]:      D[i%k][j] = D[i%k][j-1]+w[1]
            else:                                   D[i%k][j] = D[(i-1)%k][j]+w[2]
    return [D[(u+1)%k][v],v,u]

@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.wraparound(False) #This version needs shared memory arrays: xys, ss, ls, s_dist, l_dist---------------
def shared_lcs(int [:,:] xys, int[:,:,:] ss, int[:] ls, float[:,:] s_dist, float[:,:,:] l_dist,
               float lim=0.5, list w=[1.0,0.0,0.0], bint pos=0):
    cdef int a,i,j,k,u,v,x,y,d1,d2
    k,l = 2,ss.shape[1] #longest sequence in ss is the actual dimension
    cdef np.ndarray D = np.zeros([k,l+1],dtype=np.float32) #reuse a circular DP table
    for a in range(xys.shape[0]):
        x,y = xys[a,:]
        u,v = ls[x],ls[y]
        l_dist[x][x][0] = l_dist[x][x][1] = l_dist[x][x][2] = u*w[0]
        l_dist[y][y][0] = l_dist[y][y][1] = l_dist[y][y][2] = v*w[0]
        if u<v: u,v,x,y = v,u,y,x
        D[:,:] = 0.0
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
#no return for shared memory version--------------------------------------------------------------------------

#computes the LCTS for wiating and walking transfers
#xys is a  nx2 array of trip_id pairs to compute lcswt on,
#ss is the tail-padded shared-memory version of the stop sequences, ls is the length of each sequence
#s_dist is the rectangular stop to stop distance in miles, with lim acting as the limiter
#l_dist is shared memory buffer containing all the pairs that are being computed in ||
#mw is matching weights for mw[0] perfect/approx match, mw[1] delete, mw[2] extended (number of matching stops)
@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.wraparound(False) #This version needs shared memory arrays: xys, ss, ls, s_dist, l_dist------------------------------------------------
def shared_lcts(int [:,:] xys, int[:,:,:] ss, int[:] ls, float[:,:] s_dist, float[:,:,:] l_dist,
                list mw=[1.0,0.0,0.0], float lim=0.5, float walk_speed=3.0, float min_v=1.0):
    cdef int i,j,k,t,u,v,x,y,z,s1,s2,ti_a,ti_b,tj_a,tj_b,w_secs,w_time,buff_time
    k,l = 2,ss.shape[1] #longest sequence in ss is the actual dimension
    cdef np.ndarray D = np.zeros([k,l+1,2],dtype=np.float32) #reuse a circular DP table buffer
    w_secs,buff_time = int((60*60)/walk_speed+0.5),int(lim*(60*60)/walk_speed+0.5)
    for t in range(xys.shape[0]):
        x,y = xys[t,:]
        u,v = ls[x],ls[y]
        l_dist[x][x][0] = l_dist[x][x][1] = l_dist[x][x][2] = l_dist[x][x][3] = u*mw[0] #sequences match themselves
        l_dist[y][y][0] = l_dist[y][y][1] = l_dist[y][y][2] = l_dist[y][y][3] = v*mw[0] #by their nature
        for z in range(2): #tx to ty and then switch to ty to tx ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
            D[:,:,:] = 0.0 # dim(x) x dim(y) x 2: first outer dim is for waiting transfers, second is walking
            for i in range(1,u+1):
                s1,ti_a,ti_b = ss[x,i-1,0],ss[x,i-1,1],ss[x,i-1,1]+buff_time
                for j in range(1,v+1):
                    s2,tj_a,tj_b = ss[y,j-1,0],ss[y,j-1,1],ss[y,j-1,1]+buff_time
                    w_time = int(ti_a+s_dist[s1][s2]*w_secs+0.5)
                    #waiting transfer table--------------------------------------------------------------------------------------------------
                    if s1==s2 and ti_a<=tj_a and ti_b>=tj_a:
                        D[i%k][j][0] = D[(i-1)%k][j-1][0]+mw[0]*(1.0-(1.0-min_v)*(tj_a-ti_a)/buff_time)
                    elif D[i%k][j-1][0]>=D[(i-1)%k][j][0]:
                        D[i%k][j][0] = D[i%k][j-1][0]+mw[1]
                    else:
                        D[i%k][j][0] = D[(i-1)%k][j][0]+mw[2]
                    #walking transfer table--------------------------------------------------------------------------------------------------
                    if s_dist[s1][s2]<lim and w_time>=tj_a and w_time<=tj_b:
                        D[i%k][j][1] = D[(i-1)%k][j-1][1]+mw[0]*(1.0-(1.0-min_v)*s_dist[s1][s2]/lim)
                    elif D[i%k][j-1][1]>=D[(i-1)%k][j][1]:
                        D[i%k][j][1] = D[i%k][j-1][1]+mw[1]
                    else:
                        D[i%k][j][1] = D[(i-1)%k][j][1]+mw[2]
            l_dist[x][y][0] = D[(u+1)%k][v][0]
            l_dist[x][y][1] = D[(u+1)%k][v][1]
            l_dist[x][y][2] = v*mw[0]
            l_dist[x][y][3] = u*mw[0]
            u,v,x,y = v,u,y,x #switch :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#no return for shared memory version---------------------------------------------------------------------------------------------------------


@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.wraparound(False)
def pairwise_rectangular(list stops, double max_dist=120.0, int lon=0, int lat=1, bint mi=True):
    cdef int i,j,k,n
    cdef double a,b,c,d
    #cdef dict D = {}
    k,n,c = 1,len(stops),math.pi/180.0
    cdef np.ndarray D = np.zeros([n,n],dtype=np.float32)
    if mi:
        for i in range(n):
            for j in range(k,n,1):
                a = c*(stops[j][1][lon]-stops[i][1][lon])*cos(0.5*c*(stops[j][1][lat]+stops[i][1][lat]))
                b = c*(stops[j][1][lat]-stops[i][1][lat])
                d = 6371*sqrt(a*a + b*b)*0.621371#distance in miles is default
                if d<=max_dist: D[i][j] = D[j][i] = <float>d
            k += 1
    else:
        for i in range(n):
            for j in range(k,n,1):
                a = c*(stops[j][1][lon]-stops[i][1][lon])*cos(0.5*c*(stops[j][1][lat]+stops[i][1][lat]))
                b = c*(stops[j][1][lat]-stops[i][1][lat])
                d = 6371*sqrt(a*a + b*b) #distance in km is the alternate
                if d<=max_dist: D[i][j] = D[j][i] = <float>d
            k += 1
    return D #return 32 bit float distances

@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.wraparound(False)
def shape_dist_correction(list stops, dict E, int NN=2, float max_nn=0.25, verbose=True):
    cdef int a,b,i,j,m,x,y
    cdef bint one_link,two_link
    cdef float d_ab,d_ax,d_by
    cdef tuple k,q
    cdef list e,ms,st_a,st_b,u,v
    cdef dict C = {}
    ms,e = [],[0,0] #grab NN less than shape_max_dist
    for a in range(len(stops)):
        m = 0
        while m<len(stops[a][NN]) and stops[a][NN][m][1]<=max_nn: m+=1
        m -= 1
        ms += [m]
    C = {}
    for a in range(len(stops)): #one NN point away from s_a
        if ms[a]>=0: #filter out stops that don't have a close NN to estimate with
            for i in range(len(stops[a][NN])): #stop_id b
                b,d_ab = stops[a][NN][i]
                k = ((a,b) if a<=b else (b,a))
                one_link = False
                if k not in E:
                    for j in range(ms[a]+1): #look for a NN stop to s_a that has a trip to s_b
                        if i!=j:
                            x,d_ax = stops[a][NN][j]
                            q = ((x,b) if x<=b else (b,x))
                            if q in E and d_ax<d_ab:
                                if (a,i) not in C:
                                    C[(a,i)] = E[q]+d_ax
                                    one_link = True
                                elif C[(a,i)]>E[q]+d_ax: C[(a,i)] = E[q]+d_ax
                    if one_link: e[0] += 1
                    else: #get all the stops[a][NN][:ms[a]] and stops[b][NN][:ms[b]] combinations
                        if ms[b]>=0: #ms[a] and ms[b] >=0
                            two_link = False
                            st_a,st_b = stops[a][NN][:ms[a]],stops[b][NN][:ms[b]]
                            for u,v in sorted([(u,v) for u in st_a for v in st_b],key=lambda x: x[0]+x[1]):
                                if u[0]!=v[0]:
                                    x,d_ax,y,d_by = u[0],u[1],v[0],v[1]
                                    q = ((x,y) if x<=y else (y,x))
                                    if q in E and d_ax+d_by<d_ab:
                                        if (a,i) not in C:            C[(a,i)] = d_ax+E[q]+d_by
                                        elif C[(a,i)]>d_ax+E[q]+d_by: C[(a,i)] = d_ax+E[q]+d_by
                                        two_link = True
                            if two_link: e[1] += 1
    if verbose: print('%s shape distance corrections made'%(e))
    return C

@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.wraparound(False)
def scan_trip_trans(int[:,:] S, int start_i, int end_i, float[:,:] s_dist,
                    int sit=10*60, int walk=10*60, float speed=3.0):
    cdef int a,b,i,j,k,x,y,s,w,s_time,w_time,w_secs,w_dist,idx,idx_a,idx_b
    cdef tuple tx,ty
    cdef set set_ab
    cdef dict T,A,B
    w_secs = int(round((60*60)/speed))
    T = {'walk':{},'sit':{}}
    i,j,s,w = start_i,start_i,start_i,start_i
    while i<min(end_i,len(S)):
        s_time,w_time = S[i,0]+sit,S[i,0]+walk
        while j<len(S) and S[j,0]==S[i,0]: j += 1
        s = j
        while s<len(S) and S[s,0]<=s_time: s += 1
        A,B = {},{}
        for x in range(i,j,1):                      #all the stops encountered from time i=>j
            if S[x,1] in A: A[S[x,1]] += [x]
            else:           A[S[x,1]]  = [x]
        for x in range(j,s,1):                      #all the stops encountered from time j=>s
            if S[x,1] in B: B[S[x,1]] += [x]
            else:           B[S[x,1]]  = [x]
        for idx in A: #for the == pairs in A with 0 time diff
            k = 1
            for x in range(len(A[idx])):
                a = A[idx][x]
                for y in range(k,len(A[idx]),1):
                    b = A[idx][y]
                    if S[b,4]>=0: #trip off transfers to b when b is a terminal
                        tx,ty = (S[a,2],S[a,3]),(S[b,2],S[b,3])
                        if tx in T['sit']: T['sit'][tx].add(ty+(0,))
                        else:              T['sit'][tx] = set([ty+(0,)])
                        if ty in T['sit']: T['sit'][ty].add(tx+(0,))
                        else:              T['sit'][ty] = set([tx+(0,)])
                k += 1
        for idx in set(A).intersection(set(B)): #sids are prematched and x goes to y in time
            for x in range(len(A[idx])):
                a = A[idx][x]
                for y in range(len(B[idx])):
                    b = B[idx][y]
                    if S[b,4]>=0:
                        tx,ty = (S[a,2],S[a,3]),(S[b,2],S[b,3],S[b,0]-S[i,0])
                        if tx in T['sit']: T['sit'][tx].add(ty)
                        else:              T['sit'][tx] = set([ty])
        if walk==sit: w = s
        else:         w = j
        while w<len(S) and S[w,0]<=w_time+s_time: w += 1  #w is 1 past walk buffer
        for x in range(j,w,1):                      #all the stops encountered from time j=>w
            if S[x,1] in B: B[S[x,1]] += [x]
            else:           B[S[x,1]]  = [x]
        #walking buffer---------------------------------------------------
        B = {b:B[b] for b in set(B).difference(set(A))} #set of stops in B that are not in A
        for idx_a in A: #check the NN of the stops in A with the stops in B
            for idx_b in B:
                w_dist = int(round(s_dist[idx_a,idx_b]*w_secs))
                if w_dist<=walk: #time bound is secinds walking from a to b
                    time_a = S[i,0]+w_dist
                    for x in range(len(A[idx_a])):
                        a = A[idx_a][x]
                        for y in range(len(B[idx_b])): #get to the stop before it arrives and don't wait too long
                            b = B[idx_b][y]
                            if S[b,4]>=0 and time_a>=S[b,0] and time_a<=S[b,0]+sit: #can't be last stop
                                tx,ty = (S[a,2],S[a,3]),(S[b,2],S[b,3],S[b,0]-S[i,0])
                                if tx in T['walk']: T['walk'][tx].add(ty)
                                else:               T['walk'][tx] = set([ty])
        i = j
    return T #T = {'sit':{(tid_a,tdx_a):[(tid_b,tdx_b,time_ab), ...], ...},'walk':{...}}

@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.wraparound(False)
def DFS(int c_tid, int c_tdx, set d_stops, int d_time, list stops, dict seqs, dict graph, float[:,:] s_dist, float[:,:,:] l_dist,
        int trans=5,int buff_time=10, int walk_speed=3, list pw=[0,10,20], bint verbose=False):
    cdef int i,j,c_stop,c_time,out_sdx,in_sdx,w_secs,wt_stop,time_a,time_b
    cdef list D,B,gidx,scores
    cdef np.ndarray G,ws,wt
    c_stop,c_time,out_sdx,in_sdx = seqs[c_tid][c_tdx-1] #will be graph indexes for in and out links
    D,B,scores = [],[],[] #L is for prepending links, F is for managing returning branches that are collected
    if seqs[c_tid][c_tdx][1]>=d_time or trans<=0: #this path has failed=> exheded time or number of transfers
        D = [[c_tid,c_tdx,seqs[c_tid][c_tdx][0],seqs[c_tid][c_tdx][1],(seqs[c_tid][c_tdx][1]-c_time)+pw[0]*60]]
        return D
    else: #search using [(1)] direct [(2)] stoping [(3)] walking, followed by optimization [(4)]
        if seqs[c_tid][c_tdx][0] in d_stops:
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
