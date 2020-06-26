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
    return [D[u%k][v],v,u]

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
def sdist_seq_trend(int [:,:] ss, float [:,:] s_dist, int tdx, int sid, int time_max):
    cdef int i,edx,min_idx,delta_time
    cdef float min_val,delta_dist
    edx = tdx
    while edx<len(ss) and ss[edx][1]<=time_max: edx += 1
    if edx>tdx:
        min_val = np.finfo(np.float32).max
        for i in range(tdx,edx,1):
            if s_dist[ss[i][0],sid]<min_val:
                min_val,min_idx = s_dist[ss[i][0],sid],i
        delta_dist = s_dist[ss[tdx][0],sid]-min_val
        delta_time = ss[min_idx][1]-ss[tdx][1]+1
    else: delta_dist,delta_time = 0.0,1
    return (60*60)*delta_dist/delta_time

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
    T = {}
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
                        if tx in T: T[tx].add(ty+(-1,0))
                        else:       T[tx] = set([ty+(-1,0)])
                        if ty in T: T[ty].add(tx+(-1,0))
                        else:       T[ty] = set([tx+(-1,0)])
                k += 1
        for idx in set(A).intersection(set(B)): #sids are prematched and x goes to y in time
            for x in range(len(A[idx])):
                a = A[idx][x]
                for y in range(len(B[idx])):
                    b = B[idx][y]
                    if S[b,4]>=0:
                        tx,ty = (S[a,2],S[a,3]),(S[b,2],S[b,3],-1,S[b,0]-S[i,0])
                        if tx in T: T[tx].add(ty)
                        else:       T[tx] = set([ty])
        if walk==sit: w = s
        else:         w = j
        while w<len(S) and S[w,0]<=w_time+sit: w += 1  #w is 1 past walk buffer
        for x in range(j,w,1):                         #all the stops encountered from time j=>w
            if S[x,1] in B: B[S[x,1]] += [x]
            else:           B[S[x,1]]  = [x]
        #walking buffer---------------------------------------------------
        B = {b:B[b] for b in set(B).difference(set(A))} #set of stops in B that are not in A
        for idx_a in A: #check the NN of the stops in A with the stops in B
            for idx_b in B:
                w_dist = int(round(s_dist[idx_a,idx_b]*w_secs))
                if w_dist<=walk: #time bound is seconds walking from a to b
                    time_a = S[i,0]+w_dist
                    for x in range(len(A[idx_a])):
                        a = A[idx_a][x]
                        if S[a,3]>=0: #don't allow walking transfers from the first stop
                            for y in range(len(B[idx_b])): #get to the stop before it arrives and don't wait too long
                                b = B[idx_b][y]                          #don't allow a transfer to a last stop
                                if S[b,4]>=0 and S[b,0]>=time_a and S[b,0]<=time_a+sit: #S[b,4]=-1 => last stop
                                    tx,ty = (S[a,2],S[a,3]),(S[b,2],S[b,3],-2,S[b,0]-S[i,0])
                                    if tx in T: T[tx].add(ty)
                                    else:       T[tx] = set([ty])
        i = j
    return T #T = {'sit':{(tid_a,tdx_a):[(tid_b,tdx_b,time_ab), ...], ...},'walk':{...}}

# @cython.boundscheck(False)
# @cython.nonecheck(False)
# @cython.wraparound(False)
# def scan_trip_trans(int[:,:] S, int start_i, int end_i, float[:,:] s_dist,
#                     int sit=10*60, int walk=10*60, float speed=3.0):
#     cdef int a,b,i,j,k,x,y,s,w,s_time,w_time,w_secs,w_dist,idx,idx_a,idx_b
#     cdef tuple tx,ty
#     cdef set set_ab
#     cdef dict T,A,B
#     w_secs = int(round((60*60)/speed))
#     T = {'sit':{},'walk':{}}
#     i,j,s,w = start_i,start_i,start_i,start_i
#     while i<min(end_i,len(S)):
#         s_time,w_time = S[i,0]+sit,S[i,0]+walk
#         while j<len(S) and S[j,0]==S[i,0]: j += 1
#         s = j
#         while s<len(S) and S[s,0]<=s_time: s += 1
#         A,B = {},{}
#         for x in range(i,j,1):                      #all the stops encountered from time i=>j
#             if S[x,1] in A: A[S[x,1]] += [x]
#             else:           A[S[x,1]]  = [x]
#         for x in range(j,s,1):                      #all the stops encountered from time j=>s
#             if S[x,1] in B: B[S[x,1]] += [x]
#             else:           B[S[x,1]]  = [x]
#         for idx in A: #for the == pairs in A with 0 time diff
#             k = 1
#             for x in range(len(A[idx])):
#                 a = A[idx][x]
#                 for y in range(k,len(A[idx]),1):
#                     b = A[idx][y]
#                     if S[b,4]>=0: #trip off transfers to b when b is a terminal
#                         tx,ty = (S[a,2],S[a,3]),(S[b,2],S[b,3])
#                         if tx in T['sit']: T['sit'][tx].add(ty+(0,))
#                         else:              T['sit'][tx] = set([ty+(0,)])
#                         if ty in T['sit']: T['sit'][ty].add(tx+(0,))
#                         else:              T['sit'][ty] = set([tx+(0,)])
#                 k += 1
#         for idx in set(A).intersection(set(B)): #sids are prematched and x goes to y in time
#             for x in range(len(A[idx])):
#                 a = A[idx][x]
#                 for y in range(len(B[idx])):
#                     b = B[idx][y]
#                     if S[b,4]>=0:
#                         tx,ty = (S[a,2],S[a,3]),(S[b,2],S[b,3],S[b,0]-S[i,0])
#                         if tx in T['sit']: T['sit'][tx].add(ty)
#                         else:              T['sit'][tx] = set([ty])
#         if walk==sit: w = s
#         else:         w = j
#         while w<len(S) and S[w,0]<=w_time+s_time: w += 1  #w is 1 past walk buffer
#         for x in range(j,w,1):                      #all the stops encountered from time j=>w
#             if S[x,1] in B: B[S[x,1]] += [x]
#             else:           B[S[x,1]]  = [x]
#         #walking buffer---------------------------------------------------
#         B = {b:B[b] for b in set(B).difference(set(A))} #set of stops in B that are not in A
#         for idx_a in A: #check the NN of the stops in A with the stops in B
#             for idx_b in B:
#                 w_dist = int(round(s_dist[idx_a,idx_b]*w_secs))
#                 if w_dist<=walk: #time bound is secinds walking from a to b
#                     time_a = S[i,0]+w_dist
#                     for x in range(len(A[idx_a])):
#                         a = A[idx_a][x]
#                         for y in range(len(B[idx_b])): #get to the stop before it arrives and don't wait too long
#                             b = B[idx_b][y]
#                             if S[b,4]>=0 and time_a>=S[b,0] and time_a<=S[b,0]+sit: #can't be last stop
#                                 tx,ty = (S[a,2],S[a,3]),(S[b,2],S[b,3],S[b,0]-S[i,0])
#                                 if tx in T['walk']: T['walk'][tx].add(ty)
#                                 else:               T['walk'][tx] = set([ty])
#         i = j
#     return T #T = {'sit':{(tid_a,tdx_a):[(tid_b,tdx_b,time_ab), ...], ...},'walk':{...}}

@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.wraparound(False)
cdef list sub_seq_leg(dict seqs, int tid, int tdx, int sid, dict pw={0:0,-1:10*60,-2:20*60}):
    cdef a,i,j
    cdef list L = []
    i = -1
    for a in range(tdx,len(seqs[tid]),1):
        if seqs[tid][a][0]==sid: i = a; break
    if i>=0:
        L = [[tid,j,seqs[tid][j][0],seqs[tid][j][1],pw[0]] for j in range(tdx,i+1,1)]
    return L

@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.wraparound(False)
cdef list sample_branches(dict T, dict seqs,tuple s,float p=1.0):
    cdef a,b
    cdef tuple k,t
    cdef list B = []
    cdef list b_idx
    cdef set C = set([])
    cdef dict A = {}
    for a in range(s[1]+1,len(seqs[s[0]]),1): #can't do a transfer from a transfer
        if (s[0],a) in T:
            for t in T[(s[0],a)]:
                if t[0:3] not in A or t[3]<A[t[0:3]][0]:
                    A[t[0:3]] = [t[3],a]
    for k in A: C.add((A[k][1],k+(A[k][0],)))
    if len(C)>1:
        b_idx = list(np.random.choice(range(len(C)),max(1,int(len(C)*p)),replace=False))
        B = list(C)
        B = sorted([B[b] for b in b_idx])
    return B

@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.wraparound(False)
def RTS_FULL(list C,dict T,dict F,dict seqs,dict pw={0:0,-1:10*60,-2:20*60},int t_max=4,list t_p=[1.0,1.0,1.0,1.0]):
    cdef int sid,a1,a2,a3,a4
    cdef tuple s0,s1,s2,s3,s4
    cdef list L
    cdef np.ndarray Y
    cdef dict X = {i:{} for i in range(t_max+1)}
    for i in range(len(C)):
        s0,sid = (C[i][0],C[i][1]),C[i][2]
        if s0[0:2] in F and sid in F[s0[0:2]]:
            L = sub_seq_leg(seqs,s0[0],s0[1],sid,pw)
            Y  = np.array(L,dtype=np.int32)
            X[0][tuple(Y[:,2])] = Y
        elif len(X[0])<1 and t_max>0: #will get up to 1-transfer more than optimal
            for a1,s1 in sample_branches(T,seqs,s0,p=t_p[0]): #a1 is the last stop before transfer-1
                if s1[0:2] in F and sid in F[s1[0:2]]:
                    L  = sub_seq_leg(seqs,s0[0],s0[1],seqs[s0[0]][a1][0],pw)                #before transfer-1
                    L += [[s1[2],0,seqs[s1[0]][s1[1]][0],seqs[s1[0]][s1[1]][1],pw[s1[2]]]]  #transfer-1
                    L += sub_seq_leg(seqs,s1[0],s1[1],sid,pw)                               #before destination
                    Y  = np.array(L,dtype=np.int32)                                         #np array
                    X[1][tuple(Y[:,2])] = Y
                elif len(X[1])<1 and t_max>1:
                    for a2,s2 in sample_branches(T,seqs,s1,p=t_p[1]): #a2 is the last stop before transfer-2
                        if s2[0:2] in F and sid in F[s2[0:2]]:
                            L  = sub_seq_leg(seqs,s0[0],s0[1],seqs[s0[0]][a1][0],pw)                #before transfer-1
                            L += [[s1[2],0,seqs[s1[0]][s1[1]][0],seqs[s1[0]][s1[1]][1],pw[s1[2]]]]  #transfer-1
                            L += sub_seq_leg(seqs,s1[0],s1[1],seqs[s1[0]][a2][0],pw)                #before transfer-2
                            L += [[s2[2],0,seqs[s2[0]][s2[1]][0],seqs[s2[0]][s2[1]][1],pw[s2[2]]]]  #transfer-2
                            L += sub_seq_leg(seqs,s2[0],s2[1],sid,pw)                               #before destination
                            Y  = np.array(L,dtype=np.int32)                                         #np array
                            X[2][tuple(Y[:,2])] = Y
                        elif len(X[2])<1 and t_max>2: #if len(X[1])<1
                            for a3,s3 in sample_branches(T,seqs,s2,p=t_p[2]):
                                if s3[0:2] in F and sid in F[s3[0:2]]:
                                    L  = sub_seq_leg(seqs,s0[0],s0[1],seqs[s0[0]][a1][0],pw)                #before transfer-1
                                    L += [[s1[2],0,seqs[s1[0]][s1[1]][0],seqs[s1[0]][s1[1]][1],pw[s1[2]]]]  #transfer-1
                                    L += sub_seq_leg(seqs,s1[0],s1[1],seqs[s1[0]][a2][0],pw)                #before transfer-2
                                    L += [[s2[2],0,seqs[s2[0]][s2[1]][0],seqs[s2[0]][s2[1]][1],pw[s2[2]]]]  #transfer-2
                                    L += sub_seq_leg(seqs,s2[0],s2[1],seqs[s2[0]][a3][0],pw)                #before transfer-3
                                    L += [[s3[2],0,seqs[s3[0]][s3[1]][0],seqs[s3[0]][s3[1]][1],pw[s3[2]]]]  #transfer-3
                                    L += sub_seq_leg(seqs,s3[0],s3[1],sid,pw)                               #before destination
                                    Y  = np.array(L,dtype=np.int32)                                         #np array
                                    X[3][tuple(Y[:,2])] = Y
                                elif len(X[3])<1 and t_max>3: #if len(X[2])<1
                                    for a4,s4 in sample_branches(T,seqs,s3,p=t_p[3]):
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
                                            X[4][tuple(Y[:,2])] = Y
    return X
