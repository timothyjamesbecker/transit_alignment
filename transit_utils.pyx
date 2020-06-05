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
def pairwise_lcs(int[:,:,:] ss, list ls, float[:,:] dist, float lim=0.5,list w=[1.0,0.0,0.0], int pos=0):
    cdef int i,j,k,n,u,v,x,y,z,d1,d2
    cdef np.ndarray D,L
    n,l,z,k = len(ss),ss.shape[0],1,2
    L = np.zeros([n,n,2],dtype=np.float32)
    D = np.zeros([k,l+1],dtype=np.float32)
    for x in range(n):
        u = ls[x]
        for y in range(z,n,1):
            v = ls[y]
            D[:] = 0.0
            for i in range(1,u+1):
                d1 = ss[x,i-1,pos]
                for j in range(1,v+1):
                    d2 = ss[y,j-1,pos]
                    if dist[d1][d2]<=lim:              D[i%k][j] = D[(i-1)%k][j-1]+w[0]*(1.0-dist[d1][d2]/lim)
                    elif D[i%k][j-1] >= D[(i-1)%k][j]: D[i%k][j] = D[i%k][j-1]+w[1]
                    else:                              D[i%k][j] = D[(i-1)%k][j]+w[2]
            L[x][y] = L[x][y] = [D[u%k][v],u*w[0]]
        L[x][x] = [u*w[0],u*w[0]]
        z += 1
    return L

@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.wraparound(False) #This version needs shared memory arrays: xys, ss, ls, s_dist, l_dist---------------
def shared_lcs(int [:,:] xys, int[:,:,:] ss, int[:] ls, float[:,:] s_dist, float[:,:,:] l_dist,
               float lim=0.5, list w=[1.0,0.0,0.0], int pos=0):
    cdef int a,i,j,k,u,v,x,y,d1,d2
    k,l = 1,ss.shape[1] #longest sequence in ss is the actual dimension
    cdef np.ndarray D = np.zeros([k,l+1],dtype=np.float32) #reuse a circular DP table
    for a in range(xys.shape[0]):
        x,y = xys[a,:]
        u,v = ls[x],ls[y]
        l_dist[x][x][0] = l_dist[x][x][1] = u*w[0]
        l_dist[y][y][0] = l_dist[y][y][1] = v*w[0]
        if u<v: u,v,x,y = v,u,y,x
        D[:] = 0.0
        for i in range(1,u+1):
            d1 = ss[x,i-1,pos]
            for j in range(1,v+1):
                d2 = ss[y,j-1,pos]
                if s_dist[d1][d2]<=lim:            D[i%k][j] = D[(i-1)%k][j-1]+w[0]*(1.0-s_dist[d1][d2]/lim)
                elif D[i%k][j-1] >= D[(i-1)%k][j]: D[i%k][j] = D[i%k][j-1]+w[1]
                else:                              D[i%k][j] = D[(i-1)%k][j]+w[2]
        l_dist[x][y][0] = l_dist[y][x][0] = D[u%k][v]
        l_dist[x][y][1] = l_dist[y][x][1] = u*w[0]
#no return for shared memory version--------------------------------------------------------------------------


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
