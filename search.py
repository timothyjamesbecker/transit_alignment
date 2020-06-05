import os
import datetime
import gzip
import pickle
import numpy as np

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

n_base,d_base = 'ha_network/','ha_demand/' #may need to glob them now...
with gzip.GzipFile(n_base+'/network.pickle.gz','rb') as f:
    D = pickle.load(f)
persons = read_person_trip_list(d_base+'csts.txt')
stops,s_names,trips,seqs,graph  = D['stops'],D['s_names'],D['trips'],D['seqs'],D['graph']
s_idx,t_idx,l_idx,service_id = D['stop_idx'],D['trip_idx'],D['l_idx'],D['service_id']
s_dist,v_dist,l_dist,w_dist = D['s_dist'],D['v_dist'],D['l_dist'],D['w_dist']
