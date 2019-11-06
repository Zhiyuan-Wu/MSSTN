import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.misc import imresize
import requests
import json
from sys import stdout

class dataset(object):
    def __init__(self,args):
        self.data = []
        self.tr = []
        self.te = []

        self.data_mid = np.load(args['data_path_mid'])
        self.tr_mid = self.data_mid[:81*90+1]
        self.te_mid = self.data_mid[81*90+1:81*117+2]
        self.tr_mid = np.transpose(self.tr_mid,(1,0,2))
        self.te_mid = np.transpose(self.te_mid,(1,0,2))
        self.tr_mid = np.concatenate((self.tr_mid,np.concatenate((self.tr_mid,self.tr_mid[:,-1:]),1)[:,1:,1:]),-1)
        self.te_mid = np.concatenate((self.te_mid,np.concatenate((self.te_mid,self.te_mid[:,-1:]),1)[:,1:,1:]),-1)

        for i in range(args['city_number']):
            data = np.load(args['data_path_'+str(i)])
            tr = data[:81*90+1]
            te = data[81*90+1:81*117+2]
            tr = np.transpose(tr,(1,0,2))
            te = np.transpose(te,(1,0,2))
            tr = np.concatenate((tr,np.concatenate((tr,tr[:,-1:]),1)[:,1:,1:]),-1)
            te = np.concatenate((te,np.concatenate((te,te[:,-1:]),1)[:,1:,1:]),-1)
            self.data.append(data)
            self.tr.append(tr)
            self.te.append(te)

        self.batch_size = args['seq_len']-1
        self.tr_batch_counter = 0
        self.tr_batch_num = (self.tr[0].shape[1]-1)//(args['seq_len']-1)
        self.tr_batch_perm = np.linspace(0,self.tr_batch_num-1,self.tr_batch_num)
        np.random.shuffle(self.tr_batch_perm)
        self.te_batch_counter = 0
        self.te_batch_num = (self.te[0].shape[1]-1)//(args['seq_len']-1)
        self.te_batch_perm = np.linspace(0,self.te_batch_num-1,self.te_batch_num)
        np.random.shuffle(self.te_batch_perm)

    def tr_get_batch(self,id=None):
        if id:
            idx_start = self.batch_size*id
            idx_end = self.batch_size*(id+1)
        else:
            id = int(self.tr_batch_perm[self.tr_batch_counter])
            idx_start = self.batch_size*id
            idx_end = self.batch_size*(id+1)
            self.tr_batch_counter = (self.tr_batch_counter+1)%self.tr_batch_num
            if self.tr_batch_counter==0:
                np.random.shuffle(self.tr_batch_perm)
        
        batch_mid = self.tr_mid[:,idx_start:idx_end+1]
        batch_set = []
        for i in range(len(self.tr)):
            batch_set.append(self.tr[i][:,idx_start:idx_end+1])
        
        return batch_mid,batch_set

    def te_get_batch(self,id=None):
        if id:
            idx_start = self.batch_size*id
            idx_end = self.batch_size*(id+1)
        else:
            id = int(self.te_batch_perm[self.te_batch_counter])
            idx_start = self.batch_size*id
            idx_end = self.batch_size*(id+1)
            self.te_batch_counter = (self.te_batch_counter+1)%self.te_batch_num
            if self.te_batch_counter==0:
                np.random.shuffle(self.te_batch_perm)
        
        batch_mid = self.te_mid[:,idx_start:idx_end+1]
        batch_set = []
        for i in range(len(self.te)):
            batch_set.append(self.te[i][:,idx_start:idx_end+1])
        
        return batch_mid,batch_set

def Generate_adj(location_file='./data/BJSiteLocation.csv',file_name='',thr=0.3):
    def haversine(loc1, loc2):
        loc1 = np.array(loc1)/180.0*np.pi
        loc2 = np.array(loc2)/180.0*np.pi
        R = 6378.137
        h = np.sin((loc1[1]-loc2[1])/2)**2 + np.cos(loc1[1])*np.cos(loc2[1])*np.sin((loc1[0]-loc2[0])/2)**2
        d = 2*R*np.arcsin(np.sqrt(h))
        return d    
    SiteLocation = pd.read_csv(location_file).sort_values('id')
    station_list = list(SiteLocation[['id']].to_dict()['id'].values())
    location = SiteLocation[['lon','lat']].as_matrix()
    num_nodes = len(station_list)
    dis = np.zeros((num_nodes,num_nodes))
    for i in range(num_nodes):
        for j in range(i+1,num_nodes):
            dis[i,j] = haversine(location[i,:], location[j,:])
            dis[j,i] = dis[i,j]
            
    sigma = np.std(dis)
    adj = np.exp(-dis/sigma) - np.eye(num_nodes)
    #h = plt.hist(adj.reshape([35*35,1]),bins = 100)
    #thr = 0.3
    adj[adj<thr] = 0
    np.save('./data/adj_matrix'+file_name+'.npy',adj)

def point_IDW(_t,location,target,scale=0.1):
    t = np.array(_t)
    mask = np.isnan(t)
    t[mask] = 0
    d = np.sum(np.square(location-np.array(target)),1)
    d = np.exp(-d/scale)
    d[mask] = 0
    d = d/np.sum(d)
    return np.sum(d*t)

def ZYFeature(filename='./data/bjdata2_PM25.npy'):
    data = np.load(filename)
    SiteLocation = pd.read_csv('./data/BJSiteLocation.csv').sort_values('id')
    station_list = list(SiteLocation[['id']].to_dict()['id'].values())
    location = SiteLocation[['lon','lat']].as_matrix()
    num_nodes = len(station_list)
    time_length = data.shape[0]
    channel_num = data.shape[2]
    R1 = 0.2
    R2 = 0.5
    n = 8
    scale = 0.1
    target_degree = np.linspace(0,2*np.pi,n+1)[:-1]
    sample_position = np.concatenate(([R1/2*np.sin(target_degree),R1/2*np.cos(target_degree)],
                                [(R1+R2)/2*np.sin(target_degree),(R1+R2)/2*np.cos(target_degree)]),axis=1).T
    target_location = np.transpose(np.array([location for _ in range(n*2)]),(1,0,2))
    target_location = target_location + sample_position
    target_location = np.reshape(target_location,(num_nodes*n*2,2))
    weight = np.transpose(np.array([location for _ in range(num_nodes*n*2)]),(1,0,2))
    weight = np.sum(np.square(weight-target_location),2)
    weight = np.exp(-weight/scale)
    weight = weight/np.expand_dims(np.sum(weight,0),0)
    
    data = data.transpose(0,2,1).reshape(time_length*channel_num,num_nodes)
    result = np.matmul(data,weight)
    result = result.reshape(time_length,channel_num,num_nodes,n*2)
    result = result.transpose(0,2,1,3).reshape(time_length,num_nodes,channel_num*n*2)
    np.save(filename+'.zyf',result)    
    _debug = np.array([2,3,3])

def Generate_Data():    
    city = {'bj':1,'ld':0}['bj']
    fname = 'bjdata2_PM25.npy'
    gas_type = 'PM25'
    mem = pd.read_csv('./data/Beijing_historical_meo_grid.csv')
    aq = pd.read_csv('./data/beijing_17_18_aq.csv')
    SiteLocation = pd.read_csv('./data/BJSiteLocation.csv').sort_values('id')
    station_list = list(SiteLocation[['id']].to_dict()['id'].values())
    location = SiteLocation[['lon','lat']].as_matrix()

    date = mem.utc_time.drop_duplicates()
    part = True
    location_int = np.array(location)
    img_height = 31
    img_width = 21
    x_min = 115.0
    x_max = 118.0
    y_min = 39.0
    y_max = 41.0
    _img_width = img_width - 1
    _img_height = img_height - 1
    location_int[:,0] = (location_int[:,0]-x_min)*_img_height/(x_max-x_min)
    location_int[:,1] = (location_int[:,1]-y_min)*_img_width/(y_max-y_min)
    location_int = location_int.astype(int)

    c = None
    data = []
    for t in range(date.shape[0]):
        if t%100 == 0:
            print(str(t)+'/'+str(date.shape[0])+' Done.')
        
        if city:
            _b = aq.loc[aq.utc_time==date.iloc[t]].drop_duplicates()[gas_type].as_matrix()
        else:
            _b = aq.loc[aq.MeasurementDateGMT==date.iloc[t]].drop_duplicates()[gas_type].as_matrix()
        if _b.shape[0]==0 or np.sum(np.isnan(_b))==35:
            if date.iloc[t]>'2018-01-31 15:00:00':
                print('Arrive 2018-01-31 15:00:00, Stop.')
                break
            elif c is not None:
                data.append(c)
                _word = ' Use last one instead.' 
            else:
                _word = ' Skip to continue.'
            print(date.iloc[t]+': aq data missing.'+_word)
            continue
        for i in range(_b.shape[0]):
            if np.isnan(_b[i]):
                _b[i] = point_IDW(_b,location,location[i])
        
        a = mem.iloc[0+651*t:651+651*t,-5:].as_matrix().reshape((31,21,5))
        _a = a[location_int[:,0],location_int[:,1]]
        
        e = time.mktime((time.strptime(date.iloc[t],'%Y-%m-%d %H:%M:%S')))/3600.0
        _e = np.ones((35,2))
        _e[:,0] = np.cos(e%24/24.0*2*np.pi)
        _e[:,1] = np.sin(e%24/24.0*2*np.pi)

        g = int(time.mktime((time.strptime(date.iloc[t],'%Y-%m-%d %H:%M:%S')))/3600.0/24.0)
        _g = np.ones((35,2))
        _g[:,0] = np.cos(g%7/7.0*2*np.pi)
        _g[:,1] = np.sin(g%7/7.0*2*np.pi)
            
        c = np.concatenate([_b.reshape((35,1)),_a,_e,_g,location],-1)
        data.append(c)

    data = np.array(data)
    derection = data[:,:,4]
    speed = data[:,:,5]
    vx = speed*np.cos(derection*np.pi/180.0)
    vy = speed*np.sin(derection*np.pi/180.0)
    data[:,:,4] = vx
    data[:,:,5] = vy

    data2 = np.array(data)
    data2[:,:,0][data2[:,:,0]<0]=0
    data_min = np.min(data2,axis=(0,1))
    data_max = np.max(data2,axis=(0,1))
    data_min = np.array([0.0,-18.71,907.22,0.0,-29.186,-46.444,-1.0,-1.0,-1.0,-1.0,115.972,39.52])
    data_max = np.array([500.0,36.06,1038.81,100.0,40.37,33.41,1.0,1.0,1.0,1.0,117.12,40.50])
    data2 = (np.array(data2)-data_min)/(data_max-data_min)
    np.save(fname,data2)


def show_progress(message,counter,counter_max):
    stdout.write(message+', progress: {0:.2f}%     \r'.format(float(counter/counter_max*100.0)))
    stdout.flush()
def one_hot(x,depth):
    result = np.matmul(np.ones((x.shape[0],1)),np.arange(depth).reshape((1,depth)))
    x = np.matmul(x.reshape((x.shape[0],1)),np.ones((1,depth)))
    return (result==x)*1.0
def mid_scale_weather_spyder(idx, secret_key):
    # setup request pool
    site_list = pd.read_csv('data/targetSite.csv').sort_values('id')
    site_id = list(site_list[['id']].to_dict()['id'].values())
    location = site_list[['lon','lat']].values
    req_pool = []
    file_pool = []
    for k in range(location.shape[0]):
        start_date = '2017-01-01'
        end_date = '2018-01-31'
        start_time = time.mktime((time.strptime(start_date+' 12:00:00','%Y-%m-%d %H:%M:%S')))
        end_time = time.mktime((time.strptime(end_date+' 12:00:00','%Y-%m-%d %H:%M:%S')))
        req_loc = '{0:f},{1:f},'.format(location[k,1],location[k,0])
        time_now = start_time
        while time_now<=end_time:
            req = 'https://api.darksky.net/forecast/'+secret_key+'/'+req_loc+str(round(time_now))
            file = 'data/weather/'+site_id[k]+'/'+time.strftime('%Y-%m-%d',time.gmtime(time_now))+'.json'
            req_pool.append(req)
            file_pool.append(file)
            time_now += 3600*24

    # len(req_pool)=9900
    start_idx = 990*idx
    end_idx = start_idx+990
    for k in range(start_idx,end_idx):
        show_progress('Job '+str(idx), k-start_idx, end_idx-start_idx)
        try:
            r = requests.get(req_pool[k])
            with open(file_pool[k], 'w') as result_file:
                result_file.write(r.text)
        except:
            print('Error: ', req_pool[k])
            with open('ERROR_LOG_file', 'a') as ERROR_LOG_file:
                ERROR_LOG_file.write(req_pool[k]+'\n')            
    _debug = np.array([2,3,3])

def inner_mid_scale_weather_generator(site_id):
    #Collect from file
    start_date = '2017-01-01'
    end_date = '2018-01-31'
    start_time = time.mktime((time.strptime(start_date+' 12:00:00','%Y-%m-%d %H:%M:%S')))
    end_time = time.mktime((time.strptime(end_date+' 12:00:00','%Y-%m-%d %H:%M:%S')))
    time_now = start_time
    stamp_bias = time.mktime((time.strptime(start_date+' 00:00:00','%Y-%m-%d %H:%M:%S')))
    data_list = []
    time_stamp = []
    while time_now<=end_time:
        file = 'data/weather/'+site_id+'/'+time.strftime('%Y-%m-%d',time.gmtime(time_now))+'.json'
        try:
            with open(file) as f:
                data = json.load(f)
                for k in range(len(data['hourly']['data'])):
                    _a = np.array([(data['hourly']['data'][k]['temperature']-32)/1.8,
                                  data['hourly']['data'][k]['humidity'],
                                  data['hourly']['data'][k]['windSpeed'],
                                  data['hourly']['data'][k]['windBearing']])
                    _stamp = int((data['hourly']['data'][k]['time']-stamp_bias)/3600)
                    data_list.append(_a)
                    time_stamp.append(_stamp)
        except:
            pass
        time_now += 24*3600
    
    # Linear interploation
    data_ratio = len(data_list)/9504.0
    print('site '+site_id+' Data Collection Done, with integrity ratio: ',round(data_ratio,4))
    data = np.array(data_list)
    time_stamp = np.array(time_stamp)
    data_interp = []
    for i in range(4):
        data_interp.append(np.interp(np.arange(9504), time_stamp, data[:,i]))
    data = np.stack(data_interp,-1)
    
    # Additional Feature
    windx = data[:,2]*np.cos(data[:,3]/360*2*np.pi)
    windy = data[:,2]*np.sin(data[:,3]/360*2*np.pi)
    wind = np.stack([windx,windy],-1)
    hour = one_hot(np.arange(9504)%24,24)
    week = one_hot((np.arange(9504)//24)%7,7)
    data = np.concatenate([data[:,:2],wind,hour,week],-1)
    _debug = np.array([2,3,3])
    return data,data_ratio

def mid_scale_weather_generator():
    site_list = pd.read_csv('data/targetSite.csv').sort_values('id')
    site_id = list(site_list[['id']].to_dict()['id'].values())
    data = []
    for s in site_id:
        d,r = inner_mid_scale_weather_generator(s)
        data.append(d)
    data = np.stack(data,1)
    data[:,:,0] = (data[:,:,0]+26.7)/(39.18+26.7)
    data[:,:,2] = (data[:,:,2]+33)/(31.71+33)
    data[:,:,3] = (data[:,:,3]+29.5)/(25.26+29.5)
    return data[22:]

def mid_scale_concentration_collector(site_id):
    # Input: A list of sites id; Output: a numpy array
    #Collect from file
    start_date = '2017-01-01'
    end_date = '2018-01-31'
    start_time = time.mktime((time.strptime(start_date+' 12:00:00','%Y-%m-%d %H:%M:%S')))
    end_time = time.mktime((time.strptime(end_date+' 12:00:00','%Y-%m-%d %H:%M:%S')))
    time_now = start_time
    template = pd.DataFrame({'hour':[i for i in range(24)]})
    data_list = []
    while time_now<=end_time:
        file = 'data/site'+str(time.gmtime(time_now).tm_year)+'/china_sites_'+time.strftime('%Y%m%d',time.gmtime(time_now))+'.csv'
        try:
            data = pd.read_csv(file)
            data = data.loc[data['type']=='PM2.5'][['hour']+site_id]
            data = template.merge(data,on='hour',how='left')
            data_list.append(data[site_id].fillna(-1).values)
        except:
            data_list.append(-np.ones((24,len(site_id))))
        time_now += 24*3600
    
    # Linear interploation
    data = np.concatenate(data_list,0)
    data_ratio = 1-np.mean(data==-1,0)
    print('Data Collection Done, with integrity ratio: ')
    print(site_id)
    print(data_ratio)
    data_interp = []
    for i in range(data.shape[1]):
        time_stamp = np.arange(data.shape[0])[data[:,i]!=-1]
        data_interp.append(np.interp(np.arange(data.shape[0]), time_stamp, data[:,i][data[:,i]!=-1]))
    data = np.stack(data_interp,-1)
    
    return data
        
def mid_scale_concentration_generator(): 
    site_list = pd.read_csv('data/targetSite.csv').sort_values('id')
    site_id = list(site_list[['id']].to_dict()['id'].values())
    data = mid_scale_concentration_collector(site_id)
    data = data/500.0
    data = np.expand_dims(data,-1)
    return data[22:]    

def mid_scale_data():
    concentration = mid_scale_concentration_generator()
    weather = mid_scale_weather_generator()
    data = np.concatenate([concentration,weather],-1)
    np.save('data/bj_pm25_midscale_feature.npy',data)
    print('Done')

if __name__=='__main__':
    #secret_key = ['3636efb586ef9346635b4c2481ada25a','f8ef889a2a0afc7375bfd8300cac9e65'
                  #,'2f72a1c990c72ec5da35331917588182','e594c6be526dce88753dfda10df2311f'
                  #,'8e44624d195d316309ca1d910e622a70']
    #job_idx = 9
    #sk_idx = 4
    #print(job_idx,sk_idx)
    #mid_scale_weather_spyder(job_idx,secret_key[sk_idx])
    #print('Done.')
    Generate_adj('data/targetSite.csv','_midscale',0.23)
    plt.imshow(np.load('data/adj_matrix_midscale.npy'))
    plt.show()
    _debug = np.array([2,3,3])