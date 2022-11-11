#this is updated functions.py

# reasons for high risklevel??
for order_num in range(len(all_merchants_shopify)):
    if all_merchants_shopify[order_num]['shopify']['riskLevel']=='HIGH':
            print(order_num, "-", len([risk['message'] for risk in all_merchants_shopify[order_num]['shopify']['risks'] if risk['level']=="HIGH"]))
            for risk in all_merchants_shopify[order_num]['shopify']['risks']:
                if risk['level']=="HIGH":
                    #if ("Address" not in risk['message']) and ("address" not in risk['message']):
                    print(risk)


# updates pincodes and cities in city state pincode file

temp = pd.DataFrame()
res_without_cities = []
for res in missing_pincodes:
    
    if res['results'] == []:
        continue
    cities = []
    state = ''
    localities = []
    
    try:
        localities = res['results'][0]['postcode_localities']
    except:
        pass
        #print(res)
    for add_type in res['results'][0]['address_components']:
        if 'postal_code' in add_type['types']:
            pincode = int(add_type['long_name'])
        if 'administrative_area_level_1' in add_type['types']:
            state = add_type['long_name']
        if 'administrative_area_level_2' in add_type['types'] or \
           'administrative_area_level_3' in add_type['types'] or \
           'administrative_area_level_4' in add_type['types'] or \
           'administrative_area_level_5' in add_type['types'] or \
           'administrative_area_level_6' in add_type['types'] or \
           'administrative_area_level_7' in add_type['types'] or \
            'locality' in add_type['types'] or \
           'postal_town' in add_type['types']:
            temp = temp.append({'pincode':pincode, 'city_list':add_type['long_name'], 'postcode_localities':localities, 'statename':state}, ignore_index=True)
            cities.append(add_type['long_name'])
    if cities == []:
        res_without_cities.append(res)
        print(cities, pincode, state)


# use soundex/jellyfish function
from asyncore import file_dispatcher
from audioop import add
from email.mime import image
from operator import truediv
from platform import python_branch
from subprocess import STARTF_USESHOWWINDOW
from xml.dom import pulldom
import jellyfish
fuzz.ratio(jellyfish.soundex(word.lower()),jellyfish.soundex(state.lower()))


def catch_json(js):
    import ast
    try:
        return json.loads(js)
    except:
            try:
                return ast.literal_eval(js)
            except:
                return js #None


for col in raw_data.columns:
    raw_data[col] = raw_data[col].apply(catch_json)


def json_to_col(df):
    def is_datatype_dict(val):
        if type(val)==dict:
            return 1
        return 0
    def first_non_null_value(df,col):
        for i in df[col]:
            if type(i)==list:
                return i
            if pd.isna(i)==False:
                return i
        return i
    for col in df.columns:
        if is_datatype_dict(first_non_null_value(df,col))==0:
            continue
        df = pd.concat([df,pd.json_normalize(df[col])],axis=1)
        if list(df.columns).count(col)==1:
            del df[col]
    return df

def shopify_risk_features(risks_array):
    for risk in risks_array:
        proxy_or_not = 0
        address_incomplete = 0
        address_low_length = 0
        address_seems_incomplete = 0
        fraudulent_order_characteristics = 0
        high_rto_history = 0
        phone_blacklisted = 0
        email_blacklisted = 0
#         phone_valid = 1 
        safe_order = np.nan
        payment_attempts = np.nan
        different_country = 0
        shipping_ip_distance = np.nan
        same_item_same_order = 0
        same_item_same_week = 0
        customer_rto_count = np.nan
        mix_history = 0
        same_customer_multiple_orders_past = 0
        if "A high risk internet connection (web proxy) was used to place the order" == risk['message']:
            proxy_or_not = 1
        if 'Address Incomplete' in risk['message']:
            address_incomplete = 1
        if 'Address Low Length' == risk['message'] or 'Short shipping address' == risk['message']:
            address_low_length = 1
        if 'Address seems incomplete' in risk['message'] or 'Address looks incomplete' in risk['message']:
            address_seems_incomplete = 1
        if 'Characteristics of this order are similar to fraudulent orders observed in the past' in risk['message']:
            fraudulent_order_characteristics = 1
        if 'phone is black listed' == risk['message']:
            phone_blacklisted = 1
        if 'email is black listed' == risk['message']:
            email_blacklisted = 1
        if 'The order is safe for delivery' in risk['message']:
            safe_order = 1
        if 'There were' in risk['message'] and 'payment attempts' in risk['message']:
            try:
                payment_attempts = int([x for x in risk['message'].split() if x.isnumeric()][0])
            except:
                payment_attempts = 0
        if 'The billing address is listed as' in risk['message'] and 'but the order was placed from' in risk['message']:
            different_country = 1
        if 'Shipping address is' in risk['message'] and 'from location of IP address' in risk['message']:
            shipping_ip_distance = int([x for x in risk['message'].split() if x.isnumeric()][0])
        if 'has a high RTO history' in risk['message']:
            high_rto_history = 1
#         if 'invalid phone number' in risk['message'] or 'Mobile Number is invalid or too long' in risk['message'] or 'Mobile Number entered starts with invalid characters' in risk['message']:
#             phone_valid = 0
        if 'Multiple same item from' in risk['message'] and 'one order' in risk['message']:
            same_item_same_order = 1
        if 'Multiple same item from' in risk['message'] and 'one week' in risk['message']:
            same_item_same_week = 1
        if 'instances of RTO detected for this user' in risk['message']:
            customer_rto_count = int([x for x in risk['message'].split() if x.isnumeric()][0])
        if 'Customer has a mixed history of accepting and rejecting COD orders' == risk['message']:
            mix_history = 1
        if 'Multiple orders by same customer in the last 15 days' in risk['message']:
            same_customer_multiple_orders_past = 1
    return proxy_or_not, address_incomplete, address_low_length, address_seems_incomplete, fraudulent_order_characteristics, high_rto_history, phone_blacklisted, email_blacklisted, \
            safe_order, payment_attempts, different_country, shipping_ip_distance, same_item_same_order, same_item_same_week, user_rto_count, mix_history, same_customer_multiple_orders_past

    
# change language of the text   
unaccented_string = unidecode.unidecode(name).strip()

from cleantext.clean import clean
clean(text)

from translate import Translator
translator= Translator(to_lang="english")
# try:
translation = translator.translate("पाथर्डी")
print(translation)
# except:
#     print("Gibberish")

# add new city row the pincode in city state pincode table
def add_new_city(pin,city,city_state_pin):
    temp = city_state_pin.loc[city_state_pin['pincode']==pin,:]
    temp['city'] = city
    city_state_pin = city_state_pin.append(temp)
    return city_state_pin

from pprint import pprint


from tqdm.auto import tqdm
tqdm.pandas()
all_gsfid['sardine_check'] = all_gsfid.progress_apply(lambda x: sardine_fingerprint(x['gsfid'],sardine),axis=1)

import time
for i in tqdm(range(10)):
    time.sleep(3)

# hit google maps api
import requests
import time

GOOGLE_API_KEY = 'AIzaSyBfhpjZFA3zFZDyrDNEu1YHqYfNR7S1TbA' 

result_list = {}

def extract_lat_long_via_address(address_or_zipcode):
#     print(address_or_zipcode)
    lat, lng = None, None
    api_key = GOOGLE_API_KEY
    base_url = "https://maps.googleapis.com/maps/api/geocode/json"
    endpoint = f"{base_url}?address={address_or_zipcode}&key={api_key}"
#     time.sleep(0.5)

    try:
        r = requests.get(endpoint,timeout=10.0)
        if r.status_code not in range(200, 299):
            result_list[address_or_zipcode]="wrong address"
            return "wrong address"
        result_list[address_or_zipcode]=r.json()
        return r.json()
#         results = r.json()['results'][0]
#         lat = results['geometry']['location']['lat']
#         lng = results['geometry']['location']['lng']
    except:
        result_list[address_or_zipcode]="timeout error"
        return "timeout error"
    
    
    
    
    
    
    
    

# feature analysis function
def feature_analysis(feature,demand='equal',label = 'rto_or_not',df = df,method='percentile',total_count_threshold=0,rto_pct_threshold=0):
    
    print(feature)
    def not_null_columns(df):
        a=[]
        for i in df.columns:
            if df[i].isnull().sum()==0:
                a.append(i)
        return a



    def link(feature=feature,label=label, df = df,total_count_threshold=total_count_threshold,rto_pct_threshold=rto_pct_threshold):
        pivot=df.pivot_table(values=[i for i in not_null_columns(df) if i not in [feature,label]][0],index=feature,columns=label,aggfunc='count')
        pivot['sum']=pivot.sum(axis=1)
        pivot.fillna(0,inplace=True)
        pivot['rto_pct']=(pivot[1])/(pivot['sum'])
        return pivot.loc[(pivot['sum']>=total_count_threshold)&(pivot['rto_pct']>=rto_pct_threshold),:]
    
    if demand=='equal':
        return print(link(feature))
    
    
    if method == 'percentile':
        if demand == 'lower':
            table = pd.DataFrame(columns=['percentile','value_less/equal_than','total','rto_pct'])
            for i in [x/100 for x in range(5,100,5)]:
                total = len(df.loc[df[feature]<=df[feature].quantile(i),:])
                rto_pct = len(df.loc[(df[feature]<=df[feature].quantile(i))&(df[label]==1),:])/total if total>0 else 0
                table = table.append({'percentile':i,'value_less/equal_than':df[feature].quantile(i),'total':total,'rto_pct':rto_pct},ignore_index=True)
        else:
            table = pd.DataFrame(columns=['percentile','value_more/equal_than','total','rto_pct'])
            for i in [x/100 for x in range(0,100,5)]:
                total = len(df.loc[df[feature]>=df[feature].quantile(i),:])
                rto_pct = len(df.loc[(df[feature]>=df[feature].quantile(i))&(df[label]==1),:])/total if total>0 else 0
                table = table.append({'percentile':i,'value_more/equal_than':df[feature].quantile(i),'total':total,'rto_pct':rto_pct},ignore_index=True)
    else:
        if demand == 'lower':
            table = pd.DataFrame(columns=['value_less/equal_than','total','rto_pct'])
            for i in range(1,int(df[feature].max()+1),1):
                total = len(df.loc[df[feature]<=i,:])
                rto_pct = len(df.loc[(df[feature]<=i)&(df[label]==1),:])/total if total>0 else 0
                table = table.append({'value_less/equal_than':i,'total':total,'rto_pct':rto_pct},ignore_index=True)
        else:
            table = pd.DataFrame(columns=['value_more/equal_than','total','rto_pct'])
            for i in range(0,int(df[feature].max()+1),1):
                total = len(df.loc[df[feature]>=i,:])
                rto_pct = len(df.loc[(df[feature]>=i)&(df[label]==1),:])/total if total>0 else 0
                table = table.append({'value_more/equal_than':i,'total':total,'rto_pct':rto_pct},ignore_index=True)
    table['feature'] = feature
    

    
    if demand == 'lower':
        return print(table.loc[(table['total']>total_count_threshold)&(table['rto_pct']>rto_pct_threshold),:][['value_less/equal_than','total','rto_pct']].drop_duplicates())
    else:
        return print(table.loc[(table['total']>total_count_threshold)&(table['rto_pct']>rto_pct_threshold),:][['value_more/equal_than','total','rto_pct']].drop_duplicates())        
            
    
        
        
# user agent parser
s = "Mozilla/5.0 (Linux; Android 10; M2007J17I) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.79 Mobile Safari/537.36"

import httpagentparser

def device_details(user_agent):
#     return = ['platform','os']
    if pd.isna(user_agent):
        return None
    x =  httpagentparser.detect(user_agent)
#     print(x)
    return x['platform']['name'],x['os']['name']




def is_valid_phone_number(phone):
    if len(re.findall("^(\+91[\-\s]?)?[0]?(91)?[789]\d{9}$",str(phone)))!=0:
        return 1
    else:
        return 0

is_valid_phone_number(91706033452)



# check whether given text is in english or not
import re
def english_or_not(text):
    pattern = re.findall("[A-Za-z0-9 -,.]+",text)
    return (0 if len(pattern)==0 else 1)


# prefer not using it -> makes jupyter notebook slow
pd.set_option('display.max_columns',None)
pd.set_option('display.max_rows',10)



from math import cos, asin, sqrt, pi
def lat_long_distance(lat1, lon1, lat2, lon2):
    p = pi/180
    a = 0.5 - cos((lat2-lat1)*p)/2 + cos(lat1*p) * cos(lat2*p) * (1-cos((lon2-lon1)*p))/2
    return 12742 * asin(sqrt(a)) 

from sklearn.metrics import mean_squared_error,r2_score,mean_squared_log_error,accuracy_score,confusion_matrix,precision_score,recall_score
def performance_metrics(model, X_test, y_test, X_train, y_train, threshold=0.5):
    #y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)
    y_pred = [(y_prob[i][1]>=threshold)*1 for i in range(len(y_prob))]
    #y_pred_train = model.predict(X_train)
    y_prob_train = model.predict_proba(X_train)
    y_pred_train = [(y_prob_train[i][1]>=threshold)*1 for i in range(len(y_prob_train))]
    print("precision_test: ",precision_score(y_test,y_pred), "|| precision_train: ",precision_score(y_train, y_pred_train))
    print("recall_test: ",recall_score(y_test,y_pred), "|| recall_train: ",recall_score(y_train, y_pred_train))
    print("accuracy_test: ",accuracy_score(y_test,y_pred), "|| accuracy_train: ",accuracy_score(y_train, y_pred_train))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred)) 


# threhsold on pred_probabilities
from sklearn.metrics import mean_squared_error,r2_score,mean_squared_log_error,accuracy_score,confusion_matrix,precision_score,recall_score
def threshold_play(y_prob, y_test, threshold=0.5):
    y_pred = [(y_prob[i][1]>=threshold)*1 for i in range(len(y_prob))]
    print("precision_test: ",precision_score(y_test,y_pred))
    print("recall_test: ",recall_score(y_test,y_pred))
    print("accuracy_test: ",accuracy_score(y_test,y_pred))



# working with pipeline??
pipe = Pipeline([('scaler',StandardScaler()), \
          ('pca', PCA(n_components=0.99)), \
          ('knn', KNeighborsClassifier(n_neighbors=100))])

pipe.fit(X, y)


# text to speech in python
import pyttsx3
engine = pyttsx3.init()
engine.say("code is executed")
engine.runAndWait()


def insert_data_into_db(query):
    ## Connect to prd db and insert the data
    from sqlalchemy import create_engine
    import psycopg2
    connection_string_prod = create_engine('postgresql://beau-metrics-prd-redshift-adminuser:vD3NJQKQhuch8yE@10.30.24.48:5439/dev')
    connection_string_prod.execute(query)

# get today and now
from datetime import datetime
today = datetime.today().strftime('%Y-%m-%d')
now = datetime.today().strftime('%Y-%m-%d#%H:%M:%S')


# get column datatypes in sql
SELECT
COLUMN_NAME, DATA_TYPE
FROM
INFORMATION_SCHEMA.COLUMNS
WHERE TABLE_NAME = 'supplierapitransactionevent'



from sklearn.model_selection import train_test_split
X, X_test, y, y_test = train_test_split(X, y, test_size=0.33, random_state=42)



#plotting the indian states map using plotly
import pandas as pd
import plotly.graph_objects as go

df = pd.read_csv("https://gist.githubusercontent.com/jbrobst/56c13bbbf9d97d187fea01ca62ea5112/raw/e388c4cae20aa53cb5090210a42ebb9b765c0a36/active_cases_2020-07-17_0800.csv")

fig = go.Figure(data=go.Choropleth(
    geojson="https://gist.githubusercontent.com/jbrobst/56c13bbbf9d97d187fea01ca62ea5112/raw/e388c4cae20aa53cb5090210a42ebb9b765c0a36/india_states.geojson",
    featureidkey='properties.ST_NM',
    locationmode='geojson-id',
    locations=df['state'],
    z=df['active cases'],

    autocolorscale=False,
    colorscale='Reds',
    marker_line_color='peachpuff',

    colorbar=dict(
        title={'text': "Active Cases"},

        thickness=15,
        len=0.35,
        bgcolor='rgba(255,255,255,0.6)',

        tick0=0,
        dtick=20000,

        xanchor='left',
        x=0.01,
        yanchor='bottom',
        y=0.05
    )
))

fig.update_geos(
    visible=False,
    projection=dict(
        type='conic conformal',
        parallels=[12.472944444, 35.172805555556],
        rotation={'lat': 24, 'lon': 80}
    ),
    lonaxis={'range': [68, 98]},
    lataxis={'range': [6, 38]}
)

fig.update_layout(
    title=dict(
        text="Active COVID-19 Cases in India by State as of July 17, 2020",
        xanchor='center',
        x=0.5,
        yref='paper',
        yanchor='bottom',
        y=1,
        pad={'b': 10}
    ),
    margin={'r': 0, 't': 30, 'l': 0, 'b': 0},
    height=550,
    width=550
)

fig.show()



# bar plot with labels

# red flags chart
import seaborn as sns
import matplotlib.pyplot as plt
# sns.catplot(data = red_flags,x='red_flag',y='count',kind='bar')




count = list(red_flags['count devices'])

freq_series = pd.Series(count)

x_labels = list(red_flags['red flag'])

# Plot the figure.
plt.figure(figsize=(8, 8))
ax = freq_series.plot(kind="bar",color='b')
ax.set_title("Red Flags")
ax.set_xlabel("Red Flag")
ax.set_ylabel("Unique Devices")
ax.set_xticklabels(x_labels)


rects = ax.patches


labels = freq_series

for rect, label in zip(rects, labels):
    height = rect.get_height()
    ax.text(
        rect.get_x() + rect.get_width() / 2, height + 5, label, ha="center", va="bottom"
    )
    
ax.set_facecolor('xkcd:grey')
# ax.set_facecolor((1.0, 0.47, 0.42))

plt.show()



# change table themes
red_flags.style.set_properties(**{'background-color': 'black',
                           'color': 'yellow'})



def linear_interpolation(x1,y1,x2,y2,x):
    """
    simple linear interpolation function
    """
    y=(y2-y1)/(x2-x1)*(x-x1)+y1
    return y




def scaled_to_score(scaled):
    """
    this function converts scaled distance to score from 0 to 100.
    Higher the scaled distance -> less is the score
    Its a combination of linear interpolation using dict_ and x*y=c for values more than 3
    """
    dict_ = {0:100,1:70,2:45,3:25}
    if scaled<=1:
        return linear_interpolation(0,dict_[0],1,dict_[1],scaled)
    elif scaled<=2:
        return linear_interpolation(1,dict_[1],2,dict_[2],scaled)
    elif scaled <=3:
        return linear_interpolation(2,dict_[2],3,dict_[3],scaled)
    else:
        #this is x*y=constant curve
        return 3*dict_[3]/scaled




# -*- coding: utf-8 -*-
"""
Created on Mon Dec 20 20:48:21 2021

@author: Shubham_Mantri
"""

def mileage_bucket(mileage):
    if mileage<10000:
        return '0k-10k'
    elif mileage<30000:
        return '10k-30k'
    elif mileage<50000:
        return '30k-50k'
    elif mileage<75000:
        return '50k-75k'
    elif mileage<100000:
        return '75k-100k'
    else:
        return '>100k'
    

def mileage_bucket(mileage):
    if mileage<25000:
        return '0k-25k'
    elif mileage<50000:
        return '25k-50k'
    elif mileage<75000:
        return '50k-75k'
    elif mileage<100000:
        return '75k-100k'
    else:
        return '>100k'
    
def year_bucket(year):
    if year < 2010:
        return '2008-2009'
    elif year > 2009 and year < 2013:
        return '2010-2012'
    elif year > 2012 and year < 2016:
        return '2013-2015'
    elif year > 2015 and year < 2019:
        return '2016-2018'
    elif year > 2018 and year < 2022:
        return '2019-2021'
    


for city, name in [("Bangalore","Bangalore"), ("delhi", "Delhi-NCR"), ("hyderabad","Hyderabad"), ("pune", 'Pune'), ("mumbai", 'Mumbai'), ("chennai", "Chennai"), ("kolkata", "Kolkata"), ("ahmedabad", "Ahmedabad")]:
    temp = getDataFromSheets("Mission sales PAN india", city)
    temp["City"] = name
    PANIndiadata = PANIndiadata.append(temp)
    
 def Year_Bucket(year):
    if year < 2011:
        return '2008-2010'
    elif year < 2014:
        return '2011-2014'
    elif year < 2019:
        return '2015-2018'
    elif year < 2022:
        return '2019-2021'

    
import gspread
import pandas as pd
import numpy as np
import datetime
from gspread_dataframe import get_as_dataframe, set_with_dataframe
import time
from oauth2client.service_account import ServiceAccountCredentials
import pygsheets, json
from tqdm import tqdm
from pymongo import MongoClient
import datetime, time
import ssl
import warnings
warnings.filterwarnings("ignore")
import pymysql as sql
# debugger pdb.set_trace()
import pdb

scope = ['https://spreadsheets.google.com/feeds','https://www.googleapis.com/auth/drive']
# add credentials to the account
global creds
creds = ServiceAccountCredentials.from_json_keyfile_name(r'C:\Users\Shubham Mantri\Downloads\coastal-airlock-315709-d614c773b55d.json', scope)
pd.set_option('max_columns',None)


import pandas as pd
import pygsheets, math
import numpy as np
import time, datetime
import pymysql as sql
from datetime import date,timedelta
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

def MongoConnection(mongoURL = 'mongodb+srv://etl_read_user_1:FRfbJCgVpfUPQshE@sp-backend-mongo-3-pl-0.gfsjm.mongodb.net/sp-datalake?replicaSet=atlas-9q66dq-shard-0&authSource=admin&w=majority',
                    mongoDB = 'sp-datalake', 
                    days = 60, 
                    valueFile = r"C:\Users\Shubham Mantri\Downloads\crawlerdata.json"):
    """This function will fetch the data from mongoDB"""
    
    try:
        print("fetching data from mongodb... ")
                
        today = (datetime.datetime.today() - datetime.timedelta(days=days))
        lastDay = datetime.datetime(today.year, today.month, today.day)
        
        with open(valueFile, 'r') as file:
            fields = json.loads(file.read())
            
        query = {
            "Platform": {
                "$in": ["Cars24"] 
                },
            "Created At":{
                "$gte": lastDay
                }
            }
        
        mongoClient = MongoClient(mongoURL, ssl_cert_reqs=ssl.CERT_NONE)
        collection = mongoClient[mongoDB]["pricing_data"]
        jsonData = list(collection.find(query, fields))
        print("data fetched from mongodb...")
                
        inspectionData = pd.DataFrame(jsonData)
        data =  pd.concat([inspectionData.drop(['_id','Inspection Report', 'Insurance Details'], axis=1), inspectionData['Inspection Report'].apply(pd.Series), inspectionData['Insurance Details'].apply(pd.Series)], axis=1)
        print("datafrom mongodb converted to dataframe...")
        
        
        return data
    
    except Exception as E:
        print(E)


def getDataFromSheets(workBookName, worksheetName, colStart = None, colEnd = None, authorisToken = r'C:\Users\Shubham Mantri\Downloads\coastal-airlock-315709-d614c773b55d.json'):
    """
    This function will return the worksheet as dataframe.
    """
    workSheetData = None
    print(f"Getting data from sheet {workBookName}...")
    
    try:
        Connection = pygsheets.authorize(service_file = authorisToken)
        workBook = Connection.open(workBookName)
        workSheet = workBook.worksheet_by_title(worksheetName)
        workSheetData = workSheet.get_as_df(start = colStart, end = colEnd)
        print(f"Data Fetched from workBook:{workBookName} > WorkSheet:{worksheetName}...")
    
    except Exception as E:
        print(E)
    
    return workSheetData



        
def insertDataIntoSheets(data, workBookName, worksheetName, colStart = None, colEnd = None, start = (1, 1), authorisToken = r"C:/Users/Lenovo/Downloads/sheetsjk-a6c30b661099.json"):
    """
    This function will insert add the data of dataframe to the given sheet.
    """
    try:
        print(f"Inserting data into sheet {workBookName}...")
        Connection = pygsheets.authorize(service_file = authorisToken)
        workBook = Connection.open(workBookName)
        workSheet = workBook.worksheet_by_title(worksheetName)
        workSheet.clear(start = colStart, end = colEnd)
        workSheet.set_dataframe(data, start = start)
        print(f"Data inserted into workBook:{workBookName} > WorkSheet:{worksheetName}...")
        data_check=getDataFromSheets(workBookName, worksheetName)
        if data_check.equals(data)==True:
            print("Insertion is correct")
        else:
            raise Exception('Wrong Insertion!!')
    except Exception as E:
        print(E)


def insertDataIntoSheets(data, workBookName, worksheetName, colStart = None, colEnd = None,start=(1,1), authorisToken = r'C:\Users\Shubham Mantri\Downloads\coastal-airlock-315709-d614c773b55d.json'):
    """
    This function will insert add the data of dataframe to the given sheet.
    """
    
    try:
        print(f"Inserting data into sheet {workBookName}...")
        Connection = pygsheets.authorize(service_file = authorisToken)
        workBook = Connection.open(workBookName)
        workSheet = workBook.worksheet_by_title(worksheetName)
        workSheet.clear(start = colStart, end = colEnd)
        workSheet.set_dataframe(data, start=start,copy_index=False)
        print(f"Data inserted into workBook:{workBookName} > WorkSheet:{worksheetName}...")
    
    except Exception as E:
        print(E)

def getNorthSouthData():
    dataNorth = getDataFromSheets('New Assortment Data North', 'North Assorted Data')
    dataSouth = getDataFromSheets('New Assortment Data South', 'South Assorted Data')
    data = dataNorth.append(dataSouth)
    return data


def power_float_list(a,power):
    b=[]
    for i in list(a):
        b.append(i**power)
    return b

def positive_error(a):
    count=0
    sum_=0
    for x in list(a):
        if x>0:
            count+=1
            sum_+=x
        else:
            pass
    return sum_/count
def negative_error(a):
    count=0
    sum_=0
    for x in list(a):
        if x<0:
            count+=1
            sum_+=x
        else:
            pass
    return sum_/count
# a = np.subtract(predict_LinearRegression,train_set_labels)
# error = min(abs(negative_error(a)),positive_error(a))

def count(a):
    i=0
    for x in a:
        i=i+1
    return i



def changeFromPercentage(value):
    if "%" in str(value):
        value = value.strip("%")
        value = (float(value))/100
    else:
        value = float(value)
    return value



def columns_unique(df):
    for x in list(df.columns):
        print(x,len(df[x].unique()))


def count_unique(x):
    return pd.Series(x).nunique()

count_unique.__name__ = 'count_unique'

# def relation(categories_column, continuous_column):
#     categories = np.unique(categories_column)
#     print(categories)
#     TSS = (np.var(continuous_column))*len(continuous_column)
#     print(TSS)
#     df = pd.DataFrame(list(zip(categories_column,continuous_column)),columns=['categories_column','continuous_column'])
#     print(df)
#     RSS=0
#     for x in categories:
#         a=df.loc[df['categories_column']==x,'continuous_column']
#         print(a)
#         RSS+=np.var(np.array(a))*len(np.array(a))
#         print(RSS)
#     r_square = 1 - (RSS/TSS)
#     print(r_square) list,range_list,new_column):
#     i=-1
#     for x in search_list:
#         i=i+1
#         for y in range_list:
#             if x==y:
#                 new_column[i]='yes'
#                 break
#             else:
#                 pass

def relation_entropy(categories_column1,categories_column2):
    categories = np.unique(categories_column1)
    print(categories)
    a,b = np.unique(np.array(categories_column2),return_counts=True)
    print(a,b)
    c = np.divide(b,np.sum(b))
    print(c)
    total_entropy = sp.stats.entropy(c,base=len(c))
    print(total_entropy)
    df = pd.DataFrame(list(zip(categories_column1,categories_column2)), columns=['categories_column1','categories_column2'])
    print(df)
    total_residual_entropy=0
    for x in categories:        
        d = df.loc[df['categories_column1']==x,'categories_column2']
        print(np.array(d))
        e,f = np.unique(np.array(d),return_counts=True)
        print(e,f)
        g = np.divide(f,np.sum(f))
#         print(len(g))
        if len(g)==1:
            pass
        else:
            total_residual_entropy+= sp.stats.entropy(np.array(g),base=len(g))
        print(total_residual_entropy)
#     print(total_residual_entropy)
    mean_residual_entropy=total_residual_entropy/len(categories_column1)
    print(mean_residual_entropy)
    accuracy = 1-(mean_residual_entropy/total_entropy)
    print(accuracy)

def correct_city(data):
    data.loc[data["city"].isin(['Faridabad','Gurgaon','Delhi','Noida','New Delhi','Ghaziabad','Delhi West','Meerut','Rohtak','Panipat','Sonipat','Goutam  Budd  Nagar']), "city"] = "Delhi NCR"
    data.loc[data["city"].isin(['BANGALORE CENTRAL','BANGALORE SOUTH','BANGLORE-CAFE DLY','BANGALORE K R PURAM','Bengaluru','Bangaloresouth','Electronic city']), "city"] = "Bangalore"
    data.loc[data["city"].isin(['Pune','PUNE EAST','PUNE SOUTH']), "city"] = "Pune"
    data.loc[data["city"].isin(['Hyderabad','HYDERABAD NORTH','HYDERABAD WEST']), "city"] = "Hyderabad"
    data.loc[data["city"].isin(['AHMEDABAD EAST','Ahmedabad','Vadodara','GANDHI NAGAR','Gandhinagar']), "city"] = "Ahmedabad"
    data.loc[data["city"].isin(['Chennai','CHENNAI WEST','CHENNAI SOUTH','CHENNAI ATC']), "city"] = "Chennai"
    data.loc[data["city"].isin(['Kolkata','Kolkatta','Kolkatta South','Howrah']), "city"] = "Kolkata"
    data.loc[data["city"].isin(['Mumbai','Navi Mumbai','NAVIMUMBAI','MUMBAI ATC','MUMBAI EAST','MUMBAI WEST','Thane (W)','Thane','Panvel','Kalyan','Vasai','Anand','Dombivali']), "city"] = "Mumbai"
    data.loc[data["city"].isin( ['Kochi','Ernakulam']), "city"] = "Kochi"

# Classification_Models

from sklearn.ensemble import RandomForestClassifier
model_RandomForestClassifier = RandomForestClassifier(max_depth=2,n_estimators=10,random_state=0)

from sklearn.tree import DecisionTreeClassifier
model_DecisionTreeClassifier = DecisionTreeClassifier()

from sklearn.linear_model import LogisticRegression
model_LogisticRegression = LogisticRegression()

from sklearn.svm import SVC
model_SVC = SVC()



# Regression_Models

from sklearn.linear_model import LinearRegression
model_LinearRegression = LinearRegression()

from sklearn.ensemble import RandomForestRegressor
model_RandomForestRegressor = RandomForestRegressor(n_estimators=10,max_depth=2,random_state=0)

from sklearn.tree import DecisionTreeRegressor
model_DecisionTreeClassifier = DecisionTreeRegressor()

from sklearn.svm import SVR
model_SVR = SVR()



# Metrics
from sklearn.metrics import mean_squared_error,r2_score,mean_squared_log_error,accuracy_score,confusion_matrix,precision_score,recall_score
from scipy.stats import pearsonr

# standard scaler
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()



def match(value,range_):
    ans = 1
    for x in range_:
        if x==value :
            return ans
        else:
            ans = ans+1


            
# to remove outliers of the given column
def remove_outliers(data,column,method='percentile',percentile=5,std=3):
    import numpy as np
    if method=='percentile':
        return data.loc[(data[column]>np.percentile(data[column],percentile))&(data[column]<np.percentile(data[column],100-percentile)),:]
    elif method=='interquartile_range':
        percentile_25=np.percentile(data[column],25)
        percentile_75=np.percentile(data[column],75)
        iqr = percentile_75-percentile_25
        cutoff = iqr*1.5
        upper=percentile_75+cutoff
        lower=percentile_25-cutoff
        return data.loc[(data[column]>lower)&(data[column]<upper),:]
    elif method=='standard_deviation':
        deviation=np.std(data[column])
        cutoff=deviation*std
        mean=np.mean(data[column])
        upper=mean+cutoff
        lower=mean-cutoff
        return data.loc[(data[column]>lower)&(data[column]<upper),:]
    else:
        return None
        
        
        
        
# scaling the data
def scaling_data(data):
    from sklearn.preprocessing import StandardScaler
    scaler=StandardScaler()
    scaled_data=pd.DataFrame(scaler.fit_transform(data))
    scaled_data.columns=data.columns
    return scaled_data



def rescaling_data(output,data,label):
    import numpy as np
    mean=np.mean(data[label])
    std = np.std(data[label])
    return output*std+mean

def models_regression(train_set_features,train_set_labels,test_set_features,test_set_labels):
    from sklearn.metrics import mean_squared_error,mean_squared_log_error,accuracy_score,confusion_matrix,r2_score
    from sklearn.linear_model import LinearRegression
    model_LinearRegression = LinearRegression()
    model_LinearRegression.fit(train_set_features,train_set_labels)
    predict_LinearRegression = model_LinearRegression.predict(test_set_features)
    rmse_LinearRegression  = np.sqrt(mean_squared_error(test_set_labels,predict_LinearRegression))
    error_percent_LinearRegression = rmse_LinearRegression/np.median(test_set_labels)*100
    print(f"LinearRegression-> error_percent:{error_percent_LinearRegression} , rmse:{rmse_LinearRegression} , median:{np.median(test_set_labels)}")
    from sklearn.svm import SVR
    model_SVR = SVR()
    model_SVR.fit(train_set_features,train_set_labels)
    predict_SVR = model_SVR.predict(test_set_features)
    rmse_SVR  = np.sqrt(mean_squared_error(test_set_labels,predict_SVR))
    error_percent_SVR = rmse_SVR/np.median(test_set_labels)*100
    print(f"SVR-> error_percent:{error_percent_SVR} , rmse:{rmse_SVR} , median:{np.median(test_set_labels)}")
    from sklearn.tree import DecisionTreeRegressor
    model = DecisionTreeRegressor()
    model.fit(train_set_features,train_set_labels)
    predict = model_DecisionTreeClassifier.predict(test_set_features)
    rmse  = np.sqrt(mean_squared_error(test_set_labels,predict_DecisionTreeClassifier))
    error_percent_DecisionTreeClassifier = rmse_DecisionTreeClassifier/np.median(test_set_labels)*100
    print(f"DecisionTreeClassifier-> error_percent:{error_percent_DecisionTreeClassifier} , rmse:{rmse_DecisionTreeClassifier} , median:{np.median(test_set_labels)}")
    from sklearn.ensemble import RandomForestRegressor
    model_RandomForestRegressor = RandomForestRegressor(n_estimators=10,max_depth=2,random_state=0)
    model_RandomForestRegressor.fit(train_set_features,train_set_labels)
    predict_RandomForestRegressor = model_RandomForestRegressor.predict(test_set_features)
    rmse_RandomForestRegressor  = np.sqrt(mean_squared_error(test_set_labels,predict_RandomForestRegressor))
    error_percent_RandomForestRegressor = rmse_RandomForestRegressor/np.median(test_set_labels)*100
    print(f"RandomForestRegressor-> error_percent:{error_percent_RandomForestRegressor} , rmse:{rmse_RandomForestRegressor} , median:{np.median(test_set_labels)}")


def models_regression(train_set_features,train_set_labels,test_set_features,test_set_labels):
    from sklearn.metrics import mean_squared_error,mean_squared_log_error,accuracy_score,confusion_matrix,r2_score
    from sklearn.linear_model import LinearRegression
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.ensemble import RandomForestRegressor
    import xgboost as xgb
    for model in [LinearRegression(),DecisionTreeRegressor(max_depth=10),RandomForestRegressor(n_estimators=50,max_depth=10,random_state=0),xgb.XGBRegressor(n_estimators=50,reg_lambda=1,gamma=0,max_depth=10)]
        model.fit(train_set_features,train_set_labels)
        predict = model.predict(test_set_features)
        rmse = np.sqrt(mean_squared_error(test_set_labels,predict))
        error_percent = rmse/np.median(test_set_labels)*100
        r_square=r2_score(test_set_labels,predict)
        print(f"model-> error_percent:{error_percent} , rmse:{rmse} , median:{np.median(test_set_labels)}, r-square:{r_square}")



    
def models_classification(train_set_features,train_set_labels,test_set_features,test_set_labels):
    from sklearn.metrics import mean_squared_error,mean_squared_log_error,accuracy_score,confusion_matrix,r2_score
    from sklearn.linear_model import LogisticRegression
    model_LogisticRegression = LogisticRegression()
    model_LogisticRegression.fit(train_set_features,train_set_labels)
    predict_LogisticRegression = model_LogisticRegression.predict(test_set_features)
    accuracy_LogisticRegression= accuracy_score(y_true=test_set_labels,y_pred=predict_LogisticRegression)
    confusion_matrix_LogisticRegression = pd.DataFrame(confusion_matrix(y_true=test_set_labels,y_pred=predict_LogisticRegression))
    print(f"LogisticRegression-> accuracy:{accuracy_LogisticRegression}, confusion_matrix:{confusion_matrix_LogisticRegression}")
    from sklearn.svm import SVC
    model_SVC = SVC()
    model_SVM.fit(train_set_features,train_set_labels)
    predict_SVM = model_SVM.predict(test_set_features)
    accuracy_SVM= accuracy_score(y_true=test_set_labels,y_pred=predict_SVM)
    confusion_matrix_SVM = pd.DataFrame(confusion_matrix(y_true=test_set_labels,y_pred=predict_SVM))
    print(f"SVM-> accuracy:{accuracy_SVM}, confusion_matrix:{confusion_matrix_SVM}")
    from sklearn.tree import DecisionTreeClassifier
    model_DecisionTreeClassifier = DecisionTreeClassifier()
    model_DecisionTreeClassifier.fit(train_set_features,train_set_labels)
    predict_DecisionTreeClassifier = model_DecisionTreeClassifier.predict(test_set_features)
    accuracy_DecisionTreeClassifier= accuracy_score(y_true=test_set_labels,y_pred=predict_DecisionTreeClassifier)
    confusion_matrix_DecisionTreeClassifier = pd.DataFrame(confusion_matrix(y_true=test_set_labels,y_pred=predict_DecisionTreeClassifier))
    print(f"DecisionTreeClassifier-> accuracy:{accuracy_DecisionTreeClassifier}, confusion_matrix:{confusion_matrix_DecisionTreeClassifier}")
    from sklearn.ensemble import RandomForestClassifier
    model_RandomForestClassifier = RandomForestClassifier(max_depth=2,n_estimators=10,random_state=0)
    model_RandomForestClassifier.fit(train_set_features,train_set_labels)
    predict_RandomForestClassifier = model_RandomForestClassifier.predict(test_set_features)
    accuracy_RandomForestClassifier= accuracy_score(y_true=test_set_labels,y_pred=predict_RandomForestClassifier)
    confusion_matrix_RandomForestClassifier = pd.DataFrame(confusion_matrix(y_true=test_set_labels,y_pred=predict_RandomForestClassifier))
    print(f"RandomForestClassifier-> accuracy:{accuracy_RandomForestClassifier}, confusion_matrix:{confusion_matrix_RandomForestClassifier}")

    

select * from stl_load_errors order by starttime desc limit 10;


# copy/create table from another table in redshift
CREATE TABLE TestTable AS
SELECT customername, contactname
FROM customers;

    
    
    
        
        
# models selection loop
def models_iteration(model_type,train_set_features,train_set_labels,test_set_features,test_set_labels):
    def models_classification(train_set_features,train_set_labels,test_set_features,test_set_labels):
        from sklearn.metrics import mean_squared_error,mean_squared_log_error,accuracy_score,confusion_matrix,r2_score
        from sklearn.linear_model import LogisticRegression
        model_LogisticRegression = LogisticRegression()
        model_LogisticRegression.fit(train_set_features,train_set_labels)
        predict_LogisticRegression = model_LogisticRegression.predict(test_set_features)
        accuracy_LogisticRegression= accuracy_score(y_true=test_set_labels,y_pred=predict_LogisticRegression)
        confusion_matrix_LogisticRegression = pd.DataFrame(confusion_matrix(y_true=test_set_labels,y_pred=predict_LogisticRegression))
        print(f"LogisticRegression-> accuracy:{accuracy_LogisticRegression}, confusion_matrix:{confusion_matrix_LogisticRegression}")
        from sklearn.svm import SVC
        model_SVC = SVC()
        model_SVM.fit(train_set_features,train_set_labels)
        predict_SVM = model_SVM.predict(test_set_features)
        accuracy_SVM= accuracy_score(y_true=test_set_labels,y_pred=predict_SVM)
        confusion_matrix_SVM = pd.DataFrame(confusion_matrix(y_true=test_set_labels,y_pred=predict_SVM))
        print(f"SVM-> accuracy:{accuracy_SVM}, confusion_matrix:{confusion_matrix_SVM}")
        from sklearn.tree import DecisionTreeClassifier
        model_DecisionTreeClassifier = DecisionTreeClassifier()
        model_DecisionTreeClassifier.fit(train_set_features,train_set_labels)
        predict_DecisionTreeClassifier = model_DecisionTreeClassifier.predict(test_set_features)
        accuracy_DecisionTreeClassifier= accuracy_score(y_true=test_set_labels,y_pred=predict_DecisionTreeClassifier)
        confusion_matrix_DecisionTreeClassifier = pd.DataFrame(confusion_matrix(y_true=test_set_labels,y_pred=predict_DecisionTreeClassifier))
        print(f"DecisionTreeClassifier-> accuracy:{accuracy_DecisionTreeClassifier}, confusion_matrix:{confusion_matrix_DecisionTreeClassifier}")
        from sklearn.ensemble import RandomForestClassifier
        model_RandomForestClassifier = RandomForestClassifier(max_depth=2,n_estimators=10,random_state=0)
        model_RandomForestClassifier.fit(train_set_features,train_set_labels)
        predict_RandomForestClassifier = model_RandomForestClassifier.predict(test_set_features)
        accuracy_RandomForestClassifier= accuracy_score(y_true=test_set_labels,y_pred=predict_RandomForestClassifier)
        confusion_matrix_RandomForestClassifier = pd.DataFrame(confusion_matrix(y_true=test_set_labels,y_pred=predict_RandomForestClassifier))
        print(f"RandomForestClassifier-> accuracy:{accuracy_RandomForestClassifier}, confusion_matrix:{confusion_matrix_RandomForestClassifier}")
    def models_regression(train_set_features,train_set_labels,test_set_features,test_set_labels):
        from sklearn.metrics import mean_squared_error,mean_squared_log_error,accuracy_score,confusion_matrix,r2_score
        from sklearn.linear_model import LinearRegression
        model_LinearRegression = LinearRegression()
        model_LinearRegression.fit(train_set_features,train_set_labels)
        predict_LinearRegression = model_LinearRegression.predict(test_set_features)
        rmse_LinearRegression  = np.sqrt(mean_squared_error(test_set_labels,predict_LinearRegression))
        error_percent_LinearRegression = rmse_LinearRegression/np.median(test_set_labels)*100
        print(f"LinearRegression-> error_percent:{error_percent_LinearRegression} , rmse:{rmse_LinearRegression} , median:{np.median(test_set_labels)}")
        from sklearn.svm import SVR
        model_SVR = SVR()
        model_SVR.fit(train_set_features,train_set_labels)
        predict_SVR = model_SVR.predict(test_set_features)
        rmse_SVR  = np.sqrt(mean_squared_error(test_set_labels,predict_SVR))
        error_percent_SVR = rmse_SVR/np.median(test_set_labels)*100
        print(f"SVR-> error_percent:{error_percent_SVR} , rmse:{rmse_SVR} , median:{np.median(test_set_labels)}")
        from sklearn.tree import DecisionTreeRegressor
        model_DecisionTreeClassifier = DecisionTreeRegressor()
        model_DecisionTreeClassifier.fit(train_set_features,train_set_labels)
        predict_DecisionTreeClassifier = model_DecisionTreeClassifier.predict(test_set_features)
        rmse_DecisionTreeClassifier  = np.sqrt(mean_squared_error(test_set_labels,predict_DecisionTreeClassifier))
        error_percent_DecisionTreeClassifier = rmse_DecisionTreeClassifier/np.median(test_set_labels)*100
        print(f"DecisionTreeClassifier-> error_percent:{error_percent_DecisionTreeClassifier} , rmse:{rmse_DecisionTreeClassifier} , median:{np.median(test_set_labels)}")
        from sklearn.ensemble import RandomForestRegressor
        model_RandomForestRegressor = RandomForestRegressor(n_estimators=10,max_depth=2,random_state=0)
        model_RandomForestRegressor.fit(train_set_features,train_set_labels)
        predict_RandomForestRegressor = model_RandomForestRegressor.predict(test_set_features)
        rmse_RandomForestRegressor  = np.sqrt(mean_squared_error(test_set_labels,predict_RandomForestRegressor))
        error_percent_RandomForestRegressor = rmse_RandomForestRegressor/np.median(test_set_labels)*100
        print(f"RandomForestRegressor-> error_percent:{error_percent_RandomForestRegressor} , rmse:{rmse_RandomForestRegressor} , median:{np.median(test_set_labels)}")
    if model_type=='classification':
        return models_classification(train_set_features,train_set_labels,test_set_features,test_set_labels)
    elif model_type=='regression':
        return models_regression(train_set_features,train_set_labels,test_set_features,test_set_labels)
    else:
        None


###################################################################################################################

#feature importance/ feature selection - idea is to calculate score -> drop one feature -> feature_importance = reduction in score

###################################################################################################################
def imp_df(column_names, importances):
    df = pd.DataFrame({'feature': column_names,
    'feature_importance': importances})
    .sort_values('feature_importance', ascending = False)
    .reset_index(drop = True)
    return df

from sklearn.base import clone 

def drop_col_feat_imp(model, X_train, y_train, random_state = 42):
    
    # clone the model to have the exact same specification as the one initially trained
    model_clone = clone(model)
    # set random_state for comparability
    model_clone.random_state = random_state
    # training and scoring the benchmark model
    model_clone.fit(X_train, y_train)
    benchmark_score = model_clone.score(X_train, y_train)
    # list for storing feature importances
    importances = []
    
    # iterating over all columns and storing feature importance (difference between benchmark and new model)
    for col in X_train.columns:
        model_clone = clone(model)
        model_clone.random_state = random_state
        model_clone.fit(X_train.drop(col, axis = 1), y_train)
        drop_col_score = model_clone.score(X_train.drop(col, axis = 1), y_train)
        importances.append(benchmark_score - drop_col_score)
    
    importances_df = imp_df(X_train.columns, importances)
    return importances_df
#%%


        
# defines whether the columns is numerical/continuous vs categorical/discrete
def column_type(data,column):
    if (('int' in str(type(list(data[column])[0]))) or ('float' in str(type(list(data[column])[0])))):
        return 'numerical'
    else:
        return 'categorical'




# returns the respective parameter value and the graph
# used to understand relationship between feature and the label columns
def feature_importance(data,feature_column,label_column):
    def column_type(data,column):
        if (('int' in str(type(list(data[column])[0]))) or ('float' in str(type(list(data[column])[0])))):
            return 'numerical'
        else:
            return 'categorical'
    def gini_impurity(data,feature_column,label_column):
        data=data[[feature_column,label_column]]
        data.dropna(how='any',inplace=True)
        gini_impurity_list=[]
        for col_index in data[feature_column].value_counts().index:
            data1=data.loc[data[feature_column]==col_index,:]
            sum_p=0
            for index in data1[label_column].value_counts().index:
                p=data1[label_column].value_counts()[index]/data1[label_column].value_counts().sum()
                sum_p = sum_p+p**2
            gini_impurity=1-sum_p
            gini_impurity_list.append(gini_impurity)
        feature_value_count = np.array(data[feature_column].value_counts())/len(data[feature_column])
        gini_impurity_feature=np.sum(np.array(gini_impurity_list)*feature_value_count)
        return gini_impurity_feature
    def standard_deviation_reduction(data,feature_column,label_column):
        data=data[[feature_column,label_column]]
        data.dropna(how='any',inplace=True)
        std_before_split=np.std(np.array(data[label_column]))
        print(f'std_before_split:{std_before_split}')
        std_list=[]
        for col_index in data[feature_column].value_counts().index:
            data1=data.loc[data[feature_column]==col_index,label_column]
            std=np.std(np.array(data1))
            std_list.append(std)
            print(f'col_index:{col_index} , std:{std}, std_list:{std_list}')
        feature_value_count = np.array(data[feature_column].value_counts())/len(data[feature_column])
        std_weighted=np.sum(np.array(std_list)*feature_value_count)
        std_reduction=std_before_split-std_weighted
        return std_reduction
    if column_type(data=data,column=feature_column)=='numerical' and column_type(data=data,column=label_column)=='numerical':
        return sns.relplot(data=data,x=feature_column,y=label_column,kind='scatter'),np.corrcoef(x=data[feature_column],y=data[label_column])
    elif column_type(data=data,column=feature_column)=='numerical' and column_type(data=data,column=label_column)=='categorical':
        return sns.catplot(data=data,x=label_column,y=feature_column,kind='violin'),standard_deviation_reduction(data=data,feature_column=label_column,label_column=feature_column)
    elif column_type(data=data,column=feature_column)=='categorical' and column_type(data=data,column=label_column)=='numerical':
        return sns.catplot(data=data,x=feature_column,y=label_column,kind='violin'),standard_deviation_reduction(data=data,feature_column=feature_column,label_column=label_column)
    elif column_type(data=data,column=feature_column)=='categorical' and column_type(data=data,column=label_column)=='categorical':
        return sns.catplot(data=data,x=feature_column,kind='bar',hue=label_column),gini_impurity(data,feature_column,label_column)
    else:
        pass
        
   
pd.datetime(listing_lead['time'],error='coerce')
     

def get_cellvalue_from_sheets(sheet,tab,cell):
    # authorize the clientsheet 
    client = gspread.authorize(creds)
    sheet = client.open(sheet)
    worksheet=sheet.worksheet(tab)

    ##final_code

    return worksheet.acell(cell).value


def update_cellvalue_in_sheets(sheet,tab,cell,value_to_be_updated):
    # authorize the clientsheet 
    client = gspread.authorize(creds)
    sheet = client.open(sheet)
    worksheet=sheet.worksheet(tab)

    ##final_code

    worksheet.update(cell,value_to_be_updated)


def eda_graphs(data,feature_column,label_column):
    def column_type(data,column):
        if (('int' in str(type(list(data[column])[0]))) or ('float' in str(type(list(data[column])[0])))):
            return 'numerical'
        else:
            return 'categorical'
    if column_type(data=data,column=feature_column)=='numerical' and column_type(data=data,column=label_column)=='numerical':
        return sns.relplot(data=data,x=feature_column,y=label_column,kind='scatter')
    elif column_type(data=data,column=feature_column)=='numerical' and column_type(data=data,column=label_column)=='categorical':
        return sns.catplot(data=data,x=label_column,y=feature_column,kind='violin')
    elif column_type(data=data,column=feature_column)=='categorical' and column_type(data=data,column=label_column)=='numerical':
        return sns.catplot(data=data,x=feature_column,y=label_column,kind='violin')
    elif column_type(data=data,column=feature_column)=='categorical' and column_type(data=data,column=label_column)=='categorical':
        return sns.catplot(data=data,x=feature_column,kind='bar',hue=label_column)
    else:
        pass


def get_cellrange_from_sheets(sheet,tab,row_initial=1,col_initial=1,row_final=10,col_final=10):
    client = gspread.authorize(creds)
    sheet = client.open(sheet)
    worksheet=sheet.worksheet(tab)
    list2=[]
    for row in range(row_initial,row_final+1,1):
        time.sleep(1.2)
        list1=[]
        for col in range(col_initial,col_final+1,1):
            time.sleep(1.2)
            list1.append(worksheet.cell(row,col).value)
        list2.append(list1)
    df = pd.DataFrame(list2)
    return df

def importing_libraries():
    import gspread
    import pandas as pd
    import numpy as np
    import datetime
    from gspread_dataframe import get_as_dataframe, set_with_dataframe
    import time
    from oauth2client.service_account import ServiceAccountCredentials
    import pygsheets, json
    from tqdm import tqdm
    from pymongo import MongoClient
    import datetime, time
    import ssl
    import warnings
    warnings.filterwarnings("ignore")
    scope = ['https://spreadsheets.google.com/feeds','https://www.googleapis.com/auth/drive']
    # add credentials to the account
    global creds
    creds = ServiceAccountCredentials.from_json_keyfile_name(r'C:\Users\Shubham Mantri\Downloads\coastal-airlock-315709-d614c773b55d.json', scope)
    pd.set_option('max_columns',None)
    

def set_dark_theme():
    import jupyterthemes as jt
    from jupyterthemes.stylefx import set_nb_theme
    set_nb_theme('onedork')

# import jupyterthemes as jt
# from jupyterthemes.stylefx import set_nb_theme
# set_nb_theme('onedork')

from tqdm import tqdm_notebook, tnrange
import time 
for i in tqdm_notebook(range(0,10,1),desc='1st loop'):
    for j in tqdm_notebook(range(0,10,1),desc='2nd loop',leave=False):
        time.sleep(1)

success = False
# while not success:
try:
    value = input('please enter an integer')
    a = int(value)
    success = True
except:
    pass
a

def dropna_numpy(array):
    array = array[~np.isnan(array)]
    return array

# just give arguments in string format

def plot_graph(y,range_default='[x/10 for x in range(-300,300,1)]'):
    import matplotlib.pyplot as plt
    import numpy as np
    z=np.array(eval(range_default))
    x=z
    y=eval(y)
    plt.plot(z,y)
    

def input_updated(input_statement):
    try:
        x=input(input_statement)
        x=int(x)
        return x
    except:
        try:
            x=float(x)
            return x
        except:
            x
    

def left(s, amount):
    return s[:amount]

def right(s, amount):
    return s[-amount:]

def mid(s, offset, amount):
    return s[offset:offset+amount]

# mid('abcdeghij',4,2)->'eg'


def date_difference(date1,date2,requirement='days'):
    import math
    diff=date2-date1
    total_seconds=diff.days*24*60*60+diff.seconds
    if requirement=='months':
        math.floor(total_seconds/(30*24*60*60))
    elif requirement=='days':
        return math.floor(total_seconds/(24*60*60))
    elif requirement=='hours':
        return math.floor(total_seconds/(60*60))
    elif requirement=='minutes':
        return math.floor(total_seconds/(60))
    elif requirement=='seconds':
        return total_seconds

# return the dataframe for the given key
# key should be given in tuple form


def groupby_personal(data,by,key):
    return data.iloc[list(data.groupby(by=by).groups[key]),:]

def return_multiple_values(*argv):
    dictionary={}
    for arg in argv:
        dictionary[arg]=arg
    return dictionary
        
        
def train_validation_test(data,label_column,validation_size=0.25,test_size=0.2):
    from sklearn.model_selection import train_test_split
    X=data.drop(columns=label_column)
    y=data[label_column]
    # train_feature,train_label,test_feature,test_label --> use this for variable
    X_train, X_test, y_train, y_test=train_test_split(X,y, test_size=test_size, random_state=42)
    return X_train,y_train,X_test,y_test


def train_validation_test(data,label_column,validation_size=0.25,test_size=0.2):
    from sklearn.model_selection import train_test_split
    X=data.drop(columns=label_column)
    y=data[label_column]
    # train_feature,train_label,validation_feature,validation_label,test_feature,test_label --> use this for variable
    X_train, X_test, y_train, y_test=train_test_split(X,y, test_size=test_size, random_state=42)
    X_train_actual,X_validation,y_train_actual,y_validation=train_test_split(X_train,y_train, test_size=validation_size, random_state=42)
    return X_train_actual,y_train_actual,X_validation,y_validation,X_test,y_test


# data_to_add should be in transposed form
def insert_row(data_main,index,data_to_add):
    df1=data_main.iloc[:(index+1),:]
    df2=data_main.iloc[(index+1):,:]
    data_updated = pd.concat([df1,data_to_add,df2]).reset_index(drop=True)
    return data_updated


ott['i1_finish_date']=pd.to_datetime(ott['i1_finish_date'],errors='coerce')
lti['initial_run_date']=pd.to_datetime(lti['initial_run_date'],errors='coerce').dt.date


# one hot encoding
data=pd.get_dummies(data)

def col_level_selection(df,level):
    new_df=pd.DataFrame()
    for i in range(len(df.columns)):
        new_df[df.columns[i][level]]=df[df.columns[i]]
    return new_df


def checkIfPercentage(trigger):
    if "%" in str(trigger):
        trigger = trigger.replace(r"%", "")
        trigger = float(trigger)/100
        return trigger
    else:
        return trigger  
    #%%
    
# principal component analysis PCA for feature importance/ feature selection 
def pca(data,target_column,n_components_pca)
    from sklearn.decomposition import PCA
    x=data.drop(columns=[target_column])
    pca = PCA(n_components=n_components_pca)
    principalComponents = pca.fit_transform(x)
    principalDf = pd.DataFrame(data = principalComponents, columns = [f'principal_component_{i}' for i in range(1,n_components_pca+1)])
    finalDf = pd.concat([principalDf, df[[target_column]]], axis = 1)
        
#%%


# chi2 and chi-square test for feature importance/ feature selection
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
chi2_selector = SelectKBest(chi2, k=2)
X_kbest = chi2_selector.fit(X, y)




#%%

# GridSearchCV for Hyperparameter tuning
from sklearn.model_selection import GridSearchCV
param_grid={'eta':[0.1,0.3],'max_depth':[6,10,14,18],'lambda':[1,2],'gamma':[0,0.02]}
grid = GridSearchCV(xgb.XGBRegressor(), param_grid, refit = True, verbose = 3)
grid.fit(train_feature,train_label)
print(grid.best_params_)


# personal hyperparameter tuning
array_final=[]

for n_estimators in [50,75,100,150]:
    for max_depth in [6,10,15,20]:
        for gamma in [0,0.02]:
            for reg_lambda in [1,2]:
                for eta in [0.1,0.2,0.3]:
                    print(n_estimators,'/',max_depth,'/',gamma,'/',reg_lambda,'/',eta)
                    model = xgb.XGBRegressor(eta=eta,n_estimators=n_estimators,reg_lambda=reg_lambda,gamma=gamma,max_depth=max_depth)
                    model.fit(train_feature,train_label)
                    predict=model.predict(test_feature)
                    rmse = np.sqrt(mean_squared_error(test_label,predict))
                    error_percent = rmse/np.median(test_label)*100
                    r_square=r2_score(test_label,predict)
                    array=[n_estimators,max_depth,gamma,reg_lambda,eta,rmse,r_square]
                    array_final.append(array)
                    print(f"XGBRegressor-> error_percent:{error_percent} , rmse:{rmse} , median:{np.median(test_label)}, r-square:{r_square}")
        
result=pd.DataFrame(array_final,columns=['n_estimators','max_depth','gamma','reg_lambda','eta','rmse','r_square'])     



#%%

# fisher score for feature importance/ feature selection
from skfeature.function.similarity_based import fisher_score
rank = fisher_score.fisher_score(X,y)


def max_rows_columns_visible(a):
    pd.set_option('display.max_columns',a)
    pd.set_option('display.max_rows',a)
    
    
def unique_in_col_df(data)    
    unique={}
    for i in range(0,data.shape[1],1):
        # print(i)
        unique[data.columns[i]]=len(data.iloc[:,i].unique())
        
    unique_df=pd.DataFrame(unique,index=[0]).T
    return unique_df


    
def interpolated_value(tppChangePct,winpct_needed):  
    b=np.array(tppChangePct)
    length= len(b)
    gap=round(100/length,2)
    pct = np.array([i*gap for i in range(1,100) if i*gap < 200])[0:length]
    index=length-(pct>=winpct_needed).sum()-1
    result=tppChangePct[index]+(tppChangePct[index+1]-tppChangePct[index])/gap*(winpct_needed-pct[index])
    return round(result,2)

def payAttention():
    import time
    import winsound
    frequency = 2500  # Set Frequency To 2500 Hertz
    duration = 10000  # Set Duration To 1000 ms == 1 second
    winsound.Beep(frequency, duration)
    time.sleep(10)
    
    
def status(CEP,TPP):
    if pd.isna(CEP) or pd.isna(TPP):
        return ""
    elif CEP > TPP:
        return "Loss"
    elif TPP >= CEP:
        return "Win"
    
def linear_interpolation(x1,y1,x2,y2,x):
    y=(y2-y1)/(x2-x1)*(x-x1)+y1
    return y


def filtered_dataframe_advance(data,*args):
    # breakpoint()
    for arg in args:
        breakpoint()
        data=data.loc[data[arg]==globals()[arg]]
    return data
model = filtered_dataframe_advance(lti_x,'Model','City')


# prefer using the basic one rather than advance
def filtered_dataframe_basic(data,**kwargs):
    # breakpoint()
    for key,value in kwargs.items():
        data=data.loc[data[key]==value]
    return data
model = filtered_dataframe_basic(lti_x,Model=model,City='Delhi NCR')



def getDataFromRedashCSV(url):
    count = 0
    try:
        requestContent = requests.get(url)
        byteData = requestContent.content
        stringData = byteData.decode('utf-8')
        stringData = StringIO(stringData)
        data = pd.read_csv(stringData)
        return data
    except Exception as E:
        count += 1
        if count == 3:
            print("Request connection error, try again after sometime...")
            print(E)
        else:
            getDataFromRedashCSV(url)
class CheckTodayData(Exception):
    """Exception raised for errors if Today data is not available
    Attributes:
        Date -- Today's date not present
        message -- explanation of the error
    """
    def __init__(self, message = "Today's data not present, refresh the redash query"):
        self.message = message
        super().__init__(self.message)
    def __str__(self):
        return f'{self.message}'
def CheckDataSanity(data):
    dates = pd.to_datetime(data["Date"]).dt.date
    todayDate = datetime.date.today()
    if (todayDate == dates.values).all():
        pass
    else:
        raise CheckTodayData()
        
       # OOPS 
class abcd():
    
    def __init__(self,name,age):
        self.name = name
        self.age = age
    
    def abc(self):
        print(f"hello {self.name} and {self.age} world")
        
        
x = abcd("Shubham",26)

x.abc()
print(x.name , x.age)




db = sql.connect(host ='spinny-web-read-replica-3.crevgiuvg8xk.ap-south-1.rds.amazonaws.com', user= 'jatinkapoor', password='cmwOS7IhX[7k-9zN', db='spinny')
query="select * from listing_lead"
pd.read_sql(query,db)
    
    
    
ott['i1_finish_date']=pd.to_datetime(ott['i1_finish_date'],errors='coerce')

data=data.loc[(data['Model']=='xuv500')&(data['fuel']=='diesel')&(data['City']=='Delhi NCR')&(data['Year_Bucket']=='2010-2012')]

del data['win_percent'],data['Gap'],data['Bucket_Gap']

data['c24bpp'] = pd.to_numeric(data["c24bpp"], errors = "coerce")

data['Trigger']=data['Trigger'].astype('float')

data["Refurb_Cost"] = data.apply(lambda x: addMinRefurbCost(minRefurbCost, x["Year_bucket"], x["Mileage"], x["Refurb_Cost"]), axis = 1)

data.loc[~(data[col].isin(['nan', "Q1", "Q2","Q3"])), col] = "Q3"

lowData = lowData.rename(columns = {"delta": "okToTokenTrigger"})


filterdate = (datetime.datetime.today() - datetime.timedelta(days=60))
filt = datetime.date(filterdate.year, filterdate.month, filterdate.day)

data['Created At'] = pd.to_datetime(data['Created At'])

dataOld30days = data[(data["City"].isin(CityOld)) & (data['Created At'] >= filt)]



group=group.replace({'':np.nan})

# sorting one column in ascending while other column is descending
data=data.sort_values(by=['quality_category','token_date'],ascending=[True,False])



# merging/joining the dataframe
dataMinMax = pd.merge(data, G1WinPerData, on = ["Model",'Year_Bucket', 'City', 'fuel'], how = 'left')


# appending/concating the dataframe
G1WinPerData = G1WinPerData.append(tempG1Data)
data = pd.concat([data1,data2],axis=0)



totalLeadsData.drop_duplicates(subset = ['Model','Year_Bucket', 'fuel', 'City'], inplace = True)

pivot=ott.pivot_table(index=['month_i1','Procurement Category','City','Year_Bucket'],aggfunc={'Model':'count','Seller_KYC_Date':'count'}).reset_index()


# city mapping

ott.loc[ott["City"].isin(['Faridabad','Gurgaon','Delhi','Noida','New Delhi','Ghaziabad','Delhi West','Meerut','Rohtak','Panipat','Sonipat','Goutam  Budd  Nagar','NCR']), "City"] = "Delhi NCR"
ott.loc[ott["City"].isin(['BANGALORE CENTRAL','BANGALORE SOUTH','BANGLORE-CAFE DLY','BANGALORE K R PURAM','Bengaluru','Bangaloresouth','Electronic City']), "City"] = "Bangalore"
ott.loc[ott["City"].isin(['Pune','PUNE EAST','PUNE SOUTH']), "City"] = "Pune"
ott.loc[ott["City"].isin(['Hyderabad','HYDERABAD NORTH','HYDERABAD WEST']), "City"] = "Hyderabad"
ott.loc[ott["City"].isin(['AHMEDABAD EAST','Ahmedabad','Vadodara','GANDHI NAGAR','Gandhinagar']), "City"] = "Ahmedabad"
ott.loc[ott["City"].isin(['Chennai','CHENNAI WEST','CHENNAI SOUTH','CHENNAI ATC']), "City"] = "Chennai"
ott.loc[ott["City"].isin(['Kolkata','Kolkatta','Kolkatta South','Howrah']), "City"] = "Kolkata"
ott.loc[ott["City"].isin(['Mumbai','Navi Mumbai','NAVIMUMBAI','MUMBAI ATC','MUMBAI EAST','MUMBAI WEST','Thane (W)','Thane','Panvel','Kalyan','Vasai','Anand','Dombivali']), "City"] = "Mumbai"
ott.loc[ott["City"].isin( ['Kochi','Ernakulam']), "City"] = "Kochi"
#%%
cityList = ['Pune', 'Jaipur', 'Kolkata', 'Mumbai', 'Delhi NCR', 'Hyderabad','Ahmedabad', 'Lucknow', 'Chandigarh', 'Indore', 'Bangalore','Chennai', 'Coimbatore','Kochi']
ott = ott[ott["City"].isin(cityList)]

CEPMean = CEPMean.groupby(['Model','fuel','City','Year_Bucket']).mean().reset_index().rename("total")

data['make_year']=data['make_year'].round(0)



# extracting month from the datetime
import datetime as dt
ott['month_i1']=ott['i1_finish_date'].dt.month

dt.datetime.today().strftime("%A")=='Monday'

data_x=data.groupby(by=["Model", "fuel",'Year_Bucket', "City"]).size().rename('Inspections').reset_index()

print(f"{'*'*50}\n{' '*10} The code ran Sucessfully\n{'*'*50}\n")


# using sql in python
lead_list = tuple(data['lead_id'].values)
query = f"""select l.id as lead_id,date(addtime(l.time,'05:30:00')) as creation_date
    from listing_lead as l
    where l.id in {lead_list}
"""
db = sql.connect(host ='spinny-web-read-replica-3.crevgiuvg8xk.ap-south-1.rds.amazonaws.com', user= 'jatinkapoor', password='cmwOS7IhX[7k-9zN', db='spinny')
df = pd.read_sql(query, db)




# comprehension
", ".join([i for i in df1 if value[col] in i])


playing with string/matching string
get_close_match

# regex explained

. -> one character
* -> zero or more characters
+ -> 1 or more characters
? -> 0 or 1 character
^a -> means string starting with a 
[0-8] -> means string containing any number 0 to 8
[^0-2] -> means string does not contain 0,1,2
b$ -> means string end with b
a|b -> means string either contains a or b
(a|b)xz -> means string contains axz or bxz == () for sub pattern
a{2,4} -> eg daat means a min repititions = 2 and max repititions = 4


def max_rows_columns():
    pd.set_option('max_columns',None)
    pd.set_option('max_rows',None)

[i for i in [2,3] if i not in [2,4]]

def profit(recall,precision,accuracy,premium,insurance_cover,no_of_orders):
    actual_fraud_predicted_fraud = 100*(1-accuracy)*(1/(1/recall + 1/precision - 2))
    actual_fraud_predicted_not_fraud = ((1-recall)/recall)*100*(1-accuracy)*(1/(1/recall + 1/precision - 2))
    actual_not_fraud_predicted_fraud = ((1-precision)/precision)*100*(1-accuracy)*(1/(1/recall + 1/precision - 2))
    actual_not_fraud_predicted_not_fraud = 100*accuracy - (100*(1-accuracy)*(1/(1/recall + 1/precision - 2)))
    profit = (((actual_fraud_predicted_fraud+actual_not_fraud_predicted_fraud+actual_not_fraud_predicted_not_fraud + actual_fraud_predicted_not_fraud)*premium - actual_fraud_predicted_not_fraud*insurance_cover)*no_of_orders)/100
    return profit

import pandas as pd

df = pd.DataFrame(columns=['recall','precision','profit'])
for recall in [x/100 for x in range(50,100,5)]:
    for precision in range(5,50,5):
        df = df.append({'recall':recall, 'precision':precision, 'profit':round(profit(0.67, precision, 0.90, premium, 300, 100),0)},ignore_index=True)
df

def profit(recall,precision,accuracy,premium,insurance_cover,no_of_orders):
    actual_fraud_predicted_fraud = 100*(1-accuracy)*(1/(1/recall + 1/precision - 2))
    actual_fraud_predicted_not_fraud = ((1-recall)/recall)*100*(1-accuracy)*(1/(1/recall + 1/precision - 2))
    actual_not_fraud_predicted_fraud = ((1-precision)/precision)*100*(1-accuracy)*(1/(1/recall + 1/precision - 2))
    actual_not_fraud_predicted_not_fraud = 100*accuracy - (100*(1-accuracy)*(1/(1/recall + 1/precision - 2)))
    profit = (((actual_fraud_predicted_fraud+actual_not_fraud_predicted_fraud+actual_not_fraud_predicted_not_fraud + actual_fraud_predicted_not_fraud)*premium - actual_fraud_predicted_not_fraud*insurance_cover)*no_of_orders)/100
    return profit

import pandas as pd

df = pd.DataFrame(columns=['precision','premium','profit'])
for precision in [x/100 for x in range(50,100,5)]:
    for premium in range(5,50,5):
        df = df.append({'precision':precision, 'premium':premium, 'profit':round(profit(0.67, precision, 0.90, premium, 300, 100),0)},ignore_index=True)
df

profit(2/3,2/3,0.8,20,100,100)

# ------------------ Predicted (horizontal) /Actual (vertical)
#|   a   |    b    |
#-------------------
#|   c   |    d    |
#-------------------

def confusion_matrix_values(recall,precision,accuracy,total_count):
    actual_fraud_predicted_fraud = total_count*(1-accuracy)*(1/(1/recall + 1/precision - 2)) # a
    actual_fraud_predicted_not_fraud = ((1-recall)/recall)*total_count*(1-accuracy)*(1/(1/recall + 1/precision - 2)) # b
    actual_not_fraud_predicted_fraud = ((1-precision)/precision)*total_count*(1-accuracy)*(1/(1/recall + 1/precision - 2)) # c
    actual_not_fraud_predicted_not_fraud = total_count*accuracy - (total_count*(1-accuracy)*(1/(1/recall + 1/precision - 2))) # d

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import re
pd.set_option('max_columns',None)
pd.set_option('max_rows',None)

def profit2(recall,precision,A,B,x,y,premium,insurance_cover, avg_order_value):
    profit = (A*recall + B - (A*recall/precision))*y - (A+B)*premium/avg_order_value - B*y + A*x + A*(1-recall)*insurance_cover/avg_order_value
    
    return profit

df = pd.DataFrame(columns=['recall','precision','profit'])
for recall in [x/100 for x in range(50,100,10)]:
    for precision in [x/100 for x in range(50,100,10)]:
        df = df.append({'recall':recall, 'precision':precision, 'profit':round(profit2(recall, precision, 3000, 7000,0.2,0.3,7.5,300,1500),0)},ignore_index=True)
df


def profit2(recall,precision,A,B,x,y,premium,insurance_cover, avg_order_value):
    a = A*recall
    b = A*(1-recall)
    c = A*recall/precision - A*recall
    d = B - (A*recall/precision - A*recall)
    profit = (A*recall + B - (A*recall/precision))*y - (A+B)*premium/avg_order_value - B*y + A*x + A*(1-recall)*insurance_cover/avg_order_value
    metric = a*(d**0.5)/(b*c)
    return profit,metric

df = pd.DataFrame(columns=['recall','precision','profit','metric'])
for recall in [x/100 for x in range(50,100,10)]:
    for precision in [x/100 for x in range(50,100,10)]:
        df = df.append({'recall':recall, 'precision':precision, 'profit':round(profit2(recall, precision, 3000, 7000,0.2,0.3,7.5,300,1500)[0],0),'metric':round(profit2(recall, precision, 3000, 7000,0.2,0.3,7.5,300,1500)[1],2)},ignore_index=True)
df.sort_values(['metric'],ascending=False)



import seaborn as sns
import matplotlib.pyplot as plt

plt.scatter(x = df['profit'] , y = df['metric'])

sns.relplot(df,x = 'profit', y='metric',kind = 'scatter')

from autocorrect import Speller
spell = Speller(lang='en')
def correct_spelling(word):
    return spell(word)

correct_spelling('jaipur')

# get details from the pincode
import requests
import json
pincode=560034
res = requests.get(f"https://api.postalpincode.in/pincode/{pincode}")
json.loads(res.text)

import requests

url = "https://api.countrystatecity.in/v1/countries/IN/states"

headers = {
  'X-CSCAPI-KEY': 'SUNqZjVLa0xMSGdudEZHT0RRQVlOTXA5N01ETlZiMnFpRFMwUEhPUw=='
}

response = requests.request("GET", url, headers=headers)



import pandas as pd
state_iso2 = pd.DataFrame(columns = ['state','iso2'])
for i in range(0,len(json.loads(response.text))):
    state_iso2 = state_iso2.append({'state':json.loads(response.text)[i]['name'],'iso2':json.loads(response.text)[i]['iso2']},ignore_index=True)
# print(json.loads(response.text)[0]['name'])

state_iso2

# get city detials from the state
import requests
import json

def city_list_of_state(state):
    url = f"https://api.countrystatecity.in/v1/countries/IN/states/{state_iso2.loc[state_iso2['state']==state.title(),'iso2'].values[0]}/cities"

    headers = {
      'X-CSCAPI-KEY': 'SUNqZjVLa0xMSGdudEZHT0RRQVlOTXA5N01ETlZiMnFpRFMwUEhPUw=='
    }

    response = requests.request("GET", url, headers=headers)
    
    city_list = []
    for i in range(0,len(json.loads(response.text))):
        city_list.append(json.loads(response.text)[i]['name'])
    return city_list

def cleaned_city(city):
    city = city.replace("*","")
    city = city.replace("."," ")
    city = city.replace("  "," ")
    city = city.replace("   "," ")
    city = city.strip()
    a=city.lower()
    b=city.lower()
    if "-" in city:
        a = city[:city.index("-")].lower()
    if "(" in city:
        b = city[:city.index("(")].lower()
    return a if len(a)<=len(b) else b

state_iso2.loc[state_iso2['state']==state,'iso2'].values[0]

state = 'Andhra Pradesh'

class RemoveOutlier():
    """
    This class is to remove outliers present using basic methodologies.
    """
    
    def __init__(self, method = 'standard_deviation'):
        self.method = method
        self.dict_ = None
        
    def fit_transform(self, X):
        """
        This first performs the fit and then transforms the given data
        """
        dict_ = {}
        for col in X.columns:
            #ignoring the categorical column
            if len(X[col].unique())<=2:
                continue
            if self.method=='interquartile_range':
                percentile_25=np.percentile(X[col],25)
                percentile_75=np.percentile(X[col],75)
                iqr = percentile_75-percentile_25
                cutoff = iqr*1.5
                upper=percentile_75+cutoff
                lower=percentile_25-cutoff
                dict_[col] = (lower, upper)
                X = X.loc[(X[col]>lower)&(X[col]<upper),:]
            if self.method == 'standard_deviation':
                std = 3
                deviation=np.std(X[col])
                cutoff=deviation*std
                mean=np.mean(X[col])
                upper=mean+cutoff
                lower=mean-cutoff
                dict_[col] = (lower, upper)                
                X = X.loc[(X[col]>lower)&(X[col]<upper),:]
        self.dict_ = dict_
        return X
    
    def fit(self, X):
        """
        this fits the X for outliers. idea is to basically find lower and upper for individual featuers.
        """
        dict_ = {}
        for col in X.columns:
            #ignoring the categorical column
            if len(X[col].unique())<=2:
                continue
            if self.method=='interquartile_range':
                percentile_25=np.percentile(X[col],25)
                percentile_75=np.percentile(X[col],75)
                iqr = percentile_75-percentile_25
                cutoff = iqr*1.5
                upper=percentile_75+cutoff
                lower=percentile_25-cutoff
                dict_[col] = (lower, upper)
                X = X.loc[(X[col]>lower)&(X[col]<upper),:]
            if self.method == 'standard_deviation':
                std = 3
                deviation=np.std(X[col])
                cutoff=deviation*std
                mean=np.mean(X[col])
                upper=mean+cutoff
                lower=mean-cutoff
                dict_[col] = (lower, upper)                
                X = X.loc[(X[col]>lower)&(X[col]<upper),:]
        #this is saved outside the function also**
        self.dict_ = dict_ 
        return self
    
    def transform(self, X):
        """
        this transforms the X using the parameters (lower, upper) based on trained model
        """
        dict_ = self.dict_
        for col in X.columns:
            if col not in dict_.keys():
                continue
            lower = dict_[col][0]
            upper = dict_[col][1]
            X = X.loc[(X[col]>lower)&(X[col]<upper),:]
        return X
        

                
            
            

class MyNaiveClassificationModel():
    """
    Do LABEL ENCODING instead of one hot encoding
    **issue to be resolved/ to do -> original X gets changed -> need to fix

    
    LOGIC:
    It first converts all columns to categorical columns then simply equate features from test data to train data (think of it as 
    creating buckets of features -> entire n-D space divided in buckets and then choose the right bucket given our test data)
    and then calculating percent and count for that bucket.
    """
    def __init__(self, num_buckets = 20, bins_method='percentile'):
        self.num_buckets = num_buckets
        self.fit_df = None
        self.dict_bins = None
        self.dict_labels = None
        self.bins_method = bins_method
        self.dict_min_ = None
        self.dict_max_ = None
        
    def fit(self, X,y):
        num_buckets = self.num_buckets
        dict_bins = {}
        dict_labels = {}
        dict_min_ = {}
        dict_max_ = {}
        bins_method = self.bins_method
        def col_bucket(X, col, num_buckets):
            if bins_method == 'percentile':
                #idea is to do binning on percentile -> make sure equal distribution along all the buckets
                bins = sorted(list(set([np.percentile(X[col],i) for i in range(0,101,int(100/num_buckets))])))
                max_ = None
                min_ = None
            else:
                #idea is to do binnig on minmaxscaled values with fixed intervals
                try:
                    max_=np.max(X[col])
                    min_=np.min(X[col])
                    X[col] = pd.Series([((i-min_)/(max_-min_)*100) for i in list(X[col])])
                    bins = [i*(100/num_buckets) for i in range(num_buckets+1)]
                except Exception as e:
                    #this condition should not come since len(X[col].unique())<=num_buckets+1 is already there
                    print(e)
                    pass
            labels = [i for i in range(1,len(bins))]
            X[col] = pd.cut(X[col],bins=bins,labels=labels)
            #to fill the lowest value in the column since it is not included in pd.cut
            X[col].fillna(1, inplace=True)
            return X, bins, labels, max_, min_

        for col in X.columns:
            #ignoring the categorical column
            if len(X[col].unique())<=num_buckets+1:
                continue
            X, dict_bins[col], dict_labels[col], dict_max_[col], dict_min_[col] = col_bucket(X,col, num_buckets)
        # print(X.info())
        X['y'] = np.array(y)
        # print(X.info())
        self.fit_df = X.astype(float)
        self.dict_bins = dict_bins
        self.dict_labels = dict_labels
        self.dict_max_ = dict_max_
        self.dict_min_ = dict_min_
        return self
    
    def predict_proba(self, X):
        dict_bins = self.dict_bins
        dict_labels = self.dict_labels
        dict_max_ = self.dict_max_
        dict_min_ = self.dict_min_
        num_buckets = self.num_buckets
        bins_method = self.bins_method
        #y_prob -> means probability of y being == '1'
        y_prob = []
        y_count = []
        for col in X.columns:
            if col not in dict_bins.keys(): #len(X[col].unique())<=num_buckets+1:
                continue
            if bins_method == 'percentile':
                pass
            else:
                try:
                    #idea is to do maxminscaling and handle values less than min in X_test
                    X[col] = pd.Series([(((i-dict_min_[col])/(dict_max_[col]-dict_min_[col])*100) if i>dict_min_[col] else 0) for i in list(X[col])])
                except Exception as e:
                    print(e)
            X[col] = pd.cut(X[col],bins=dict_bins[col],labels=dict_labels[col])
            #to handle lowest value of the column
            X[col].fillna(1, inplace=True)
        X.reset_index(drop=True, inplace=True)
        X = X.astype(float)
        """
        fit_df = self.fit_df
        fit_df = fit_df.astype(int).astype(str)
        
        for i in range(len(X)):
            row_df = X.iloc[[i]]
            row_df = row_df.astype(int).astype(str)
            filtered_df = fit_df.merge(row_df, on = list(row_df.columns), how='inner')
            y_count.append(filtered_df['y'].count())
            try:
                prob = filtered_df['y'].sum()/filtered_df['y'].count()
            except:
                prob = np.nan
            y_prob.append(prob)
        """
        #above is less optimised approach for below
        #"""
        for i, row in X.iterrows():
            fit_df = self.fit_df
            for col in X.columns:
                fit_df = fit_df.loc[fit_df[col]==row[col]]
            y_count.append(fit_df['y'].count())
            try:
                prob = fit_df['y'].sum()/fit_df['y'].count()
            except:
                prob = np.nan
            y_prob.append(prob)
        #"""
        return y_prob #, y_count
    
    def get_params(self):
        return self.fit_df, self.dict_bins, self.dict_labels


class TreeAndLinearRegressionModel():
    """
    Do LABEL ENCODING rather than ONE HOT ENCODING
    Avoid PCA or SCALING
    if PCA or scaling is done before -> then it is exactly same as linear regression
    
    LOGIC:
    First filters the data based on the categorical columns (choosed based on category_threshold) -> TREE PART
    and then does linear regression on the numerical columns after apply standard scaling -> REGRESSION PART
    Idea is to first do tree part on the categorical columns and select the right rows from the train data based on 
    these categorical values (test data) and then applying regression for numerical features.
    """
    
    def __init__(self, category_threshold=5):
        self.category_threshold = category_threshold
        self.fit_df = None
        self.categorical_columns = None
        self.numerical_columns = None
    
    def fit(self, X, y):
        category_threshold = self.category_threshold
        categorical_columns = [col for col in X.columns if len(X[col].unique())<=category_threshold]
        numerical_columns = [col for col in X.columns if col not in categorical_columns]
        X['y'] = np.array(y)
        self.fit_df = X
        self.categorical_columns = categorical_columns
        self.numerical_columns = numerical_columns
        return self
        
    def predict_proba(self, X):
        from sklearn.linear_model import LinearRegression
        from sklearn.preprocessing import StandardScaler
        category_threshold = self.category_threshold
        categorical_columns = self.categorical_columns
        numerical_columns = self.numerical_columns
        y_prob = []
        y_count = []
        X.reset_index(drop=True, inplace=True)
        for i, row in X.iterrows():
            fit_df = self.fit_df
            row_df = X.iloc[[i]]
            row_df = row_df.loc[:,[col for col in row_df.columns if col in numerical_columns]]
            #this step is for TREE part of the code -> for categorical columns
            for col in X.columns:
                #same as (col in numerical columns)
                if col not in categorical_columns:
                    continue
                else:
                    fit_df = fit_df.loc[fit_df[col]==row[col]]
                    fit_df.drop(columns = col, inplace= True)
            #this step is for REGRESSION part of the code -> for numerical columns
            X_fit = fit_df.loc[:, fit_df.columns!='y']
            y_fit = fit_df['y']
            #scaling the data before using linear regression
            scaler = StandardScaler()
            scaler.fit(X_fit)
            X_fit = pd.DataFrame(scaler.transform(X_fit))
            row_df = pd.DataFrame(scaler.transform(row_df), columns = X_fit.columns)
            #with scaling "y"
            """
            y_fit_mean = y_fit.mean()
            y_fit_std = y_fit.std()
            if y_fit_std == 0:
                y_fit_std = 1
                y_fit_mean = 0
            y_fit = pd.Series([((i-y_fit_mean)/y_fit_std) for i in list(y_fit)])
            linear_regression_model = LinearRegression()
            linear_regression_model.fit(X_fit,y_fit)
            try:
                prob = (linear_regression_model.predict(row_df)[0])*y_fit_std + y_fit_mean
            except:
                prob = np.nan
            """
            #without scaling "y"
            linear_regression_model = LinearRegression()
            linear_regression_model.fit(X_fit,y_fit)
            try:
                prob = (linear_regression_model.predict(row_df)[0])
            except:
                prob = np.nan
            y_prob.append(prob)
            y_count.append(len(X_fit))
        return y_prob #, y_count
            
    def get_params(self):
        return self.fit_df, self.categorical_columns, self.numerical_columns
            
        

class TreeAndLogisticRegressionModel():
    """
    Do LABEL ENCODING rather than ONE HOT ENCODING
    Avoid PCA or SCALING
    if PCA or scaling is done before -> then it is exactly same as logistic regression
    
    LOGIC:
    First filters the data based on the categorical columns (choosed based on category_threshold) -> TREE PART
    and then does logistic regression on the numerical columns after apply standard scaling -> REGRESSION PART
    Idea is to first do tree part on the categorical columns and select the right rows from the train data based on 
    these categorical values (test data) and then applying regression for numerical features.
    """
    
    def __init__(self, category_threshold=5):
        self.category_threshold = category_threshold
        self.fit_df = None
        self.categorical_columns = None
        self.numerical_columns = None
    
    def fit(self, X, y):
        category_threshold = self.category_threshold
        categorical_columns = [col for col in X.columns if len(X[col].unique())<=category_threshold]
        numerical_columns = [col for col in X.columns if col not in categorical_columns]
        X['y'] = np.array(y)
        self.fit_df = X
        self.categorical_columns = categorical_columns
        self.numerical_columns = numerical_columns
        return self
        
    def predict_proba(self, X):
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        category_threshold = self.category_threshold
        categorical_columns = self.categorical_columns
        numerical_columns = self.numerical_columns
        y_prob = []
        y_count = []
        X.reset_index(drop=True, inplace=True)
        for i, row in X.iterrows():
            fit_df = self.fit_df
            row_df = X.iloc[[i]]
            row_df = row_df.loc[:,[col for col in row_df.columns if col in numerical_columns]]
            #this step is for TREE part of the code -> for categorical columns
            for col in X.columns:
                #same as (col in numerical columns)
                if col not in categorical_columns:
                    continue
                else:
                    fit_df = fit_df.loc[fit_df[col]==row[col]]
                    fit_df.drop(columns = col, inplace= True)
            #this step is for REGRESSION part of the code -> for numerical columns
            X_fit = fit_df.loc[:, fit_df.columns!='y']
            y_fit = fit_df['y']
            #scaling the data before using linear regression
            scaler = StandardScaler()
            scaler.fit(X_fit)
            X_fit = pd.DataFrame(scaler.transform(X_fit))
            row_df = pd.DataFrame(scaler.transform(row_df), columns = X_fit.columns)
            logistic_regression_model = LogisticRegression()
            logistic_regression_model.fit(X_fit,y_fit)
            try:
                prob = (logistic_regression_model.predict_proba(row_df)[0])
            except:
                prob = np.nan
            y_prob.append(prob)
            y_count.append(len(X_fit))
        return y_prob #, y_count
            
    def get_params(self):
        return self.fit_df, self.categorical_columns, self.numerical_columns
            
        

def get_data_from_db(query):
    ## Connect to prd db, get order_ids
    from sqlalchemy import create_engine
    import psycopg2
    connection_string_prod = create_engine('postgresql://beau-metrics-prd-redshift-adminuser:vD3NJQKQhuch8yE@10.30.24.48:5439/dev')
    df = pd.read_sql(query, con=connection_string_prod)
    return df

## flatten and reshape

a

a.reshape(1,-1)

b = a.reshape(2,5)
b.reshape(5,2)
b.reshape(1,-1)

b.flatten()

## testing classes

class test():
    def __init__(self, var1):
        self.var1 = var1
        self.var2 = None
        
    def method1(self,var2):
        self.var2 = var2
        return self.var2
    
    def method2(self):
        return self.var2

test_class = test(var1='var1')

test_class.method1("var2")

test_class.method2()

## My optimisation algorithm


from sympy import symbols, diff
import time
x, y = symbols('x y', real=True)
def my_optimisation_algo(function_to_minimise, initialise = (0,0), learning_rate=0.1, time_out_seconds=120, error=0.01):
    """
    LOGIC:
    Its a modified version of gradient descent.
    Idea is using simple learning_rate as a step as long as function is getting minimised. Pain arise when
    point reaches increasing side of the curve, i.e, function start to increase -> Now use gradient descent 
    to find optimal values
    
    NEED:
    plane surface -> gradient descent fails -> takes very small steps
    """
    #a corresponds to value of x and same for b and y
    a_final = None
    b_final = None
    t1 = time.time()
    
    #this is partial differential of 'f' w.r.t 'x'
    dl_dx = diff(function_to_minimise,x)
    dl_dy = diff(function_to_minimise,y)
    a,b = initialise
    while True:
        t2 = time.time()
        if t2-t1>time_out_seconds:
            return "time out from first loop"
        f1 = function_to_minimise.subs({x:a,y:b})
        a = a - learning_rate
        b = b - learning_rate
        f2 = function_to_minimise.subs({x:a,y:b})
        
        if abs(f2-f1)<=error:
            return (a,b)
        
        #this condition is first part of the logic
        if f2<=f1:
            continue
        #this condition is second part of the logic
        else:
            while True:
                t3 = time.time()
                if t3-t1>time_out_seconds:
                    return "time out from second loop"
#                 f3 = function_to_minimise.subs({x:a,y:b})
                
                #idea is stop iteration on 'a' if whilee got 'a_final' value
                if abs(learning_rate*dl_dx.subs({x:a,y:b}))<=error:
                    a_final = a
                if a_final is None:
                    a = a - learning_rate*dl_dx.subs({x:a,y:b})
                    
                #idea is stop iteration on 'b' if whilee got 'b_final' value
                if abs(learning_rate*dl_dy.subs({x:a,y:b}))<=error:
                    b_final = b
                if b_final is None:
                    b = b - learning_rate*dl_dy.subs({x:a,y:b})
                
                if (a_final is not None) and (b_final is not None):
                    return (a_final,b_final)
                
#                 f4 = function_to_minimise.subs({x:a,y:b})
#                 if abs(f3-f4)<=error:
#                     return (a,b)

            
    

my_optimisation_algo((x-1)**2 + (y-10)**2,error=0.01, learning_rate=0.01)

dict_ = {}
nums = [2,7,11,15]
for i in range(len(nums)):
    dict_[nums[i]]=i

a = 3
for i in range(1,11):
    print(f"{a}X{i}={a*i}")

dict_.keys()



def feature_analysis(feature,demand='lower',label = 'rto_or_not',df = df,method='percentile',total_count_threshold=0,rto_pct_threshold=0):
    
    print(feature)
    def not_null_columns(df):
        a=[]
        for i in df.columns:
            if df[i].isnull().sum()==0:
                a.append(i)
        return a



    def link(feature=feature,label=label, df = df,total_count_threshold=total_count_threshold,rto_pct_threshold=rto_pct_threshold):
        pivot=df.pivot_table(values=[i for i in not_null_columns(df) if i not in [feature,label]][0],index=feature,columns=label,aggfunc='count')
        pivot['sum']=pivot.sum(axis=1)
        pivot.fillna(0,inplace=True)
        pivot['rto_pct']=(pivot[1])/(pivot['sum'])
        return pivot.loc[(pivot['sum']>=total_count_threshold)&(pivot['rto_pct']>=rto_pct_threshold),:]
    
    if demand=='equal':
        return print(link(feature))
    
    
    if method == 'percentile':
        if demand == 'lower':
            table = pd.DataFrame(columns=['percentile','value_less/equal_than','total','rto_pct'])
            for i in [x/100 for x in range(5,100,5)]:
                total = len(df.loc[df[feature]<=df[feature].quantile(i),:])
                rto_pct = len(df.loc[(df[feature]<=df[feature].quantile(i))&(df[label]==1),:])/total if total>0 else 0
                table = table.append({'percentile':i,'value_less/equal_than':df[feature].quantile(i),'total':total,'rto_pct':rto_pct},ignore_index=True)
        else:
            table = pd.DataFrame(columns=['percentile','value_more/equal_than','total','rto_pct'])
            for i in [x/100 for x in range(0,100,5)]:
                total = len(df.loc[df[feature]>=df[feature].quantile(i),:])
                rto_pct = len(df.loc[(df[feature]>=df[feature].quantile(i))&(df[label]==1),:])/total if total>0 else 0
                table = table.append({'percentile':i,'value_more/equal_than':df[feature].quantile(i),'total':total,'rto_pct':rto_pct},ignore_index=True)
    else:
        if demand == 'lower':
            table = pd.DataFrame(columns=['value_less/equal_than','total','rto_pct'])
            for i in range(1,int(df[feature].max()+1),1):
                total = len(df.loc[df[feature]<=i,:])
                rto_pct = len(df.loc[(df[feature]<=i)&(df[label]==1),:])/total if total>0 else 0
                table = table.append({'value_less/equal_than':i,'total':total,'rto_pct':rto_pct},ignore_index=True)
        else:
            table = pd.DataFrame(columns=['value_more/equal_than','total','rto_pct'])
            for i in range(0,int(df[feature].max()+1),1):
                total = len(df.loc[df[feature]>=i,:])
                rto_pct = len(df.loc[(df[feature]>=i)&(df[label]==1),:])/total if total>0 else 0
                table = table.append({'value_more/equal_than':i,'total':total,'rto_pct':rto_pct},ignore_index=True)
    table['feature'] = feature



    

    
    if demand == 'lower':
        return print(table.loc[(table['total']>total_count_threshold)&(table['rto_pct']>rto_pct_threshold),:][['value_less/equal_than','total','rto_pct']].drop_duplicates())
    else:
        return print(table.loc[(table['total']>total_count_threshold)&(table['rto_pct']>rto_pct_threshold),:][['value_more/equal_than','total','rto_pct']].drop_duplicates())        
            
    




def swap(arr,i,j):
    arr[i],arr[j]=arr[j],arr[i]
    return arr




def get_data_from_db(query):
    ## Connect to prd db, get order_ids
    from sqlalchemy import create_engine
    import psycopg2
    connection_string_prod = create_engine('postgresql://beau-metrics-prd-redshift-adminuser:vD3NJQKQhuch8yE@10.30.24.48:5439/dev')
    df = pd.read_sql(query, con=connection_string_prod)
    return df



# Emulator logic:
((Build.FINGERPRINT.startsWith("google/sdk_gphone_")
        && Build.FINGERPRINT.endsWith(":user/release-keys") -> this could be the issue
        && Build.MANUFACTURER == "Google" && Build.PRODUCT.startsWith("sdk_gphone_") && Build.BRAND == "google"
        && Build.MODEL.startsWith("sdk_gphone_"))
        //
        || Build.FINGERPRINT.startsWith("generic")
        || Build.FINGERPRINT.contains("generic")
        || Build.FINGERPRINT.startsWith("unknown")
        || Build.MODEL.contains("google_sdk")
        || Build.MODEL.contains("Emulator")
        || Build.MODEL.contains("Android SDK built for x86")
        //bluestacks
        || "QC_Reference_Phone" == Build.BOARD && !"Xiaomi".equals(
    Build.MANUFACTURER,
    ignoreCase = true
) //bluestacks
        || Build.MANUFACTURER.contains("Genymotion")
        || Build.HOST.startsWith("Build") //MSI App Player
        || Build.BRAND.startsWith("generic") && Build.DEVICE.startsWith("generic")
        || Build.PRODUCT == "google_sdk"
        // another Android SDK emulator check
        || SystemPropertiesInvoker.getSystemProperty("ro.kernel.qemu") == "1")




# working with mongodb and python using pymongo
def get_database():
    from pymongo import MongoClient
    import pymongo

    # Provide the mongodb atlas url to connect python to mongodb using pymongo
    CONNECTION_STRING = 'mongodb://root:UbdhxUQUrBWefTLBMMoJ@orch-prd-mb-docdb-cluster-read-replica-instance-0.c9shasfdhefj.ap-south-1.docdb.amazonaws.com/sensordata_db'

    # Create a connection using MongoClient. You can import MongoClient or use pymongo.MongoClient
    from pymongo import MongoClient
    client = MongoClient(CONNECTION_STRING)
    return client
    # Create the database for our example (we will use the same database throughout the tutorial
client = get_database()
mydb = client['db']
mycol = mydb['collection']




# create table in sql
CREATE table if not exists dbt.fingerprint_raw_data_1 (
	id INT IDENTITY(1, 1) NOT NULL,
	raw_data VARCHAR(MAX),
	created_at timestamp DEFAULT GETDATE(),
	PRIMARY KEY (id)
);


# binary vector to image 
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
plt.imsave('image.png', image_vector, cmap=cm.gray)




# creating index in mongodb
mycol.create_index([('merchantId',1)], background=True)


# scp command -> copy file from local to ec2/server
scp ~/Downloads/Bureau/witzeal/new_fingerprint.py ubuntu@neo4j:/home/ubuntu/witzeal/new_fingerprint.py
# copy file from ec2 to local
scp ubuntu@neo4j:/home/ubuntu/temp1.py ~/Downloads/temp1.py 



# pandas to redshift
from sqlalchemy import create_engine
import pandas as pd
import psycopg2
conn = create_engine('postgresql://beau-metrics-prd-redshift-adminuser:vD3NJQKQhuch8yE@10.30.24.48:5439/elt')
df = pd.DataFrame([{'a': 'foo', 'b': 'green', 'c': 11},{'a':'bar', 'b':'blue', 'c': 20}])
print(df)
df.to_sql(name='test_table_1', schema='dbt', con=conn, index=False, if_exists='append')



# Multithreading in python??
import time
import threading

def calc_square(numbers):
    print("calculate square numbers")
    for n in numbers:
        time.sleep(1)
        print('square:',n*n)

def calc_cube(numbers):
    print("calculate cube of numbers")
    for n in numbers:
        time.sleep(1)
        print('cube:',n*n*n)

arr = [2,3,8,9]

t = time.time()

t1= threading.Thread(target=calc_square, args=(arr,))
t2= threading.Thread(target=calc_cube, args=(arr,))

t1.start()
t2.start()

t1.join()
t2.join()

print("done in : ",time.time()-t)
print("Hah... I am done with all my work now!")






# mongodb/pymongo tutorial
row_count_all = sardine.count_documents({})

# pymmongo cursor to dataframe->
pd.DataFrame(list(cursor))




multiprocessing/multiprocess in python

import multiprocess
import time
# from task import task

def task():
    print("Sleeping for 1 seconds")
    time.sleep(5)
    print("Finished sleeping")

start_time = time.perf_counter()

# Creates two processes
p1 = multiprocess.Process(target=task)
p2 = multiprocess.Process(target=task)

# Starts both processes
p1.start()
p2.start()

p1.join()
p2.join()
finish_time = time.perf_counter()

print(f"Program finished in {finish_time-start_time} seconds")




# create index/composite index in pymongo
fingerprint_data.create_index([("merchantId",pymongo.ASCENDING),
                            ("deviceFingerprint",pymongo.ASCENDING)],
                            background=True)



# this is sicko!!
if np.nan:
    print("true")
 output = true

 if None:
    print("true")
output = 

if 0:
    print("true")
output=



def unique_col_check(df, col, group_on='new_fingerprint'):
    group = df.groupby(group_on).agg({col:(lambda x: pd.Series(x).nunique())}).reset_index()
    print(col,"-->")
    print("fingerprint having unique value more than 1: ",group.loc[group[col]>1,:].shape[0])
    print(group.loc[group[col]>1,:].head())

unique_col_check(df =raw_data_normalised,col='signaturesInfo_.signatureHashcode_.0')




# flask api tutorial -> run this as app.py
from flask import Flask
app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'This is my first API call!'

@app.route('/success/<name>')
def success(name):
   return 'welcome %s' % name

@app.route('/post_json', methods=['POST'])
def func():
    a = request.args.get("a")
    b = request.args.get("b")
    return sum(a,b)

if __name__ == '__main__':
   app.run(debug = True)

# post request api using flask
from flask import Flask,jsonify,request,make_response,url_for,redirect
import requests, json

app = Flask(__name__)



@app.route('/user_fingerprint_api', methods=['POST'])
def user_fingerprint_api():

    input_json = request.get_json(force=True)
    username = input_json['username']
    password = input_json['password']
    session_data = input_json['session_data']
    device_fingerprint = input_json['device_fingerprint']
    session_id = input_json['session_id']
    stage = input_json['stage']


    return jsonify(user_fingerprint_score(username, password, session_data, device_fingerprint, session_id, stage))

if __name__ == '__main__':
    app.run(debug=True)

# writing to file
with open('test_session.json','w') as f:
    f.write('str')
# reading from a file
with open('test_session.json','r') as f:
    text = f.read()
    


# best way to use github??
git stash
git pull
git stash apply #merges local changes into the latest code
git add -A
git commit 
git push


# clone a particular branch
git clone -b branch_name git_url

# issue with git pull? to remove local changes?
git stash
git stash drop
git pull


# merge branch1 to branch2
git checkout branch2
git merge branch1

# git pull from specific branch
git pull origin dev

# git push to specific branch/ even create new remote branch
git push origin localBranchName:remoteBranchName

# work on remote branches?
git fetch origin
git checkout origin/dev


# force push git?
git push --force origin

# list of branches
git branch

# checkout to new branch?
git checkout -b branch_name

# link local branch to remote branch
git branch --set-upstream-to=origin/dev dev
-----------------------------------------------------------------------------

# docker commands??

docker pull redis:version #to pull the docker image
docker images #list of all images
docker run redis:version #starts the container of the redis image

#to run the baseos image container
docker run -it centos:7 #to login -> to keep running the container

# to run container but not login
docker run -itd centos:7 

# to login in the container -> only for baseos image -> u cant enter a daemon
docker attach container_id

docker ps #list of containers that are running
docker ps -a #-> list of running and stopped/exited containers
watch -n 1 docker ps -a #every 1 second refresh

# to exit the container
exit #-> container is also stopped/exited
ctrl+p+q # container is still active
 
# terminate container?? -> ctrl+C 
docker stop container_id/container_name
docker start container_id/container_name

# which command will run after running the image
docker inspect centos:7 # then search for cmd

# copy file from local to container or vice versa
docker cp index.html(src)  container_id:/(dest) 

docker run -d -p6000:6379 --name name_of_container redis:version

docker logs container_id/container_name #to list logs of the container

docker exec -it container_id/container_name /bin/bash #to go to the container terminal (it -> interactive terminal)

# rename container name
docker rename container_id new_name

-----------------------------------------------------------------------------




# How To Deploy a Python Flask API Application on Docker??
https://adamtheautomator.com/python-flask-api/


from bson import ObjectId



# how to read/wwrite a file
f = open("/Users/shubham_mantri/Downloads/raw_data (1).txt", "r")
raw = f.read()
f.close()


# create simple index in mongodb
mycol.create_index("username", background=True)

# create composite index in mongodb
mycol.create_index([("username",1),("device_fingerprint",1), ("stage",1)],name="uds_index",background=True)



# dask tutorial/basics
import dask

import dask.dataframe as dd

ddf = dd.read_csv("/Users/shubham_mantri/Documents/docdb/sensordata_db/fingerprint_data/probo_app_cloning.csv")


ddf['signaturesInfo_.isPlayStoreInstall_'].value_counts().compute()

print("visualise partitions:")
ddf['createdAt'].max().visualize()

print("get count of partitions:")
ddf.divisions


# how to set environment variable
export ENV_VARIABLE="env variable test"

# use environment variable in python
import os
print(os.environ['ENV_VARIABLE'])


# run html file using terminal
open -a "Google Chrome" html_tutorial.html


# connect ec2 to dynamodb
# first create IAM role -> fullaccessdynamodb and attach to EC2 instance
aws dynamodb scan --table-name <tablename> --region <your-region>



# get number of arguments of the function
from inspect import signature
def someMethod(self, arg1, kwarg1=None):
    pass
sig = signature(sum_)
len(sig.parameters)



# Testing the functions:

def sum_(a,b):
    # handle two cases:
    #     1.None
    #     2.''
    #     3.datatype check
    if a is None or a=="":
        a=0
    if b is None or b=="":
        b=0
    # correcting the datatype to required
    a = float(a)
    b = float(b)
    return a+b
arguments = input("Enter arguments separated by comma:")
expected_result = input("Enter expected output:")
arguments_list=arguments.split(",")
# correcting the input type
if expected_result=='None':
    expected_result=np.nan
else:
    try:
        expected_result = int(expected_result)
    except:
        try:
            expected_result=float(expected_result)
        except:
            try:
                expected_result=bool(expected_result)
            except:
                x
# function to test -> sum_
actual_result = sum_(*arguments_list)
print("expected output: ", expected_result)
print("actual output: ",actual_result)
if actual_result==expected_result:
    print("Test case: PASS")
else:
    print("Test case: FAIL")



# how to install aws cli in system
aws configure
# configure aws with python
import os
os.environ['aws_access_key_id'] = 'xxxxxxxxx'
os.environ['aws_secret_access_key'] = 'xxxxxxxxx'



# load secrets from secret manager from the aws secret manager
import boto3
client = boto3.client('secretsmanager')
response = client.get_secret_value(
    SecretId = #secret store name
)
secretDict = json.loads(response['SecretString'])


# connect slack to python -> build slackbot using python
# https://www.pragnakalp.com/create-slack-bot-using-python-tutorial-with-examples/
"""
1. create new app in slack (app.slack.com)
2. generate a token
3. connect app to the channel
"""
import slack
import os
client = slack.WebClient(token='token')
client.chat_postMessage(
    channel='#channel_name', text="Message")


# read yaml file using python
import yaml
with open("example.yaml", "r") as stream:
    try:
        print(yaml.safe_load(stream))
    except yaml.YAMLError as exc:
        print(exc)




# convert json to yaml
yaml.dump(json_item)


# write logs to file using python
import logging
#Creating and Configuring Logger
Log_Format = "[%(asctime)s] [%(levelname)s] [%(module)s:%(funcName)s-%(lineno)d] - %(message)s"
logging.basicConfig(filename = "logging_test.log",
                    filemode = "a",
                    format = Log_Format, 
                    level = logging.DEBUG)
logger = logging.getLogger()
#Testing our Logger
def func():
    logger.error("Our First Log Message")


import logging
logging.debug('This is a debug message')
logging.info('This is an info message')
logging.warning('This is a warning message')
logging.error('This is an error message')
logging.critical('This is a critical message')


# encrypt and decrypt using python
from cryptography.fernet import Fernet
key = Fernet.generate_key() #this is your "password"
cipher_suite = Fernet(key)
encoded_text = cipher_suite.encrypt(b"Hello stackoverflow!")
decoded_text = cipher_suite.decrypt(encoded_text)
print(decoded_text.decode("utf-8") )


# cloudwatch insights query
fields @timestamp, @message
| filter (@message like "v2/auth/otp/validate" and @message like "400")
| sort @timestamp desc
| limit 20



# create new environment in python
python -m venv new-environment
source new-environment/bin/activate


# give execute access to python file
chmod a+x  python_file.py



def is_emulator(device_data: dict) -> bool:
    return #




web scraping using selenium

#!pip install chromedriver_autoinstaller
#!pip install selenium

import chromedriver_autoinstaller
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options

options = Options()
options.headless = True

chromedriver_autoinstaller.install()
driver = webdriver.Chrome()

Address_data = []
driver = webdriver.Chrome()
for i in range(1,201):
    url = f'http://www.business-yellowpages.com/indonesia/page-{i}’
    driver.get(url)
    print(i)
    elts = driver.find_elements(by=By.XPATH, value='//span[text()="Address:"]/..')
    Address_data.extend([elt.text.replace('Address:','') for elt in elts])
    print(len(Address_data))

driver.quit()

import pandas as pd
df = pd.DataFrame([{'address' : add, 'country' : 'Indonesia'} for add in Address_data])
df.to_csv('Indonesia_Address.csv', index=False)




# import function from another file in another folder
import sys
sys.path.insert(1, '/Users/shubham_mantri/Downloads/new_func')
or
sys.path.append("/Users/shubham_mantri/Downloads/new_func")# -> prefer -> folder to look for python files
from func import *

from application.app.folder.file import func_name




# mongodb query example
def get_database():
    import pymongo
    from pymongo import MongoClient

    # Provide the mongodb atlas url to connect python to mongodb using pymongo
    CONNECTION_STRING = "mongodb://root:UbdhxUQUrBWefTLBMMoJ@orch-prd-mb-docdb-cluster-read-replica-instance-0.c9shasfdhefj.ap-south-1.docdb.amazonaws.com/sensordata_db"

    # Create a connection using MongoClient. You can import MongoClient or use pymongo.MongoClient
    from pymongo import MongoClient

    client = MongoClient(CONNECTION_STRING)
    return client
    # Create the database for our example (we will use the same database throughout the tutorial
client = get_database()
mydb = client["sensordata_db"]
sardine = mydb["sardine_device_fingerprints"]
fingerprint_data = mydb["fingerprint_data"]

pd.DataFrame(list(fingerprint_data.aggregate([ 
    { "$match": { "$and":[{"deviceFingerprint": { "$ne": "" }},
                         {"createdAt": {"$gte":1664582400000000}}]}},
    {"$limit":1000},

    # Count all occurrences
    { "$group": {
        "_id": {
            "androidid": "$androidId_.id_",
            "deviceFingerprint": "$deviceFingerprint"
        },
        "count": { "$sum": 1 }
    }},
    
    # Sum all occurrences and count distinct
    { "$group": {
        "_id":"$_id.androidid",
        "totalCountSessions": { "$sum": "$count" },
        "distinctCountFingerprint": { "$sum": 1 }
    }}
    ,
    {
        "$match": {"distinctCountFingerprint": {"$gt": 2}}
    }
]))).head(10)




# print/get the current line number in python
from inspect import currentframe, getframeinfo
frameinfo = getframeinfo(currentframe())
print(frameinfo.filename, frameinfo.lineno)



# convert multiindex column to single index
df4.columns = [' '.join(col).strip() for col in df4.columns.values]


# dockerfile to build new docker image
FROM python:3.8-slim-buster
COPY resources/en_Lightsaber_Language-1.0.0.tar.gz en_Lightsaber_Language-1.0.0.tar.gz
COPY resources/requirements.txt requirements.txt
RUN pip3 install -r requirements.txt
RUN pip3 install en_Lightsaber_Language-1.0.0.tar.gz
COPY . .
RUN pip3 install -e .
EXPOSE 5000
RUN mkdir -p logs/
CMD ["python3", "LightsaberSuiteDriver.py"]



# keywords in python
from keyword import kwlist




# parse json in sql
json_extract_path_text(apiresponse,'package') as package



# parse list in sql
select json_extract_array_element_text('[111,112,113]', 2);



# redshift error logs
select * from stl_load_errors order by starttime desc limit 20


def none_or_blank(field):
    if field is None or field == "" or all(pd.isna(field)):
        return True
    return False



# get aws account id
subprocess.call(['aws','sts','get-caller-identity','--query','Account','--output','text'])



#embeddings code
import numpy as np
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Embedding

reviews = ['nice food',
        'amazing restaurant',
        'too good',
        'just loved it!',
        'will go again',
        'horrible food',
        'never go there',
        'poor service',
        'poor quality',
        'needs improvement']

sentiment = np.array([1,1,1,1,1,0,0,0,0,0])

# indexing the word to text
# eg the -> 0, too -> 1
print(one_hot("amazing restaurant",30))

vocab_size = 30
encoded_reviews = [one_hot(d, vocab_size) for d in reviews]
print(encoded_reviews)

max_length = 4
padded_reviews = pad_sequences(encoded_reviews, maxlen=max_length, padding='post')
print(padded_reviews)

embeded_vector_size = 5

model = Sequential()
model.add(Embedding(vocab_size, embeded_vector_size, input_length=max_length,name="embedding"))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))

X = padded_reviews
y = sentiment

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
print(model.summary())

model.fit(X, y, epochs=50, verbose=0)

# evaluate the model
loss, accuracy = model.evaluate(X, y)
accuracy

# getting the embeddings of the words
weights = model.get_layer('embedding').get_weights()[0]
len(weights)

# getting of embedding of the word index at 13
weights[13]







# PEP8 convention

Naming convention:
MY_CONSTANT
MyClass
my_variable
my_function
my_module.py
mypackage
my_class_method


Code structure:
Import libraries
Constants
Functions
Main code





# google scraper using python
import requests
import urllib
import pandas as pd
from requests_html import HTML
from requests_html import HTMLSession

def get_source(url):
    """Return the source code for the provided URL. 

    Args: 
        url (string): URL of the page to scrape.

    Returns:
        response (object): HTTP response object from requests_html. 
    """

    try:
        session = HTMLSession()
        response = session.get(url)
        return response

    except requests.exceptions.RequestException as e:
        print(e)
        
def get_results(query):
    
    query = urllib.parse.quote_plus(query)
    response = get_source("https://google.com/search?q=" + query)
    
    return response

def parse_results(response):
    
    css_identifier_result = ".tF2Cxc"
    css_identifier_title = "h3"
    css_identifier_link = ".yuRUbf a"
    css_identifier_text = ".VwiC3b"
    
    results = response.html.find(css_identifier_result)

    output = []
    
    for result in results:
        try:
            text = result.find(css_identifier_text, first=True).text
        except:
            text = ''
        
        item = {
            'title': result.find(css_identifier_title, first=True).text,
            'link': result.find(css_identifier_link, first=True).attrs['href'],
            'text': text
        }
        
        output.append(item)
        
    return output

def google_search(query):
    response = get_results(query)
    return parse_results(response)

google_search("web scraping")




# remove stopwords
import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords
cachedStopWords = stopwords.words("english")

def remove_stopwords(string):
    return ' '.join([word for word in string.split() if word not in cachedStopWords])


# sentence to vector using BERT
# pip install sent2vec
from sent2vec.vectorizer import Vectorizer

sentences = [
    "This is an awesome book to learn NLP.",
    "DistilBERT is an amazing NLP model.",
    "We can interchangeably use embedding, encoding, or vectorizing.",
]
vectorizer = Vectorizer()
vectorizer.run(sentences)
vectors = vectorizer.vectors





# screen command in linux
screen -S screen1 #-> create a new screen
screen -ls #-> list of screen running
screen -r screen1 #-> go to screen1
Ctrl + A + D #-> detach from screen
exit #-> terminate a screen




# python to Golang converter
https://pytago.dev/