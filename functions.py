# this is change in local
# this is change in local


# reasons for high risklevel??
for order_num in range(len(all_merchants_shopify)):
    if all_merchants_shopify[order_num]['shopify']['riskLevel']=='HIGH':
            print(order_num, "-", len([risk['message'] for risk in all_merchants_shopify[order_num]['shopify']['risks'] if risk['level']=="HIGH"]))
            for risk in all_merchants_shopify[order_num]['shopify']['risks']:
                if risk['level']=="HIGH":
                    #if ("Address" not in risk['message']) and ("address" not in risk['message']):
                    print(risk)


# updates pincodes and cities in city state pincode file
warnings.filterwarnings('ignore')
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
import jellyfish
fuzz.ratio(jellyfish.soundex(word.lower()),jellyfish.soundex(state.lower()))


def catch_json(js):
    try:
        return ast.literal_eval(js)
    except:
        return None

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





# hit google maps api
import requests
import time
from tqdm import tqdm
tqdm.pandas()
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
pd.set_option('display.max_rows',None)



from math import cos, asin, sqrt, pi
def lat_long_distance(lat1, lon1, lat2, lon2):
    p = pi/180
    a = 0.5 - cos((lat2-lat1)*p)/2 + cos(lat1*p) * cos(lat2*p) * (1-cos((lon2-lon1)*p))/2
    return 12742 * asin(sqrt(a)) 


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
