import pandas as pd
import numpy as np 

from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

import warnings
warnings.filterwarnings('ignore')

# import pydata_google_auth
# credentials = pydata_google_auth.get_user_credentials(
#     ['https://www.googleapis.com/auth/bigquery'],
# )

# from datetime import date

# current_year = date.today().year
# previous_year = str(int(current_year) - 1) 

# query = "select * from osp-dmp.intern_smu.osp_cube_vansales where EXTRACT(YEAR from invoice_date) in (" + str(previous_year) + "," + str(current_year) + ")"
# project_id = "osp-dmp"

# import pandas_gbq as pdg
# transaction = pdg.read_gbq(query, project_id=project_id, dialect='standard', credentials=credentials)

transaction = pd.read_csv("customerTransaction.csv")
transaction.sample(1000000)

# transaction.to_csv('transaction.csv',index=False)

product_master = pd.read_excel('Master_Products.xlsx')

# create a dictionary to convert product name TH to ENG later
product_name_TH_ENG = product_master.groupby('productcode')['sku_name_english'].unique().reset_index()
product_name_TH_ENG = product_name_TH_ENG.append({'productcode':0, 'sku_name_english':'NA'}, ignore_index=True)

product_code = product_name_TH_ENG['productcode'].tolist()
product_name_ENG = product_name_TH_ENG['sku_name_english'].tolist()

product_name_TH_ENG_dict = {product_code[i]: product_name_ENG[i] for i in range(len(product_code))}

# remove 25761 rows where 'Base Qty' == 0 
transactionBQ = transaction[transaction['Base Qty']>0]

# consider only Province with transaction counts >= 3% & total net amount >= 3%


majorIndex = np.array(transactionBQ['Province Name Eng'].value_counts().index)
majorString = ''
print(majorIndex)
for i in majorIndex:
    majorString += i
    majorString += '|'
    
test = transactionBQ[transactionBQ['Province Name Eng'].str.contains(majorString[:-1])]
test['Province Name Eng'].nunique()

# we dont need quantity sum 
# we need either has taken or not 
# so if user has taken that item mark as 1 else mark as 0

def convert_into_binary(x):
    if x > 0:
        return 1
    else:
        return 0

rules_mlxtend_all_provinces = pd.DataFrame()

for province in majorIndex:
    
    # limit to 40 recommendation per province, not 10 because will remove M-150 and duplicates 
    count10 = 0
    
    test_province = test[test['Province Name Eng']==province]
    
    # convert data in format which is required 
    basket = pd.pivot_table(data=test_province, index='Invoice Number', columns='Product Code', values='Base Qty', aggfunc='sum', fill_value=0)
    
    basket_sets = basket.applymap(convert_into_binary)
    
    # Find the min_support. I'm looking at the 25 Percentile for frequency of Product Code 
    total_transaction = transactionBQ['Invoice Number'].nunique()
    frequencyOfItemAt25Percentile = transactionBQ['Product Code'].value_counts().describe()[4]
    min_supportcalculted = frequencyOfItemAt25Percentile / total_transaction * 100
    
    # call apriori function
    # min_support refers to how many times the item will appear. eg 0.07 means ocurs 7 times out of 100 transactions 
    frequent_itemsets = apriori(basket_sets, min_support=min_supportcalculted, use_colnames=True)
    
    # A lift value greater than 1 means that item Y is likely to be bought if item X is bought, while a value less than 1 means that item Y is unlikely to be bought if item X is bought.
    # Apply association rules on frequent itemset
    # An antecedent is an item found within the data. 
    # A consequent is an item found in combination with the antecedent
    rules_mlxtend = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
    rules_mlxtend.sort_values(by=['lift'], ascending=False, inplace=True)
    rules_mlxtend = rules_mlxtend.head(40)
    rules_mlxtend = rules_mlxtend.iloc[::2]
    rules_mlxtend.insert(1, "antecedents_2", 0)
    rules_mlxtend.insert(3, "consequents_2", 0)
    rules_mlxtend.insert(0, "Province Name Eng", 'NA')
#     rules_mlxtend['Product Name ENG'] = 'NA'
#     rules_mlxtend['Product Name ENG_2'] = 'NA'
    rules_mlxtend['Product Name TH'] = 'NA'
    rules_mlxtend['Product Name TH_2'] = 'NA'
    rules_mlxtend['GroupNameLevel1'] = 'NA'
    rules_mlxtend['GroupNameLevel1_2'] = 'NA'
    
    # convert frozenset to integers, remove groupnamelevel1 == 'M-150'
    for i in range(len(rules_mlxtend.index)):
        
        # add province name
        rules_mlxtend.at[rules_mlxtend.index[i], 'Province Name Eng'] = province
    
        # convert antecedents into integer
        current_antecedent = rules_mlxtend.iloc[i].antecedents

        if len(current_antecedent) == 1:
            antecedent = int(str(current_antecedent)[11:-2])
            rules_mlxtend.at[rules_mlxtend.index[i],'antecedents']= antecedent
            # add in corresponding GroupNamelevel1
            rules_mlxtend.at[rules_mlxtend.index[i],'GroupNameLevel1'] = transactionBQ[transactionBQ['Product Code']==antecedent]['GroupNameLevel1'].iloc[0]

        else:
            antecedent =  int(str(current_antecedent)[11:19])
            antecedent_2 = int(str(current_antecedent)[20:29])
            rules_mlxtend.at[rules_mlxtend.index[i],'antecedents']= antecedent
            rules_mlxtend.at[rules_mlxtend.index[i],'antecedents_2']= antecedent_2
             # add in corresponding GroupNamelevel1
            rules_mlxtend.at[rules_mlxtend.index[i],'GroupNameLevel1'] = transactionBQ[transactionBQ['Product Code']==antecedent]['GroupNameLevel1'].iloc[0]
            rules_mlxtend.at[rules_mlxtend.index[i],'GroupNameLevel1_2'] = transactionBQ[transactionBQ['Product Code']==antecedent_2]['GroupNameLevel1'].iloc[0]



        # turn consequents into integer, add GroupNameLevel1, add product name TH
        current_consequent = rules_mlxtend.iloc[i].consequents

        if len(current_consequent) == 1:
            consequent = int(str(current_consequent)[11:-2])
            rules_mlxtend.at[rules_mlxtend.index[i],'consequents']= consequent
            groupnamelevel = transactionBQ[transactionBQ['Product Code']==consequent]['GroupNameLevel1'].iloc[0]
            # add in corresponding Product Name TH
            rules_mlxtend.at[rules_mlxtend.index[i],'Product Name TH'] = transactionBQ[transactionBQ['Product Code']==consequent]['Product Name TH'].iloc[0]
            # add in corresponding GroupNamelevel1
            rules_mlxtend.at[rules_mlxtend.index[i],'GroupNameLevel1'] = transactionBQ[transactionBQ['Product Code']==consequent]['GroupNameLevel1'].iloc[0]

        else:
            consequent =  int(str(current_consequent)[11:19])
            consequent_2 = int(str(current_consequent)[20:29])
            rules_mlxtend.at[rules_mlxtend.index[i],'consequents']= consequent
            rules_mlxtend.at[rules_mlxtend.index[i],'consequents_2']= consequent_2
            # add in corresponding Product Name TH
            rules_mlxtend.at[rules_mlxtend.index[i],'Product Name TH'] = transactionBQ[transactionBQ['Product Code']==consequent]['Product Name TH'].iloc[0]
            rules_mlxtend.at[rules_mlxtend.index[i],'Product Name TH_2'] = transactionBQ[transactionBQ['Product Code']==consequent]['Product Name TH'].iloc[0]
             # add in corresponding GroupNamelevel1
            rules_mlxtend.at[rules_mlxtend.index[i],'GroupNameLevel1'] = transactionBQ[transactionBQ['Product Code']==consequent]['GroupNameLevel1'].iloc[0]
            rules_mlxtend.at[rules_mlxtend.index[i],'GroupNameLevel1_2'] = transactionBQ[transactionBQ['Product Code']==consequent_2]['GroupNameLevel1'].iloc[0]

            
        # limit to 40 recommendation per province
        count10 += 1
        
        if count10 == 40:
            break
    
    # convert product name TH to product name ENG
    rules_mlxtend['sku_name_english'] = rules_mlxtend['consequents'] 
    rules_mlxtend['sku_name_english_2'] = rules_mlxtend['consequents_2']
    
    
    rules_mlxtend = rules_mlxtend[['Province Name Eng', 'antecedents', 'antecedents_2', 'consequents', 'consequents_2', 'lift', 'confidence', 'Product Name TH', 'Product Name TH_2', 'sku_name_english', 'sku_name_english_2', 'GroupNameLevel1', 'GroupNameLevel1_2']]
    rules_mlxtend = rules_mlxtend[(rules_mlxtend.GroupNameLevel1 != 'M-150') & (rules_mlxtend.GroupNameLevel1_2 != 'M-150')].head(10)
    rules_mlxtend_all_provinces = pd.concat([rules_mlxtend_all_provinces, rules_mlxtend], ignore_index=True)

# convert productcode to sku_name_english
rules_mlxtend_all_provinces['sku_name_english'] = rules_mlxtend_all_provinces['sku_name_english'].map(product_name_TH_ENG_dict)
rules_mlxtend_all_provinces['sku_name_english_2'] = rules_mlxtend_all_provinces['sku_name_english_2'].map(product_name_TH_ENG_dict)

rules_mlxtend_all_provinces.to_csv('AprioriResults.csv',index=False)
    