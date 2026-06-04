import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

df = pd.read_csv('online_retail.csv', encoding='unicode_escape')

df['Description'] = df['Description'].str.strip()
df.dropna(axis=0, subset=['InvoiceNo', 'Description', 'CustomerID'], inplace=True)
df['InvoiceNo'] = df['InvoiceNo'].astype('str')
df = df[~df['InvoiceNo'].str.contains('C')]
df = df[df['Quantity'] > 0]

basket = (df.groupby(['InvoiceNo', 'Description'])['Quantity']
          .sum().unstack().reset_index().fillna(0)
          .set_index('InvoiceNo'))

def encode_units(x):
    if x <= 0:
        return 0
    if x >= 1:
        return 1

basket_sets = basket.map(encode_units)

nilai_min_support = [0.02, 0.03, 0.05] # 2%, 3%, dan 5%
nilai_min_confidence = 0.5             # 50%

hasil_rules = {}

for min_sup in nilai_min_support:
    print(f"=== Eksperimen dengan min_support = {min_sup} ({min_sup*100}%) ===")
    
    frequent_itemsets = apriori(basket_sets, min_support=min_sup, use_colnames=True)
    
    print(f"Jumlah Frequent Itemsets ditemukan: {len(frequent_itemsets)}")
    
    if len(frequent_itemsets) > 0:
        
        rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=nilai_min_confidence)
        
        valid_rules = rules[rules['lift'] > 1.0]
        
        valid_rules = valid_rules.sort_values('lift', ascending=False).reset_index(drop=True)
        
        print(f"Jumlah Association Rules valid (Confidence >= 50% & Lift > 1): {len(valid_rules)}\n")
        
        hasil_rules[min_sup] = valid_rules
        
        if not valid_rules.empty:
            print(valid_rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head())
            print("-" * 60, "\n")
    else:
        print("Tidak ada rules yang dapat dibentuk.\n")
        hasil_rules[min_sup] = pd.DataFrame()