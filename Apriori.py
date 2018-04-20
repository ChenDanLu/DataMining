from apyori import apriori
import pandas as pd
import numpy as np

data = pd.read_csv('./data/new_Building_Permits.csv', header=0, engine='python')
data = np.array(data)
print(data.shape)
results = list(apriori(transactions=data, min_confidence=0.7, min_support=0.7))
min_con = 0.9
for relation in results:
    print(relation.items)
    asso_rules = relation.ordered_statistics
    for rule in asso_rules:
        if rule.confidence > min_con:
            print(str(rule.items_base)+' ===>> '+str(rule.items_add))
            print(rule.lift)