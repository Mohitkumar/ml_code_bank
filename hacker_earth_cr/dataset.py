import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

train = pd.read_csv("/home/mohit/Downloads/f2c2f440-8-dataset_he/train.csv")

print train.info()
print train.head()

train.drop(['Browser_Used','Device_Used','User_ID'], axis=1, inplace=True)
#sns.heatmap(train.isnull(), yticklabels=False, cbar=False)
#plt.show()
sns.set_style('darkgrid')
plt.figure(figsize=(10, 5))
sns.countplot(train['Is_Response'], alpha=.80, palette=['grey', 'orange'])
plt.title('happy vs not happy')
plt.ylabel('# examples')
plt.show()

