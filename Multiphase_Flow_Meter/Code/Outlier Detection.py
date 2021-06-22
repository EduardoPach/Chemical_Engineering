import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def MyMAD(x):
    med = np.median(x)
    dist = np.abs(x-med)
    dist = np.median(dist)
    MAD = 1.4826*dist
    return MAD

data_path = "https://raw.githubusercontent.com/EduardoPach/Chemical_Engineering/main/Multiphase_Flow_Meter/Data"

files = ['Liq06_Gas5_Ex1.csv',
 'Liq04_Gas5_Ex1.csv',
 'Liq05_Gas5_Ex1.csv',
 'Liq05_Gas4_Ex1.csv',
 'Liq06_Gas4_Ex1.csv',
 'Liq04_Gas3_Ex1.csv',
 'Liq04_Gas4_Ex1.csv',
 'Liq03_Gas3_Ex1.csv',
 'Liq03_Gas4_Ex1.csv',
 'Liq04_Gas2_Ex1.csv',
 'Liq05_Gas2_Ex1.csv',
 'Liq05_Gas3_Ex1.csv',
 'Liq06_Gas2_Ex1.csv',
 'Liq07_Gas2_Ex1.csv',
 'Liq07_Gas3_Ex1.csv',
 'Liq03_Gas3_Ex2.csv',
 'Liq03_Gas4_Ex2.csv',
 'Liq04_Gas2_Ex2.csv',
 'Liq04_Gas3_Ex2.csv',
 'Liq04_Gas4_Ex2.csv',
 'Liq04_Gas5_Ex2.csv',
 'Liq05_Gas2_Ex2.csv',
 'Liq05_Gas3_Ex2.csv',
 'Liq05_Gas4_Ex2.csv',
 'Liq05_Gas5_Ex2.csv',
 'Liq06_Gas2_Ex2.csv',
 'Liq06_Gas3_Ex2.csv',
 'Liq06_Gas4_Ex2.csv',
 'Liq06_Gas5_Ex2.csv',
 'Liq07_Gas1_Ex1.csv',
 'Liq07_Gas1_Ex2.csv',
 'Liq07_Gas2_Ex2.csv',
 'Liq07_Gas3_Ex2.csv',
 'Liq03_Gas5_Ex1.csv',
 'Liq04_Gas1_Ex1.csv',
 'Liq04_Gas1_Ex2.csv',
 'Liq03_Gas2_Ex3.csv',
 'Liq03_Gas3_Ex3.csv',
 'Liq03_Gas4_Ex3.csv',
 'Liq03_Gas5_Ex3.csv',
 'Liq04_Gas4_Ex3.csv',
 'Liq04_Gas3_Ex3.csv',
 'Liq04_Gas1_Ex3.csv',
 'Liq06_Gas3_Ex1.csv',
 'Liq02_Gas5_Ex1.csv',
 'Liq03_Gas1_Ex1.csv',
 'Liq03_Gas2_Ex2.csv',
 'Liq02_Gas4_Ex2.csv',
 'Liq02_Gas1_Ex3.csv',
 'Liq02_Gas5_Ex2.csv',
 'Liq02_Gas2_Ex1.csv',
 'Liq02_Gas2_Ex2.csv',
 'Liq02_Gas4_Ex1.csv',
 'Liq02_Gas1_Ex2.csv',
 'Liq02_Gas3_Ex1.csv',
 'Liq03_Gas1_Ex3.csv',
 'Liq02_Gas3_Ex2.csv',
 'Liq03_Gas1_Ex2.csv',
 'Liq03_Gas2_Ex1.csv',
 'Liq04_Gas2_Ex3.csv',
 'Liq02_Gas2_Ex3.csv',
 'Liq03_Gas5_Ex2.csv',
 'Liq02_Gas1_Ex1.csv']

files.sort()


X_labels = ["Piezo1", "Piezo2", "801s", "Condutance", "dP", "OutletFlow", "AccX", "AccY", "AccZ", "GyrX", "GyrY",
            "GyrZ"]

outliers = {col: 0 for col in X_labels}
size = 0

for count, val in enumerate(files):

    df_ = pd.read_csv(data_path + "/" + val)
    size = size+df_.shape[0]
    for col in X_labels:
        x = df_[[col]].values
        MAD = MyMAD(x)
        cond = np.abs(x - np.median(x)) / (MAD + 1e-8)
        x[cond > 3] = np.median(x)
        df_[col] = x
        outliers[col] = outliers[col] + len(x[cond > 3])


plt.style.use("seaborn")
labs = {"Piezo1":"VT-01","Piezo2":"VT-02","801s":"VT-03","Condutance":"CT-01","dP":"PDT-01","OutletFlow":"FT-02",
        "AccX":"GT-01","AccY":"GT-02","AccZ":"GT-03","GyrX":"ST-01","GyrY":"ST-02","GyrZ":"ST-03"}

df_outliers = pd.DataFrame(outliers,index=[1])
df_outliers.rename(columns=labs,inplace=True)
df_outliers= df_outliers.transpose().sort_values(by=1)
plt.barh(y=df_outliers.index,width=df_outliers.values[:,0]/size*100)
plt.ylabel("Sensors",fontsize=14)
plt.yticks(fontsize=13)
plt.xticks(fontsize=13)
plt.xlabel("% Outliers",fontsize = 14)
plt.title("Outliers Detected by MAD",fontsize=20,fontweight="bold")
plt.tight_layout()
plt.savefig("Images/Outliers.png",dpi=300)
plt.show()