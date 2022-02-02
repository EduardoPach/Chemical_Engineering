import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from control import Control

def main():
    URL_STEP = "https://raw.githubusercontent.com/EduardoPach/Chemical_Engineering/main/Dynamic_Identification/step.csv"
    URL_REAL = "https://raw.githubusercontent.com/EduardoPach/Chemical_Engineering/main/Dynamic_Identification/Exemplo_Dados_Reais.csv"
    data = pd.read_csv(URL)
    data['date'] = pd.to_datetime(data['date'])
    data.set_index('date', inplace=True)

    con1 = Control(data=data, u="u", y="y1")
    con2 = Control(data=data, u="u", y="y2")

    con1.FOPDT_fit(Kp=0, taup=20, thetap=30, plot=True)
    con2.FOPDT_fit(Kp=0, taup=40, thetap=40, plot=True)
    plt.show()

if __name__=="__main__":
    main()