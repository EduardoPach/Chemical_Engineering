import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns


Qs = np.array([0.283, 0.306, 0.432, 0.475 , 0.624 ,0.710, 0.743])
Puls1 = np.array([25.25, 29.86, 35.84, 42.77, 55.53 ,62.43, 62.89])
Puls2 = np.array([12.13, 10.76 , 16.33 , 17.06, 22.64, 25.06, 27.10])

mod1 = sm.OLS(Qs,sm.add_constant(Puls1)).fit()
mod2 = sm.OLS(Qs,sm.add_constant(Puls2)).fit()

p1 = mod1.params
p2 = mod2.params

rmse1 = np.sqrt(np.mean(mod1.resid**2))
rmse2 = np.sqrt(np.mean(mod2.resid**2))

plt.figure(figsize=(12,6))

plt.subplot(1,2,1)
plt.plot(Puls1,Qs,'ko',label="Experimental Points",alpha=0.75)
sns.regplot(x=Puls1,y=Qs,color="red",label="Q(L/min) = {:.4f}P(Hz)+{:.2f}".format(p1[1],p1[0]),scatter=False)
plt.ylabel("Q (L/min)",fontsize=13)
plt.xlabel("P (Hz)",fontsize = 13)
plt.ylim([0.2,0.8])
plt.title("FT - 01 Calibration",fontsize = 18,fontweight="bold")
plt.text(26,0.6,"RMSE = $\pm {:.2f}L/min$".format(rmse1),fontsize=14)
plt.legend(fontsize=14)

plt.subplot(1,2,2)
plt.plot(Puls2,Qs,'ko',label="Experimental Points",alpha=0.75)
sns.regplot(x=Puls2,y=Qs,color="red",label="Q(L/min) = {:.4f}P(Hz)+{:.2f}".format(p2[1],p2[0]),scatter=False)
plt.xlabel("P (Hz)",fontsize = 13)
plt.text(11,0.6,"RMSE = $\pm {:.2f}L/min$".format(rmse2),fontsize=14)
plt.yticks([0.2,.3,.4,.5,.6,.7,.8],labels=[])
plt.ylim([0.2,0.8])
plt.title("FT - 02 Calibration",fontsize = 18,fontweight="bold")
plt.legend(fontsize=14)

plt.tight_layout(pad=0.3)
plt.savefig("Images/Flow_Sensors_Calibration.png",dpi=300)

plt.show()