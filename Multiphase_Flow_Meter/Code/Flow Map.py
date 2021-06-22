import matplotlib.pyplot as plt
import numpy as np

# Boundaries in ft/s
Gas1 = np.array([0.1,5.0])
Liq1 = np.array([0.5,0.5])

Gas2 = np.array([7.5,40])
Liq2 = np.array([0.3,0.3])

Gas3 = np.array([0.1,230])
Liq3 = np.array([14,14])

Gas4 = np.array([35,14,10.5,2.5,2.5,3.25])
Liq4 = np.array([0.01,0.1,0.2,1.15,4.8,14])

Gas5 = np.array([70,60,38,40,50,100,230])
Liq5 = np.array([0.01,0.1,0.3,0.56,1,2.5,14])

Gas6 = np.array([230,269])
Liq6 = np.array([14,30])

# System
Dint = 8e-3
A = Dint**2*np.pi/4

# Experimental Points
Vaz_Liq = np.arange(0.2,0.71,0.1)/60/1000/A
Vaz_Gas = np.arange(1,6,1)/60/1000/A
LimVazLiq,LimVazGas = np.meshgrid(Vaz_Liq,Vaz_Gas)

# Plotting

plt.figure(figsize=(10,6))

plt.plot(Gas1*0.3048*60*1000*A,Liq1*0.3048*60*1000*A,'b-o')
plt.plot(Gas2*0.3048*60*1000*A,Liq2*0.3048*60*1000*A,'b-o')
plt.plot(Gas3*0.3048*60*1000*A,Liq3*0.3048*60*1000*A,'b-o')
plt.plot(Gas4*0.3048*60*1000*A,Liq4*0.3048*60*1000*A,'b-o')
plt.plot(Gas5*0.3048*60*1000*A,Liq5*0.3048*60*1000*A,'b-o')
plt.plot(Gas6*0.3048*60*1000*A,Liq6*0.3048*60*1000*A,'b-o')

plt.plot(LimVazGas[:-2,:]*60*1000*A,LimVazLiq[:-2,:]*60*1000*A,'o',color='green')
plt.plot(LimVazGas[-2:,:-1]*60*1000*A,LimVazLiq[-2:,:-1]*60*1000*A,'o',color='green')

plt.plot(4,0.5,'o',color="orange")
plt.plot(5,0.4,'o',color="orange")

plt.ylim([0.01*0.3048*60*1000*A,20*0.3048*60*1000*A])
plt.xlim([0.1*0.3048*60*1000*A,500*0.3048*60*1000*A])

plt.yscale("log")
plt.xscale("log")

plt.grid(True)

plt.text(0.1*60*1000*A,0.01*60*1000*A,"Stratified",fontsize=13,fontweight="bold")
plt.text(0.1*60*1000*A,1*60*1000*A,"Plug",fontsize=13,fontweight="bold")
plt.text(4*60*1000*A,1*60*1000*A,"Slug",fontsize=13,fontweight="bold")
plt.text(10*60*1000*A,0.01*60*1000*A,"Wavy",fontsize=13,fontweight="bold")
plt.text(50*60*1000*A,0.01*60*1000*A,"Annular",fontsize=13,fontweight="bold");

plt.xlabel("Gas Superficial Velocity (m/s)",fontsize=13,fontweight="bold")
plt.ylabel("Liquid Superficial Velocity (m/s)",fontsize=13,fontweight="bold");
plt.tight_layout()
plt.savefig("Images/Flow_Map_with_Experimental_Points.png",dpi=300)
plt.show()
