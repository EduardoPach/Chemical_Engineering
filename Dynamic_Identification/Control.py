import numpy as np
from scipy.optimize import minimize
from scipy.integrate import odeint
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText

class Control:
  """
  Classe para realizar aproximação de sistemas para FOPDT e realizar proposição 
  simulação de controladores.
  """
  def __init__(self, u, y,data):
      self.u = data[u].values
      self.y = data[y].values
      self.t = (data.index-data.index[0]).values/np.timedelta64(1,"s")
      self.u0 = self.u[0]
      self.y0 = self.y[0]
      self.uf = interp1d(self.t, self.u)
      self.ns = len(self.t)
  def FOPDT(self,y,t,Kp,taup,thetap):
    """
    Função que retorna valores da derivada do FOPDT
    """
    try:
        if (t-thetap) <= 0:
            up = self.uf(0.0)
        else:
            up = self.uf(t-thetap)
    except:
        up = self.u0
    dydt = (-(y-self.y0)+Kp*(up-self.u0))/taup
    return dydt
  
  def sim_fopdt(self,x):
    """
    Simulação da FOPDT em função dos parâmetros tau, K e theta.
    """
    Kp,taup,thetap=x
    ym = np.zeros(self.ns)
    ym[0] = self.y0
    for i in range(0,self.ns-1):
        ts = [self.t[i],self.t[i+1]]
        y1 = odeint(self.FOPDT,ym[i],ts,args=(Kp,taup,thetap))
        ym[i+1] = y1[-1]
    return ym
  
  def fopdt_objective(self,x):
    """
    Função objetivo (soma dos erros quadráticos) em função dos parâmetros
    tau, theta e K.
    """
    ym = self.sim_fopdt(x)
    obj = 0
    for i in range(len(ym)):
        obj = obj + (ym[i]-self.y[i])**2    
    return obj
  
  def FOPDT_fit(self,Kp,taup,thetap,plot=False):
    """
    Função que realiza o ajuste para obtenção dos parâmetros do modelo
    FOPDT.
    """
    x0 = [Kp,taup,thetap]
    Init = self.fopdt_objective(x0)
    print("Fobj inicial: "+str(round(Init,2)))
    self.solution = minimize(self.fopdt_objective,x0)
    x = self.solution.x
    self.Kp = x[0]
    self.taup = x[1]
    self.thetap = x[2]
    print("Fobj final: "+str(round(self.fopdt_objective(x))))
    print("Kp: {:.4f}".format(self.Kp))
    print("taup: {:.4f}".format(self.taup))
    print("thetap: {:.4f}".format(self.thetap))
    if plot:
        
        ym1 = self.sim_fopdt(x0)
        ym2 = self.sim_fopdt(x)
        
        plt.figure(figsize=(12,6))
        ax1 = plt.subplot(2,1,1)
        
        plt.plot(self.t,self.y,'kx-',linewidth=2,label='Dados de Processo')
        plt.plot(self.t,ym1,'b-',linewidth=2,label='Valores Iniciais')
        plt.plot(self.t,ym2,'r--',linewidth=3,label='FOPDT Ajustado')
        plt.ylabel('Output')
        plt.legend(loc='best',bbox_to_anchor=(1,1))
        
        at = AnchoredText("$K_p$: {:.2f}\n $\\tau_p$: {:.2f}\n $\\theta_p$: {:.2f}".format(self.Kp,self.taup,self.thetap),
              prop=dict(size=10), frameon=True,
              loc='lower right',
              )
        at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
        ax1.add_artist(at)
        ax1.set_title("Identificação de Dinâmica",fontsize=15,fontweight="bold")
    
        plt.subplot(2,1,2)
        plt.plot(self.t,self.u,'bx-',linewidth=2)
        plt.plot(self.t,self.uf(self.t),'r--',linewidth=3)
        plt.legend(['Medido','Interpolado'],loc='best')
        plt.ylabel('MV')
        
        plt.tight_layout()
        
  def FOPDT_SinCon(self,name,mod="PID"):
      """
      Correlações para controladores para processo FOPDT.

      name: Correlation name (ZN,IAE-S,IAE-R,ITAE-S,ITAE-R,IMC)
      mod: Controller Model (P,PI,PID)
      """
      if mod=='P':
          self.Kc = self.taup/self.Kp/self.thetap
      
      elif mod=="PI":
          if name=="ZN":
              self.Kc = 0.9*self.taup/self.Kp/self.thetap
              self.tauI = 3.33*self.thetap
          elif name == "IAE-S":
              self.Kc = 0.758/self.Kp*(self.thetap/self.taup)**(-0.861)
              self.tauI = self.taup/(1.02-0.323*(self.thetap/self.taup))
          elif name == "IAE-R":
              self.Kc = 0.984/self.Kp*(self.thetap/self.taup)**(-0.986)
              self.tauI = self.taup/(0.608*(self.thetap/self.taup)**(-0.707))
          elif name == "ITAE-S":
              self.Kc = 0.586/self.Kp*(self.thetap/self.taup)**(-0.916)
              self.tauI = self.taup/(1.03-0.165*(self.thetap/self.taup))
          elif name == "ITAE-R":
              self.Kc = 0.859/self.Kp*(self.thetap/self.taup)**(-0.977)
              self.tauI = self.taup/(0.674*(self.thetap/self.taup)**(-0.680))
          elif name == "IMC":
              self.tauC = input("Escolha o valor de TauC:\n\n")
              self.Kc = self.taup/self.Kp/(self.tauC+self.thetap)
              self.tauI = self.taup
          else:
              raise NameError("Sem Correlação com esse nome")
              
      elif mod=="PID":
          if name=="ZN":
              self.Kc = 1.2*self.taup/self.Kp/self.thetap
              self.tauI = 2*self.thetap
              self.tauD = 0.5*self.thetap 
              
          elif name == "IAE-S":
              self.Kc = 1.086/self.Kp*(self.thetap/self.taup)**(-0.869)
              self.tauI = self.taup/(0.740 - 0.130*(self.thetap/self.taup))
              self.tauD = 0.348*self.taup*(self.thetap/self.taup)**0.914
              
          elif name == "IAE-R":
              self.Kc = 1.435/self.Kp*(self.thetap/self.taup)**(-0.921)
              self.tauI = self.taup/(0.878*(self.thetap/self.taup)**(-0.749))
              self.tauD = 0.482*(self.thetap/self.taup)**1.137
              
          elif name == "ITAE-S":
              self.Kc = 0.965/self.Kp*(self.thetap/self.taup)**(-0.850)
              self.tauI = self.taup/(0.796-0.147*(self.thetap/self.taup))
              self.tauD = 0.308*self.taup*(self.thetap/self.taup)**0.929
              
          elif name == "ITAE-R":
              self.Kc = 1.357/self.Kp*(self.thetap/self.taup)**(-0.947)
              self.tauI = self.taup/(0.842*(self.thetap/self.taup)**(-0.738))
              self.tauD = 0.381*self.taup*(self.thetap/self.taup)**0.995
              
          elif name == "IMC":
              op = input("Escolha o valor de TauC:\n 1 - ThetaP\n 2 - Valor Desejado: \n")
              if op=='1':
                  self.tauC = self.thetap
              else:
                  self.tauC = float(op)
              self.Kc = (self.taup+self.thetap*0.5)/self.Kp/(self.tauC+self.thetap*0.5)
              self.tauI = self.taup+self.thetap/2
              self.tauD = self.taup*self.thetap/(self.taup*2+self.thetap)
              
          else:
              raise NameError("There's no such tunning correlation available")
  
  def Processo(self,y,t,u):
      dydt = (1/self.taup)*(-(y-0)+self.Kp*(u-0))
      return dydt

  def SimCon(self,t,Sat=[],SpStep=1,mod="PID",name="ITAE-R"):
      self.FOPDT_SinCon(mod=mod,name=name)
      
      n = len(t)
      SP = np.ones(n)*SpStep
      PV = np.zeros(n)
      MV = np.zeros(n)

      y0 = 0
      P = 0
      I = 0
      D = 0
      
      delta_t = t[1]-t[0]
      ndelay = int(np.ceil(self.thetap/delta_t))

      for i in range(1,n):
          ts = [t[i-1],t[i]]
          iop = max(0,i-ndelay)
          y = odeint(self.Processo,y0,ts,args=(MV[iop],))
          PV[i] = y[-1]
          y0 = y[-1]
          e = SP[i]-PV[i]
          dt = t[i]-t[i-1]
          P = self.Kc*e
          I += self.Kc/self.tauI*dt*e
          D = self.Kc*self.tauD*(PV[i]-PV[i-1])/dt
          MV[i] = P+I+D

      plt.figure(figsize=(12,6))
      
      ax1 = plt.subplot(211)
      title = "Simulação do Controlador "+mod+" "+name
      if name == "IMC":
          title = title+" $\\tau_c$: {:.1f}".format(self.tauC)
      ax1.set_title(title,fontsize=15,fontweight="bold")
      ax1.plot(t,PV,'b',label="PV")
      ax1.plot(t,SP,'g--',label="SP")
      ax1.legend(loc='best',fontsize=14,bbox_to_anchor=(1,1))
      
      at = AnchoredText("P: {:.0f}\nI: {:.0f}\nD: {:.0f}".format(self.Kc,self.Kc/self.tauI*60,self.tauD*self.Kc),
                prop=dict(size=10), frameon=True,
                loc='lower right',
                )
      at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
      ax1.add_artist(at)

      ax2 = plt.subplot(212)
      ax2.plot(t,MV,'b',label="MV")
      ax2.legend(loc="best")
      plt.tight_layout()