# Multiphase Flow Meter (MPFM)

In here I'll briefly share the experiments I made to create a MPFM using data-fusion to measure Water and Air flowrate in a Multiphase Flow.

## 1. Introduction to MPFMs


MPFMs are instruments used to measure flowrate of Multiphase Flows. They are able to provide real time data using embedded sensors in their structure which brings huge benefits for certain applications e.g. in the Oil & Gas Industry where an exclusive separator is used to separate the mixture coming from the well to facilitate the flowrates measure (See [Handbook of Multiphase Flow Metering](https://nfogm.no/wp-content/uploads/2014/02/MPFM_Handbook_Revision2_2005_ISBN-82-91341-89-3.pdf) and [Multiphase Flow Metering: Current Trends and Future Developments](https://www.researchgate.net/publication/241783661_Multiphase_Flow_Metering_Current_Trends_and_Future_Developments). 

![](https://github.com/EduardoPach/Chemical_Engineering/blob/main/Multiphase_Flow_Meter/Images/with_without_MPFM_Oil.png)
Oil Well Example from [Handbook of Multiphase Flow Metering](https://nfogm.no/wp-content/uploads/2014/02/MPFM_Handbook_Revision2_2005_ISBN-82-91341-89-3.pdf)

The phase distribution inside equipaments, pipes, etc for Multiphase Flows is known as Flowpatterns. These patterns depend on Operational conditions (Temperature, Pressure), Fluids properties (Viscosity, Surface tension, Density) and Geometric conditions (Pipe diameter and orientation). Diagrams were created in order to classify the flowpatterns. Those diagrams are often called as Flow Maps and there are different Flow Maps for different types of Flow (e.g. Liq-Liq and Liq-Gas) and Pipe orientation (i.e. Vertical, Horizontal and Angled).

![](https://github.com/EduardoPach/Chemical_Engineering/blob/main/Multiphase_Flow_Meter/Images/Flow_Map.png)
Horizontal Pipe Flow Map (Axis in Volumetric Flowrate for Diameter = 8mm) Adapted from [Mandhane and Aziz](https://www.sciencedirect.com/science/article/abs/pii/0301932274900068)

<figure>
  <img
  src="https://github.com/EduardoPach/Chemical_Engineering/blob/main/Multiphase_Flow_Meter/Images/Flow_Map.png"
  alt="My System.">
  <figcaption>Figure 2. Horizontal Pipe Flow Map (Axis in Volumetric Flowrate for Diameter = 8mm) Adapted from [Mandhane and Aziz](https://www.sciencedirect.com/science/article/abs/pii/0301932274900068)</figcaption>
</figure>

## 2. Experiments

To do the experiments a Flow loop was created using PU pipes, Water flow sensors, Rotameter, Vibration sensors, Accelerometer/Gyroscope sensor, Venturi tube + differential Pressure sensor and a Condutance sensor. Below is a picture of the system.

<figure>
  <img
  src="https://github.com/EduardoPach/Chemical_Engineering/blob/main/Multiphase_Flow_Meter/Images/system.jpg"
  alt="My System.">
  <figcaption>Figure 3. Flow Loop</figcaption>
</figure>

## 3. Results
