# Multiphase Flow Meter (MPFM)

In here I'll briefly share the experiments I made to create a MPFM using data-fusion to measure Water and Air flowrate in a Multiphase Flow.

## 1. Introduction to MPFMs


MPFMs are instruments used to measure flowrate of Multiphase Flows. They are able to provide real time data using embedded sensors in their structure which brings huge benefits for certain applications e.g. in the Oil & Gas Industry where an exclusive separator is used to separate the mixture coming from the well to facilitate the flowrates measure (See [Handbook of Multiphase Flow Metering](https://nfogm.no/wp-content/uploads/2014/02/MPFM_Handbook_Revision2_2005_ISBN-82-91341-89-3.pdf) and [Multiphase Flow Metering: Current Trends and Future Developments](https://www.researchgate.net/publication/241783661_Multiphase_Flow_Metering_Current_Trends_and_Future_Developments). 

<figure>
  <img
  src="https://github.com/EduardoPach/Chemical_Engineering/blob/main/Multiphase_Flow_Meter/Images/with_without_MPFM_Oil.png"
  alt="Example of well testing from the MPFM Handbook.">
  <figcaption><a href="https://nfogm.no/wp-content/uploads/2014/02/MPFM_Handbook_Revision2_2005_ISBN-82-91341-89-3.pdf">Figure 1. Oil Well Example from Handbook of Multiphase Flow Metering</a></figcaption>
</figure>

<p>&nbsp;</p> 
<p></p> 

The phase distribution inside equipaments, pipes, etc for Multiphase Flows is known as Flow patterns. These patterns depend on Operational conditions (Temperature, Pressure), Fluids properties (Viscosity, Surface tension, Density) and Geometric conditions (Pipe diameter and orientation). Diagrams were created in order to classify the Flow patterns. Those diagrams are often called as Flow Maps and there are different Flow Maps for different types of Flow (e.g. Liq-Liq and Liq-Gas) and Pipe orientation (i.e. Vertical, Horizontal and Angled).

<figure>
  <img
  src="https://github.com/EduardoPach/Chemical_Engineering/blob/main/Multiphase_Flow_Meter/Images/Flow_Map.png"
  alt="Flow Map adapted from Mandhane.">
  <figcaption><a href="https://www.sciencedirect.com/science/article/abs/pii/0301932274900068">Figure 2. Horizontal Pipe Flow Map (Axis in Volumetric Flowrate for Diameter = 8mm) Adapted from Mandhane and Aziz</a></figcaption>
</figure>




## 2. Experiments

To do the experiments a Flow loop was created using PU pipes, Water flow sensors, Rotameter, Vibration sensors, Accelerometer/Gyroscope sensor, Venturi tube + differential Pressure sensor, Condutance sensor and an Arduino Nano. Below is a picture of the system.

<figure>
  <img
  src="https://github.com/EduardoPach/Chemical_Engineering/blob/main/Multiphase_Flow_Meter/Images/system.jpg"
  alt="My System.">
  <figcaption>Figure 3. Flow Loop</figcaption>
</figure>



Since it's difficult to see from Figure 3 the system and grasp all it's details Figure 4 shows a draw of the system.

<figure>
  <img
  src="https://github.com/EduardoPach/Chemical_Engineering/blob/main/Multiphase_Flow_Meter/Images/P&ID.png"
  alt="My System draw.">
  <figcaption>Figure 4. System P&ID</figcaption>
</figure>



<table class="center">
  <caption>Table 1. Description of each TAG in the Figure 4</caption>
  <tr>
    <th>TAG</th>
    <th>Sensor</th> 
    <th>Description</th>
  </tr>
  <tr>
    <td>FT - 01</td>
    <td> <a href="https://pt.aliexpress.com/item/32973601341.html?spm=a2g0o.productlist.0.0.32d6aa679EuDiO&algo_pvid=6a3b9d3c-a6d2-4057-8bca-1ec374920d2c&algo_expid=6a3b9d3c-a6d2-4057-8bca-1ec374920d2c-6&btsid=0bb0623a16219632869212815e7c03&ws_ab_test=searchweb0_0,searchweb201602_,searchweb201603_"> YF-s401 </a> </td>
    <td>Water flow sensor based on the Hall Effect</td>
  </tr>
  <tr>
    <td>FG - 01</td>
    <td>Rotameter</td>
    <td>Rotameter with range 0 to 10 L/min</td>
  </tr>
  <tr>
    <td>PDT - 01</td>
    <td> <a href="https://www.digikey.com/en/products/detail/nxp-usa-inc/MPXV7002DP/1168436"> MPXV7002DP </a> </td>
    <td>Differential pressure sensor with range +/- 2kPa</td>
  </tr>
   <tr>
    <td>VT - 01</td>
    <td> <a href="https://pt.aliexpress.com/item/1005001267732402.html?spm=a2g0o.productlist.0.0.460d552eRCdBUJ&algo_pvid=3c29cc12-539e-4777-a70f-c33b55594d35&algo_expid=3c29cc12-539e-4777-a70f-c33b55594d35-24&btsid=0b0a555e16219632267787910e25bd&ws_ab_test=searchweb0_0,searchweb201602_,searchweb201603_"> Vibration sensor (SZYTF brand)</a></td>
    <td>Couldn't find any relevant information about it</td>
  </tr>
   <tr>
    <td>VT - 02</td>
    <td> <a href="https://pt.aliexpress.com/item/1005002010885373.html?spm=a2g0o.productlist.0.0.460d552eRCdBUJ&algo_pvid=3c29cc12-539e-4777-a70f-c33b55594d35&algo_expid=3c29cc12-539e-4777-a70f-c33b55594d35-23&btsid=0b0a555e16219632267787910e25bd&ws_ab_test=searchweb0_0,searchweb201602_,searchweb201603_"> Piezo Vibration sensor (IIC Serial brand) </a> </td>
    <td>Couldn't find any relevant information about it</td>
  </tr>
   <tr>
    <td>VT - 03</td>
    <td> <a href="https://pt.aliexpress.com/item/4001045403018.html?spm=a2g0o.search0302.0.0.3dc71473YKdRnJ&algo_pvid=30f55e7e-e460-4fa1-a9d1-4b7ec18a240a&algo_expid=30f55e7e-e460-4fa1-a9d1-4b7ec18a240a-14&btsid=0bb0622e16219627707677422e306c&ws_ab_test=searchweb0_0,searchweb201602_,searchweb201603_"> 801s </a> </td>
    <td>Couldn't find any relevant information about it</td>
  </tr>
   <tr>
    <td>ST - 01/02/03</td>
    <td> <a href="https://pt.aliexpress.com/item/1005001621877471.html?spm=a2g0o.productlist.0.0.64bd1024uzy4pK&algo_pvid=null&algo_expid=null&btsid=0b0a556216219634375406463eeeee&ws_ab_test=searchweb0_0,searchweb201602_,searchweb201603_"> GY-521 MPU-6050 </a> </td>
    <td>Angular velocity range +/- 250 °/s. (01: x, 02: y, 03: z)</td>
  </tr>
   <tr>
    <td>GT - 01/02/03</td>
    <td><a href="https://pt.aliexpress.com/item/1005001621877471.html?spm=a2g0o.productlist.0.0.64bd1024uzy4pK&algo_pvid=null&algo_expid=null&btsid=0b0a556216219634375406463eeeee&ws_ab_test=searchweb0_0,searchweb201602_,searchweb201603_"> GY-521 MPU-6050 </a> </td>
    <td>Acceleration range +/- 2g. (01: x, 02: y, 03: z)</td>
  </tr>
    <tr>
    <td>FT - 02 & CT - 01</td>
    <td> <a href="https://pt.aliexpress.com/item/32837942827.html?gclid=Cj0KCQjwwLKFBhDPARIsAPzPi-K-W0JMIPXvHhqNyVcwN94E0BFCfMCddK4aI-Mnw_yW6_rqSRDDKj4aAub7EALw_wcB"> DWS-MH-01 </a></td>
    <td>Water flow sensor based on Hall Effect with Condutance sensor embedded.</td>
  </tr>
    </tr>
    <tr>
    <td>FV - 01</td>
    <td> <a href="https://produto.mercadolivre.com.br/MLB-1793792454-valvula-pneumatica-12mm-_JM?matt_tool=35419131&matt_word=&matt_source=google&matt_campaign_id=12410582774&matt_ad_group_id=116564269605&matt_match_type=&matt_network=g&matt_device=c&matt_creative=500616071919&matt_keyword=&matt_ad_position=&matt_ad_type=pla&matt_merchant_id=305341964&matt_product_id=MLB1793792454&matt_product_partition_id=306248980482&matt_target_id=aud-1010920718118:pla-306248980482&gclid=Cj0KCQjwwLKFBhDPARIsAPzPi-KFdtpy24TuR9dJZg_y_NHVj-HQWIu_CotiNMm5UjCpOBn9f2Gip1YaAmEtEALw_wcB"> Needle Valve </a></td>
    <td>12 mm Needle Valve.</td>
  </tr>
    </tr>
    <tr>
    <td>B - 01</td>
    <td> <a href="https://pt.aliexpress.com/item/4000053282569.html?spm=a2g0s.9042311.0.0.2742b90aHtHVwp"> Submersible Water Pump </a></td>
    <td>Water pump with Nominal capacity of 400 L/h.</td>
  </tr>
    </tr>
    <tr>
    <td>C - 01</td>
    <td> <a href="https://www.amazon.in/BOYU-Electomagnetic-Air-Compressor-ACQ-003/dp/B0792DMPPF"> Air Compressor </a></td>
    <td>Air compressor with Nominal capacity of 2400 L/h</td>
  </tr>
</table>



The idea behind our selected sensors is that different Flow pattenrs exhibit completly different oscillation patterns, which means that a different load will be apply to the pipe for different patterns, therefore measures like vibration, angular speed and acceleration of the pipe might be useful. Moreover, different patterns are also seen in the Venturi tube and Condutance measures. 


Water is admitted in the system through B - 01 and parallel to that air is admitted through C - 01. Both of these lines have Flow rate instruments (FT - 01 and FG - 01) and needle valves to control these rates. After the lines are mixed the Multiphase Flow passes through the MPFM structure where it first goes through a Venturi tube, where the pressure difference is measured (PDT - 01), then it goes through three different vibration sensors (VT - 01/02/03), a Accelerometer/ Gyroscope, where the angular velocity (STs) and acceleration (GTs) on the three space axis are measured, and finally passes throguh another water flow meter with a condutance sensor embedded (FT - 02 and CT - 01).


With the system ready to be used (every sensor connected to the Arduino and the [Arduino code](https://github.com/EduardoPach/Chemical_Engineering/blob/main/Multiphase_Flow_Meter/Code/Sensors_Readings.ino) running well) the only task left was to know the possible water/air flow ranges that could be tested in the system. So, just by opening and closing the FV - 01 and FV - 02 valves and looking the air and water volumetric Flow rates given by FG - 01 and FT - 01. An interatction between air flow and water flow was detected, which reduced the possible combinations of these variables, hence the water flow range was choosen to be between 0.2 and 0.7 L/min and the air flow rate between 1 and 5 L/min (with exception to water flow rates as 0.7 L/min due to their interaction). Figure 5 shows all the experiments made on a Flow Map in order to classify the pattern.



<figure>
  <img
  src="https://github.com/EduardoPach/Chemical_Engineering/blob/main/Multiphase_Flow_Meter/Images/Flow_Map_with_Experimental_Points.png"
  alt="My Flow Map with Experimental points.">
  <figcaption>Figure 5. Flow Map for an 8 mm Horizontal Pipe. Green points are the experiments made and the Orange points are experiments that were re-classfied as Slug due to pattern visualization.  P&ID</figcaption>
</figure>


## 3. Results







