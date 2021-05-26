import serial
import datetime as dt
import os

os.chdir("Data")

port = "COM3"
baud = 9600
arduino = serial.Serial(port,baud)
print("Connected to Arduino")

filename = input("Filename: ")
air_flow = input("Air flow (L/min) for experiment: ")
flow_pattern = input("Flow pattern: ")
file = open(filename+'.csv',"a")
start_data = dt.datetime.now()
line = 0
total_time = 3

while (dt.datetime.now()-start_data).total_seconds()/60 < total_time:
    values = str(arduino.readline().decode("utf-8"))
    values = values[:-2]
    if line == 0:
        print("Starting Collection")
        values = 'data,'+values+',Airflow'+',FlowPattern'
        file.write(values+'\n')
        line = line+1
    else:
        now = dt.datetime.now().strftime("%d/%Y/%m %H:%M:%S")
        values = now+','+values+','+air_flow+','+flow_pattern
        file.write(values+'\n')

file.close()
print("Data Collection finished")