import flowmeter
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pickle
import os


def run_mpfm():
    os.environ["TF_CPP_MIN_LOG_LEVEL"] ="2"
    tf.keras.backend.set_floatx('float64')

    data = flowmeter.TwoPhaseFlow()

    x,y = data.load(drop="Condutance")
    opt = tf.keras.optimizers.RMSprop(learning_rate=0.005, momentum=0.5, centered=True)

    def BuildModel(units):
        model = tf.keras.Sequential([
        tf.keras.layers.Dense(units=units,activation="tanh"),
        tf.keras.layers.Dense(units=1)])
        model.compile(optimizer=opt,loss="mse")
        return model

    Neurons = np.arange(1,16)
    air_test_r = list()
    air_train_r = list()

    water_test_r = list()
    water_train_r = list()

    for neuron in Neurons:
        dir = f"MPFM_{neuron}_Neurons"
        fname = f"{neuron}_Neurons"

        model_air = BuildModel(neuron)
        model_water = BuildModel(neuron)

        mpfm = flowmeter.MPFM(x=x,y=y,model_air=model_air,model_water=model_water)

        mpfm.fit(n_rep=1000,printer=True)
        mpfm.save(fname=fname+'.h5',dir=dir)

        air_test_r.append(mpfm.air_test["r2"])
        air_train_r.append(mpfm.air_train["r2"])
        water_train_r.append(mpfm.water_train["r2"])
        water_test_r.append(mpfm.water_test["r2"])

        with open(dir+'/'+"air_"+fname+'.pkl',"wb") as f:
            pickle.dump(mpfm.results_air,f)

        with open(dir + '/' + "water_" + fname + '.pkl', "wb") as f:
            pickle.dump(mpfm.results_water, f)

        mpfm.plot_results(phase="water",
                          fname=dir+'/'+"water_"+fname+".png"
                          )
        mpfm.plot_results(phase="air",
                          fname=dir+'/'+"air_"+fname+".png"
                          )

    air_test_r = np.sqrt(np.array(air_test_r))
    air_train_r = np.sqrt(np.array(air_train_r))
    water_test_r = np.sqrt(np.array(water_test_r))
    water_train_r = np.sqrt(np.array(water_train_r))

    return [air_train_r,air_test_r,water_train_r,water_test_r]

if __name__ == "__main__":

    air_train_r,air_test_r,water_train_r,water_test_r = run_mpfm()
    neurons = np.arange(1,16)

    fig,(ax1,ax2) = plt.subplots(nrows=2,ncols=1,sharex=True,figsize=(12,8))

    ax1.plot(neurons,air_train_r)
    ax1.plot(neurons,air_test_r)
    ax1.set_title("Air Model - Correlation Coefficient for #Neurons", fontsize=15)

    ax2.plot(neurons,water_train_r)
    ax2.plot(neurons,water_test_r)
    ax2.set_title("Water Model - Correlation Coefficient for #Neurons",fontsize=15)

    plt.tight_layout()
    plt.show()



