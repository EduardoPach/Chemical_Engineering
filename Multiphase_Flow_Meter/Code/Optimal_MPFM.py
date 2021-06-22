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

    fname = "Optimal_MPFM"

    model_air = BuildModel(11)
    model_water = BuildModel(4)

    mpfm = flowmeter.MPFM(x=x,y=y,model_air=model_air,model_water=model_water)

    mpfm.fit(n_rep=2000,printer=True)
    mpfm.save(fname=fname+'.h5')


    with open("air_"+fname+'.pkl',"wb") as f:
        pickle.dump(mpfm.results_air,f)

    with open("water_" + fname + '.pkl', "wb") as f:
        pickle.dump(mpfm.results_water, f)

    mpfm.plot_results(phase="water",
                      fname="Images"+'/'+"water_"+fname+".png"
                      )
    mpfm.plot_results(phase="air",
                      fname="Images"+'/'+"air_"+fname+".png"
                      )
    plt.show()

    return mpfm

if __name__ == "__main__":

    mpfm = run_mpfm()