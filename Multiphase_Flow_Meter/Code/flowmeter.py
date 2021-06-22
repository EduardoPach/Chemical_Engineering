import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import statsmodels.api as sm
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class MPFM:

    def __init__(self,x,y,model_air,model_water):
        '''

        :param x:
        :param y:
        :param model_air:
        :param model_water:
        '''
        self.x = x
        self.y_air = y["Airflow"]
        self.y_water = y["InletFlow"]

        self.model_air = model_air
        self.model_water = model_water

        self.model_air.build([None,len(self.air_sensors)])
        self.model_water.build([None,len(self.water_sensors)])

        self.air_train = 0
        self.air_test = 0
        self.air_all = 0

        self.water_train = 0
        self.water_test = 0
        self.water_all = 0

        self.n_air = 0
        self.n_water = 0

    def BackwardElimination(self, Y, alpha=0.05, scaler=StandardScaler(),verbose=False):
        '''
         Selects variables using Backward Elimination method based on p-value.
         Inputs:
           X: DataFrame
           Y: array-like
         returns:
           List with variables names
        '''

        X = self.x

        if scaler:
            cols = X.columns
            X = scaler.fit_transform(X)
            X = pd.DataFrame(X, columns=cols)

        X = sm.add_constant(X)
        p = alpha

        while p >= alpha:

            p = 0
            model = sm.OLS(Y, X).fit()

            for i in model.pvalues.index:

                if model.pvalues[i] > p:
                    con = i
                    p = model.pvalues[i]

            if p >= alpha:
                X.drop(con, axis=1, inplace=True)
                if verbose:
                    print(con + ' has a p-value of {:.3f}, thus was dropped'.format(p))

        var = model.pvalues.index
        if "const" in var:
            var = var.drop("const")

        return var.values

    @property
    def air_sensors(self):
        return self.BackwardElimination(self.y_air)

    @property
    def water_sensors(self):
        return self.BackwardElimination(self.y_water)

    def fit(self,n_rep=100,scaler=StandardScaler(), printer=True):
        '''
        :param printer: boolean - If true shows the best models metrics whenever an improvement occurs
        :param n_rep: int - Number of times that the models will be estimated
        :param scaler: Scaler object that will perform that data transformation
        :return:
        '''
        X_air = self.x[self.air_sensors]
        X_water = self.x[self.water_sensors]

        verbose = False

        if scaler:
            cols_airflow = X_air.columns
            cols_waterflow = X_water.columns

            X_air = scaler.fit_transform(X_air)
            X_air = pd.DataFrame(X_air, columns=cols_airflow)

            X_water = scaler.fit_transform(X_water)
            X_water = pd.DataFrame(X_water, columns=cols_waterflow)

        Early = tf.keras.callbacks.EarlyStopping(patience=5,restore_best_weights=True)

        self.results_air = self.make_results()
        self.results_water = self.make_results()

        names = [("r2_train", "mse_train", "mape_train"),
                 ("r2_test", "mse_test", "mape_test"),
                 ("r2_all", "mse_all", "mape_all")]

        initializer = tf.keras.initializers.GlorotUniform()

        for i in range(n_rep):

            self.weights_update()

            X_train_air,X_test_air,y_train_air,y_test_air = \
                train_test_split(X_air,self.y_air, train_size=0.80)

            X_train_water, X_test_water, y_train_water,y_test_water = \
                train_test_split(X_water, self.y_water, train_size=0.80)


            self.model_air.fit(x=X_train_air,y=y_train_air,
                                      epochs=500,validation_data=(X_test_air,y_test_air),
                                      callbacks=[Early],batch_size=4,verbose = 0)

            self.model_water.fit(x=X_train_water,y=y_train_water,
                                        epochs=500,validation_data=(X_test_water,y_test_water),
                                        callbacks=[Early],batch_size=4,verbose=0)


            ########################################## AIRFLOW ##########################################
            for j,(x_,y_) in enumerate([(X_train_air,y_train_air),(X_test_air,y_test_air),(X_air,self.y_air)]):
                keys = names[j]
                vals = self.metrics(model=self.model_air,x=x_,y=y_)
                self.results_air = self.update_results(self.results_air,keys,vals)
            #############################################################################################

            ##########################################  WATER  ##########################################
            for j,(x_,y_) in enumerate([(X_train_water,y_train_water),(X_test_water,y_test_water),(X_water,self.y_water)]):
                keys = names[j]
                vals = self.metrics(model=self.model_water,x=x_,y=y_)
                self.results_water = self.update_results(self.results_water,keys,vals)
            #############################################################################################

            pos_air = self.get_best(self.results_air)
            pos_water = self.get_best(self.results_water)

            if i==0 or i==pos_air:
                air_weights = self.model_air.get_weights()

                self.air_train = {"r2":self.results_air["r2_train"][i],
                                  "mse":self.results_air["mse_train"][i],
                                  "mape":self.results_air["mape_train"][i]}

                self.air_test = {"r2":self.results_air["r2_test"][i],
                                 "mse":self.results_air["mse_test"][i],
                                 "mape":self.results_air["mape_test"][i]}

                self.air_all = {"r2":self.results_air["r2_all"][i],
                                "mse":self.results_air["mse_all"][i],
                                "mape":self.results_air["mape_all"][i]}
                self.n_air = i+1
                if printer:
                    verbose = True

            if i==0 or i==pos_water:
                water_weights = self.model_water.get_weights()

                self.water_train = {"r2": self.results_water["r2_train"][i],
                                    "mse": self.results_water["mse_train"][i],
                                    "mape": self.results_water["mape_train"][i]}

                self.water_test = {"r2": self.results_water["r2_test"][i],
                                   "mse": self.results_water["mse_test"][i],
                                   "mape": self.results_water["mape_test"][i]}

                self.water_all = {"r2": self.results_water["r2_all"][i],
                                  "mse": self.results_water["mse_all"][i],
                                  "mape": self.results_water["mape_all"][i]}

                self.n_water = i+1
                if printer:
                    verbose = True

            else:
                verbose = False

            if verbose:
                self.print_results()

        self.model_water.set_weights(water_weights)
        self.model_air.set_weights(air_weights)

    def print_results(self):
        '''
        Prints models' metrics
        '''
        print("=" * 60)
        for air,water,mode in [(self.air_train,self.water_train,"Train"),
                               (self.air_test,self.water_test,"Test "),
                               (self.air_all,self.water_all,"All  ")]:

            print("-"*20+"\t\t"+mode+"\t\t"+"-"*20)

            print(f"#Model Air: {self.n_air} \t\t\t\t\t\t    #Model H2O: {self.n_water} \n")

            print("R Air:     {:.4f} \t\t\t\t\t    R H2O:    {:.4f}".format(np.sqrt(air["r2"]),np.sqrt(water["r2"])))

            print("MSE Air:   {:.4f} \t\t\t\t\t    MSE H2O:  {:.4f}".format(air["mse"], water["mse"]))

            print("MAPE Air:  {:.2f}% \t\t\t\t\t    MAPE H2O: {:.2f}%".format(air["mape"]*100,water["mape"]*100))

        print("=" * 60)
        print("")

    def save(self,fname,dir=None):
        '''
        Saves air and water models to an specific directory or to the current directory.
        The fname argument will be used to name both models, however each model will
        start with its compound name (i.e. water or air).

        :param fname: str - File name with .h5 extension.
        :param dir: str - Directory to save both air and water models.
        '''
        if dir:
            if not os.path.exists(dir):
                os.makedirs(dir)
                self.model_air.save(dir+"/air_"+fname)
                self.model_water.save(dir+"/water_"+fname)
            else:
                self.model_air.save(dir + "/air_" + fname)
                self.model_water.save(dir + "/water_" + fname)
        else:
            self.model_air.save("air_"+fname)
            self.model_water.save("water_"+fname)

    def load(self,fname=None,dir=None):
        '''
        It loads the MPFM's models. Passing both fname and dir will load the files inside the directory
        (this option is supposed to be used if there is more than one MPFM saved in the same directory).
        When only dir is passed it's assumed that there are only two files inside the directory: one for
        air and one for water. Passing only fname will load the models in the current directory.

        :param fname: str - File name used in the save method.
        :param dir: str - Directory name where models are saved

        '''
        if dir and fname:
            self.model_air = tf.keras.models.load_model(dir + "/air_" + fname)
            self.model_water = tf.keras.models.load_model(dir + "/water_" + fname)
        elif dir:
            files = os.listdir(dir)
            self.model_air = tf.keras.models.load_model(dir+"/"+files[0])
            self.model_water = tf.keras.models.load_model(dir+"/"+files[1])
        elif fname:
            self.model_air = tf.keras.models.load_model("air_"+fname)
            self.model_water = tf.keras.models.load_model("water_"+fname)
        else:
            raise (ValueError("Please, pass at least one of the two arguments: fname or dir"))

    def predict(self,x):
        '''
        :param x: Dataframe object with the same column names as passed to create MPFM object
        :return:
        '''
        x_air = x[self.air_sensors]
        x_water = x[self.water_sensors]

        air_pred = self.model_air.predict(x_air)
        water_pred = self.model_water.predict(x_water)

        return air_pred,water_pred

    def plot_results(self,phase="water",fname=None):
        '''
          Plot MSE and R from the results dictionary from fit
        '''

        plt.style.use("seaborn")

        results = self.results_water if phase=="water" else self.results_air
        idx = self.get_best(results)

        n_ANN = len(results["r2_train"])

        test_mse = results["mse_test"]

        train_mse = results["mse_train"]

        train_r2 = np.sqrt(results["r2_train"])

        test_r2 = np.sqrt(results['r2_test'])

        x = np.arange(1,n_ANN+1)

        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 6))
        axes = axes.flatten()

        axes[0].plot(x, train_r2, label="Train")
        axes[0].plot(x, test_r2, label="Test")
        axes[0].set_title("Correlation Coefficient across Neural Networks", fontsize=15)
        axes[0].scatter(idx+1,np.sqrt(results["r2_train"][idx]),color="red",label="Best")
        axes[0].scatter(idx+1,np.sqrt(results["r2_test"][idx]),color="red")
        axes[0].set_ylabel("R", fontsize=15)
        axes[0].legend()

        axes[1].plot(train_r2, test_r2, 'o', alpha=0.5)
        axes[1].plot(np.sqrt(results["r2_train"][idx]),np.sqrt(results["r2_test"][idx]), 'o',
                     alpha=1,color="red",label="Best")
        axes[1].set_xlabel(r"$R_{train}$", fontsize=15)
        axes[1].set_ylabel(r"$R_{test}$", fontsize=15)
        axes[1].set_title("Correlation Coefficient Pairs", fontsize=15)
        axes[1].legend()

        axes[2].plot(x, train_mse, label="Train")
        axes[2].plot(x, test_mse, label="Test")
        axes[2].scatter(idx+1,results["mse_train"][idx],color="red",label="Best")
        axes[2].scatter(idx+1,results["mse_test"][idx],color="red")
        axes[2].set_title("MSE across Neural Networks", fontsize=15)
        axes[2].set_xlabel("Neural Network Number", fontsize=15)
        axes[2].set_ylabel("MSE", fontsize=15)
        axes[2].legend()

        axes[3].plot(train_mse, test_mse, 'o', alpha=0.5)
        axes[3].plot(results["mse_train"][idx], results["mse_test"][idx], 'o', alpha=1,color="red",label="Best")
        axes[3].set_ylabel(r"$MSE_{test}$", fontsize=15)
        axes[3].set_xlabel(r"$MSE_{train}$", fontsize=15)
        axes[3].set_title("MSE Pairs", fontsize=15)
        axes[3].legend()

        plt.suptitle(phase.title()+" Result Summary of " + str(n_ANN) + " Neural Networks ",
                     fontsize=20, fontweight="bold")

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        if fname:
            plt.savefig(fname,dpi=300)

    def weights_update(self):
        '''
        A way that I've found to re-initialize the weights of the neural networks in order to search
        for a global minimum in the optimization of the weights. I could have used a different approach
        with tf.keras.models.clone_model which re-initialize all layers weights.
        :return:
        '''
        initializer = tf.keras.initializers.GlorotUniform()
        for layer in self.model_air.layers:
            weights, bias = layer.get_weights()
            new_weights = initializer(shape=(weights.shape[0], weights.shape[1]))
            layer.set_weights((new_weights, bias))

        for layer in self.model_water.layers:
            weights, bias = layer.get_weights()
            new_weights = initializer(shape=(weights.shape[0], weights.shape[1]))
            layer.set_weights((new_weights, bias))

    @staticmethod
    def metrics(model,x,y):
        y_pred = model.predict(x).flatten()
        mse = mean_squared_error(y,y_pred)
        r2 = r2_score(y,y_pred)
        mape = np.mean(np.abs(y-y_pred)/y)
        return r2,mse,mape

    @staticmethod
    def get_best(results):
        v = pd.DataFrame(results)
        v = v.sort_values("r2_train",ascending=False)
        v["Rank_Train"] = np.arange(1,len(v)+1)
        v = v.sort_values("r2_test",ascending=False)
        v["Rank_Test"] = np.arange(1,len(v)+1)
        v["Final_Rank"] = v["Rank_Train"]+v["Rank_Test"]
        pos = np.sqrt(v.sort_values("Final_Rank",ascending=True) \
                          [["r2_train","r2_test","mse_train","mse_test"]]).index[0]
        return pos

    @staticmethod
    def make_results():
        names = ["r2_train","r2_test","r2_all",
                 "mse_train","mse_test","mse_all",
                 "mape_train","mape_test","mape_all"]
        results = {name:list() for name in names}
        return results

    @staticmethod
    def update_results(results,keys,vals):
        for i in range(len(keys)):
            results[keys[i]].append(vals[i])
        return results

class TwoPhaseFlow:

    def __init__(self,pattern=None):
        '''
        :param pattern: str - which flow pattern to get the data names are in pt-br ("Estratificado, Plug, Golfada).
        If None are passed to pattern then the data from all flow patterns will be returned after calling the .load
        method of the object.
        '''
        self.pattern = pattern

    @staticmethod
    def median_absolute_deviation(x):
        '''
        Median Absolute Deviation is the method used to get rid of outliers in the dataset. This method was selected
        due to median's low sensitivity to outliers. (When using 3 sigma method the variance and mean are highly
        affected by the presence of outliers, thus not being the best option).
        :param x:
        :return:
        '''
        med = np.median(x)
        dist = np.abs(x - med)
        dist = np.median(dist)
        MAD = 1.4826 * dist
        return MAD

    def Clean(self,df):
        '''
        This function loops through the dataframe's columns and check which observations are not inside the
        interval Median - 3*MAD < x < 3*MAD + Median and replace them with the median value.

        :param df: Dataframe object
        :return:
        '''
        X_labels = ["Piezo1", "Piezo2", "801s", "Condutance", "dP",
                    "OutletFlow", "AccX", "AccY", "AccZ", "GyrX", "GyrY", "GyrZ"]
        for col in X_labels:
            x = df[[col]].values
            MAD = self.median_absolute_deviation(x)
            cond = np.abs(x - np.median(x)) / (MAD + 1e-8)
            x[cond > 3] = np.median(x)
            df[col] = x
        return df

    @staticmethod
    def Read_csv(file):
        '''

        :param file: str - csv file name in the GitHub repository
        :return:
        '''
        # This variable holds the URL for the repo directory with the csv files
        data_path = "https://raw.githubusercontent.com/EduardoPach/Chemical_Engineering/main/Multiphase_Flow_Meter/Data"
        df = pd.read_csv(data_path+"/"+file)
        # Adds the name of the experiment that will be further used
        df["Name"] = file.split(".")[0]
        return df

    def load(self,drop=None):
        '''

        :param drop: str - column of the data to be dropped.
        :return:
        '''

        # files' name in the GitHub repo
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
        # mapping the Read_csv method to the files list and returning a list of dataframes
        dfs = list(map(self.Read_csv,files))
        # mapping the Clean method to the list of dfs returning a list of dfs without outliers
        dfs = list(map(self.Clean,dfs))
        # concatenate all dataframes in the list of dataframes
        df = pd.concat(dfs)
        # selects the pattern specified in the object creation
        if self.pattern:
            df = df.loc[df["FlowPattern"]==self.pattern]
        # drops the column(s) specified in this method
        if drop:
            df.drop(labels=drop,axis=1,inplace=True)
            df.drop(labels="FlowPattern",axis=1,inplace=True)
        # groups by experiment Name and returns the means and stds
        df = df.groupby("Name").agg(["mean","std"])
        # change column names to avoid a MultiIndex dataframe. Takes the higher index level and joins it
        # to the sensor name e.g. Piezo1_mean and Piezo1_std
        df.columns = ["_".join(col) for col in df.columns]
        # pops the response variables from the dataframe
        y = {"InletFlow":df.pop("InletFlow_mean").values,"Airflow":df.pop("Airflow_mean").values}
        # remove any Unnamed column that might be in the dataframe and the remaining response variables columns for std
        x = df.loc[:, (~df.columns.str.match("Unnamed")) &
                       (~df.columns.str.match("Airflow")) &
                       (~df.columns.str.match("InletFlow"))]

        return x,y