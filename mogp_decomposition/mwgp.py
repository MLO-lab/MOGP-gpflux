from matplotlib import pyplot as plt
import os
import numpy as np
import sys
from sklearn.metrics import *
data_folder_path = '../../../data'
sys.path.append(data_folder_path)
import tensorflow as tf
import gpflow
import gpflux
from tensorflow.keras.layers import Input, Dense, Flatten, Concatenate, Multiply
from tensorflow.keras.models import Model
from tensorflow.keras import layers


def get_free_gpu_idx():
    """Get the index of the GPU with current lowest memory usage."""
    os.system("nvidia-smi -q -d Memory |grep -A4 GPU|grep Used >tmp")
    memory_available = [int(x.split()[2]) for x in open("tmp", "r").readlines()]
    return np.argmin(memory_available)

gpu_idx = get_free_gpu_idx()
print("Using GPU #%s" % gpu_idx)
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)


class GPD():
    def __init__(
        self,
        I,
        J,
        K,
        M1,
        M2,
        emb_sizes,
        batch_size=512,
        obs_mean1=None,
        obs_mean2=None,
        emb_reg=1e-3,
        lr=1e-3,
        save_path="./",
    ):
        """
        :param I: integer, number of entities in the first dimension.
        :param J: integer, number of entities in the second dimension.
        :param K: integer, number of entities in the third dimension.
        :param M: integer, number of inducing points.
        :param emb_sizes: a list of embedding sizes as integers.
        :param batch_size: integer, mini batch size for training, necessary to define the placeholders.
        :param obs_mean: integer, mean of training targets, optional.
        :param emb_reg: float, regularization term for embeddings.
        :param lr: float, learning rate.
        """

        self.I = I
        self.J = J
        self.K = K
        self.M1 = M1
        self.M2 = M2
        self.emb_sizes = emb_sizes
        self.batch_size = batch_size
        self.obs_mean1 = obs_mean1
        self.obs_mean2 = obs_mean2
        self.emb_reg = emb_reg
        self.lr = lr
        self.save_path = save_path
        self.emb1 = None
        self.emb2 = None
        self.emb3 = None
        self.kernel1 = None
        self.kernel2 = None
        self.model = None
        self.Z_size1 = None
        self.Z_size2 = None


    def make_kernels(self, emb_size, kernels=None, active_dims=None):
        """
        :param emb_size: integer, embedding size for a kernel of one specific dimension.
        :param kernels: a list of strings, e.g. ['RBF', 'White], the sum of which shall form the kernel for one
        specific dimension.
        :param active_dims: active dimension of the kernel.
        :return: one kernel being the sum of all kernels required by the parameter 'kernels'.
        """ 
        kern = None
        if "RBF" in kernels:
            kern = gpflow.kernels.RBF(active_dims=active_dims, lengthscales=np.ones(emb_size))
            if "White" in kernels:
                kern = kern + gpflow.kernels.White()

        if "Linear" in kernels:
            kern = gpflow.kernels.Linear(active_dims=active_dims)  # lengthscales=np.ones(emb_size))
            if "White" in kernels:
                kern = kern + gpflow.kernels.White()

        return kern


    def build(self, kernels=["RBF"], first_pcs=None, second_pcs=None, third_pcs=None):  # , **kwargs    
        """
        Building the GP-Decomposition model, partially by calling build_svgp or build_sgpr.
        :param kernels: list of strings, defining the kernel structure for each embedding.
        :param optimiser: string, currently either 'adam' or 'adagrad'.
        :return: None.
        """
        
        if first_pcs is None:
            self.emb1 = tf.keras.Sequential(tf.keras.layers.Embedding(
                input_dim=self.I+1,
                output_dim=self.emb_sizes[0],
                dtype=tf.float64,
                embeddings_regularizer=tf.keras.regularizers.l2(self.emb_reg),
                name="emb1",
                mask_zero=True
                
            ))
        else:
            self.emb1 = tf.keras.Sequential(tf.keras.layers.Embedding(
                input_dim=self.I+1,
                output_dim=self.emb_sizes[0],
                dtype=tf.float64,
                embeddings_regularizer=tf.keras.regularizers.l2(self.emb_reg),
                name="emb1",
                mask_zero=True,
                embeddings_initializer = tf.keras.initializers.Constant(value=first_pcs)
            ))
            
        if second_pcs is None:    
            self.emb2 = tf.keras.Sequential(tf.keras.layers.Embedding(
                input_dim=self.J+1,
                output_dim=self.emb_sizes[1],
                dtype=tf.float64,
                embeddings_regularizer=tf.keras.regularizers.l2(self.emb_reg),
                name="emb2",
                mask_zero=True
            ))
        else:
            self.emb2 = tf.keras.Sequential(tf.keras.layers.Embedding(
                input_dim=self.J+1,
                output_dim=self.emb_sizes[1],
                dtype=tf.float64,
                embeddings_regularizer=tf.keras.regularizers.l2(self.emb_reg),
                name="emb2",
                mask_zero=True,
                embeddings_initializer = tf.keras.initializers.Constant(value=second_pcs)
            ))
        self.emb1.build()
        self.emb2.build()
        
        if self.K is not None:
            if third_pcs is None:
                self.emb3 = tf.keras.Sequential(tf.keras.layers.Embedding(
                    input_dim=self.K+1,
                    output_dim=self.emb_sizes[2],
                    dtype=tf.float64,
                    embeddings_regularizer=tf.keras.regularizers.l2(self.emb_reg),
                    name="emb3",
                    mask_zero=True
                ))
            else:
                self.emb3 = tf.keras.Sequential(tf.keras.layers.Embedding(
                    input_dim=self.K+1,
                    output_dim=self.emb_sizes[2],
                    dtype=tf.float64,
                    embeddings_regularizer=tf.keras.regularizers.l2(self.emb_reg),
                    name="emb3",
                    mask_zero=True,
                    embeddings_initializer = tf.keras.initializers.Constant(value=third_pcs)
                ))
            self.emb3.build()
            

        kernels1_1 = self.make_kernels(
            emb_size=self.emb_sizes[0],
            kernels=kernels,
            active_dims=np.arange(0, self.emb_sizes[0]),
        )
        
        kernels1_2 = self.make_kernels(
            emb_size=self.emb_sizes[0],
            kernels=kernels,
            active_dims=np.arange(0, self.emb_sizes[0]),
        )

        kernels2 = self.make_kernels(
            emb_size=self.emb_sizes[1],
            kernels=kernels,
            active_dims=np.arange(
                self.emb_sizes[0], self.emb_sizes[0] + self.emb_sizes[1]
            ),
        )
    
        self.kernel1 = kernels1_1 * kernels2
        if self.K is not None:
            kernels3 = self.make_kernels(
                emb_size=self.emb_sizes[2],
                kernels=kernels,
                active_dims=np.arange(
                    self.emb_sizes[0], self.emb_sizes[0] + self.emb_sizes[2]
                ),
            )
            self.kernel2 = kernels1_2 * kernels3
            
        if self.obs_mean1 is not None:
            observations_mean = tf.constant([self.obs_mean1], dtype=tf.float64)
            self.mean_fn1 = lambda _: observations_mean[:, None]
        else:
            self.mean_fn1 = self.obs_mean1
            
        if self.obs_mean2 is not None:
            observations_mean = tf.constant([self.obs_mean2], dtype=tf.float64)
            self.mean_fn2 = lambda _: observations_mean[:, None]
        else:
            self.mean_fn2 = self.obs_mean2   

        self.Z_size1 = self.emb_sizes[0] + self.emb_sizes[1]
        if self.K is not None:
            self.Z_size2 = self.emb_sizes[0] + self.emb_sizes[2]

    def set_inducing_points(self, X_tr1, X_tr2=None):
        ind = np.random.choice(range(X_tr1.shape[0]), self.M1, replace=False)
        inducing_variable = X_tr1[ind, :]
        inducing_variable_emb1 = self.emb1(inducing_variable[:, 0])
        inducing_variable_emb2 = self.emb2(inducing_variable[:, 1])
        hZ1 = tf.concat([inducing_variable_emb1, inducing_variable_emb2], axis=1)
        self.Z_0_1 = gpflow.inducing_variables.InducingPoints(hZ1)
        if self.K is not None:
            ind = np.random.choice(range(X_tr2.shape[0]), self.M2, replace=False)
            inducing_variable = X_tr2[ind, :]
            inducing_variable_emb1 = self.emb1(inducing_variable[:, 0])
            inducing_variable_emb3 = self.emb3(inducing_variable[:, 1])
            hZ2 = tf.concat([inducing_variable_emb1, inducing_variable_emb3], axis=1)  
            self.Z_0_2 = gpflow.inducing_variables.InducingPoints(hZ2)
    
    
    def make_model(self):
        def myMask(x):
            import keras.backend as Kbackend
            mask= Kbackend.greater(x,0) #will return boolean values
            mask= Kbackend.cast(mask, dtype=Kbackend.floatx()) 
            return mask
    
        likelihood = gpflow.likelihoods.Gaussian(0.1)
        likelihood_container = gpflux.layers.TrackableLayer()
        likelihood_container.likelihood = likelihood
        loss = gpflux.losses.LikelihoodLoss(likelihood)

        inputA1 = Input(shape=1)
        xA = self.emb1(inputA1)
        tmp = tf.keras.layers.Lambda(lambda val: myMask(val))(inputA1)
        xA = Multiply(name = 'masked_embedding_cells1')([xA,tmp])
        
        inputB1 = Input(shape=1)
        xB = self.emb2(inputB1)
        tmp = tf.keras.layers.Lambda(lambda val: myMask(val))(inputB1)
        xB = Multiply(name = 'masked_embedding_genes')([xB,tmp])

        modelA1 = Model(inputs=inputA1, outputs=xA)
        modelB1 = Model(inputs=inputB1, outputs=xB)
        combined = tf.reshape(tf.keras.layers.concatenate([modelA1.output, modelB1.output]),[-1, self.emb_sizes[0]+self.emb_sizes[1]])
        gp_layer1 = gpflux.layers.GPLayer(self.kernel1, self.Z_0_1, num_data=self.I*self.J, num_latent_gps=self.emb_sizes[0]+self.emb_sizes[1])
        xC = gp_layer1(combined)
        Output1 = likelihood_container(xC)
        
        if self.K is not None:
            inputA2 = Input(shape=1)
            xA = self.emb1(inputA2)
            tmp = tf.keras.layers.Lambda(lambda val: myMask(val))(inputA2)
            xA = Multiply(name = 'masked_embedding_cells2')([xA,tmp])
            
            inputB2 = Input(shape=1)
            xB = self.emb3(inputB2)
            tmp = tf.keras.layers.Lambda(lambda val: myMask(val))(inputB2)
            xB = Multiply(name = 'masked_embedding_peaks')([xB,tmp])
            
            modelA2 = Model(inputs=inputA2, outputs=xA)
            modelB2 = Model(inputs=inputB2, outputs=xB)
            combined = tf.reshape(tf.keras.layers.concatenate([modelA2.output, modelB2.output]),[-1, self.emb_sizes[0]+self.emb_sizes[2]])
            gp_layer2 = gpflux.layers.GPLayer(self.kernel2, self.Z_0_2, num_data=self.I*self.K, num_latent_gps=self.emb_sizes[0]+self.emb_sizes[2])
            xC = gp_layer2(combined)
            Output2 = likelihood_container(xC)
            self.model = Model(inputs=[inputA1,inputB1,inputA2,inputB2], outputs=[Output1,Output2])
            self.model.compile(loss=loss, optimizer="adam")
        else:    
            self.model = Model(inputs=[inputA1,inputB1], outputs=Output1)
            self.model.compile(loss=loss, optimizer="adam")    
              
    def train(self, X_tr1, Y_tr1, X_tr2=None, Y_tr2=None, labels=None, n_iter=None):
        
        checkpoint_path = os.path.join(self.save_path, "model_params.ckpt")  
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path ,
                                                 save_weights_only=True,
                                                 verbose=1)
        self.set_inducing_points(X_tr1, X_tr2)
        self.make_model()
        
        if self.K is not None:
            raw_inputs = [X_tr1, X_tr2]
            padded_inputs = tf.keras.preprocessing.sequence.pad_sequences(raw_inputs, padding="post")
            X_tr1_0padded = padded_inputs[0]
            X_tr2_0padded = padded_inputs[1]
            
            raw_inputs = [Y_tr1, Y_tr2]
            padded_inputs = tf.keras.preprocessing.sequence.pad_sequences(raw_inputs, padding="post", dtype=np.float64)
            Y_tr1_0padded = padded_inputs[0]
            Y_tr2_0padded = padded_inputs[1]
            
            
            shuffle_ids1 = np.random.choice(range(X_tr1_0padded.shape[0]), X_tr1_0padded.shape[0], replace=False)
            X_tr_Shuffeled1 = np.copy(X_tr1_0padded[shuffle_ids1])
            Y_tr_Shuffeled1 = np.copy(Y_tr1_0padded[shuffle_ids1])
            shuffle_ids2 = np.random.choice(range(X_tr2_0padded.shape[0]), X_tr2_0padded.shape[0], replace=False)
            X_tr_Shuffeled2 = np.copy(X_tr2_0padded[shuffle_ids2])
            Y_tr_Shuffeled2 = np.copy(Y_tr2_0padded[shuffle_ids2])
            hist = self.model.fit((X_tr_Shuffeled1[:, 0],X_tr_Shuffeled1[:, 1],X_tr_Shuffeled2[:, 0],X_tr_Shuffeled2[:, 1]), (Y_tr_Shuffeled1,Y_tr_Shuffeled2), epochs=n_iter, batch_size=self.batch_size, callbacks=[cp_callback])
            plt.plot(hist.history["loss"])
        else:    
            shuffle_ids1 = np.random.choice(range(X_tr1.shape[0]), X_tr1.shape[0], replace=False)
            X_tr_Shuffeled1 = np.copy(X_tr1[shuffle_ids1])
            Y_tr_Shuffeled1 = np.copy(Y_tr1[shuffle_ids1])
            hist = self.model.fit([X_tr_Shuffeled1[:, 0],X_tr_Shuffeled1[:, 1]], Y_tr_Shuffeled1, epochs=n_iter, batch_size=self.batch_size, callbacks=[cp_callback])
            plt.plot(hist.history["loss"])
        return hist
        
        
    def load_model(self, X_tr1, X_tr2):
        self.set_inducing_points(X_tr1, X_tr2)
        self.make_model()    
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        checkpoint_path = os.path.join(self.save_path, "model_params.ckpt")  
        self.model.load_weights(checkpoint_path)
    