#%% 
import numpy as np
import matplotlib.pyplot as plt
import h5py
import tensorflow as tf
import datetime

#%% 
f_data = 'PRIOR_NL_10000.h5'
#f_data = 'PRIOR_NL_200000.h5'
# load 'M1' and 'D1' from f_data
with h5py.File(f_data, 'r') as f:
    M = f['M1'][:, ::4]
    D = f['D1'][:]


M=np.log10(M)
D=np.real(np.log10(D))
# the index of all data in D that are not NaN
idx = np.where(~np.isnan(D).any(axis=1))[0]

M=M[idx]
D=D[idx]


#%%
N, Nm = M.shape
N, Nd = D.shape

print('Number of samples:', N)
print('Number of input features:', Nm)
print('Number of output features:', Nd)


# split data into training,validation and test data
N_train = int(N * 0.8)
N_val = int(N * 0.1)
N_test = N - N_train - N_val

M_train = M[:N_train]
D_train = D[:N_train]

M_val = M[N_train:N_train + N_val]
D_val = D[N_train:N_train + N_val]

M_test = M[N_train + N_val:]
D_test = D[N_train + N_val:]

# %% 
# Setup a simple Feed Forward Neural Network to learn the mapping from M to D
# Use an input layer using M.shape[1] as input shape
# Use 2 hiden layers with 64 units each
# Use an output layer with D.shape[1] units

# Consider using
# Noramalization !
# Regularizatoin / Dropout
# Handling negative values in the output
# Early Stopping / adaptive learning rate

# Tensorboard callback to visualize the training process
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

nunits = 2**6;2**8
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(nunits, input_shape=(Nm,), activation='relu'))   
model.add(tf.keras.layers.Dense(nunits, activation='relu'))
model.add(tf.keras.layers.Dense(nunits, activation='relu'))
model.add(tf.keras.layers.Dense(Nd))

model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(M_train, D_train, 
          epochs=100, 
          batch_size=32, 
          validation_data=(M_val, D_val), 
          callbacks=[tensorboard_callback])

#%%

# Evaluate the model
loss = model.evaluate(M_test, D_test)
print('Test Loss:', loss)

# Compute the Mean Squared Error
D_pred = model.predict(M_test)
mse = np.mean((D_test - D_pred)**2)
print('Test MSE:', mse)

# Compute the relative error
rel_err = np.mean(np.abs(D_test - D_pred) / np.abs(D_test))
print('Test Relative Error:', rel_err)



#%%
# Plot the predicted vs true output
plt.figure()
plt.plot(D_test, D_pred, '.', 'markersize', 0.1)
plt.xlabel('True D')
plt.ylabel('Predicted D')
plt.show()
# %%
plt.figure()
plt.plot(D_test[0:30],'k-')
plt.plot(D_pred[0:30],'r:')
plt.xlabel('Sample #')
plt.ylabel('D')
plt.grid()
plt.show()



# %%
