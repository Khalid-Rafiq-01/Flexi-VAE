#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import torch
import torch.nn as nn
import time
import math
import torch

num_time_steps = 500
x = np.linspace(0.0,1.0,num=128)
dx = 1.0/np.shape(x)[0]
tsteps = np.linspace(0.0,2.0,num=num_time_steps)
dt = 2.0/np.shape(tsteps)[0]

class AE_Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim=2, feats=[512, 256, 128, 64, 32]):
        super(AE_Encoder, self).__init__()
        self.latent_dim = latent_dim
        self._net = nn.Sequential(
            nn.Linear(input_dim, feats[0]),
            nn.GELU(),
            nn.Linear(feats[0], feats[1]),
            nn.GELU(),
            nn.Linear(feats[1], feats[2]),
            nn.GELU(),
            nn.Linear(feats[2], feats[3]),
            nn.GELU(),
            nn.Linear(feats[3], feats[4]),
            nn.GELU(),
            nn.Linear(feats[4], latent_dim)
        )

    def forward(self, x):
      Z = self._net(x)
      return Z


class AE_Decoder(nn.Module):
    def __init__(self, latent_dim, output_dim, feats=[32, 64, 128, 256, 512]):
        super(AE_Decoder, self).__init__()
        self.output_dim = output_dim
        self._net = nn.Sequential(
            nn.Linear(latent_dim, feats[0]),
            nn.GELU(),
            nn.Linear(feats[0], feats[1]),
            nn.GELU(),
            nn.Linear(feats[1], feats[2]),
            nn.GELU(),
            nn.Linear(feats[2], feats[3]),
            nn.GELU(),
            nn.Linear(feats[3], feats[4]),
            nn.GELU(),
            nn.Linear(feats[4], output_dim),
        )

    def forward(self, x):
      y = self._net(x)
      return y


class AE_Model(nn.Module):
    def __init__(self, encoder, decoder):
        super(AE_Model, self).__init__()
        self.encoder = encoder
        self.decoder = decoder # decoder for x(t)

    def forward(self, x):
        z = self.encoder(x)
        # Reconstruction
        x_hat = self.decoder(z)  # Reconstruction of x(t)

        return x_hat
        
        
class PytorchLSTM(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=40, output_dim=2):
        super().__init__()
        # First LSTM: simulates return_sequences=True
        self.lstm1 = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        # Second LSTM: simulates return_sequences=False
        self.lstm2 = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        # Dense layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        """
        x shape: [batch_size, time_window, input_dim]
        """
        # LSTM1 (return_sequences=True)
        out1, (h1, c1) = self.lstm1(x)
        # out1 shape: [batch_size, time_window, hidden_dim]

        # LSTM2 (return_sequences=False -> we only use the last time step)
        out2, (h2, c2) = self.lstm2(out1)
        # out2 shape: [batch_size, time_window, hidden_dim]
        # Last timestep (since we didn't set return_sequences=True)
        # is effectively out2[:, -1, :], but PyTorch LSTM always returns full seq unless you slice.

        last_timestep = out2[:, -1, :]  # shape: [batch_size, hidden_dim]

        # Dense -> 2 outputs
        output = self.fc(last_timestep)  # shape: [batch_size, 2]
        return output
        
# def measure_lstm_prediction_time(
#     decoder,
#     lstm_model,
#     lstm_testing_data,
#     sim_num,
#     final_time,
#     time_window=10
# ):
#     """
#     Predicts up to `final_time` in a walk-forward manner for simulation `sim_num`,
#     measures the elapsed time, and returns the final predicted latent + the true latent.

#     Parameters
#     ----------
#     decoder : torch.nn.Module
#     	The trained weights of the decoder
#     model : torch.nn.Module
#         Trained PyTorch LSTM model. We'll set model.eval() inside.
#     lstm_testing_data : np.ndarray
#         Shape (num_test_snapshots, num_time_steps, 3).
#         The last dimension typically holds (2 latents + 1 param) or similar.
#     sim_num : int
#         Which simulation index to use (e.g., 0 for the first).
#     final_time : int
#         The final timestep index you want to predict up to (>= time_window).
#         For example, if time_window=10 and final_time=20, we will predict from t=10..19.
#     time_window : int
#         Size of the rolling window (default=10).

#     Returns
#     -------
#     float
#         Elapsed time (seconds) for performing the predictions from t=time_window up to t=final_time.
#     np.ndarray
#         The final predicted latent at time=final_time (shape (2,)).
#     np.ndarray
#         The true latent at time=final_time (shape (2,)).
#     """

#     # Basic shape info
#     num_time_steps = lstm_testing_data.shape[1]
#     if final_time > num_time_steps:
#         raise ValueError(
#             f"final_time={final_time} exceeds available time steps={num_time_steps}."
#         )
#     if final_time < time_window:
#         raise ValueError(
#             f"final_time={final_time} is less than time_window={time_window}, no prediction needed."
#         )

#     # Initialize the rolling window with first `time_window` steps
#     input_seq = np.zeros((1, time_window, 3), dtype=np.float32)
#     input_seq[0, :, :] = lstm_testing_data[sim_num, 0:time_window, :]

#     lstm_model.eval()  # inference mode

#     final_pred = None  # store the final predicted latent
#     start_time = time.time()

#     with torch.no_grad():
#         # Predict from t=time_window to t=final_time-1
#         # so that at the end of the loop we've generated a prediction for index final_time.
#         # If you want the model's prediction at final_time itself, we do a loop up to final_time.
#         for t in range(time_window, final_time):
#             inp_tensor = torch.from_numpy(input_seq).float()  # shape [1, 10, 3]
#             pred = lstm_model(inp_tensor)  # shape [1, 2]
#             pred_np = pred.numpy()[0, :]  # shape (2,)

#             # Shift the rolling window
#             temp = input_seq[0, 1:time_window, :].copy()
#             input_seq[0, 0:time_window - 1, :] = temp
#             input_seq[0, time_window - 1, 0:2] = pred_np

#             # Keep track of the last prediction
#             final_pred = pred_np
            
#     x_hat_tau_pred = decoder(torch.tensor(final_pred, dtype = torch.float32))
    
#     end_time = time.time()
    
#     elapsed = end_time - start_time

#     # final_pred is the LSTM's predicted latent for step `final_time`.
#     # The *true* latent at that time is:
#     final_true = lstm_testing_data[sim_num, final_time, 0:2]  # shape (2,)
    
#     return elapsed, final_pred, final_true

import torch
import time
import numpy as np

def measure_lstm_prediction_time(
    decoder,
    lstm_model,
    lstm_testing_data,
    sim_num,
    final_time,
    time_window=10
):
    """
    Predicts up to `final_time` in a walk-forward manner for simulation `sim_num`,
    measures the elapsed time, and returns the final predicted latent + the true latent.

    Parameters
    ----------
    decoder : torch.nn.Module
    	The trained weights of the decoder.
    lstm_model : torch.nn.Module
        Trained PyTorch LSTM model. We'll set model.eval() inside.
    lstm_testing_data : np.ndarray
        Shape (num_test_snapshots, num_time_steps, 3).
        The last dimension typically holds (2 latents + 1 param) or similar.
    sim_num : int
        Which simulation index to use (e.g., 0 for the first).
    final_time : int
        The final timestep index you want to predict up to (>= time_window).
    time_window : int
        Size of the rolling window (default=10).

    Returns
    -------
    float
        Elapsed time (seconds) for performing the predictions from t=time_window up to t=final_time.
    np.ndarray
        The final predicted latent at time=final_time (shape (2,)).
    np.ndarray
        The true latent at time=final_time (shape (2,)).
    """

    num_time_steps = lstm_testing_data.shape[1]
    if final_time > num_time_steps:
        raise ValueError(
            f"final_time={final_time} exceeds available time steps={num_time_steps}."
        )
    if final_time < time_window:
        raise ValueError(
            f"final_time={final_time} is less than time_window={time_window}, no prediction needed."
        )

    input_seq = np.zeros((1, time_window, 3), dtype=np.float32)
    input_seq[0, :, :] = lstm_testing_data[sim_num, 0:time_window, :]

    lstm_model.eval()

    final_pred = None

    # Ensure GPU operations are completed before starting the timer
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    start_time = time.time()

    with torch.no_grad():
        for t in range(time_window, final_time):
            inp_tensor = torch.from_numpy(input_seq).float()

            # Move to GPU if available
            if torch.cuda.is_available():
                inp_tensor = inp_tensor.cuda()
                lstm_model = lstm_model.cuda()
                decoder = decoder.cuda()

            pred = lstm_model(inp_tensor)

            # Move back to CPU for further processing
            pred_np = pred.cpu().numpy()[0, :]

            temp = input_seq[0, 1:time_window, :].copy()
            input_seq[0, 0:time_window - 1, :] = temp
            input_seq[0, time_window - 1, 0:2] = pred_np

            final_pred = pred_np

    x_hat_tau_pred = decoder(torch.tensor(final_pred, dtype=torch.float32).unsqueeze(0))

    # Ensure all GPU computations are finished before stopping the timer
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    end_time = time.time()
    elapsed = end_time - start_time

    final_true = lstm_testing_data[sim_num, final_time, 0:2]

    return elapsed, final_pred, final_true

    
    
def collect_snapshots(Rnum):
    snapshot_matrix = np.zeros(shape=(np.shape(x)[0],np.shape(tsteps)[0]))

    trange = np.arange(np.shape(tsteps)[0])
    for t in trange:
        snapshot_matrix[:,t] = exact_solution(Rnum,tsteps[t])[:]

    return snapshot_matrix



def collect_multiparam_snapshots_train():
    rnum_vals_1 = np.arange(600,850,100)[:, None]
    rnum_vals_2 = np.arange(1100,2250,100)[:, None]
    rnum_vals = np.concatenate((rnum_vals_1, rnum_vals_2), axis = 0).squeeze()
    
    rsnap = 0
    for rnum_val in rnum_vals:
        snapshots_temp = np.transpose(collect_snapshots(rnum_val))
        
        if rsnap == 0:
            all_snapshots = snapshots_temp
        else:
            
            all_snapshots = np.concatenate((all_snapshots,snapshots_temp),axis=0)
            
        rsnap = rsnap + 1    
    return all_snapshots, rnum_vals/1000

def collect_multiparam_snapshots_test():
    rnum_vals = np.arange(450,2755,280)
    
    rsnap = 0
    for rnum_val in rnum_vals:
        snapshots_temp = np.transpose(collect_snapshots(rnum_val))
        
        if rsnap == 0:
            all_snapshots = snapshots_temp
        else:
            
            all_snapshots = np.concatenate((all_snapshots,snapshots_temp),axis=0)
            
        rsnap = rsnap + 1    
    return all_snapshots, rnum_vals/1000

# def collect_multiparam_snapshots_train():
#     rnum_vals = np.arange(900,2900,100)
    
#     rsnap = 0
#     for rnum_val in rnum_vals:
#         snapshots_temp = np.transpose(collect_snapshots(rnum_val))
        
#         if rsnap == 0:
#             all_snapshots = snapshots_temp
#         else:
            
#             all_snapshots = np.concatenate((all_snapshots,snapshots_temp),axis=0)
            
#         rsnap = rsnap + 1    
#     return all_snapshots, rnum_vals/1000

# def collect_multiparam_snapshots_test():
#     rnum_vals = np.arange(1050,2850,200)
    
#     rsnap = 0
#     for rnum_val in rnum_vals:
#         snapshots_temp = np.transpose(collect_snapshots(rnum_val))
        
#         if rsnap == 0:
#             all_snapshots = snapshots_temp
#         else:
            
#             all_snapshots = np.concatenate((all_snapshots,snapshots_temp),axis=0)
            
#         rsnap = rsnap + 1    
#     return all_snapshots, rnum_vals/1000
    
    

    # return elapsed, final_pred, final_true

def exact_solution(Rnum,t):
    x = np.linspace(0.0,1.0,num=128)
    t0 = np.exp(Rnum/8.0)
    return (x/(t+1))/(1.0+np.sqrt((t+1)/t0)*np.exp(Rnum*(x*x)/(4.0*t+4)))
    
        
