from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy as np
import torch
from torch import nn as nn
from torch.nn import functional as F
import tensorflow as tf
from tqdm import trange


TORCH_DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
TORCH_DEVICE = torch.device('cpu')

def get_required_argument(dotmap, key, message, default=None):
    val = dotmap.get(key, default)
    if val is default:
        raise ValueError(message)
    return val

def truncated_normal(size, std):
    # We use TF to implement initialization function for neural network weight because:
    # 1. Pytorch doesn't support truncated normal
    # 2. This specific type of initialization is important for rapid progress early in training in cartpole

    # Do not allow tf to use gpu memory unnecessarily
    # cfg = tf.compat.v1.ConfigProto()
    # cfg.gpu_options.allow_growth = True
    #
    # sess = tf.compat.v1.Session(config=cfg)
    # val = sess.run(tf.compat.v1.truncated_normal(shape=size, stddev=std))
    val = tf.random.truncated_normal(shape=size, stddev=std)

    # Close the session and free resources
    # sess.close()

    return torch.tensor(val.numpy(), dtype=torch.float32)


def swish(x):
    return x * torch.sigmoid(x)

def shuffle_rows(arr):
    idxs = np.argsort(np.random.uniform(size=arr.shape), axis=-1)
    return arr[np.arange(arr.shape[0])]

def get_affine_params(ensemble_size, in_features, out_features):

    w = truncated_normal(size=(ensemble_size, in_features, out_features),
                         std=1.0 / (2.0 * np.sqrt(in_features)))
    w = nn.Parameter(w)

    b = nn.Parameter(torch.zeros(ensemble_size, 1, out_features, dtype=torch.float32))

    return w, b

class PtModel(nn.Module):

    def __init__(self, ensemble_size, in_features, out_features):
        super().__init__()

        self.num_nets = 1

        self.in_features = in_features
        self.out_features = out_features

        self.lin0_w, self.lin0_b = get_affine_params(ensemble_size, in_features, 500)

        self.lin1_w, self.lin1_b = get_affine_params(ensemble_size, 500, 500)

        self.lin2_w, self.lin2_b = get_affine_params(ensemble_size, 500, 500)

        self.lin3_w, self.lin3_b = get_affine_params(ensemble_size, 500, out_features)

        self.inputs_mu = nn.Parameter(torch.zeros(in_features), requires_grad=False)
        self.inputs_sigma = nn.Parameter(torch.zeros(in_features), requires_grad=False)

        self.max_logvar = nn.Parameter(torch.ones(1, out_features // 2, dtype=torch.float32) / 2.0)
        self.min_logvar = nn.Parameter(- torch.ones(1, out_features // 2, dtype=torch.float32) * 10.0)

    def compute_decays(self):

        lin0_decays = 0.0001 * (self.lin0_w ** 2).sum() / 2.0
        lin1_decays = 0.00025 * (self.lin1_w ** 2).sum() / 2.0
        lin2_decays = 0.00025 * (self.lin2_w ** 2).sum() / 2.0
        lin3_decays = 0.0005 * (self.lin3_w ** 2).sum() / 2.0

        return lin0_decays + lin1_decays + lin2_decays + lin3_decays

    def fit_input_stats(self, data):

        mu = np.mean(data, axis=0, keepdims=True)
        sigma = np.std(data, axis=0, keepdims=True)
        sigma[sigma < 1e-12] = 1.0

        self.inputs_mu.data = torch.from_numpy(mu).to(TORCH_DEVICE).float().squeeze(0)
        self.inputs_sigma.data = torch.from_numpy(sigma).to(TORCH_DEVICE).float().squeeze(0)

    def forward(self, inputs, ret_logvar=False):

        # Transform inputs
        # inputs = (inputs - self.inputs_mu) / self.inputs_sigma

        inputs = inputs.matmul(self.lin0_w) + self.lin0_b
        inputs = swish(inputs)

        inputs = inputs.matmul(self.lin1_w) + self.lin1_b
        inputs = swish(inputs)

        inputs = inputs.matmul(self.lin2_w) + self.lin2_b
        inputs = swish(inputs)

        inputs = inputs.matmul(self.lin3_w) + self.lin3_b

        mean = inputs[:, :, :self.out_features // 2]

        logvar = inputs[:, :, self.out_features // 2:]
        logvar = self.max_logvar - F.softplus(self.max_logvar - logvar)
        logvar = self.min_logvar + F.softplus(logvar - self.min_logvar)

        if ret_logvar:
            return mean, logvar

        return mean.squeeze(0), torch.exp(logvar).squeeze(0)

    def train(self, X, Z, X_new, Z_new, nEpochs):
        """Trains the internal model of this controller. Once trained,
        this controller switches from applying random actions to using MPC.

        Arguments:
            obs_trajs: A list of observation matrices, observations in rows.
            acs_trajs: A list of action matrices, actions in rows.
            rews_trajs: A list of reward arrays.

        Returns: None.
        """

        # Construct new training points and add to training set
        # new_train_in, new_train_targs = [], []
        # for obs, acs in zip(obs_trajs, acs_trajs):
        #     new_train_in.append(np.concatenate([self.obs_preproc(obs[:-1]), acs], axis=-1))
        #     new_train_targs.append(self.targ_proc(obs[:-1], obs[1:]))
        train_in = np.concatenate((X,  X_new), axis=0)
        train_targs = np.concatenate((Z , Z_new), axis=0)



        # Train the model
        # self.has_been_trained = True

        # Train the pytorch model
        self.fit_input_stats(train_in)

        idxs = np.random.randint(train_in.shape[0], size=[train_in.shape[0]])

        epochs = nEpochs

        # TODO: double-check the batch_size for all env is the same
        batch_size = 32

        epoch_range = trange(epochs, unit="epoch(s)", desc="Network training")
        num_batch = int(np.ceil(idxs.shape[-1] / batch_size))

        for _ in epoch_range:

            for batch_num in range(num_batch):
                batch_idxs = idxs[batch_num * batch_size : (batch_num + 1) * batch_size]

                loss = 0.01 * (self.max_logvar.sum() - self.min_logvar.sum())
                loss += self.compute_decays()

                # TODO: move all training data to GPU before hand
                train_in_batch = torch.from_numpy(train_in[batch_idxs]).to(TORCH_DEVICE).float()
                train_targ_batch = torch.from_numpy(train_targs[batch_idxs]).to(TORCH_DEVICE).float()

                mean, logvar = self.forward(train_in_batch, ret_logvar=True)
                mean = mean[0]
                logvar = logvar[0]
                inv_var = torch.exp(-logvar)

                train_losses = ((mean - train_targ_batch) ** 2) * inv_var + logvar
                train_losses = train_losses.mean(-1).mean(-1).sum()
                # Only taking mean over the last 2 dimensions
                # The first dimension corresponds to each model in the ensemble

                loss += train_losses

                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

            idxs = shuffle_rows(idxs)

            val_in = torch.from_numpy(train_in[idxs[:5000]]).to(TORCH_DEVICE).float()
            val_targ = torch.from_numpy(train_targs[idxs[:5000]]).to(TORCH_DEVICE).float()

            mean, _ = self.forward(val_in)
            mse_losses = ((mean - val_targ) ** 2).mean(-1).mean(-1)

            epoch_range.set_postfix({
                "Training loss(es)": mse_losses.detach().cpu().numpy()

            })
        return mse_losses.detach().numpy(), mse_losses.detach().numpy(), mse_losses.detach().numpy()

    def _expand_to_ts_format(self, mat):
        dim = np.array(mat).shape[-1]

        # Before, [10, 5] in case of proc_obs
        reshaped = mat.view(-1, self.num_nets, 20 // self.num_nets, dim)
        # After, [2, 5, 1, 5]

        transposed = reshaped.transpose(0, 1)
        # After, [5, 2, 1, 5]

        reshaped = transposed.contiguous().view(self.model.num_nets, -1, dim)
        # After. [5, 2, 5]

        return reshaped

    def predict_next_obs(self, obs, acs, many_in_parallel):
        proc_obs = obs

        proc_obs = self._expand_to_ts_format(proc_obs)
        acs = self._expand_to_ts_format(acs)

        inputs = torch.cat((proc_obs, acs), dim=-1)

        mean, var = self.model(inputs)

        predictions = mean + torch.randn_like(mean, device=TORCH_DEVICE) * var.sqrt()

        # TS Optimization: Remove additional dimension
        predictions = self._flatten_to_matrix(predictions)

        return state_list

def nn_constructor(model_init_cfg):

    ensemble_size = get_required_argument(model_init_cfg, "num_nets", "Must provide ensemble size")

    load_model = model_init_cfg.get("load_model", False)

    assert load_model is False, 'Has yet to support loading model'

    model = PtModel(1,
                    model_init_cfg.model_in, model_init_cfg.model_out * 2).to(TORCH_DEVICE)
    # * 2 because we output both the mean and the variance

    model.optim = torch.optim.Adam(model.parameters(), lr=0.001)

    return model

