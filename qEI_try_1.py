# https://botorch.org/tutorials/closed_loop_botorch_only
import numpy as np
import os
import torch

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # GPU加速
dtype = torch.double
SMOKE_TEST = os.environ.get("SMOKE_TEST")  # 烟雾测试，检查系统是否能执行最基本的功能。该词来源于电子设备测试，电子设备在测试的时候如果冒烟，说明它无法运行一些的功能。

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
np.set_printoptions(precision=12,linewidth=500)
from botorch.test_functions import Branin

branin =Branin(negate=True)  # https://botorch.org/api/test_functions.html?highlight=hartmann


def weighted_obj(X):
    """Feasibility weighted objective; zero if not feasible."""
    return branin(X)

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


from botorch.models import FixedNoiseGP, ModelListGP
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood

NOISE_SE = 0  # 噪声 σ^2

# A torch.Tensor is a multi-dimensional matrix containing elements of a single data type.
train_yvar = torch.tensor(NOISE_SE ** 2, device=device, dtype=dtype)


# 生成初始点
def generate_initial_data(n):
    # generate training data
    train_x = torch.rand(n, 2, device=device, dtype=dtype)
    exact_obj = branin(train_x).unsqueeze(-1)  # add output dimension

    train_obj = exact_obj

    best_observed_value = weighted_obj(train_x).max().item()
    return train_x, train_obj,  best_observed_value


def initialize_model(train_x, train_obj,state_dict=None):
    # define models for objective and constraint
    model_obj = FixedNoiseGP(train_x, train_obj, train_yvar.expand_as(train_obj)).to(
        train_x)  # A single-task exact GP model using fixed noise levels.
    # combine into a multi-output GP model
    # A multi-output GP model with independent GPs for the outputs.
    # This model supports different-shaped training inputs for each of its sub-models.  It can be used with any BoTorch models.
    model = ModelListGP(model_obj)

    mll = SumMarginalLogLikelihood(model.likelihood, model)
    # load state dict if it is passed
    if state_dict is not None:
        model.load_state_dict(state_dict)
    return mll, model



from botorch.optim import optimize_acqf

bounds = torch.tensor([[-5.0 ,0.0] , [10.0,15.0]], device=device, dtype=dtype)


BATCH_SIZE = 4 if not SMOKE_TEST else 2
NUM_RESTARTS = 20 if not SMOKE_TEST else 2
RAW_SAMPLES = 512 if not SMOKE_TEST else 32


def optimize_acqf_and_get_observation(acq_func):
    """Optimizes the acquisition function, and returns a new candidate and a noisy observation."""
    # optimize
    candidates, _ = optimize_acqf(
        acq_function=acq_func,
        bounds=bounds,
        q=BATCH_SIZE,
        num_restarts=NUM_RESTARTS,
        raw_samples=RAW_SAMPLES,
        # used for intialization heuristic
        # The number of samples for initialization. This is required if batch_initial_conditions is not specified.
        options={"batch_limit": 4, "maxiter": 200},
    )
    # observe new values
    new_x = candidates.detach()
    exact_obj = branin(new_x).unsqueeze(-1)  # add output dimension
    new_obj = exact_obj

    return new_x, new_obj




# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


from botorch import fit_gpytorch_model
from botorch.acquisition.monte_carlo import qExpectedImprovement, qNoisyExpectedImprovement
from botorch.sampling.samplers import SobolQMCNormalSampler
from botorch.exceptions import BadInitialCandidatesWarning

import time
import warnings

warnings.filterwarnings('ignore', category=BadInitialCandidatesWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

N_TRIALS = 45 if not SMOKE_TEST else 2
N_BATCH = 4 if not SMOKE_TEST else 2
MC_SAMPLES = 256 if not SMOKE_TEST else 32




for i in range(10):
    best_observed_all_ei = []
    best_observed_ei = []

    # call helper functions to generate initial training data and initialize model
    train_x_ei, train_obj_ei, best_observed_value_ei = generate_initial_data(n=20)
    mll_ei, model_ei = initialize_model(train_x_ei, train_obj_ei)

    best_observed_ei.append(best_observed_value_ei)

    min_y = []
    # average over multiple trials
    for trial in range(1, N_TRIALS + 1):

        print(f"\nTrial {trial:>2} of {N_TRIALS} ", end="")
        min_bitch_y = []
        # run N_BATCH rounds of BayesOpt after the initial random batch
        for iteration in range(1, N_BATCH + 1):

            t0 = time.time()

            # fit the models
            fit_gpytorch_model(mll_ei)


            # define the qEI and qNEI acquisition modules using a QMC sampler
            qmc_sampler = SobolQMCNormalSampler(num_samples=MC_SAMPLES)

            # for best_f, we use the best observed noisy values as an approximation
            qEI = qExpectedImprovement(
                model=model_ei,
                best_f=(train_obj_ei ).max(),
                sampler=qmc_sampler,

            )

            # optimize and get new observation
            new_x_ei, new_obj_ei = optimize_acqf_and_get_observation(qEI)


            # update training points
            train_x_ei = torch.cat([train_x_ei, new_x_ei])
            train_obj_ei = torch.cat([train_obj_ei, new_obj_ei])



            # update progress

            best_value_ei = weighted_obj(train_x_ei).max().item()

            best_observed_ei.append(best_value_ei)


            # reinitialize the models so they are ready for fitting on next iteration
            # use the current state dict to speed up fitting
            mll_ei, model_ei = initialize_model(
                train_x_ei,
                train_obj_ei,
                model_ei.state_dict(),
            )
            min_bitch_y.append(-best_value_ei)


            t1 = time.time()
            verbose =True
            if verbose:
                print(
                    f"\nBatch {iteration:>2}: best_value qEI = "
                    f"{best_value_ei:>4.12f}, "
                    f"time = {t1 - t0:>4.12f}.", end=""
                )
            else:
                print(".", end="")
        min_y.append(np.array(min_bitch_y).min())
        best_observed_all_ei.append(best_observed_ei)

    np.savetxt('min_y_qEI_'+str(i)+'.txt', min_y, delimiter=',', fmt='%s')
    # ---------------------------------------------------------------------------
    # ---------------------------------------------------------------------------
    #
    # import numpy as np
    # from matplotlib import pyplot as plt
    #
    #
    # # matplotlib inline
    #
    # def ci(y):
    #     return 1.96 * y.std(axis=0) / np.sqrt(N_TRIALS)
    #
    #
    # GLOBAL_MAXIMUM = branin.optimal_value
    #
    # iters = np.arange(N_BATCH + 1) * BATCH_SIZE
    # y_ei = np.asarray(best_observed_all_ei)
    #
    #
    # fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    #
    # ax.errorbar(iters, y_ei.mean(axis=0), yerr=ci(y_ei), label="qEI", linewidth=1.5)
    #
    # plt.plot([0, N_BATCH * BATCH_SIZE], [GLOBAL_MAXIMUM] * 2, 'k', label="true best objective", linewidth=2)
    # ax.set_ylim(bottom=0.5)
    # ax.set(xlabel='number of observations (beyond initial points)', ylabel='best objective value')
    # ax.legend(loc="lower right")
    # plt.show()




