# Learned Reference-based Diffusion Sampling for multi-modal distributions: `sde_sampler_lrds`

![LRDS](assets/lrds.gif)

This is the official code for **"Learned Reference-based Diffusion Sampler for Multi-Modal Distributions"** ([ICLR'25](https://openreview.net/forum?id=fmJUYgmMbL)). This repository is a fork of [github.com/juliusberner/sde_sampler](https://github.com/juliusberner/sde_sampler) and extends its capabilities.  

### Features:  
- Implements previous diffusion samplers [**PIS**](https://openreview.net/forum?id=_uCb2ynRu7Y) ([`solver.oc.PIS`](./sde_sampler/solver/oc.py)), [**DDS**](https://openreview.net/forum?id=8pvnfTAbu1f) ([`solver.oc.DDS`](./sde_sampler/solver/oc.py)), [**DIS**](https://openreview.net/forum?id=h4pNROsO06) ([`solver.oc.Bridge`](./sde_sampler/solver/oc.py)), along with the generic **RDS** ([`solver.oc.RDS`](./sde_sampler/solver/oc.py)) class proposed in our paper, and custom implementations of [**CMCD**](https://openreview.net/forum?id=PP1rudnxiW) ([`solver.oc.CMCD`](./sde_sampler/solver/oc.py)) and an alternative **DIS** ([`loss.oc.DiscreteTimeReversalLossEI`](./sde_sampler/loss/oc.py)). We also implement the different metrics from [**Beyond ELBOs**](https://proceedings.mlr.press/v235/blessing24a.html) (see `compute_eubo` in [`loss.oc`](./sde_sampler/loss/oc.py)).
- Provides all **multi-modal target distributions** and related **metrics** from the paper
	* Mixture of Two Gaussians ([`distr.gauss.TwoModes`](./sde_sampler/distr/gauss.py)) and ([`distr.gauss.TwoModesFull`](./sde_sampler/distr/gauss.py))
	* Mixture of multiple Gaussians ([`distr.gauss.ManyModes`](./sde_sampler/distr/gauss.py))
	* Checkerboard ([`distr.checkerboard.Checkernoard`](./sde_sampler/distr/checkerboard.py))
	* Rings ([`distr.rings.Rings`](./sde_sampler/distr/rings.py))
	* Phi Four ([`distr.phi_four.PhiFour`](./sde_sampler/distr/phi_four.py)) based on the implementation of [github.com/marylou-gabrie/flonaco](https://github.com/marylou-gabrie/flonaco)
	* Bayesian Logisitic Regression ([`distr.logistic_regression.LogisticRegression`](./sde_sampler/distr/logistic_regression.py)) with datasets stored in [`data/`](./data/)
	* Mixture of NICE normalizing flows ([`distr.nice.MixtureNice`](./sde_sampler/distr/nice.py)) with flows trained on MNIST in [`data/`](./data/)
- Includes **SMC** ([`additions.ebm_mle.smc_sampler`](./sde_sampler/additions/ebm_mle.py)), **RE** ([`additions.ebm_mle.re_sampler`](./sde_sampler/additions/ebm_mle.py)), [**PDDS**](https://proceedings.mlr.press/v235/phillips24a.html) ([`additions.ebm_mle.smc_sampler`](./sde_sampler/additions/ebm_mle.py) with `use_pdds_weights=True`), and standard **MCMC samplers** ([`additions.mcmc.mala_step/ula_step/rwmh_step`](./sde_sampler/additions/mcmc.py)).  
- Implements **Energy-Based Model (EBM) training** methods: [**DA-EBM**](https://arxiv.org/abs/2304.10707) ([`additions.da_ebm.DAEBM`](./sde_sampler/additions/da_ebm.py)), [**DRL**](https://openreview.net/forum?id=v_1Soh8QUNc) ([`additions.drl.DiffusionRecoveryLikelihood`](./sde_sampler/additions/drl.py)) as well as our own RE-based algorithm ([`additions.ebm_mle.MaximumLikelihoodEBM`](./sde_sampler/additions/ebm_mle.py)), and score matching techniques [**DSM**](https://openreview.net/forum?id=PxTIG12RRHS) ([`additions.sm.ScoreMatching`](./sde_sampler/additions/sm.py)), [**TSM**](https://arxiv.org/abs/2402.08667) ([`additions.sm.TargetScoreMatching`](./sde_sampler/additions/sm.py)).  

This repository serves as a comprehensive toolkit for diffusion-based sampling in complex multi-modal settings.

## Installation

### 1. Clone the Repository  
```bash
git clone git@github.com:h2o64/sde_sampler_lrds.git
cd sde_sampler
```  

### 2. Set Up the Environment  
We recommend using [Conda](https://conda.io/docs/user-guide/install/download.html):  
```bash
conda create -n sde_sampler python=3.9 pip --yes  
conda activate sde_sampler
```  

### 3. Install Dependencies  

#### **For GPU Users**  
Check your CUDA version with `nvidia-smi` and install the appropriate `cuda` (for `pykeops`) and `torch`/`torchvision` packages using the [PyTorch install guide](https://pytorch.org/get-started).  
For example, if your CUDA version is **>=11.7**, install:  
```bash
conda install pytorch=2.0.1 torchvision=0.15.2 pytorch-cuda=11.7 cuda-minimal-build=11.7 -c pytorch -c nvidia --yes  
```  

#### **For CPU Users**  
If you don’t have a GPU, install the CPU-only version:  
```bash
conda install pytorch torchvision cpuonly -c pytorch --yes  
```  

### 4. Install the `sde_sampler_lrds` Package  
```bash
pip install -e .
```

## Reproducing the experiments

All the scripts to rerun the experiments are in [`experiments/`](./experiments/) and are completely self contained (they are entirely determined by the arguments given to the script). The key auxiliary function is `experiments.benchmark_utils.make_model` which abstracts the Hydra config. It takes the following arguments
- `solver_type` : Type of solver (among `dds_orig`, `pis_orig`, `dis_orig`, `cmcd`, `vp-ref` or `pbm-ref`)
- `ref_type` : Type of reference for RDS (among `default`, `gaussian`, `gmm` or `nn`)
- `loss_type` : Variant of the divergence loss (among `kl` or `lv`) 
- `integrator_type`: Variant of the SDE integrator for RDS (among `em`, `ei` or `ddpm_like`)
- `model_type`: Type of neural network parametrization (among `target_informed_zero_init`, `target_informed_unet_zero_init`, `target_informed_langevin_init`, `target_informed_lerp_tempering`, `base_zero_init`or `unet_zero_init`)
- `time_type`: Type of time discretization (among `uniform` or `snr`)
- `solver_details`: Dictionnary containing the reference details
- `target_details`: Parameters of the target (generated by `experiments.benchmark_utils.make_target_details`)
- `training_details`: Dictionnary with attributes `train_steps`, `train_batch_size` and `eval_batch_size`

A demo notebook is available at [`notebooks/demo_gmm_lrds.ipynb`](./notebooks/demo_gmm_lrds.ipynb).

## References

If you found the codebase usefull, please consider citing us!
```
@inproceedings{noble2025learned,
	title={Learned Reference-based Diffusion Sampler for multi-modal distributions},
	author={Maxence Noble and Louis Grenioux and Marylou Gabri{\'e} and Alain Oliviero Durmus},
	booktitle={The Thirteenth International Conference on Learning Representations},
	year={2025},
	url={https://openreview.net/forum?id=fmJUYgmMbL}
}
```