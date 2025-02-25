# Libraries
import time
import torch
import math
from tqdm import trange


def list_of_dict_2_dict_of_list(l):
    return {
        k: [x[k] for x in l if k in x] for k in l[0].keys()
    }


def evaluate_eubo(trainable, results, compute_eubo_last_arg, use_ema):
    eval_samples_from_target = trainable.target.sample((trainable.eval_batch_size,))
    with torch.no_grad():
        rnd_target = trainable.loss.compute_eubo(
            trainable.eval_ts.to(trainable.device),
            eval_samples_from_target,
            trainable.clipped_target_unnorm_log_prob,
            compute_eubo_last_arg,
            use_ema=use_ema
        )
    neg_rnd_target = -rnd_target
    # Compute importance weights
    weights = torch.nn.functional.softmax(neg_rnd_target, dim=0)
    # Compute EUBO-based metrics
    results.metrics['eval/log_norm_const_is_f'] = -rnd_target.logsumexp(dim=0).item() + math.log(len(weights))
    results.metrics['eval/eubo'] = neg_rnd_target.mean().item()
    results.metrics['eval/effective_sample_size_f'] = (1. / (weights ** 2).sum()).item()
    results.metrics['eval/norm_effective_sample_size_f'] = results.metrics["eval/effective_sample_size_f"] / \
        len(weights)
    return results


class TrainableWrapper(torch.nn.Module):

    def __init__(self, trainable, verbose=True):
        super().__init__()
        self.trainable = trainable
        self.verbose = verbose

    def run(self, keep_training_metrics=False):
        self.trainable.train()
        if keep_training_metrics:
            training_metrics = []
        if self.verbose:
            r = trange(self.trainable.n_steps, self.trainable.train_steps)
        else:
            r = range(self.trainable.n_steps, self.trainable.train_steps)
        training_time = 0.0
        for i in r:
            step_start_time = time.time()
            metrics = self.trainable.step(i)
            step_end_time = time.time()
            training_time += step_end_time - step_start_time
            if keep_training_metrics:
                training_metrics.append(metrics)
            if self.verbose:
                r.set_description("loss={:.2e}".format(metrics['train/loss']))
        results = self.trainable.evaluate(use_ema=self.trainable.use_ema)
        results = self.compute_results_eubo(results, use_ema=self.trainable.use_ema)
        results.metrics['eval/training_time'] = training_time
        if keep_training_metrics:
            return results, list_of_dict_2_dict_of_list(training_metrics)
        else:
            return results

    def compute_results_eubo(self, results, use_ema=True):
        if hasattr(self.trainable.loss, 'compute_eubo') and self.trainable.eubo_available and hasattr(self.trainable.target, 'sample'):
            if hasattr(self.trainable, 'reference_distr'):
                results = evaluate_eubo(self.trainable, results,
                                        self.trainable.reference_distr.log_prob,
                                        use_ema=use_ema)
            else:
                results = evaluate_eubo(self.trainable, results,
                                        self.trainable.prior.log_prob,
                                        use_ema=use_ema)
        return results

    @torch.no_grad()
    def evaluate(self, use_ema=True, log=True):
        use_ema_ = self.trainable.use_ema and use_ema
        training = self.trainable.training
        self.trainable.eval()
        self.trainable.to(self.trainable.eval_device)
        results = self.trainable.compute_results(use_ema=use_ema_)
        results = self.compute_results_eubo(results, use_ema=use_ema_)
        self.trainable.train(training)
        self.trainable.to(self.trainable.device)
        return results


class TrainableWrapperWithIntermediates(TrainableWrapper):
    def run(self, results_freq=16, n_seeds=1, bonus_metrics=None):
        self.trainable.train()
        intermediates_eval_results = []
        intermediates_training_results = []
        if self.verbose:
            r = trange(self.trainable.n_steps, self.trainable.train_steps)
        else:
            r = range(self.trainable.n_steps, self.trainable.train_steps)
        training_time = 0.0
        for i in r:
            step_start_time = time.time()
            metrics = self.trainable.step(i)
            step_end_time = time.time()
            training_time += step_end_time - step_start_time
            intermediates_training_results.append(metrics)
            if (i + 1) % results_freq == 0:
                all_results = []
                for _ in range(n_seeds):
                    results = self.trainable.evaluate(use_ema=self.trainable.use_ema)
                    results = self.compute_results_eubo(results, use_ema=self.trainable.use_ema)
                    if (bonus_metrics is not None) and (isinstance(bonus_metrics, list)):
                        for metric_name, metric in bonus_metrics:
                            results.metrics['eval/' + metric_name] = metric(results.samples)
                    all_results.append(dict(**results.metrics, ))
                intermediates_eval_results.append(list_of_dict_2_dict_of_list(all_results))
            if self.verbose:
                r.set_description("loss={:.2e}".format(metrics['train/loss']))
        results = self.trainable.evaluate(use_ema=self.trainable.use_ema)
        results = self.compute_results_eubo(results, use_ema=self.trainable.use_ema)
        results.metrics['eval/training_time'] = training_time
        return results, list_of_dict_2_dict_of_list(intermediates_training_results), list_of_dict_2_dict_of_list(intermediates_eval_results)
