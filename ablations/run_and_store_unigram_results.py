# %% 
import os
os.environ["CUDA_ARGS_ORDER"]="PCI_BUS_ID"  
os.environ["CUDA_VISIBLE_ARGS"]="0"
import sys
sys.path.append('../')
from transformer_lens import HookedTransformer
from sklearn.linear_model import LinearRegression
import argparse
import json
from scipy.stats import ttest_rel, ttest_ind
from neel.imports import *
from neel_plotly import * 
import neel 
import tqdm
import math 
from datasets import Dataset
import pathlib
import json
import pandas as pd
import plotly.express as px
from functools import partial
from utils import *
import math
from scipy.stats import spearmanr
import rbo
import plotly.graph_objects as go
import hydra
from omegaconf import DictConfig, OmegaConf
from torch.nn.functional import kl_div


def adjust_vectors_3dim(v, u, target_values):
    """
    Adjusts a batch of vectors v such that their projections along the unit vector u equal the target values.

    Parameters:
    - v: A 3D tensor of shape (n, m, d), representing the batch of vectors to be adjusted.
    - u: A 1D unit tensor of shape (d,), representing the direction along which the adjustment is made.
    - target_values: A 2D tensor of shape (n, m), representing the desired projection values of the vectors in v along u.

    Returns:
    - adjusted_v: The adjusted batch of vectors such that their projections along u are equal to the target values.
    """
    current_projections = (v @ u.unsqueeze(-1)).squeeze(-1)  # Current projections of v onto u
    delta = target_values - current_projections  # Differences needed to reach the target projections
    adjusted_v = v + delta.unsqueeze(-1) * u  # Adjust v by the deltas along the direction of u
    return adjusted_v


def mean_ablate_components(components_to_ablate=None,
                           unigram_distrib=None,
                            tokenized_data=None,
                            entropy_df=None,
                            model=None,
                            k=10,
                            device='mps',
                            type='indirect',
                            chunk_size=20,
                            save_path=None):
    
    # sample a set of random batch indices
    random_sequence_indices = np.random.choice(entropy_df.batch.unique(), k, replace=False)

    print(f'ablate_components: ablate {components_to_ablate} with k = {k}')
    
    pbar = tqdm.tqdm(total=k, file=sys.stdout)

    # new_entropy_df with only the random sequences
    filtered_entropy_df = entropy_df[entropy_df.batch.isin(random_sequence_indices)].copy()

    results = {}
    final_df = None

    activation_mean_values = torch.tensor(entropy_df[[f'{component_name}_activation' for component_name in components_to_ablate]].mean())

    unigram_direction_vocab = unigram_distrib.log() - unigram_distrib.log().mean()
    unigram_direction_vocab /= unigram_direction_vocab.norm()
    
    # get neuron indices
    neuron_indices = [int(neuron_name.split('.')[1]) for neuron_name in components_to_ablate]

    # get layer indices
    layer_indices = [int(neuron_name.split('.')[0]) for neuron_name in components_to_ablate]
    layer_idx = layer_indices[0]

    for batch_n in filtered_entropy_df.batch.unique():
        tok_seq = tokenized_data['tokens'][batch_n]

        # get unaltered logits
        model.reset_hooks()
        inp = tok_seq.unsqueeze(0).to(device)
        logits, cache = model.run_with_cache(inp)
        logprobs = logits[0, :, :].log_softmax(dim=-1)

        res_stream = cache[utils.get_act_name("resid_post", layer_idx)][0]
        
        # get the entropy_df entries for the current sequence
        rows = filtered_entropy_df[filtered_entropy_df.batch == batch_n]
        assert len(rows) == len(tok_seq), f'len(rows) = {len(rows)}, len(tok_seq) = {len(tok_seq)}'

        # get the value of the logits projected onto the b_U direction
        unigram_projection_values = logits @ unigram_direction_vocab
        unigram_projection_values = unigram_projection_values.squeeze()

        previous_activation = cache[utils.get_act_name("post", layer_idx)][0, :, neuron_indices]
        del cache
        activation_deltas = activation_mean_values.to(previous_activation.device) - previous_activation
        # activation deltas is seq_n x n_neurons

        # multiple deltas by W_out
        res_deltas = activation_deltas.unsqueeze(-1) * model.W_out[layer_idx, neuron_indices, :]
        res_deltas = res_deltas.permute(1, 0, 2)

        # print the size of res_deltas in mb
        #print(f'res_deltas size: {res_deltas.element_size() * res_deltas.nelement() / 1024 / 1024} MB')

        loss_post_ablation = []
        entropy_post_ablation = []

        loss_post_ablation_with_frozen_unigram = []
        entropy_post_ablation_with_frozen_unigram = []

        kl_divergence_after = []
        kl_divergence_after_frozen_unigram = []

        log_unigram_distrib = unigram_distrib.log()

        kl_divergence_before = kl_div(logprobs, log_unigram_distrib, reduction='none', log_target=True).sum(axis=-1).cpu().numpy()

        for i in range(0, res_deltas.shape[0], chunk_size):
            res_deltas_chunk = res_deltas[i:i+chunk_size]
            updated_res_stream_chunk = res_stream.repeat(res_deltas_chunk.shape[0], 1, 1) + res_deltas_chunk
            # apply ln_final
            updated_res_stream_chunk = model.ln_final(updated_res_stream_chunk)

            # print the size of the updated_res_stream in mb
            #print(f'updated_res_stream size: {updated_res_stream_chunk.element_size() * updated_res_stream_chunk.nelement() / 1024 / 1024} MB')

            ablated_logits_chunk = updated_res_stream_chunk @ model.W_U + model.b_U
            del updated_res_stream_chunk
            ablated_logits_with_frozen_unigram_chunk = adjust_vectors_3dim(ablated_logits_chunk, unigram_direction_vocab, unigram_projection_values)
            # count the number of zero elements in the chunk < 0.0001
            #print(f'chunk {i} has {torch.sum(ablated_logits_chunk < 0.0001)} zero elements')

            #print(f'ablated logits last chunk : {ablated_logits_chunk[-1]}')

            # compute loss for the chunk
            loss_post_ablation_chunk = model.loss_fn(ablated_logits_chunk, inp.repeat(res_deltas_chunk.shape[0], 1), per_token=True).cpu()
            loss_post_ablation_chunk = np.concatenate((loss_post_ablation_chunk, np.zeros((loss_post_ablation_chunk.shape[0], 1))), axis=1)
            loss_post_ablation.append(loss_post_ablation_chunk)

            # compute entropy for the chunk
            entropy_post_ablation_chunk = get_entropy(ablated_logits_chunk)
            entropy_post_ablation.append(entropy_post_ablation_chunk.cpu())

            abl_logprobs = ablated_logits_chunk.log_softmax(dim=-1)

            del ablated_logits_chunk

            # compute loss for ablated_logits_with_frozen_unigram_chunk
            loss_post_ablation_with_frozen_unigram_chunk = model.loss_fn(ablated_logits_with_frozen_unigram_chunk, inp.repeat(res_deltas_chunk.shape[0], 1), per_token=True).cpu()
            loss_post_ablation_with_frozen_unigram_chunk = np.concatenate((loss_post_ablation_with_frozen_unigram_chunk, np.zeros((loss_post_ablation_with_frozen_unigram_chunk.shape[0], 1))), axis=1)
            loss_post_ablation_with_frozen_unigram.append(loss_post_ablation_with_frozen_unigram_chunk)

            # compute entropy for ablated_logits_with_frozen_unigram_chunk
            entropy_post_ablation_with_frozen_unigram_chunk = get_entropy(ablated_logits_with_frozen_unigram_chunk)
            entropy_post_ablation_with_frozen_unigram.append(entropy_post_ablation_with_frozen_unigram_chunk.cpu())

            #kl_divergence = kl_div(single_token_abl_probs, single_token_probs) # this is element-wise 
            #total_kl_divergence = kl_divergence.sum(axis=1)

            # compute KL divergence between the distribution ablated with frozen unigram and the og distribution
            abl_logprobs_with_frozen_unigram = ablated_logits_with_frozen_unigram_chunk.log_softmax(dim=-1)
            #kl_divergence = kl_div(single_token_abl_probs_with_frozen_ln, single_token_probs) # this is element-wise
            #kl_divergence_with_frozen_ln = kl_divergence.sum(axis=1)

            # compute KL divergence between the ablated distribution and the distribution from the unigram direction
            kl_divergence_after_chunk = kl_div(abl_logprobs, log_unigram_distrib.expand_as(abl_logprobs), reduction='none', log_target=True).sum(axis=-1).cpu().numpy()
            #print(f'kl_divergence_after_chunk: {kl_divergence_after_chunk}')
            del abl_logprobs
            kl_divergence_after.append(kl_divergence_after_chunk)

            kl_divergence_after_frozen_unigram_chunk = kl_div(abl_logprobs_with_frozen_unigram, log_unigram_distrib.expand_as(abl_logprobs_with_frozen_unigram), reduction='none', log_target=True).sum(axis=-1).cpu().numpy()
            del abl_logprobs_with_frozen_unigram
            kl_divergence_after_frozen_unigram.append(kl_divergence_after_frozen_unigram_chunk)

            del ablated_logits_with_frozen_unigram_chunk

        loss_post_ablation = np.concatenate(loss_post_ablation, axis=0)
        entropy_post_ablation = np.concatenate(entropy_post_ablation, axis=0)

        loss_post_ablation_with_frozen_unigram = np.concatenate(loss_post_ablation_with_frozen_unigram, axis=0)
        entropy_post_ablation_with_frozen_unigram = np.concatenate(entropy_post_ablation_with_frozen_unigram, axis=0)

        kl_divergence_after = np.concatenate(kl_divergence_after, axis=0)
        kl_divergence_after_frozen_unigram = np.concatenate(kl_divergence_after_frozen_unigram, axis=0)

        # compute loss
        # loss_post_ablation = model.loss_fn(ablated_logits, inp.repeat(len(neuron_indices), 1), per_token=True).cpu()
        # loss_post_ablation = np.concatenate((loss_post_ablation, np.zeros((loss_post_ablation.shape[0], 1))), axis=1)
        # entropy_post_ablation = get_entropy(ablated_logits)

        del res_deltas
        torch.cuda.empty_cache()  # Empty the cache

        # loss_post_ablation_with_frozen_ln = model.loss_fn(ablated_logits_with_frozen_ln, inp.repeat(len(neuron_indices), 1), per_token=True).cpu()
        # loss_post_ablation_with_frozen_ln = np.concatenate((loss_post_ablation_with_frozen_ln, np.zeros((loss_post_ablation_with_frozen_ln.shape[0], 1))), axis=1)
        # entropy_post_ablation_with_frozen_ln = get_entropy(ablated_logits_with_frozen_ln)

        # compute KL divergence between the ablated distribution and the og distribution
        #single_token_abl_probs = ablated_logits[0, :, :].softmax(dim=-1).cpu().numpy()
        #single_token_probs = logits[0, :, :].softmax(dim=-1).cpu().numpy()
        #kl_divergence = kl_div(single_token_abl_probs, single_token_probs) # this is element-wise 
        #total_kl_divergence = kl_divergence.sum(axis=1)

        # compute KL divergence between the distribution ablated with frozen ln and the og distribution
        #single_token_abl_probs_with_frozen_ln = ablated_logits_with_frozen_ln[0, :, :].softmax(dim=-1).cpu().numpy()
        #kl_divergence = kl_div(single_token_abl_probs_with_frozen_ln, single_token_probs) # this is element-wise
        #kl_divergence_with_frozen_ln = kl_divergence.sum(axis=1)

        # update df_to_append with the computed KL divergences
        #df_to_append.loc[df_to_append.batch == batch_n, 'total_kl_divergence'] = total_kl_divergence
        #df_to_append.loc[df_to_append.batch == batch_n, 'kl_divergence_with_frozen_ln'] = kl_divergence_with_frozen_ln
        #df_to_append.loc[df_to_append.batch == batch_n, 'delta_kl_divergence'] = kl_divergence_with_frozen_ln - total_kl_divergence

        # store the final_df as a feather file
        #pathlib.Path(f'{save_path}/k{k}/').mkdir(parents=True, exist_ok=True)
        #df_to_append.to_feather(f'{save_path}/k{k}/neuron{component_name}.feather')

        for i, component_name in enumerate(components_to_ablate):
            df_to_append = filtered_entropy_df[filtered_entropy_df.batch == batch_n].copy()

            # drop all the columns that are not the component_name
            df_to_append = df_to_append.drop(columns=[f'{neuron}_activation' for neuron in components_to_ablate if neuron != component_name])

            # rename the component_name column to 'activation'
            df_to_append = df_to_append.rename(columns={f'{component_name}_activation': 'activation'})

            df_to_append['component_name'] = component_name
            df_to_append[f'loss_post_ablation'] = loss_post_ablation[i]
            df_to_append[f'loss_post_ablation_with_frozen_unigram'] = loss_post_ablation_with_frozen_unigram[i]
            df_to_append[f'entropy_post_ablation'] = entropy_post_ablation[i]
            df_to_append[f'entropy_post_ablation_with_frozen_unigram'] = entropy_post_ablation_with_frozen_unigram[i]
            df_to_append[f'kl_divergence_before'] = kl_divergence_before
            df_to_append[f'kl_divergence_after'] = kl_divergence_after[i]
            df_to_append[f'kl_divergence_after_frozen_unigram'] = kl_divergence_after_frozen_unigram[i]

            if final_df is None:
                final_df = df_to_append
            else:
                final_df = pd.concat([final_df, df_to_append])

        results[batch_n] = final_df
        final_df = None

        pbar.update(1)
    
    return results


@hydra.main(config_path='./conf', config_name='config_ln_ablations')
def run_and_store_ablation_results(args: DictConfig):

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.set_grad_enabled(False)

    os.chdir(args.chdir)
    save_path = f'./{args.output_dir}/{args.model}/unigram/{args.dataset.replace("/","_")}_{args.data_range_start}-{args.data_range_end}'

    # check if save_path exists, if not create it
    pathlib.Path(save_path).mkdir(parents=True, exist_ok=True)

    with open(args.hf_token_path, 'r') as f:
        hf_token = f.read()

    model, tokenizer = load_model_from_tl_name(args.model, args.device, args.transformers_cache_dir, hf_token=hf_token)
    model = model.to(args.device)

    # Set the model in evaluation mode
    model.eval()

    #data = load_dataset("stas/openwebtext-10k", split='train')
    data = load_dataset(args.dataset, split='train')
    first_1k = data.select([i for i in range(args.data_range_start, args.data_range_end)])

    if 'qwen' in args.model.lower(): 
        tokenized_data = qwen_tokenize_and_concatenate(first_1k, tokenizer, max_length=256, column_name='text', add_bos_token=False)
        #seems to work, but need to check that there aren't weird bugs
        # repo suggests that we don't begin with a bos token: https://github.com/QwenLM/Qwen - 'Running Qwen'
    else: 
        tokenized_data = utils.tokenize_and_concatenate(first_1k, tokenizer, max_length=256, column_name='text')

    tokenized_data = tokenized_data.shuffle(args.seed)
    token_df = nutils.make_token_df(tokenized_data['tokens'], model=model)

    entropy_neuron_layer = model.cfg.n_layers - 1
    if args.neuron_range is not None:
        start = args.neuron_range.split('-')[0]
        end = args.neuron_range.split('-')[1]
        all_neuron_indices = list(range(int(start), int(end)))
    else:
        all_neuron_indices = list(range(0, model.cfg.d_mlp))
    all_neurons = [f"{entropy_neuron_layer}.{i}" for i in all_neuron_indices]

    if args.dry_run:
        all_neurons = all_neurons[:10]

    if 'pythia' in args.model:
        print('loading unigram distribution for pythia...')
        unigram_distrib = get_pile_unigram_distribution(device=args.device, file_path='../datasets/pythia-unigrams.npy')
    elif 'gpt' in args.model:
        print('loading unigram distribution for gpt2...')
        unigram_distrib = get_pile_unigram_distribution(device=args.device, file_path='../datasets/gpt2-small-unigrams_openwebtext-2M_rows_500000.npy', pad_to_match_W_U=False)
    else:
        raise Exception(f'No unigram distribution for {args.model}')

    # =============================================================================
    # Compute entropy and activation for each neuron
    # =============================================================================
    entropy_dim_layer = model.cfg.n_layers - 1
    entropy_df = get_entropy_activation_df(all_neurons,
                                                    tokenized_data,
                                                    token_df,
                                                    model,
                                                    batch_size=args.batch_size,
                                                    device=args.device,
                                                    cache_residuals=False,
                                                    cache_pre_activations=False,
                                                    compute_kl_from_bu=False,
                                                    residuals_layer=entropy_dim_layer,
                                                    residuals_dict={},)


    # =============================================================================
    # Ablate the dimensions
    # =============================================================================
    model.set_use_attn_result(False)
    ablation_type = 'direct'
    results = mean_ablate_components(components_to_ablate=all_neurons,
                                    tokenized_data=tokenized_data,
                                    entropy_df=entropy_df,
                                    model=model,
                                    k=args.k,
                                    device=args.device,
                                    type=ablation_type,
                                    unigram_distrib=unigram_distrib,
                                    save_path=save_path)
    
    # concatenate the results
    final_df = pd.concat(results.values())

    final_df = filter_entropy_activation_df(final_df.reset_index(), model_name=args.model, tokenizer=tokenizer, start_pos=3, end_pos=-1)

    # store the final_df as a feather file
    final_df.to_feather(f'{save_path}/k{args.k}.feather')


# %%
os.chdir('../large_scale_exp')
# args = OmegaConf.load('./conf/config_ln_ablations.yaml')
# args['dry_run'] = True
# args['device'] = 'mps'
# args['chdir'] = './'
# args['transformers_cache_dir'] = None
# args['batch_size'] = 4
# args['data_range_start'] = 0
# args['data_range_end'] = 10
# args['model'] = 'pythia-410m'
# args['k'] = 2
# args['chunk_size'] = 10
# run_and_store_ablation_results(args)

# %%
if __name__ == '__main__':
    print(f'current dir: {os.getcwd()}')
    run_and_store_ablation_results()
# %%