# %% 


import os

import sys
sys.path.append('../')
import numpy as np
import torch
from datasets import load_dataset
from utils import get_entropy, load_model_from_tl_name, filter_entropy_activation_df, get_entropy_activation_df
import neel.utils as nutils
import transformer_lens.utils as utils
import tqdm
import pathlib
import pandas as pd
import hydra
from omegaconf import DictConfig, OmegaConf


def neuron_ablation_hook(value, hook, neuron_idx, ablation_value):
    value[0, :, neuron_idx] = ablation_value
    return value

def head_bos_ablation_hook(value, hook, head_idx):
    # value has shape batch x n_heads x seq_len x seq_len
    new_pattern = torch.zeros_like(value[:, head_idx, :])
    new_pattern[:, 0] = 1.0
    value[:, head_idx, :, :] = new_pattern
    return value

def head_random_ablation_hook(value, hook, head_idx):
    # value has shape batch x n_heads x seq_len x seq_len
    # generate random pattern
    new_pattern = torch.rand_like(value[:, head_idx], device=value.args.device)
    # make sure that upper triangular part is - np.inf
    mask = torch.ones_like(new_pattern)
    mask = mask.to(value.args.device)
    mask *= -1e9
    mask = torch.triu(mask, diagonal=1)
    mask += 1
    new_pattern = new_pattern * mask
    # apply softmax to make sure that the pattern is a valid probability distribution
    new_pattern = new_pattern.softmax(dim=-1)
    value[:, head_idx, :, :] = new_pattern
    return value

def head_mean_output_hook(value, hook, head_idx, mean_head_output):
    # value has shape batch x seq_len x n_heads x d_model
    value[:, :, head_idx] += mean_head_output
    return value

def ln_final_scale_hook(value, hook, scale_values):
    #scale hook (batch, seq, 1)
    #only applies it to current in theory we could be efficient and do it on the whole sequence + pass in batches
    value[0, :, 0] = scale_values
    return value

def mean_ablate_components(components_to_ablate=None,
                      tokenized_data=None,
                      entropy_df=None,
                      model=None,
                      k=10,
                      device='mps',
                      get_rank_of_correct_token=False,
                      chunk_size=20):
    
    # sample a set of random batch indices
    random_sequence_indices = np.random.choice(entropy_df.batch.unique(), k, replace=False)

    print(f'ablate_components: ablate {components_to_ablate} with k = {k}')
    
    pbar = tqdm.tqdm(total=k, file=sys.stdout)

    # new_entropy_df with only the random sequences
    filtered_entropy_df = entropy_df[entropy_df.batch.isin(random_sequence_indices)].copy()

    results = {}
    final_df = None

    activation_mean_values = torch.tensor(entropy_df[[f'{component_name}_activation' for component_name in components_to_ablate]].mean())
    
    # get neuron indices
    neuron_indices = [int(neuron_name.split('.')[1]) for neuron_name in components_to_ablate]

    # get layer indices
    layer_indices = [int(neuron_name.split('.')[0]) for neuron_name in components_to_ablate]
    layer_idx = layer_indices[0]

    for batch_n in filtered_entropy_df.batch.unique():
        tok_seq = tokenized_data['tokens'][batch_n]

        # get unaltered logits
        inp = tok_seq.unsqueeze(0).to(device)
        logits, cache = model.run_with_cache(inp)
        
        # get the entropy_df entries for the current sequence
        rows = filtered_entropy_df[filtered_entropy_df.batch == batch_n]
        assert len(rows) == len(tok_seq), f'len(rows) = {len(rows)}, len(tok_seq) = {len(tok_seq)}'

        ln_scales  = torch.tensor(rows[f'ln_final_scale'].values).to(device)

        res_stream = cache[utils.get_act_name("resid_post", layer_idx)][0]

        previous_activation = cache[utils.get_act_name("post", layer_idx)][0, :, neuron_indices]
        del cache
        activation_deltas = activation_mean_values.to(previous_activation.device) - previous_activation
        # activation deltas is seq_n x n_neurons

        if get_rank_of_correct_token:
                logits_ranking = logits.argsort(dim=-1, descending=True)

                rank_of_correct_token = (logits_ranking[:, :-1] == inp[:, 1:].unsqueeze(-1)).nonzero(as_tuple=True)[2]
                rank_of_correct_token = rank_of_correct_token.view(inp[:, :-1].shape).cpu().numpy()
                rank_of_correct_token_before = np.concatenate((rank_of_correct_token, np.zeros((rank_of_correct_token.shape[0], 1))), axis=1)

        loss_post_ablation = []
        entropy_post_ablation = []
        top_logits_post_ablation = []
        ranks_of_correct_token_post_ablation = []

        for i in range(0, len(neuron_indices), chunk_size):
            neuron_indices_chunk = neuron_indices[i:i+chunk_size]

            # multiple deltas by W_out
            res_deltas_chunk = activation_deltas[:, neuron_indices_chunk].unsqueeze(-1) * model.W_out[layer_idx, neuron_indices_chunk, :]
            res_deltas_chunk = res_deltas_chunk.permute(1, 0, 2)

            updated_res_stream_chunk = res_stream.repeat(res_deltas_chunk.shape[0], 1, 1) + res_deltas_chunk
            
            # apply ln_final
            updated_res_stream_chunk = model.ln_final(updated_res_stream_chunk)

            ablated_logits_chunk = updated_res_stream_chunk @ model.W_U + model.b_U
            
            del updated_res_stream_chunk
            # compute loss for the chunk
            loss_post_ablation_chunk = model.loss_fn(ablated_logits_chunk, inp.repeat(res_deltas_chunk.shape[0], 1), per_token=True).cpu()
            loss_post_ablation_chunk = np.concatenate((loss_post_ablation_chunk, np.zeros((loss_post_ablation_chunk.shape[0], 1))), axis=1)
            loss_post_ablation.append(loss_post_ablation_chunk)

            # compute entropy for the chunk
            entropy_post_ablation_chunk = get_entropy(ablated_logits_chunk)
            entropy_post_ablation.append(entropy_post_ablation_chunk.cpu())

            # get top logit for each token
            top_logits_post_ablation_chunk = ablated_logits_chunk.argmax(dim=-1)
            top_logits_post_ablation.append(top_logits_post_ablation_chunk.cpu())

            # get rank of correct token
            if get_rank_of_correct_token:
                logits_ranking = ablated_logits_chunk.argsort(dim=-1, descending=True)

                rank_of_correct_token = (logits_ranking[:, :-1] == inp[:, 1:].unsqueeze(-1)).nonzero(as_tuple=True)[2]
                rank_of_correct_token = rank_of_correct_token.view(logits_ranking[:, :-1].shape[0:2]).cpu().numpy()
                rank_of_correct_token = np.concatenate((rank_of_correct_token, np.zeros((rank_of_correct_token.shape[0], 1))), axis=1)
                ranks_of_correct_token_post_ablation.append(rank_of_correct_token)

            del ablated_logits_chunk

        loss_post_ablation = np.concatenate(loss_post_ablation, axis=0)
        entropy_post_ablation = np.concatenate(entropy_post_ablation, axis=0)
        top_logits_post_ablation = np.concatenate(top_logits_post_ablation, axis=0)

        if get_rank_of_correct_token:
            ranks_of_correct_token_post_ablation = np.concatenate(ranks_of_correct_token_post_ablation, axis=0)
        
        # forward with frozen ln    
        loss_post_ablation_with_frozen_ln = []
        entropy_post_ablation_with_frozen_ln = []
        top_logits_post_ablation_with_frozen_ln = []
        ranks_of_correct_token_post_ablation_with_frozen_ln = []

        for i in range(0, len(neuron_indices), chunk_size):
            neuron_indices_chunk = neuron_indices[i:i+chunk_size]

            # multiple deltas by W_out
            res_deltas_chunk = activation_deltas[:, neuron_indices_chunk].unsqueeze(-1) * model.W_out[layer_idx, neuron_indices_chunk, :]
            res_deltas_chunk = res_deltas_chunk.permute(1, 0, 2)

            updated_res_stream_chunk = res_stream.repeat(res_deltas_chunk.shape[0], 1, 1) + res_deltas_chunk
            
            # apply ln_final
            if model.cfg.normalization_type == 'RMSPre' or model.cfg.normalization_type == 'RMSPre' or model.cfg.final_rms:
                updated_res_stream_chunk = updated_res_stream_chunk / ln_scales.unsqueeze(-1).to(updated_res_stream_chunk.device)
            elif model.cfg.normalization_type == 'LNPre' or model.cfg.normalization_type == 'LN':
                updated_res_stream_chunk = updated_res_stream_chunk - updated_res_stream_chunk.mean(dim=-1, keepdim=True)
                updated_res_stream_chunk = updated_res_stream_chunk / ln_scales.unsqueeze(-1).to(updated_res_stream_chunk.device)
            else:
                raise ValueError(f'Normalization type {model.cfg.normalization_type} not supported')

            ablated_logits_chunk = updated_res_stream_chunk @ model.W_U + model.b_U

            del updated_res_stream_chunk

            # compute loss for the chunk
            loss_post_ablation_chunk = model.loss_fn(ablated_logits_chunk, inp.repeat(res_deltas_chunk.shape[0], 1), per_token=True).cpu()
            loss_post_ablation_chunk = np.concatenate((loss_post_ablation_chunk, np.zeros((loss_post_ablation_chunk.shape[0], 1))), axis=1)
            loss_post_ablation_with_frozen_ln.append(loss_post_ablation_chunk)

            # compute entropy for the chunk
            entropy_post_ablation_chunk = get_entropy(ablated_logits_chunk)
            entropy_post_ablation_with_frozen_ln.append(entropy_post_ablation_chunk.cpu())

            # get top logit for each token
            top_logits_post_ablation_chunk = ablated_logits_chunk.argmax(dim=-1)
            top_logits_post_ablation_with_frozen_ln.append(top_logits_post_ablation_chunk.cpu())

            # get rank of correct token
            if get_rank_of_correct_token:
                logits_ranking = ablated_logits_chunk.argsort(dim=-1, descending=True)

                rank_of_correct_token = (logits_ranking[:, :-1] == inp[:, 1:].unsqueeze(-1)).nonzero(as_tuple=True)[2]
                rank_of_correct_token = rank_of_correct_token.view(logits_ranking[:, :-1].shape[0:2]).cpu().numpy()
                rank_of_correct_token = np.concatenate((rank_of_correct_token, np.zeros((rank_of_correct_token.shape[0], 1))), axis=1)
                ranks_of_correct_token_post_ablation_with_frozen_ln.append(rank_of_correct_token)

            del ablated_logits_chunk

        torch.cuda.empty_cache()  # Empty the cache

        loss_post_ablation_with_frozen_ln = np.concatenate(loss_post_ablation_with_frozen_ln, axis=0)
        entropy_post_ablation_with_frozen_ln = np.concatenate(entropy_post_ablation_with_frozen_ln, axis=0)
        top_logits_post_ablation_with_frozen_ln = np.concatenate(top_logits_post_ablation_with_frozen_ln, axis=0)

        if get_rank_of_correct_token:
            ranks_of_correct_token_post_ablation_with_frozen_ln = np.concatenate(ranks_of_correct_token_post_ablation_with_frozen_ln, axis=0)

        for i, component_name in enumerate(components_to_ablate):
            df_to_append = filtered_entropy_df[filtered_entropy_df.batch == batch_n].copy()

            # drop all the columns that are not the component_name
            df_to_append = df_to_append.drop(columns=[f'{neuron}_activation' for neuron in components_to_ablate if neuron != component_name])

            # rename the component_name column to 'activation'
            df_to_append = df_to_append.rename(columns={f'{component_name}_activation': 'activation'})

            df_to_append['component_name'] = component_name
            df_to_append[f'loss_post_ablation'] = loss_post_ablation[i]
            df_to_append[f'loss_post_ablation_with_frozen_ln'] = loss_post_ablation_with_frozen_ln[i]
            df_to_append[f'entropy_post_ablation'] = entropy_post_ablation[i]
            df_to_append[f'entropy_post_ablation_with_frozen_ln'] = entropy_post_ablation_with_frozen_ln[i]
            df_to_append[f'top_logits_post_ablation'] = top_logits_post_ablation[i]
            df_to_append[f'top_logits_post_ablation_with_frozen_ln'] = top_logits_post_ablation_with_frozen_ln[i]
            if get_rank_of_correct_token:
                df_to_append[f'rank_of_correct_token_post_ablation'] = ranks_of_correct_token_post_ablation[i]
                df_to_append[f'rank_of_correct_token_post_ablation_with_frozen_ln'] = ranks_of_correct_token_post_ablation_with_frozen_ln[i]
                df_to_append[f'rank_of_correct_token_before_ablation'] = rank_of_correct_token_before[0]

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
    torch.set_grad_enabled(False)

    os.chdir(args.chdir)
    save_path = f'./{args.output_dir}/{args.model}/ln_scale/{args.dataset.replace("/","_")}_{args.data_range_start}-{args.data_range_end}'

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
        all_neurons = all_neurons[:20]

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
    results = mean_ablate_components(components_to_ablate=all_neurons,
                                    tokenized_data=tokenized_data,
                                    entropy_df=entropy_df,
                                    model=model,
                                    k=args.k,
                                    device=args.device,
                                    get_rank_of_correct_token=args.get_rank_of_correct_token)
    
    # concatenate the results
    final_df = pd.concat(results.values())
    final_df = filter_entropy_activation_df(final_df.reset_index(), model_name=args.model, tokenizer=tokenizer, start_pos=3, end_pos=-1)

    # store the final_df as a feather file
    final_df = final_df.reset_index(drop=True)
    final_df.to_feather(f'{save_path}/k{args.k}.feather')


# %%
if __name__ == '__main__':
    print(f'current dir: {os.getcwd()}')
    run_and_store_ablation_results()
# %%
