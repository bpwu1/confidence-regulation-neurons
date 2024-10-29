import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import numpy as np
import torch
import einops
from transformers import AutoModelForCausalLM, AutoTokenizer
import transformer_lens
from transformer_lens import HookedTransformer, ActivationCache
import transformer_lens.utils as tl_utils
import re
import pickle
import datasets
from datasets import load_dataset
import neel.utils as nutils
from typing import List
import tqdm
import math
from datasets import Dataset
import pandas as pd
import plotly.express as px
from functools import partial
import scipy.stats
from torch.nn.functional import kl_div
from collections import Counter


def neuron_str_to_neuron(input_str):
    return int(input_str.split(".")[0]), int(input_str.split(".")[1])

# from logits
def get_entropy(logits, use_log2=False):
    '''
    logits (batch seq d_vocab)
    
    return 
    entropy (batch seq)
    ''' 
    entropy = logits.softmax(dim=-1) * logits.log_softmax(dim=-1)

    if use_log2: 
        entropy = entropy / torch.log(torch.tensor(2.0))

    return - einops.einsum(entropy, "batch seq d_vocab -> batch seq")

def get_entropy_activation_df(neuron_names, 
                              tokenized_data, 
                              token_df, 
                              model, 
                              batch_size=32, 
                              device='mps',
                              cache_residuals=False, 
                              cache_pre_activations=False,
                              residuals_dict=None,
                              compute_kl_from_bu=False,
                              cache_resid_norm=False, # caches post norm. Requires cache_residuals=True 
                              residuals_layer=None,
                              apply_ln_to_cache=False,
                              apply_ln_location_layer=-1, #means final ln_scale
                              apply_ln_location_mlp_input=False,
                              unigram_distrib=None,
                              model2_for_kl=None,
                              bos_attn_heads_list = [] # list of attention heads to cache attn heads to BOS
                              ):


    if cache_resid_norm: 
        assert(cache_residuals), "To cache resid norms, you must also set cache_residuals=True" 

    entropy = []
    top_logit = []
    preds = []
    losses = []
    top_p = []
    token_ids = []
    neuron_activations_cache_dict = {}
    ln_final_scale = []
    kl_from_bu = []
    kl_from_unigram = []
    neuron_pre_activations_cache_dict = {}
    attention_to_bos_dict = {}
    ranks_of_correct_token = []
    model2_kl_div = []

    if residuals_dict is None: 
        residuals_dict = {
            'resid_pre': [],
            'resid_mid': [],
            'resid_post': []
        }

    for neuron in neuron_names:
        neuron_activations_cache_dict.update({neuron:[]})
        if cache_pre_activations: 
            neuron_pre_activations_cache_dict.update({neuron:[]})

    for act_name in residuals_dict.keys(): 
        if re.match(r"L(\d+)H(\d+)", act_name):
            attention_to_bos_dict.update({act_name:[]})

    for attn_head_name in bos_attn_heads_list:
        if attn_head_name not in attention_to_bos_dict.keys():
            attention_to_bos_dict.update({attn_head_name:[]})
    
    num_batches = math.ceil(len(tokenized_data) / batch_size)
    for i in tqdm.tqdm(range(num_batches)): 
        start = i * batch_size 
        end = start + batch_size
        if end > len(tokenized_data):
            end = len(tokenized_data)
        batch = tokenized_data['tokens'][start:end]
        token_ids.append(batch)

        logits, cache = model.run_with_cache(batch.to(device))

        top_logit.append(logits.max(dim=-1).values.cpu().numpy())
        preds.append(logits.argmax(dim=-1).cpu().numpy())
        top_p.append(np.log(logits.softmax(dim=-1).max(dim=-1).values.cpu().numpy()))

        
        # Get the ranking of the logits
        logits_ranking = logits.argsort(dim=-1, descending=True).cpu()

        # Get the rank of each token in the batch
        rank_of_correct_token = (logits_ranking[:, :-1] == batch[:, 1:].unsqueeze(-1)).nonzero(as_tuple=True)[2]
        rank_of_correct_token = rank_of_correct_token.view(batch[:, :-1].shape).cpu().numpy()
        rank_of_correct_token = np.concatenate((rank_of_correct_token, np.zeros((rank_of_correct_token.shape[0], 1))), axis=1)


        # Append the ranks to the list
        ranks_of_correct_token.append(rank_of_correct_token)
        if model2_for_kl is not None:
            model2_logits = model2_for_kl(batch.to(device))
            kl_divergence = torch.nn.functional.kl_div(logits.log_softmax(dim=-1), model2_logits.log_softmax(dim=-1), log_target=True, reduction="none").sum(dim=-1)
            model2_kl_div.append(kl_divergence.cpu().numpy())

        if compute_kl_from_bu:
            # compute KL divergence between the ablated distribution and the distribution from the model
            single_token_probs = logits.log_softmax(dim=-1).cpu().numpy()
            
            # compute KL divergence between the ablated distribution and the distribution from the model.b_U
            b_U_probs = model.b_U.softmax(dim=0).cpu()
            kl_divergence = kl_div(single_token_probs, b_U_probs.log(), reduction='none').sum(dim=-1)
            kl_from_bu.append(kl_divergence.numpy())

        if unigram_distrib is not None:
            # compute KL divergence between the ablated distribution and the unigram distribution
            single_token_probs = logits.log_softmax(dim=-1).cpu()

            kl_divergence = kl_div(single_token_probs, unigram_distrib.log().cpu(), reduction='none', log_target=True).sum(dim=-1)
            kl_from_unigram.append(kl_divergence.numpy())
            
        # pad loss at the end of the sequence
        losses_shorter  = model.loss_fn(logits, batch.to(device), per_token=True).cpu().numpy()
        loss = np.concatenate((losses_shorter, np.zeros((losses_shorter.shape[0], 1))), axis=1)
        losses.append(loss)

        for neuron_name in neuron_names:
            neuron_layer, neuron_index = neuron_str_to_neuron(neuron_name)
            
            neuron_activations_cache_dict[neuron_name].append(cache[tl_utils.get_act_name("post", neuron_layer)][..., neuron_index].cpu().numpy())

            if cache_pre_activations: 
                neuron_pre_activations_cache_dict[neuron_name].append(cache[tl_utils.get_act_name("pre", neuron_layer)][..., neuron_index].cpu().numpy())

        entropy.append(get_entropy(logits, use_log2=False).cpu().numpy())

        ln_final_scale.append(cache["ln_final.hook_scale"].cpu().numpy())

        if cache_residuals:
            for act_name in residuals_dict.keys():
                if re.match(r"L(\d+)H(\d+)", act_name):
                    #assumes we want head output 

                    cache.compute_head_results()
                    
                    head_results, labels = cache.stack_head_results(layer=-1, return_labels=True, apply_ln=False)
                    head_results = head_results.cpu()
                    index = labels.index(act_name)
                    activation = head_results[index]

                elif "." not in act_name:
                    #assumes dict key is something like 'resid_post"
                    activation = cache[tl_utils.get_act_name(act_name, residuals_layer)]

                else:
                    # assumes that the dict key is the raw string of form e.g. 'blocks.7.hook_attn_out'
                    activation = cache[act_name]
                
                if apply_ln_to_cache: 
                    activation = cache.apply_ln_to_stack(activation.to(device), layer=apply_ln_location_layer, mlp_input=apply_ln_location_mlp_input)

                residuals_dict[act_name].append(activation.cpu().numpy())

        # attention to bos
        for attn_head in attention_to_bos_dict.keys():
            pattern = r"L(\d+)H(\d+)"
            match = re.search(pattern, attn_head)

            layer_idx, head_idx = map(int, match.groups())
            if attn_head not in attention_to_bos_dict:
                attention_to_bos_dict[attn_head] = []
            attention_to_bos_dict[attn_head].append(cache[tl_utils.get_act_name("pattern", layer_idx)][:,head_idx,:,0].cpu().numpy())


        del logits
        del cache

    concat_entropy = np.concatenate(entropy, axis = 0)
    concat_top_logit = np.concatenate(top_logit, axis = 0)
    concat_preds = np.concatenate(preds, axis = 0)
    concat_losses = np.concatenate(losses, axis=0)
    concat_top_p = np.concatenate(top_p, axis=0)
    concat_ln_final_scale = np.concatenate(ln_final_scale)
    concat_token_ids = np.concatenate(token_ids, axis=0)
    ranks_of_correct_token = np.concatenate(ranks_of_correct_token, axis=0)

    for neuron_name in neuron_activations_cache_dict.keys(): 
        neuron_activations_cache_dict[neuron_name] = np.concatenate(neuron_activations_cache_dict[neuron_name], axis=0)

        if cache_pre_activations: 
            neuron_pre_activations_cache_dict[neuron_name] = np.concatenate(neuron_pre_activations_cache_dict[neuron_name], axis=0)

        if cache_pre_activations: 
            neuron_pre_activations_cache_dict[neuron_name] = np.concatenate(neuron_pre_activations_cache_dict[neuron_name], axis=0)

    token_df['token_id'] = concat_token_ids.flatten()
    token_df['entropy'] = concat_entropy.flatten()
    token_df['top_logit'] = concat_top_logit.flatten()
    token_df['pred'] = concat_preds.flatten()
    token_df['loss'] = concat_losses.flatten()
    token_df['top_logp'] = concat_top_p.flatten()
    token_df['ln_final_scale'] = concat_ln_final_scale.flatten()
    token_df['rank_of_correct_token'] = ranks_of_correct_token.flatten()
    token_df['correct_token_rank'] = token_df['rank_of_correct_token'] #same as above, because some scripts use one name and some use the other
    #used to compute top1 accuracy
    token_df['pred_in_top1'] = token_df['correct_token_rank'] == 0
    token_df['pred_in_top5'] = token_df['correct_token_rank'] < 5

    if model2_for_kl is not None:
        concat_model2_kl_div = np.concatenate(model2_kl_div, axis=0)
        token_df['kl_from_xl'] = concat_model2_kl_div.flatten()
    if compute_kl_from_bu:
        concat_kl_from_bu = np.concatenate(kl_from_bu, axis=0)
        token_df['kl_from_bu'] = concat_kl_from_bu.flatten()
    if unigram_distrib is not None:
        concat_kl_from_unigram = np.concatenate(kl_from_unigram, axis=0)
        token_df['kl_from_unigram'] = concat_kl_from_unigram.flatten()

    for neuron_name in neuron_activations_cache_dict.keys(): 
        token_df[f'{neuron_name}_activation'] = neuron_activations_cache_dict[neuron_name].flatten()

        if cache_pre_activations: 
            token_df[f'{neuron_name}_pre_activation'] = neuron_pre_activations_cache_dict[neuron_name].flatten()

    for attn_head in attention_to_bos_dict.keys():
        token_df[f'{attn_head}_bos_attn'] = np.concatenate(attention_to_bos_dict[attn_head], axis=0).flatten()


    if cache_residuals:
        for act_name in residuals_dict.keys():
            residuals_dict[act_name] = np.concatenate(residuals_dict[act_name]).reshape(len(token_df), model.cfg.d_model)

        if cache_resid_norm: 
            for act_name in residuals_dict.keys(): 
                token_df[f'{residuals_layer}.{act_name}_norm'] = np.linalg.norm(residuals_dict[act_name], axis=-1)

        return token_df, residuals_dict
    else:
        return token_df


def filter_entropy_activation_df(entropy_df, model_name=None, tokenizer=None, start_pos=3, end_pos=-1):
    '''
    removes tokens at start and end of sequence
    removes BOS tokens
    removes tokens where next prediction is BOS

    '''
    #filtering token_ids
    newline_token_id = None


    bos_token_id = tokenizer.bos_token_id
    eos_token_id = tokenizer.eos_token_id

    if 'pythia' in model_name.lower(): 
        assert(bos_token_id==0 and eos_token_id==0)

    if 'stanford' in model_name.lower(): 
        newline_token_id = tokenizer.encode("\n")[0]
        assert(bos_token_id==50256 and eos_token_id==50256)
    elif 'gpt' in model_name.lower(): 
        newline_token_id = tokenizer.encode("\n")[0]

    bos_mask = entropy_df['token_id'] == bos_token_id
    eos_mask = entropy_df['token_id'] == eos_token_id
    combined_mask = bos_mask | eos_mask 

    if newline_token_id is not None:
        combined_mask = combined_mask | (entropy_df['token_id'] == newline_token_id)

    preceding_token_mask = combined_mask.iloc[1:].reset_index(drop=True)
    combined_mask = combined_mask | preceding_token_mask

    filtered_entropy_df = entropy_df[~combined_mask]

    #filtering token positions
    if end_pos < 0: 
        last_seq_pos = entropy_df['pos'].max()
        end_pos_val = last_seq_pos + 1 + end_pos
    else:
        end_pos_val = end_pos

    filtered_entropy_df = filtered_entropy_df[filtered_entropy_df['pos'].between(start_pos, end_pos_val, inclusive="left")]
    return filtered_entropy_df

def filter_resid_stack(resid_stack, filtered_entropy_df):
    # either takes in numpy stack of residuals or dict of numpy stack of residuals
    # relies on index of filtered_entropy_df to filter resid_dict

    filtered_indices = filtered_entropy_df.index.tolist()
    if isinstance(resid_stack, dict): 
        for key in resid_stack.keys():
            resid_stack[key] = resid_stack[key][filtered_indices]

    else: 
        # assumes resid_dict is numpy array 
        resid_stack = resid_stack[filtered_indices]

    return resid_stack


def average_absolute_rank_change(list1, list2):
    """
    Compute the average absolute change in the rank of an item between two lists.
    The lists may contain different items.
    """
    # Create a set of unique items from both lists
    unique_items = set(list1) | set(list2)

    # Create dictionaries to store the ranks of each item in both lists
    rank_dict1 = {item: rank for rank, item in enumerate(list1, start=1)}
    rank_dict2 = {item: rank for rank, item in enumerate(list2, start=1)}

    # Calculate the rank changes
    total_change = 0
    for item in unique_items:
        rank1 = rank_dict1.get(item, len(list1) + 1)  # Default to max rank if item not in list1
        rank2 = rank_dict2.get(item, len(list2) + 1)  # Default to max rank if item not in list2
        rank_change = abs(rank1 - rank2)
        total_change += rank_change

    # Calculate average change
    average_change = total_change / len(unique_items)
    return average_change


def adjust_vector(v, u, target_value):
    """
    Adjusts a vector v such that its projection along the unit vector u equals the target value.

    Parameters:
    - v: A 1D tensor of shape (d,), representing the vector to be adjusted.
    - u: A 1D unit tensor of shape (d,), representing the direction along which the adjustment is made.
    - target_value: A scalar representing the desired projection value of v along u.

    Returns:
    - adjusted_v: The adjusted vector such that its projection along u is equal to the target value.
    """
    current_projection = v @ u  # Current projection of v onto u
    delta = target_value - current_projection  # Difference needed to reach the target projection
    adjusted_v = v + delta * u  # Adjust v by the delta along the direction of u
    return adjusted_v

def adjust_vectors(v, u, target_values):
    """
    Adjusts a batch of vectors v such that their projections along the unit vector u equal the target values.

    Parameters:
    - v: A 2D tensor of shape (n, d), representing the batch of vectors to be adjusted.
    - u: A 1D unit tensor of shape (d,), representing the direction along which the adjustment is made.
    - target_values: A 1D tensor of shape (n,), representing the desired projection values of the vectors in v along u.

    Returns:
    - adjusted_v: The adjusted batch of vectors such that their projections along u are equal to the target values.
    """
    current_projections = v @ u  # Current projections of v onto u
    delta = target_values - current_projections  # Differences needed to reach the target projections
    adjusted_v = v + delta[:, None] * u  # Adjust v by the deltas along the direction of u
    return adjusted_v


#WARNING -> this function doesn't update all columns of entropy df. Some columns in returned new_entropy_df will not reflect the true probs/metrics for the ablated resid stream
def bos_ablate_attn_heads(attn_head_names,
                      tokenized_data=None,
                      entropy_df=None,
                      model=None,
                      select='all',
                      k=10,
                      device='mps',
                      cache_pre_activations=True,
                      compute_resid_norm_change=False, # requires entropy_df to have cached pre-ablation norm. currently hard-coded to do "final_layer".resid_post_norm 
                      subtract_b_U=False,
                      seed = 42,
                      compute_kl = False
                      ):

    post_ablation_neuron_activations_cache_dict = {}
    post_ablation_neuron_pre_activations_cache_dict = {}

    post_ablation_ln_final_scale = []

    post_ablation_entropy = []
    post_ablation_resid_norm = []
    post_ablation_top_logit_delta = []
    pred_changes = []
    post_ablation_losses = []
    post_ablation_top_ps = []
    kl_before_after = []
    kl_from_bu = []
    post_ablation_resid_norm = []

    if select == 'all':
        new_entropy_df = entropy_df.copy()
    else:
        new_entropy_df = entropy_df.sample(frac=1, random_state=seed).iloc[:k].copy()

    if compute_resid_norm_change: 
        #currently hard-coded to do "final_layer".resid_post_norm 
        print("Reminder: resid_norm change is currently hardcoded for final layer resid post only")
        assert(f"{model.cfg.n_layers - 1}.resid_post_norm" in entropy_df.columns), "To compute resid norm change, entropy_df must have column final_layer.resid_post_norm in entropy_df"



    neuron_names = []
    for col_name in entropy_df.columns: 
        pattern = r'^\d+\.\d+_activation$'

        if re.search(pattern, col_name):
            neuron_names.append(col_name.split("_")[0]) # ideally use regex instead of split


    for neuron in neuron_names:
        post_ablation_neuron_activations_cache_dict.update({neuron:[]})
        if cache_pre_activations: 
            post_ablation_neuron_pre_activations_cache_dict.update({neuron:[]})


    def ablation_hook(value, hook, position, head):
        value[..., head, position, :] = 0.0
        value[..., head, position, 0] = 1.0
        return value

    pbar = tqdm.tqdm(total=len(new_entropy_df))
    for i, r in new_entropy_df.iterrows():
        inp = tokenized_data['tokens'][r.batch][:].to(device)
        
        position = r.pos

        hooks = []
        for attn_head in attn_head_names: 
            pattern = r"L(\d+)H(\d+)"
            match = re.search(pattern, attn_head)
            layer_idx, head_idx = map(int, match.groups())

            hooks.append((tl_utils.get_act_name("pattern", layer_idx), partial(ablation_hook, position=position, head=head_idx)))

        model.reset_hooks()
        with model.hooks(fwd_hooks=hooks):
            ablated_logits, ablated_cache = model.run_with_cache(inp)
            post_ablation_ln_final_scale.append(ablated_cache["ln_final.hook_scale"][0,position,:].cpu().numpy())

            if compute_resid_norm_change: 
                #TODO: probably best way to calculate resid norm is to add a hook. for now, we recompute from the cache
            
                post_ablation_resid_norm.append(ablated_cache[tl_utils.get_act_name("resid_post", model.cfg.n_layers - 1)][:,position].norm(dim=-1).cpu().numpy()) #hard coded for final layer


        for neuron_name in neuron_names:
            neuron_layer = int(neuron_name.split(".")[0])
            neuron_index = int(neuron_name.split(".")[1])

            post_ablation_neuron_activations_cache_dict[neuron_name].append(ablated_cache[tl_utils.get_act_name("post", neuron_layer)][..., position, neuron_index].cpu().numpy())

            if cache_pre_activations: 
                post_ablation_neuron_pre_activations_cache_dict[neuron_name].append(ablated_cache[tl_utils.get_act_name("pre", neuron_layer)][..., position, neuron_index].cpu().numpy())


        ablated_entropy = get_entropy(ablated_logits[:,position,:].unsqueeze(1), use_log2=False).cpu().numpy()
        post_ablation_entropy.append(ablated_entropy)
        post_ablation_pred  = ablated_logits.argmax(dim=-1).cpu().numpy()
        post_ablation_top_logit_delta.append(ablated_logits[0, position, r.pred].item() - r.top_logit)
        pred_changes.append(1 if post_ablation_pred[0, position] != r.pred else 0)
        post_ablation_top_ps.append(np.log(ablated_logits.softmax(dim=-1).max(dim=-1).values.cpu().numpy()[:, position]))

        loss_array = model.loss_fn(ablated_logits, inp.unsqueeze(0), per_token=True).cpu().numpy()
        loss_array = np.concatenate((loss_array, np.zeros((loss_array.shape[0], 1))), axis=1)
        post_ablation_losses.append(loss_array[:, position])

        if compute_kl: 
            logits = model(inp)

            # compute KL divergence between the ablated distribution and the distribution from the model
            single_token_abl_probs = ablated_logits[0, position, :].softmax(dim=-1).cpu()
            single_token_probs = logits[0, position, :].softmax(dim=-1).cpu()
            kl_divergence = kl_div(single_token_abl_probs, single_token_probs, reduction='none') # this is element-wise 
            kl_before_after.append(kl_divergence.sum().item())
            
            # compute KL divergence between the ablated distribution and the distribution from the model.b_U
            b_U_probs = model.b_U.softmax(dim=0).cpu()
            kl_divergence_after = kl_div(single_token_abl_probs, b_U_probs, reduction='none').sum().item()
            kl_divergence_before = kl_div(single_token_probs, b_U_probs, reduction='none').sum().item()
            kl_from_bu.append(kl_divergence_after - kl_divergence_before)            

        pbar.update(1)


    new_entropy_df['post_ablation_entropy'] = np.concatenate(post_ablation_entropy, axis=0)
    new_entropy_df['entropy_diff'] = new_entropy_df['post_ablation_entropy'] - new_entropy_df['entropy']
    new_entropy_df['absolute_entropy_diff'] = np.abs(new_entropy_df['entropy_diff'])
    new_entropy_df['top_logit_decrease'] = np.array(post_ablation_top_logit_delta)
    new_entropy_df['pred_change'] = np.array(pred_changes)
    new_entropy_df['post_ablation_loss'] = np.concatenate(post_ablation_losses, axis=0)
    new_entropy_df['loss_diff'] = new_entropy_df['post_ablation_loss'] - new_entropy_df['loss']
    new_entropy_df['post_ablation_top_logp'] = np.concatenate(post_ablation_top_ps, axis=0)
    new_entropy_df['top_logp_diff'] = new_entropy_df['post_ablation_top_logp'] - new_entropy_df['top_logp']

    if compute_kl:
        new_entropy_df['kl_before_after'] = np.array(kl_before_after)
        new_entropy_df['kl_from_bu'] = np.array(kl_from_bu)
        new_entropy_df['absolute_kl_from_bu'] = np.abs(new_entropy_df['kl_from_bu'])
        

    new_entropy_df['post_ablation_ln_final_scale'] = np.concatenate(post_ablation_ln_final_scale).flatten()
    new_entropy_df['ln_final_scale_diff'] = new_entropy_df['post_ablation_ln_final_scale'] - new_entropy_df['ln_final_scale']


    for neuron_name in post_ablation_neuron_activations_cache_dict.keys(): 
        new_entropy_df[f'{neuron_name}_activation_post_abl'] = np.concatenate(post_ablation_neuron_activations_cache_dict[neuron_name], axis=0).flatten()

        if cache_pre_activations: 
            new_entropy_df[f'{neuron_name}_pre_activation_post_abl'] = np.concatenate(post_ablation_neuron_pre_activations_cache_dict[neuron_name], axis=0).flatten()


    if compute_resid_norm_change: 
        new_entropy_df[f"post_ablation_{model.cfg.n_layers - 1}.resid_post_norm"] = np.concatenate(post_ablation_resid_norm, axis=0)
        new_entropy_df[f"{model.cfg.n_layers - 1}.resid_post_norm_change"] = new_entropy_df[f"post_ablation_{model.cfg.n_layers - 1}.resid_post_norm"] - new_entropy_df[f"{model.cfg.n_layers - 1}.resid_post_norm"]

    return new_entropy_df


def mean_ablate_attn_heads(attn_head_names,
                      tokenized_data=None,
                      entropy_df=None,
                      model=None,
                      select='all',
                      k=10,
                      device='mps',
                      cache_pre_activations=True,
                      compute_resid_norm_change=False, # requires entropy_df to have cached pre-ablation norm. currently hard-coded to do "final_layer".resid_post_norm 
                      subtract_b_U=False,
                      seed = 42,
                      compute_kl = False, 
                      ablation_values_dict = None
                      ):

    post_ablation_neuron_activations_cache_dict = {}
    post_ablation_neuron_pre_activations_cache_dict = {}

    post_ablation_ln_final_scale = []

    post_ablation_entropy = []
    post_ablation_resid_norm = []
    post_ablation_top_logit_delta = []
    pred_changes = []
    post_ablation_losses = []
    post_ablation_top_ps = []
    kl_before_after = []
    kl_from_bu = []
    post_ablation_resid_norm = []

    if select == 'all':
        new_entropy_df = entropy_df.copy()
    else:
        new_entropy_df = entropy_df.sample(frac=1, random_state=seed).iloc[:k].copy()


    if compute_resid_norm_change: 
        #currently hard-coded to do "final_layer".resid_post_norm 
        print("Reminder: resid_norm change is currently hardcoded for final layer resid post only")
        assert(f"{model.cfg.n_layers - 1}.resid_post_norm" in entropy_df.columns), "To compute resid norm change, entropy_df must have column final_layer.resid_post_norm in entropy_df"



    neuron_names = []
    for col_name in entropy_df.columns: 
        pattern = r'^\d+\.\d+_activation$'

        if re.search(pattern, col_name):
            neuron_names.append(col_name.split("_")[0]) # ideally use regex instead of split


    for neuron in neuron_names:
        post_ablation_neuron_activations_cache_dict.update({neuron:[]})
        if cache_pre_activations: 
            post_ablation_neuron_pre_activations_cache_dict.update({neuron:[]})


    def mean_ablation_hook(value, hook, position, head, attn_z):
        value[..., position, head,:] = attn_z 
        return value

    pbar = tqdm.tqdm(total=len(new_entropy_df))


    for i, r in new_entropy_df.iterrows():
        inp = tokenized_data['tokens'][r.batch][:].to(device)
        
        #logits, cache = model.run_with_cache(inp)
        #test_act = cache[tl_utils.get_act_name("post", neuron_layer)][0, -1, neuron_index].cpu().numpy()
        position = r.pos

        hooks = []
        for attn_head in attn_head_names: 
            pattern = r"L(\d+)H(\d+)"
            match = re.search(pattern, attn_head)
            layer_idx, head_idx = map(int, match.groups())

            hooks.append((f"blocks.{layer_idx}.attn.hook_z", partial(mean_ablation_hook, position=position, head=head_idx, attn_z=ablation_values_dict[attn_head])))

        model.reset_hooks()
        with model.hooks(fwd_hooks=hooks):
            ablated_logits, ablated_cache = model.run_with_cache(inp)
            post_ablation_ln_final_scale.append(ablated_cache["ln_final.hook_scale"][0,position,:].cpu().numpy())

            if compute_resid_norm_change:             
                post_ablation_resid_norm.append(ablated_cache[tl_utils.get_act_name("resid_post", model.cfg.n_layers - 1)][:,position].norm(dim=-1).cpu().numpy()) #hard coded for final layer

        for neuron_name in neuron_names:
            neuron_layer = int(neuron_name.split(".")[0])
            neuron_index = int(neuron_name.split(".")[1])

            post_ablation_neuron_activations_cache_dict[neuron_name].append(ablated_cache[tl_utils.get_act_name("post", neuron_layer)][..., position, neuron_index].cpu().numpy())

            if cache_pre_activations: 
                post_ablation_neuron_pre_activations_cache_dict[neuron_name].append(ablated_cache[tl_utils.get_act_name("pre", neuron_layer)][..., position, neuron_index].cpu().numpy())

        ablated_entropy = get_entropy(ablated_logits[:,position,:].unsqueeze(1), use_log2=False).cpu().numpy()
        post_ablation_entropy.append(ablated_entropy)
        post_ablation_pred  = ablated_logits.argmax(dim=-1).cpu().numpy()
        post_ablation_top_logit_delta.append(ablated_logits[0, position, r.pred].item() - r.top_logit)
        pred_changes.append(1 if post_ablation_pred[0, position] != r.pred else 0)
        post_ablation_top_ps.append(np.log(ablated_logits.softmax(dim=-1).max(dim=-1).values.cpu().numpy()[:, position]))

        loss_array = model.loss_fn(ablated_logits, inp.unsqueeze(0), per_token=True).cpu().numpy()
        loss_array = np.concatenate((loss_array, np.zeros((loss_array.shape[0], 1))), axis=1)
        post_ablation_losses.append(loss_array[:, position])

        if compute_kl: 
            logits = model(inp)

            # compute KL divergence between the ablated distribution and the distribution from the model
            single_token_abl_probs = ablated_logits[0, position, :].softmax(dim=-1).cpu()
            single_token_probs = logits[0, position, :].softmax(dim=-1).cpu()
            kl_divergence = kl_div(single_token_abl_probs, single_token_probs, reduction='none') # this is element-wise 
            kl_before_after.append(kl_divergence.sum().item())
            
            # compute KL divergence between the ablated distribution and the distribution from the model.b_U
            b_U_probs = model.b_U.softmax(dim=0).cpu()
            kl_divergence_after = kl_div(single_token_abl_probs, b_U_probs, reduction='none').sum().item()
            kl_divergence_before = kl_div(single_token_probs, b_U_probs, reduction='none').sum().item()
            kl_from_bu.append(kl_divergence_after - kl_divergence_before)            
        
        pbar.update(1)

    new_entropy_df['post_ablation_entropy'] = np.concatenate(post_ablation_entropy, axis=0)
    new_entropy_df['entropy_diff'] = new_entropy_df['post_ablation_entropy'] - new_entropy_df['entropy']
    new_entropy_df['absolute_entropy_diff'] = np.abs(new_entropy_df['entropy_diff'])
    new_entropy_df['top_logit_decrease'] = np.array(post_ablation_top_logit_delta)
    new_entropy_df['pred_change'] = np.array(pred_changes)
    new_entropy_df['post_ablation_loss'] = np.concatenate(post_ablation_losses, axis=0)
    new_entropy_df['loss_diff'] = new_entropy_df['post_ablation_loss'] - new_entropy_df['loss']
    new_entropy_df['post_ablation_top_logp'] = np.concatenate(post_ablation_top_ps, axis=0)
    new_entropy_df['top_logp_diff'] = new_entropy_df['post_ablation_top_logp'] - new_entropy_df['top_logp']

    if compute_kl:
        new_entropy_df['kl_before_after'] = np.array(kl_before_after)
        new_entropy_df['kl_from_bu'] = np.array(kl_from_bu)
        new_entropy_df['absolute_kl_from_bu'] = np.abs(new_entropy_df['kl_from_bu'])

    new_entropy_df['post_ablation_ln_final_scale'] = np.concatenate(post_ablation_ln_final_scale).flatten()
    new_entropy_df['ln_final_scale_diff'] = new_entropy_df['post_ablation_ln_final_scale'] - new_entropy_df['ln_final_scale']

    for neuron_name in post_ablation_neuron_activations_cache_dict.keys(): 
        new_entropy_df[f'{neuron_name}_activation_post_abl'] = np.concatenate(post_ablation_neuron_activations_cache_dict[neuron_name], axis=0).flatten()

        if cache_pre_activations: 
            new_entropy_df[f'{neuron_name}_pre_activation_post_abl'] = np.concatenate(post_ablation_neuron_pre_activations_cache_dict[neuron_name], axis=0).flatten()


    if compute_resid_norm_change: 
        new_entropy_df[f"post_ablation_{model.cfg.n_layers - 1}.resid_post_norm"] = np.concatenate(post_ablation_resid_norm, axis=0)
        new_entropy_df[f"{model.cfg.n_layers - 1}.resid_post_norm_change"] = new_entropy_df[f"post_ablation_{model.cfg.n_layers - 1}.resid_post_norm"] - new_entropy_df[f"{model.cfg.n_layers - 1}.resid_post_norm"]

    return new_entropy_df




# ====
# hf model name
# ====

def tl_name_to_hf_name(model_name): 
    hf_model_name = transformer_lens.loading_from_pretrained.get_official_model_name(model_name)
    return hf_model_name


def load_model_from_tl_name(model_name, device='cuda', cache_dir=None, hf_token=None): 
    hf_model_name = tl_name_to_hf_name(model_name)

    #loading tokenizer
    if "qwen" in model_name.lower():
        tokenizer = AutoTokenizer.from_pretrained(hf_model_name, trust_remote_code=True, pad_token='<|extra_0|>', eos_token='<|endoftext|>', cache_dir=cache_dir)
        # following the example given in their github repo: https://github.com/QwenLM/Qwen
    else: 
        tokenizer = AutoTokenizer.from_pretrained(hf_model_name, trust_remote_code=True, cache_dir=cache_dir, token=hf_token)

    #loading model 
    if "llama" in model_name.lower() or "gemma" in model_name.lower() or "mistral" in model_name.lower(): 
        hf_model = AutoModelForCausalLM.from_pretrained(hf_model_name, token=hf_token, cache_dir=cache_dir)
        model = HookedTransformer.from_pretrained(model_name=model_name, hf_model=hf_model, tokenizer=tokenizer, device=device, cache_dir=cache_dir)
    else: 
        model = HookedTransformer.from_pretrained(model_name, device=device, cache_dir=cache_dir, token=hf_token)

    return model, tokenizer


# Induction functions

def generate_induction_examples(model, tokenizer, seq_length=100, num_examples=500, seed=42, device='cuda', use_natural_text=False, use_separator=None, num_repetitions=1): 
    torch_seed = torch.Generator(device=device)
    torch_seed.manual_seed(seed)

    if use_natural_text: 
        data = load_dataset("stas/c4-en-10k", split='train')
        first_1k = data.select([i for i in range(4000, 8000)])
        max_len = 256 # used to ensure we select samples that are at least 256 tokens
        def tok_function(examples):
            return {'tokens':tokenizer(examples['text'], add_special_tokens=False, max_length=max_len, truncation=True).input_ids}
        tok_dataset = first_1k.map(tok_function, batched=True, remove_columns=['text'])
        filtered = tok_dataset.filter(lambda example: len(example['tokens']) >= max_len)
        filtered = filtered.shuffle(seed=seed)
        filtered.set_format(type="torch", columns=['tokens'])
        sequences = filtered['tokens'][:num_examples, :seq_length].to(device)
    else:
        sequences = torch.randint(0, model.cfg.d_vocab, size=(num_examples, seq_length), generator=torch_seed, device=device)
    
    #wont work for models without single token bos
    if tokenizer.bos_token_id is not None:
        bos_prefix = torch.tensor([tokenizer.bos_token_id]*num_examples, device=device).unsqueeze(1)
        first_sequence = [bos_prefix, sequences]
    else: 
        first_sequence = [sequences]

    if use_separator:
        separator = einops.repeat(model.to_tokens(use_separator, prepend_bos=False), "1 seq -> num_ex seq", num_ex=num_examples)
        things_to_repeat = [separator, sequences]
    else:
        things_to_repeat = [sequences]
    
    things_to_concat = first_sequence + (things_to_repeat*num_repetitions)
    tokens = torch.concat(things_to_concat, dim=-1)

    return tokens


def get_induction_data_and_token_df(model, tokenizer, seq_length, num_examples, seed=42, device='cuda', use_natural_text=False, use_separator=None, num_repetitions=1): 
    artificial_data = Dataset.from_dict({'tokens': generate_induction_examples(model, tokenizer, seq_length=seq_length, num_examples=num_examples, seed=seed, device=device, use_natural_text=use_natural_text, use_separator=use_separator, num_repetitions=num_repetitions)})
    artificial_data.set_format(type="torch", columns=["tokens"])

    artificial_token_df = nutils.make_token_df(artificial_data["tokens"])

    return artificial_data, artificial_token_df


# ====== 
# unigram stuff 
# ====== 

# get pile unigram count - specifically for Pythia
def get_pile_unigram_distribution(file_path="datasets/pythia-unigrams.npy", pad_to_match_W_U=True, device="cuda", model_name="pythia-410m"): 
    
    unigram_count = np.load(file_path)

    if "pythia" in model_name:
        W_U_SIZE = 50304
        TRUE_VOCAB_SIZE = 50277
    elif "phi-2" in model_name: 
        W_U_SIZE = 51200
        TRUE_VOCAB_SIZE = 50295
    token_discrepancy = W_U_SIZE - TRUE_VOCAB_SIZE
    if pad_to_match_W_U:
        unigram_count = np.concatenate([unigram_count, [0] * token_discrepancy])

    unigram_distrib = unigram_count + 1 
    unigram_distrib = unigram_distrib / unigram_distrib.sum()
    unigram_distrib = torch.tensor(unigram_distrib, dtype=torch.float32).to(device)

    return unigram_distrib


def generate_unigram_df(data, model, tokenizer, model_name, save_file=False, save_path="datasets/unigram_df.pkl"):
    
    def tokenization(example): 
        return tokenizer(example["text"], truncation=False, padding=False, add_special_tokens=False, return_attention_mask=False)

    # num_proc currently doesn't work for streamed, IterableDatasets
    if isinstance(data, datasets.IterableDataset):
        data = data.map(tokenization, batched=True)
    else:
        data = data.map(tokenization, batched=True, num_proc=10)

    # handle discrepancy in d_vocab between pythia tokenizer and W_U
    if 'pythia' in model_name: 
        vocab_dict = {k:0 for k in range(model.cfg.d_vocab)}
    else: 
        vocab_dict = {k:0 for k in tokenizer.get_vocab().values() }
    
    unigram_count = Counter(vocab_dict)

    # this could be sped up by processing in batches
    for tokenized_example in tqdm.tqdm(data['input_ids']):
        unigram_count.update(tokenized_example)


    df = pd.DataFrame({'token_id':unigram_count.keys(), 'count':unigram_count.values()})

    if 'phi' in model_name: 
        # model.to_single_str_token doesn't work for last n tokens since they have no corresponding str
        # final real token is 50294 which corresponds to \t\t
        df['str_token'] = df['token_id'].apply(lambda x: tokenizer.decode(x))
    else:
        df['str_token'] = df['token_id'].apply(lambda x: model.to_single_str_token(x))
    df['token_id_as_str'] = df['token_id'].apply(lambda x: str(x))

    df = df.sort_values("token_id", ascending=True)
    df["count_rank"] = scipy.stats.rankdata(df["count"])
    # data is ranked smallest to largest. i.e. count=0 is rank=1

    if save_file: 
        with open(save_path, "wb") as f:
            print("Saving unigram df to: ", save_path)
            pickle.dump(df, f)

    return df

def load_unigram_df(filepath):
    with open(filepath, "rb") as f:
        df = pickle.load(f)

    # should already be saved in correct order, but just in case
    df = df.sort_values("token_id", ascending=True)
    return df


def get_unigram_distrib(unigram_df, device="cuda"): 
    unigram_distrib = unigram_df['count'].values + 1

    if unigram_distrib.min() == 0:
        unigram_distrib += 1

    unigram_distrib = unigram_distrib / unigram_distrib.sum()
    unigram_distrib = torch.tensor(unigram_distrib, dtype=torch.float32).to(device)
    return unigram_distrib


#====
# natural induction stuff
#====
# used in a dataset.map() to add info to tokenized data about induction

def n_gram_counter(tensor, n=2):
    """
    Counts n-grams in a 1D PyTorch tensor of integers.
    
    Parameters:
        tensor (torch.Tensor): A 1D tensor of integers.
        n (int): The length of the n-gram sequence.
    
    Returns:
        Counter: A Counter object mapping each n-gram to its frequency in the tensor.
    """
    # Convert the tensor to a list of integers
    if isinstance(tensor, torch.Tensor): 
        items = tensor.tolist()
    else:
        items = tensor
    # Generate n-grams
    n_grams = [tuple(items[i:i+n]) for i in range(len(items)-n+1)]
    
    # Count n-grams
    n_gram_counts = Counter(n_grams)
    
    return n_gram_counts

def add_induction_info(example, n=4, banned_tokens=set()): 

    ngrams = n_gram_counter(example['tokens'], n)
    # filter for ngrams that contain repeated tokens inside the ngram
    # filter for ngrams that contain banned tokens
    # filter for ngrams that only appear once
    filtered_ngrams = Counter({k: v for k, v in ngrams.items() if (len(set(k).intersection(banned_tokens)) == 0) and (len(set(k))==len(k)) and (v>1)})

    if len(filtered_ngrams) == 0:
        example['is_valid'] = False
        example['induction_ngram'] = None
        example['induction_ngram_first_pos'] = None
        example['induction_ngram_second_pos'] = None
        example['induction_ngram_count'] = None
        example['n_tokens_between_induction_ngrams'] = None
        return example
    else: 
        example['is_valid'] = True
        # now we need to choose which repeated n-gram we treat as our induction prefix
        # we could choose either via the prefix that appears first or the one that is most commonly repeated in the sequence, or just randomly

        # we select the one that appears first
        n_grams_with_pos = get_n_grams_with_pos_dict(example['tokens'], n)
        for pos, ngram in n_grams_with_pos.items():
            if ngram in filtered_ngrams: #first occurence of an ngram that appears in filtered ngrams
                induction_ngram = ngram
                induction_ngram_first_pos = pos
                break

        ngram_occurences = [k for k, v in n_grams_with_pos.items() if v == induction_ngram]
        assert(len(ngram_occurences) == filtered_ngrams[induction_ngram])
        #get pos of second occurence of the ngram
        #also record how many times this ngram appears.

        example['induction_ngram'] = induction_ngram
        example['induction_ngram_first_pos'] = induction_ngram_first_pos
        example['induction_ngram_second_pos'] = ngram_occurences[1]
        example['induction_ngram_count'] = filtered_ngrams[induction_ngram]
        example['n_tokens_between_induction_ngrams'] = ngram_occurences[1] - (induction_ngram_first_pos + n)

        return example


# used to record position of ngrams in a sequence
def get_n_grams_with_pos_dict(tensor, n=2): 
    '''
    returns dict where key is the pos_index and value is the n-gram
    '''
    if isinstance(tensor, torch.Tensor): 
        items = tensor.tolist()
    else:
        items = tensor
    
    n_grams_with_pos = {i:tuple(items[i:i+n]) for i in range(len(items)-n+1)}
    return n_grams_with_pos


def get_banned_tokens_for_induction(model, tokenizer): 
    unknown_token_list = [i for i in range(model.cfg.d_vocab) if '�' in model.to_single_str_token(i)] # this covers cases like 447 and 227 which are jointly tokenised as an apostrophe in gpt2-small
    bos_token_list = [tokenizer.bos_token_id] # this may cause issues for models without bos token

    if 'Llama-2' in model.cfg.model_name:
        other_tokens = [13, 29871] 

    else: 
        other_tokens = [model.to_single_token('\n')]
    banned_tokens = set(unknown_token_list + bos_token_list + other_tokens)
    return banned_tokens


def get_natural_induction_data(tokenized_data, tokenizer, induction_prefix_length=4, max_induction_ngram_count=2, min_distance_between_induction_ngrams=1, banned_tokens=set()):
    # takes in a hf dataset with column 'tokens' of format torch
    # returns a filtered dataset with extra info about induction

    # remove samples where eos token is in the middle of the text
    filtered_data = tokenized_data.filter(lambda example: len(example['tokens'][example['tokens']==tokenizer.bos_token_id]) < 2)

    # add info about induction (used to filter out non-induction samples)
    filtered_data = filtered_data.map(lambda x: add_induction_info(x, n=induction_prefix_length, banned_tokens=banned_tokens), batched=False)

    #necessary because otherwise columns are torch tensors which means we can't filter
    filtered_data.reset_format()
    filtered_data = filtered_data.filter(lambda example: example['is_valid'], batched=False)

    # additional filtering
    # making sure induction prefix doesn't repeat too many times
    filtered_data = filtered_data.filter(lambda example: example["induction_ngram_count"]
    <=max_induction_ngram_count, batched=False)
    # making sure there are tokens in between repetitions of induction prefix 
    filtered_data = filtered_data.filter(lambda example: example["n_tokens_between_induction_ngrams"] >= min_distance_between_induction_ngrams, batched=False)

    filtered_data.set_format(type="torch", columns=["tokens"])

    return filtered_data


def get_potential_entropy_neurons_udark(model, select_mode="top_n", percentage_threshold=0.01, select_top_n=10, udark_start=-40, udark_end=0, plot_graph=False): 

    # take svd of W_U
    U, S, V = torch.linalg.svd(model.W_U, full_matrices=False)

    # make scatter plot of W_out[-1] @ U_entropy and W_out[-1].norm()
    U_entropy = range(udark_start, udark_end)
    norm = model.W_out[-1].norm(dim=-1)
    norm_fraction_on_U_entropy = (model.W_out[-1:] @ U)[-1, :, U_entropy].norm(dim=-1) / norm
    # make dataframe
    df = pd.DataFrame({'norm_fraction_on_U_entropy': norm_fraction_on_U_entropy.cpu(), 'norm': norm.cpu(), 'neuron_index': np.arange(model.cfg.d_mlp)})
    df['component_name'] = df['neuron_index'].apply(lambda x: f"{model.cfg.n_layers - 1}.{x}")

    top_percentage = int(percentage_threshold * model.cfg.d_mlp)
    # sort by fraction of norm in null space
    sorted_df = df.sort_values('norm_fraction_on_U_entropy', ascending=False)

    if select_mode == "top_n":
        top_percent_neuron_names = sorted_df['component_name'].iloc[:select_top_n].to_list()
    else: 
        top_percent_neuron_names = sorted_df['component_name'].iloc[:top_percentage].to_list()

    if plot_graph:
        fig = px.scatter(df, x='norm', y='norm_fraction_on_U_entropy', title=f'Entropy neurons, {len(top_percent_neuron_names)}', hover_name="component_name")

        min_thresold_value = df[df['component_name'] == top_percent_neuron_names[-1]]['norm_fraction_on_U_entropy'].item()
        fig.add_hline(y=min_thresold_value, line_dash="dash", line_color="black", line_width=1)

        fig.show()
    return top_percent_neuron_names



def induction_attn_detector(cache: ActivationCache, threshold=0.7, model=None) -> List[str]:
    '''
    Returns a df of induction scores 

    Remember - the tokens used to generate rep_cache are (bos_token, *rand_tokens, *rand_tokens)
    '''
    induction_heads = []
    attn_heads_labels = []
    induction_scores = []
    for layer in range(model.cfg.n_layers):
        for head in range(model.cfg.n_heads):
            if cache.has_batch_dim: 
                attention_pattern = cache["pattern", layer][:, head]
                attention_pattern = attention_pattern.mean(dim=0)
            else: 
                attention_pattern = cache["pattern", layer][head]

            # take avg of (-seq_len+1)-offset elements
            seq_len = (attention_pattern.shape[-1] - 1) // 2
            score = attention_pattern.diagonal(-seq_len+1).mean()

            attn_heads_labels.append(f"L{layer}H{head}")
            induction_scores.append(score.item())

    induction_df = pd.DataFrame({'labels':attn_heads_labels, 'induction_scores':induction_scores})
    induction_df['is_induction'] = induction_df['induction_scores'] > threshold

    return induction_df


def generate_induction_df(model, tokens, batch_size=1, num_batches=1, threshold=0):
    '''
    tokens is a tensor of dim (batch, seq_len)
    '''

    induction_df = pd.DataFrame()
    for i in tqdm.tqdm(range(num_batches)):
        rep_logits, rep_cache = model.run_with_cache(tokens[i*batch_size:(i+1)*batch_size])
        tmp_df = induction_attn_detector(rep_cache, threshold=threshold, model=model)
        tmp_df["batch"] = i
        induction_df = pd.concat([induction_df, tmp_df])

    aggregated_induction_df = induction_df.groupby("labels").mean()
    induction_df['is_induction'] = induction_df['induction_scores'] > threshold

    return aggregated_induction_df