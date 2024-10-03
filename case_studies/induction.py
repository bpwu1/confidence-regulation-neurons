# %% 
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  
os.environ["CUDA_VISIBLE_DEVICES"]="0"
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
import plotly
from plotly.express.colors import qualitative

# %%
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.set_grad_enabled(False)

transformers_cache_dir = None
#check if cuda is available
if torch.cuda.is_available():
    device = 'cuda'
    
else:
    device = 'mps'


# %%
model_name = "gpt2-small"
# model_name = "Llama-2-7B"

print_summary_info = False
use_log2_entropy = False 

entropy_type = 'base e'
if use_log2_entropy:
    entropy_type = 'base 2'

# %%
model, tokenizer = load_model_from_tl_name(model_name, device, transformers_cache_dir)
model = model.to(device)

# %%
#data = load_dataset("stas/openwebtext-10k", split='train')
data = load_dataset("stas/c4-en-10k", split='train')
first_1k = data.select([i for i in range(0, 2000)])


tokenized_data = utils.tokenize_and_concatenate(first_1k, tokenizer, max_length=256, column_name='text')

tokenized_data = tokenized_data.shuffle(SEED)
token_df = nutils.make_token_df(tokenized_data['tokens'])

# %%
if model_name == "Llama-2-7b":
    udark_start = -40
elif model_name == "gpt2-small":
    udark_start = -12

entropy_neurons = get_potential_entropy_neurons_udark(model, select_mode="top_n", select_top_n=10,udark_start=udark_start, udark_end=0, plot_graph=True)

possible_random_neuron_indices = list(range(0, model.cfg.d_mlp))
for neuron_name in entropy_neurons:
    possible_random_neuron_indices.remove(int(neuron_name.split('.')[1]))

entropy_neuron_layer = int(entropy_neurons[0].split('.')[0])
random_neuron_indices = random.sample(possible_random_neuron_indices, 10)
random_neurons = [f"{entropy_neuron_layer}.{i}" for i in random_neuron_indices]

# %%
# =============================================================================
# Induction
# =============================================================================

num_seq = 1000
induction_tokenized_data = {}
induction_tokenized_data['tokens'] = []
for i in range(num_seq):
    index = random.randint(0, len(tokenized_data['tokens']))
    sequence_enc = tokenized_data['tokens'][index]
    sequence_enc = torch.cat((sequence_enc[:100], sequence_enc[1:100]), dim=0)
    induction_tokenized_data['tokens'].append(sequence_enc.numpy())
induction_tokenized_data['tokens'] = torch.tensor(induction_tokenized_data['tokens'])
new_token_df = nutils.make_token_df(induction_tokenized_data['tokens'])
# convert to dataset, preserving the type of 'tokens' as torch.Tensor
induction_tokenized_data = Dataset.from_dict(induction_tokenized_data)
induction_tokenized_data.set_format(type="torch", columns=["tokens"])
# %%
n_of_baselines = 5
# sample n_of_baselines random lists of neurons to ablate
random_baselines = []
for i in range(n_of_baselines):
    random_baseline = random.sample(possible_random_neuron_indices, len(entropy_neurons))
    random_baseline = [f"{entropy_neuron_layer}.{i}" for i in random_baseline]
    random_baselines.append(random_baseline)

# %%

unigram_distrib = None
# %%
# check if random_baseline is a defined variable
components_to_track = entropy_neurons + random_neurons
try:
    components_to_track += [neuron for neuron_list in random_baselines for neuron in neuron_list]
except NameError:
    pass

print(components_to_track)

batch_size = 8
entropy_dim_layer = model.cfg.n_layers - 1
component_output_to_cache = {'resid_post': []}
entropy_df, resid_dict = get_entropy_activation_df(entropy_neurons + [f'{entropy_neuron_layer}.{i}' for i in possible_random_neuron_indices],
                                                   induction_tokenized_data,
                                                   new_token_df,
                                                   model,
                                                   batch_size=batch_size,
                                                   device=device,
                                                   cache_residuals=True,
                                                   cache_pre_activations=False,
                                                   compute_kl_from_bu=False,
                                                   residuals_layer=entropy_dim_layer,
                                                   residuals_dict=component_output_to_cache,
                                                   unigram_distrib=unigram_distrib)
# %%
# average neuron activations across sequences
neuron_activations_cache_dict = {}
for neuron_name in entropy_neurons +random_neurons:
    neuron_activations_cache_dict[neuron_name] = entropy_df[[f'{neuron_name}_activation']].values.reshape((num_seq, -1)).mean(axis=0)

# average entropy across sequences
entropy = entropy_df['entropy'].values.reshape((num_seq, -1)).mean(axis=0)

# average ln_final_scale across sequences
ln_final_scale = entropy_df['ln_final_scale'].values.reshape((num_seq, -1)).mean(axis=0)

#kl_from_unigram = entropy_df['kl_from_unigram'].values.reshape((num_seq, -1)).mean(axis=0)

# average loss across sequences
loss = entropy_df['loss'].values.reshape((num_seq, -1)).mean(axis=0)
# %%
# plot activations for each neuron, along with entropy and loss
x_axis = list(range(len(neuron_activations_cache_dict[neuron_name])))
fig = go.Figure()
for neuron_name, neuron_activations in neuron_activations_cache_dict.items():
    fig.add_trace(go.Scatter(x=x_axis, y=neuron_activations, mode='lines', name=f'{neuron_name}_activation'))

fig.add_trace(go.Scatter(x=x_axis, y=entropy, mode='lines', name='entropy'))
fig.add_trace(go.Scatter(x=x_axis, y=loss, mode='lines', name='loss'))
fig.add_trace(go.Scatter(x=x_axis, y=ln_final_scale, mode='lines', name='ln_final_scale'))
#fig.add_trace(go.Scatter(x=x_axis, y=kl_from_unigram, mode='lines', name='kl_from_unigram'))
# add vertical lines to indicate the end of the first sequence
fig.add_vline(x=99, line_dash="dash", line_color="black", annotation_text="end of first occurrence of sequence")
# add title
fig.update_layout(title=f"Average activations, entropy, and loss across {num_seq} sequences. Model: {model_name}")
# set axis labels
fig.update_xaxes(title_text='Position in sequence')

fig.show()
# %%
# =============================================================================
# Plots for paper: activations on induction
# =============================================================================
# plot activations for each neuron, along with entropy and loss
x_axis = list(range(len(neuron_activations_cache_dict[neuron_name])))
fig = go.Figure()
fig.add_trace(go.Scatter(x=x_axis, y=entropy, mode='lines', name='Entropy', line=dict(width=3, dash='solid')))
fig.add_trace(go.Scatter(x=x_axis, y=loss, mode='lines', name='Loss', line=dict(width=3, dash='solid')))
for neuron_name in entropy_neurons:
    neuron_activations = neuron_activations_cache_dict[neuron_name]
    fig.add_trace(go.Scatter(x=x_axis, y=neuron_activations, mode='lines', name=f'{neuron_name}', line=dict(width=2.5, dash='dot')))

#fig.add_trace(go.Scatter(x=x_axis, y=ln_final_scale, mode='lines', name='ln_final_scale'))
#fig.add_trace(go.Scatter(x=x_axis, y=kl_from_unigram, mode='lines', name='kl_from_unigram'))
# add vertical lines to indicate the end of the first sequence
fig.add_vline(x=99, line_dash="dash", line_color="black", annotation_text="start of induction", line_width=2)
# add title
fig.update_layout(title=f"(a) Induction: Activations, Entropy, Loss")
# set axis labels
fig.update_xaxes(title_text='Position in Sequence')
#fig.update_yaxes(title_text='Value')

# remove padding
fig.update_layout(margin=dict(l=0, r=3, t=30, b=0))

# decrease the width of the plot
fig.update_layout(width=350*1, height=275/1)

# decrease title font size
fig.update_layout(title_font_size=16)

# save the plot as a pdf
#fig.write_image('../img/induction_activations.pdf')

fig.show()

# %%
# get mean values for each entropy neuron
mean_values_on_random_text = {neuron_name : entropy_df[entropy_df.pos < 100][f'{neuron_name}_activation'].mean() for neuron_name in entropy_neurons + random_neurons}
mean_values_on_induction = {neuron_name : entropy_df[entropy_df.pos >= 100][f'{neuron_name}_activation'].mean() for neuron_name in entropy_neurons + random_neurons}
# %%
threshold = 0.1
selected_neurons = []
weighted_diffs = {}
for neuron_name, mean_value in mean_values_on_random_text.items():
    # compute the difference between the mean value of the neuron on the random text and the mean value of the neuron on the induction text
    diff = mean_values_on_induction[neuron_name] - mean_value
    neuron_idx = int(neuron_name.split('.')[1])
    neuron_layer = int(neuron_name.split('.')[0])
    weighted_diff = diff * model.W_out[neuron_layer, neuron_idx].norm()
    weighted_diffs[neuron_name] = weighted_diff
    if abs(weighted_diff) > threshold:
        is_entropy = neuron_name in entropy_neurons
        if is_entropy:
            print(f'neuron_name = {neuron_name}, diff = {diff}, weighted_diff = {weighted_diff}')
            selected_neurons.append(neuron_name)
        else:
            print(f'NOT ENTROPY: neuron_name = {neuron_name}, diff = {diff}, weighted_diff = {weighted_diff}')




# %%
# =============================================================================
# Induction: Ablations
# =============================================================================

def neurons_clipped_ablation_hook(value, hook, neuron_indices, ablation_values):
    # todo update this for neruons that "deactivate" on induction
    for neuron_idx, ablation_value in zip(neuron_indices, ablation_values):
        value[0, :, neuron_idx] = torch.min(value[0, :, neuron_idx], ablation_value)

    return value

def strict_neurons_clipped_ablation_hook(value, hook, neuron_indices, ablation_values):

    if len(neuron_indices) == 1:
        ablation_value = ablation_values[0]
        neuron_index = neuron_indices[0]

        if weighted_diffs[f'{entropy_neuron_layer}.{neuron_index}'] < 0:
            value[0, :, neuron_index] = torch.max(ablation_value, value[0, :, neuron_index])
        else:
            value[0, :, neuron_index] = torch.min(ablation_value, value[0, :, neuron_index])
    else:
        value[0, :, neuron_indices] = ablation_values
    return value



def ln_final_scale_hook(value, hook, scale_values):
    #scale hook (batch, seq, 1)
    value[0, :, 0] = scale_values
    return value


def clip_ablate_neurons(list_of_components_to_ablate=None,
                      unigram_distrib=None,
                      tokenized_data=None,
                      entropy_df=None,
                      model=None,
                      k=10,
                      device='mps',
                      ablation_type='mean'):
    
    # sample a set of random batch indices
    random_sequence_indices = np.random.choice(entropy_df.batch.unique(), k, replace=False)

    final_df = None
    
    pbar = tqdm.tqdm(total=len(list_of_components_to_ablate) * k, file=sys.stdout)

    for components_to_ablate in list_of_components_to_ablate:

        # check if components_to_ablate is string
        if isinstance(components_to_ablate, str):
            components_to_ablate = [components_to_ablate]

        print(f'ablate_components: ablate {components_to_ablate} with k = {k}')

        # new_entropy_df with only the random sequences
        filtered_entropy_df = entropy_df[entropy_df.batch.isin(random_sequence_indices)]

        df_to_append = filtered_entropy_df.copy()

        neuron_indices = []
        neuron_layers = []
        for component_name in components_to_ablate:
            pattern = r"(\d+).(\d+)"
            match = re.search(pattern, component_name)
            layer_idx, neuron_idx = map(int, match.groups())
            neuron_indices.append((neuron_idx))
            neuron_layers.append((layer_idx))

        # check that all entries in neuron_layers are the same
        assert all(x == neuron_layers[0] for x in neuron_layers), f'neuron_layers = {neuron_layers}'

        neuron_layer = neuron_layers[0]

        # check if dimension name is formatted as {layer}.{neuron}
        if re.match(r"(\d+).(\d+)", components_to_ablate[0]):
            # compute mean only on the first part of the sequences
            entropy_df_f = entropy_df[entropy_df.pos < entropy_df.pos.max() / 2]
            if model.cfg.act_fn == 'relu' and ablation_type == 'mean':
                print(f'ablation_type {ablation_type} does not make sense for relu activations')
            if ablation_type == 'zero':
                ablation_values = torch.zeros((len(components_to_ablate), 1), device=device)
            elif ablation_type == 'mean':
                ablation_values = torch.tensor(entropy_df_f[[f'{neuron_layer}.{neuron_idx}_activation' for neuron_idx in neuron_indices]].mean(axis=0).values)
            elif ablation_type == 'min':
                ablation_values = torch.tensor(entropy_df_f[[f'{neuron_layer}.{neuron_idx}_activation' for neuron_idx in neuron_indices]].min(axis=0).values)
            else:
                raise ValueError(f'ablation_type {ablation_type} not supported')
        else:
            raise ValueError(f'component_name {components_to_ablate[0]} is not formatted as layer_idx.neuron_idx')

        for batch_n in filtered_entropy_df.batch.unique():
            tok_seq = tokenized_data['tokens'][batch_n]

            inp = tok_seq.unsqueeze(0).to(device)

            # get the entropy_df entries for the current sequence
            rows = filtered_entropy_df[filtered_entropy_df.batch == batch_n]
            assert len(rows) == len(tok_seq), f'len(rows) = {len(rows)}, len(tok_seq) = {len(tok_seq)}'

            ln_scales  = torch.tensor(rows[f'ln_final_scale'].values)
    
            hooks = [(utils.get_act_name("post", neuron_layer), partial(strict_neurons_clipped_ablation_hook, neuron_indices=neuron_indices, ablation_values=ablation_values.to(device)))]

            model.reset_hooks()

            with model.hooks(fwd_hooks=hooks):
                ablated_logits, loss = model(inp, return_type='both', loss_per_token=True)

            # compute loss
            loss = loss.cpu().numpy()
            loss = np.concatenate((loss[0], np.zeros((1))), axis=0)
            df_to_append.loc[df_to_append.batch == batch_n, 'loss_post_ablation'] = loss
            entropy_post_ablation = get_entropy(ablated_logits)
            df_to_append.loc[df_to_append.batch == batch_n, 'entropy_post_ablation'] = entropy_post_ablation.squeeze().cpu().numpy()

            # forward with frozen ln
            hooks +=[("ln_final.hook_scale", partial(ln_final_scale_hook, scale_values=ln_scales))]
            with model.hooks(fwd_hooks=hooks):
                ablated_logits_with_frozen_ln, loss = model(inp, return_type='both', loss_per_token=True)

            loss = model.loss_fn(ablated_logits_with_frozen_ln, inp, per_token=True).cpu().numpy()
            loss = np.concatenate((loss[0], np.zeros((1))), axis=0)
            df_to_append.loc[df_to_append.batch == batch_n, 'loss_post_ablation_with_frozen_ln'] = loss
            entropy_post_ablation_with_frozen_ln = get_entropy(ablated_logits_with_frozen_ln)
            df_to_append.loc[df_to_append.batch == batch_n, 'entropy_post_ablation_with_frozen_ln'] = entropy_post_ablation_with_frozen_ln.squeeze().cpu().numpy()

            # compute KL divergence between the ablated distribution and the distribution from the unigram direction
            if unigram_distrib is not None:
                # get unaltered logits
                model.reset_hooks()
                logits, cache = model.run_with_cache(inp)

                abl_probs = ablated_logits[0, :, :].softmax(dim=-1).cpu().numpy()
                probs = logits[0, :, :].softmax(dim=-1).cpu().numpy()

                kl_divergence_after = np.sum(kl_div(abl_probs, unigram_distrib.cpu().numpy()), axis=1)
                kl_divergence_before = np.sum(kl_div(probs, unigram_distrib.cpu().numpy()), axis=1)
                
                single_token_abl_probs_with_frozen_ln = ablated_logits_with_frozen_ln[0, :, :].softmax(dim=-1).cpu().numpy()
                df_to_append.loc[df_to_append.batch == batch_n, 'kl_from_unigram_diff'] = (kl_divergence_after - kl_divergence_before)  
                kl_divergence_after_frozen_ln = np.sum(kl_div(single_token_abl_probs_with_frozen_ln, unigram_distrib.cpu().numpy()), axis=1)
                df_to_append.loc[df_to_append.batch == batch_n, 'kl_from_bu_diff_with_frozen_bu'] = (kl_divergence_after_frozen_ln - kl_divergence_before)  

            pbar.update(1)

        df_to_append['component_name'] = '-'.join(components_to_ablate)

        # stack the df_to_append to final_df
        if final_df is None:
            final_df = df_to_append
        else:
            final_df = pd.concat([final_df, df_to_append])
    
    return final_df

# %%
# =============================================================================
# ablate multiple neurons at the same time
# =============================================================================
k = 50
ablation_type = 'mean'
#components_to_ablate = [entropy_neurons] + [[neuron] for neuron in entropy_neurons] #+ random_baselines
components_to_ablate = [selected_neurons] + random_baselines #+ [[neuron] for neuron in selected_neurons] #+ random_baselines
induction_multiple_ablation_df = clip_ablate_neurons(list_of_components_to_ablate=components_to_ablate,
                                tokenized_data=induction_tokenized_data,
                                entropy_df=entropy_df,
                                model=model,
                                k=k,
                                device=device,
                                ablation_type=ablation_type)
# add is_entropy column
induction_multiple_ablation_df['is_entropy'] = induction_multiple_ablation_df['component_name'].isin(entropy_neurons)
induction_multiple_ablation_df['delta_loss_post_ablation'] = induction_multiple_ablation_df['loss_post_ablation'] - induction_multiple_ablation_df['loss']
induction_multiple_ablation_df['loss_post_ablation/loss'] = induction_multiple_ablation_df['loss_post_ablation'] / induction_multiple_ablation_df['loss']
induction_multiple_ablation_df['abs_delta_loss_post_ablation'] = np.abs(induction_multiple_ablation_df['loss_post_ablation'] - induction_multiple_ablation_df['loss'])
induction_multiple_ablation_df['abs_delta_loss_post_ablation_with_frozen_ln'] = np.abs(induction_multiple_ablation_df['loss_post_ablation_with_frozen_ln'] - induction_multiple_ablation_df['loss'])
induction_multiple_ablation_df['delta_entropy'] = induction_multiple_ablation_df['entropy_post_ablation'] - induction_multiple_ablation_df['entropy']
induction_multiple_ablation_df['delta_entropy_with_frozen_ln'] = induction_multiple_ablation_df['entropy_post_ablation_with_frozen_ln'] - induction_multiple_ablation_df['entropy']
induction_multiple_ablation_df['abs_delta_entropy'] = np.abs(induction_multiple_ablation_df['delta_entropy'])
induction_multiple_ablation_df['abs_delta_entropy_with_frozen_ln'] = np.abs(induction_multiple_ablation_df['delta_entropy_with_frozen_ln'])
columns_to_aggregate =list(induction_multiple_ablation_df.columns[-17:]) + ['loss', 'pos', 'rank_of_correct_token']
agg_results = induction_multiple_ablation_df[columns_to_aggregate].groupby(['pos', 'component_name']).mean().reset_index()
agg_results_std = induction_multiple_ablation_df[columns_to_aggregate].groupby(['pos', 'component_name']).std().reset_index()

# %%
induction_multiple_ablation_df['1/rank_of_correct_token'] = induction_multiple_ablation_df['rank_of_correct_token'].apply(lambda x: 1/(x+1))

#%%

# get mean values for each entropy neuron
mean_values_on_random_text = {neuron_name : entropy_df[entropy_df.pos < entropy_df.pos.max() / 2][f'{neuron_name}_activation'].mean() for neuron_name in entropy_neurons}

print(mean_values_on_random_text)

# %%
#simpler avg loss diff
neuron_selection = '11.3030'
neuron_df = induction_multiple_ablation_df[induction_multiple_ablation_df.component_name == neuron_selection]

neuron_df = neuron_df[neuron_df[f'{neuron_selection}_activation'] > mean_values_on_random_text[neuron_selection]]

neuron_df = neuron_df[columns_to_aggregate].groupby(['pos', 'component_name']).mean().reset_index()


px.line(neuron_df, x="pos", y="loss_post_ablation/loss", title=f"Change in loss when ablating {neuron_selection}. <br> Ablation type: {ablation_type}")
# %%
neuron_selection = '23.1332'
neuron_df = induction_multiple_ablation_df[induction_multiple_ablation_df.component_name == neuron_selection]
neuron_df = neuron_df[neuron_df["pos"]>100]
px.scatter(neuron_df, x="pos", y="loss_post_ablation/loss", title=f"Change in loss when ablating {neuron_selection}. <br> Ablation type: {ablation_type}", color="1/rank_of_correct_token", color_continuous_scale=plotly.colors.diverging.Picnic,)
# %%
px.scatter(neuron_df, x=f"{neuron_selection}_activation", y="loss_post_ablation/loss", title=f"Change in loss when ablating {neuron_selection}. <br> Ablation type: {ablation_type}", color="1/rank_of_correct_token", color_continuous_scale=plotly.colors.diverging.Picnic)
# %%
px.scatter(neuron_df, x=f"{neuron_selection}_activation", y="delta_loss_post_ablation", title=f"Change in loss when ablating {neuron_selection}. <br> Ablation type: {ablation_type}", color="1/rank_of_correct_token", color_continuous_scale=plotly.colors.diverging.Picnic)


# %%
# plot the average loss difference for each neuron for each position as line plots, one line per neuron
fig = go.Figure()

neuron_selection = '-'.join(selected_neurons)

neuron_df = induction_multiple_ablation_df[induction_multiple_ablation_df.component_name == neuron_selection]
neuron_df = neuron_df
#for neuron_name, mean_value in zip(entropy_neurons, mean_values_on_random_text):
#    neuron_df = neuron_df[neuron_df[f'{neuron_name}_activation'] > mean_value]
neuron_df = neuron_df[columns_to_aggregate].groupby(['pos', 'component_name']).mean().reset_index()
neuron_df_std = neuron_df[columns_to_aggregate].groupby(['pos', 'component_name']).std().reset_index()

fig.add_trace(go.Scatter(x=neuron_df.pos, y=neuron_df['loss_post_ablation/loss'], mode='lines', name=f'{neuron_selection}'))
'''
# # Add upper bound line (mean + std)
# fig.add_trace(go.Scatter(
#     x=neuron_df.pos,
#     y=neuron_df['delta_loss_post_ablation'] + neuron_df_std['delta_loss_post_ablation'],
#     mode='lines',
#     line=dict(width=0),
#     showlegend=False
# ))

# Add lower bound line (mean - std)
fig.add_trace(go.Scatter(
    x=neuron_df.pos,
    y=neuron_df['delta_loss_post_ablation'] - neuron_df_std['delta_loss_post_ablation'],
    mode='lines',
    line=dict(width=0),
    fill='tonexty',
    fillcolor='rgba(0,0,255,0.2)',  # Change this to the color you want
    showlegend=False
))'''

# add title
fig.update_layout(title=f"Change in loss when ablating {neuron_selection}. Ablation type: {ablation_type}")
# add label to y axis
fig.update_yaxes(title_text='loss_post_ablation/loss')
# add label to x axis
fig.update_xaxes(title_text='position in the sequence')
fig.add_vline(x=99, line_dash="dash", line_color="black", annotation_text="end of first occurrence of sequence")
fig.show()
# %%
fig = go.Figure()
for i, neuron_list in enumerate(components_to_ablate):
    neuron_selection = '-'.join(neuron_list)

    if neuron_list == entropy_neurons:
        label = 'entropy'
    elif len(neuron_list) == 1 and neuron_list[0] in entropy_neurons:
        label = neuron_list[0]
    else:
        label = f'random {i}'

    neuron_df = agg_results[agg_results.component_name == neuron_selection]
    neuron_df_std = agg_results_std[agg_results_std.component_name == neuron_selection]
    fig.add_trace(go.Scatter(x=neuron_df.pos, y=neuron_df['loss_post_ablation/loss'], mode='lines', name=label))

    #fig.add_trace(go.Scatter(x=neuron_df.pos, y=neuron_df['loss_post_ablation/loss'], mode='lines', name=f'{neuron_name}'))

    # Add upper bound line (mean + std)
    '''fig.add_trace(go.Scatter(
        x=neuron_df.pos,
        y=neuron_df['delta_loss_post_ablation'] + neuron_df_std['delta_loss_post_ablation'],
        mode='lines',
        line=dict(width=0),
        showlegend=False
    ))

    # Add lower bound line (mean - std)
    fig.add_trace(go.Scatter(
        x=neuron_df.pos,
        y=neuron_df['delta_loss_post_ablation'] - neuron_df_std['delta_loss_post_ablation'],
        mode='lines',
        line=dict(width=0),
        fill='tonexty',
        fillcolor='rgba(0,0,255,0.2)',  # Change this to the color you want
        showlegend=False
    ))'''

# add title
fig.update_layout(title=f"Change in loss when ablating neurons simultaneously. Ablation type: {ablation_type}")
# add label to y axis
fig.update_yaxes(title_text='delta_loss_post_ablation')
# add label to x axis
fig.update_xaxes(title_text='position in the sequence')
fig.add_vline(x=99, line_dash="dash", line_color="black", annotation_text="end of first occurrence of sequence")
fig.show()


# %%
fig = go.Figure()
for i, neuron_list in enumerate(components_to_ablate):
    neuron_selection = '-'.join(neuron_list)

    if neuron_list == entropy_neurons:
        label = 'entropy'
    elif len(neuron_list) == 1 and neuron_list[0] in entropy_neurons:
        label = neuron_list[0]
    else:
        label = f'random {i}'

    neuron_df = agg_results[agg_results.component_name == neuron_selection]
    fig.add_trace(go.Scatter(x=neuron_df.pos, y=neuron_df['loss_post_ablation/loss'], mode='lines', name=label))

# add title
fig.update_layout(title=f"Change in loss when ablating neurons simultaneously. Ablation type: {ablation_type}")
# add label to y axis
fig.update_yaxes(title_text='loss_post_ablation/loss')
# add label to x axis
fig.update_xaxes(title_text='position in the sequence')
fig.add_vline(x=99, line_dash="dash", line_color="black", annotation_text="end of first occurrence of sequence")
fig.show()


# %%
fig = go.Figure()
# plot both the loss and the loss_post_ablation 
neuron_df = agg_results[agg_results.component_name == '-'.join(components_to_ablate)]
fig.add_trace(go.Scatter(x=neuron_df.pos, y=neuron_df['loss'], mode='lines', name=f'loss'))
fig.add_trace(go.Scatter(x=neuron_df.pos, y=neuron_df['loss_post_ablation'], mode='lines', name=f'loss_post_ablation'))

# add title
fig.update_layout(title=f"Change in loss when ablating {components_to_ablate} simultaneously. Ablation type: {ablation_type}")
# add label to y axis
fig.update_yaxes(title_text='delta_loss_post_ablation')
# add label to x axis
fig.update_xaxes(title_text='position in the sequence')
fig.add_vline(x=99, line_dash="dash", line_color="black", annotation_text="end of first occurrence of sequence")
fig.show()

# %%
# neuron_selection = '31.7335'
neuron_selection = '23.1332'
induction_multiple_ablation_df_f = induction_multiple_ablation_df[induction_multiple_ablation_df.component_name == neuron_selection]
for neuron_name, mean_value in zip(entropy_neurons, mean_values_on_random_text):
    induction_multiple_ablation_df_f = induction_multiple_ablation_df_f[induction_multiple_ablation_df_f[f'{neuron_name}_activation'] > mean_value]
induction_multiple_ablation_df_f = induction_multiple_ablation_df_f[induction_multiple_ablation_df_f.pos > 100]
#induction_multiple_ablation_df_f = induction_multiple_ablation_df_f[induction_multiple_ablation_df_f['rank_of_correct_token'] == 0]

print(len(induction_multiple_ablation_df_f))
print(len(induction_multiple_ablation_df_f)/len(induction_multiple_ablation_df) * len(components_to_ablate))

induction_multiple_ablation_df_f['1/rank_of_correct_token'] = induction_multiple_ablation_df_f['rank_of_correct_token'].apply(lambda x: 1/(x+1))
fig = px.scatter(induction_multiple_ablation_df_f, x='loss', y='delta_loss_post_ablation', color='pos', hover_data=['str_tokens', 'loss'], opacity=1, color_continuous_scale=plotly.colors.diverging.Picnic, title=f'Change in loss upon ablation of {"-".join(entropy_neurons)} on induction. Ablation type: {ablation_type}', marginal_x='histogram', marginal_y='histogram')
# make it log scale
fig.show()
# %%
induction_multiple_ablation_df_f = induction_multiple_ablation_df[induction_multiple_ablation_df.component_name == '-'.join(entropy_neurons)]
for neuron_name, mean_value in zip(entropy_neurons, mean_values_on_random_text):
    induction_multiple_ablation_df_f = induction_multiple_ablation_df_f[induction_multiple_ablation_df_f[f'{neuron_name}_activation'] > mean_value]
induction_multiple_ablation_df_f = induction_multiple_ablation_df_f[induction_multiple_ablation_df_f.pos > 100]
#induction_multiple_ablation_df_f = induction_multiple_ablation_df_f[induction_multiple_ablation_df_f['rank_of_correct_token'] == 0]
# make histogram of the delta loss, make it density
fig = px.histogram(induction_multiple_ablation_df_f, x='delta_loss_post_ablation', marginal='box', title=f'Change in loss upon ablation of {"-".join(entropy_neurons)} on induction. Ablation type: {ablation_type}')
fig.show()
# %%
# check wheter delta_loss_post_ablation is significantly different from 0
print(ttest_rel(induction_multiple_ablation_df_f['delta_loss_post_ablation'], np.zeros(len(induction_multiple_ablation_df_f))))
print(induction_multiple_ablation_df_f['delta_loss_post_ablation'].mean())

# %%
induction_multiple_ablation_df_f = induction_multiple_ablation_df[induction_multiple_ablation_df.component_name == '-'.join(entropy_neurons)]
for neuron_name, mean_value in zip(entropy_neurons, mean_values_on_random_text):
    induction_multiple_ablation_df_f = induction_multiple_ablation_df_f[induction_multiple_ablation_df_f[f'{neuron_name}_activation'] > mean_value]
induction_multiple_ablation_df_f = induction_multiple_ablation_df_f[induction_multiple_ablation_df_f.pos > 100]
induction_multiple_ablation_df_f = induction_multiple_ablation_df_f[induction_multiple_ablation_df_f['rank_of_correct_token'] > 0]
# make histogram of the delta loss, make it density
fig = px.histogram(induction_multiple_ablation_df_f, x='delta_loss_post_ablation', marginal='box', title=f'Change in loss upon ablation of {"-".join(entropy_neurons)} on induction. Ablation type: {ablation_type}')
fig.show()
# %%
# check wheter delta_loss_post_ablation is significantly different from 0
print(ttest_rel(induction_multiple_ablation_df_f['delta_loss_post_ablation'], np.zeros(len(induction_multiple_ablation_df_f))))
print(induction_multiple_ablation_df_f['delta_loss_post_ablation'].mean())


# %%
induction_multiple_ablation_df['1/rank_of_correct_token'] = induction_multiple_ablation_df['rank_of_correct_token'].apply(lambda x: 1/(x+1))
fig = px.scatter(induction_multiple_ablation_df, x=f'loss', y='loss_post_ablation/loss', color='1/rank_of_correct_token', hover_data=['str_tokens', 'loss'], opacity=1, color_continuous_scale=plotly.colors.diverging.Picnic, title=f'Change in loss upon ablation of {"-".join(components_to_ablate)} on induction. Ablation type: {ablation_type}', marginal_x='histogram', marginal_y='histogram')
# make it log scale
fig.show()

# %%
# =============================================================================
# Ablation of single neurons
# =============================================================================

components_to_ablate = []

# add entropy neurons W_out
if model_name == "Llama-2-7b":
    udark_start = -40
elif model_name == "gpt2-small":
    udark_start = -12

potential_entropy_neurons = get_potential_entropy_neurons_udark(model, select_mode="top_n", select_top_n=20,udark_start=udark_start, udark_end=0, plot_graph=True)

components_to_ablate += selected_neurons

# add random neurons
components_to_ablate += random_neurons[:]

components_to_ablate = [[component] for component in components_to_ablate]

# %%
model.set_use_attn_result(False)
k = 1000
ablation_type = 'mean'
induction_ablation_df = clip_ablate_neurons(list_of_components_to_ablate=components_to_ablate,
                                tokenized_data=induction_tokenized_data,
                                unigram_distrib=unigram_distrib,
                                entropy_df=entropy_df,
                                model=model,
                                k=k,
                                device=device,
                                ablation_type=ablation_type)
# %%
# add is_entropy column
#induction_ablation_df['is_entropy'] = induction_ablation_df['component_name'].isin(potential_entropy_neurons)
induction_ablation_df['delta_loss_post_ablation'] = induction_ablation_df['loss_post_ablation'] - induction_ablation_df['loss']
induction_ablation_df['delta_loss_post_ablation_with_frozen_ln'] = induction_ablation_df['loss_post_ablation_with_frozen_ln'] - induction_ablation_df['loss']
induction_ablation_df['loss_post_ablation/loss'] = induction_ablation_df['loss_post_ablation'] / induction_ablation_df['loss']
induction_ablation_df['loss_post_ablation_with_frozen_ln/loss'] = induction_ablation_df['loss_post_ablation_with_frozen_ln'] / induction_ablation_df['loss']
induction_ablation_df['abs_delta_loss_post_ablation'] = np.abs(induction_ablation_df['loss_post_ablation'] - induction_ablation_df['loss'])
induction_ablation_df['abs_delta_loss_post_ablation_with_frozen_ln'] = np.abs(induction_ablation_df['loss_post_ablation_with_frozen_ln'] - induction_ablation_df['loss'])
induction_ablation_df['delta_entropy'] = induction_ablation_df['entropy_post_ablation'] - induction_ablation_df['entropy']
induction_ablation_df['entropy_post_ablation/entropy'] = induction_ablation_df['entropy_post_ablation'] / induction_ablation_df['entropy']
induction_ablation_df['delta_entropy_with_frozen_ln'] = induction_ablation_df['entropy_post_ablation_with_frozen_ln'] - induction_ablation_df['entropy']
induction_ablation_df['abs_delta_entropy'] = np.abs(induction_ablation_df['delta_entropy'])
induction_ablation_df['abs_delta_entropy_with_frozen_ln'] = np.abs(induction_ablation_df['delta_entropy_with_frozen_ln'])
columns_to_aggregate =list(induction_ablation_df.columns[-17:]) + ['loss', 'pos']
agg_results = induction_ablation_df[columns_to_aggregate].groupby(['pos', 'component_name']).mean().reset_index()
#%%
# get mean values for each entropy neuron
mean_values_on_random_text = {neuron_name : entropy_df[entropy_df.pos < entropy_df.pos.max() / 2][f'{neuron_name}_activation'].mean() for neuron_name in entropy_neurons}

print(mean_values_on_random_text)
# %%

metric = 'entropy_post_ablation/entropy'
#metric = 'loss_post_ablation/loss'

# plot the average loss difference for each neuron for each position as line plots, one line per neuron
fig = go.Figure()
correct_token_df = induction_ablation_df[induction_ablation_df['rank_of_correct_token'] > -1]
correct_token_df = correct_token_df[columns_to_aggregate].groupby(['pos', 'component_name']).mean().reset_index()
for neuron_name in potential_entropy_neurons + random_neurons:
    neuron_df = correct_token_df[correct_token_df.component_name == neuron_name]
    fig.add_trace(go.Scatter(x=neuron_df.pos, y=neuron_df[metric], mode='lines', name=f'{neuron_name}'))

# add title
fig.update_layout(title=f"Metric Change Upon Ablation")
# add label to y axis
fig.update_yaxes(title_text=metric)
# add label to x axis
fig.update_xaxes(title_text='position in the sequence')
fig.add_vline(x=99, line_dash="dash", line_color="black", annotation_text="end of first occurrence of sequence")
fig.show()
# %%
# =============================================================================
# Plot for paper: relative change in entropy
# =============================================================================
metric = 'entropy_post_ablation/entropy'
#metric = 'loss_post_ablation/loss'

# plot the average loss difference for each neuron for each position as line plots, one line per neuron
fig = go.Figure()
correct_token_df = induction_ablation_df[induction_ablation_df['rank_of_correct_token'] > -1]
correct_token_df = correct_token_df[columns_to_aggregate].groupby(['pos', 'component_name']).mean().reset_index()
color_dict = {'potential_entropy_neurons': 'blue', 'random_neurons': 'red'}
i = 2
first_baseline = True
for neuron_name in potential_entropy_neurons + random_neurons:
    neuron_df = correct_token_df[correct_token_df.component_name == neuron_name]
    if neuron_name in potential_entropy_neurons:
        fig.add_trace(go.Scatter(x=neuron_df.pos, y=neuron_df[metric], mode='lines', name=f'{neuron_name}', line=dict(color=qualitative.Plotly[i], dash='dot', width=2.5)))
        i += 1
    else:
        fig.add_trace(go.Scatter(x=neuron_df.pos, y=neuron_df[metric], mode='lines', name='Baselines', line=dict(color=qualitative.Plotly[0]), showlegend=first_baseline))
        first_baseline = False

    #fig.add_trace(go.Scatter(x=neuron_df.pos, y=neuron_df[metric], mode='lines', name=f'{neuron_name}'))

fig.add_vline(x=99, line_dash="dash", line_color="black", annotation_text="start of induct.")

# add title
fig.update_layout(title=f"(b) Change in Entropy Upon Ablation")
# set axis labels
fig.update_xaxes(title_text='Position in Sequence')
# add label to y axis
fig.update_yaxes(title_text='Entropy Post Ablation / Entropy')

# remove padding
fig.update_layout(margin=dict(l=0, r=3, t=30, b=0))

# decrease the width of the plot
rescaling_factor = 1.0
fig.update_layout(width=350*rescaling_factor, height=275/rescaling_factor)

# decrease title font size
fig.update_layout(title_font_size=16)

# save the plot as a pdf
#fig.write_image('../img/induction_entropy_change.pdf')

fig.show()


# %%
neuron = '11.2378'
# plot loss_post_ablation_with_frozen_ln/loss and loss_post_ablation/loss for neuron 23.1332
fig = go.Figure()
correct_token_df = induction_ablation_df[induction_ablation_df['rank_of_correct_token'] == 0]
correct_token_df = correct_token_df[columns_to_aggregate].groupby(['pos', 'component_name']).mean().reset_index()
neuron_df = correct_token_df[correct_token_df.component_name == neuron]
#neuron_df = agg_results[agg_results.component_name == neuron]
fig.add_trace(go.Scatter(x=neuron_df.pos, y=neuron_df['loss_post_ablation_with_frozen_ln/loss'], mode='lines', name=f'loss_post_ablation_with_frozen_ln/loss'))
fig.add_trace(go.Scatter(x=neuron_df.pos, y=neuron_df['loss_post_ablation/loss'], mode='lines', name=f'loss_post_ablation/loss'))
fig.add_trace(go.Scatter(x=neuron_df.pos, y=neuron_df['delta_loss_post_ablation'], mode='lines', name=f'delta_loss_post_ablation'))
fig.add_trace(go.Scatter(x=neuron_df.pos, y=neuron_df['delta_loss_post_ablation_with_frozen_ln'], mode='lines', name=f'delta_loss_post_ablation_with_frozen_ln'))
fig.add_trace(go.Scatter(x=neuron_df.pos, y=neuron_df['delta_entropy'], mode='lines', name=f'delta_entropy'))
fig.add_trace(go.Scatter(x=neuron_df.pos, y=neuron_df['delta_entropy_with_frozen_ln'], mode='lines', name=f'delta_entropy_with_frozen_ln'))
# add title
fig.update_layout(title=f"Ablation of {neuron} in {model_name}. Ablation type: clipped {ablation_type}")



# %%
# plot the average loss difference for each neuron for each position as line plots, one line per neuron
fig = go.Figure()
correct_token_df = induction_ablation_df[induction_ablation_df['rank_of_correct_token'] > -1]
correct_token_df = correct_token_df[columns_to_aggregate].groupby(['pos', 'component_name']).mean().reset_index()
for neuron_name in potential_entropy_neurons + random_neurons:
    neuron_df = correct_token_df[agg_results.component_name == neuron_name]
    fig.add_trace(go.Scatter(x=neuron_df.pos, y=neuron_df['delta_loss_post_ablation'], mode='lines', name=f'{neuron_name}'))

# add title
fig.update_layout(title=f"Change in loss upon ablation on correct tokens")
# add label to y axis
fig.update_yaxes(title_text='delta_loss_post_ablation')
# add label to x axis
fig.update_xaxes(title_text='position in the sequence')
fig.add_vline(x=99, line_dash="dash", line_color="black", annotation_text="end of first occurrence of sequence")
fig.show()

# %%
induction_ablation_df['1/rank_of_correct_token'] = induction_ablation_df['rank_of_correct_token'].apply(lambda x: 1/(x+1))
# filter out the rows in for which 1/rank_of_correct_token is < 1
#induction_ablation_df_f = induction_ablation_df[induction_ablation_df['1/rank_of_correct_token'] >=1]
induction_ablation_df_f = induction_ablation_df[induction_ablation_df.pos > 101]
#induction_ablation_df_f = induction_ablation_df_f[induction_ablation_df_f['31.7335_activation'] >0 ]
neuron_selection = '11.2378'
induction_ablation_df_f = induction_ablation_df_f[induction_ablation_df_f[f'{neuron_selection}_activation'] > mean_values_on_random_text[neuron_selection]]
fig = px.scatter(induction_ablation_df_f[induction_ablation_df_f['component_name'] == neuron_selection], x=f'loss', y='delta_loss_post_ablation', color='1/rank_of_correct_token', hover_data=['str_tokens', 'loss'], opacity=1, color_continuous_scale=plotly.colors.diverging.Picnic, title=f'Change in loss upon ablation of {neuron_selection} on induction.', marginal_y='histogram', ) #range_color=[-0.3, 0.3])

fig.show()
# %%
# =============================================================================
# plots for paper: scatter plot of 2378
# =============================================================================
neuron_selection = '11.2378'
induction_ablation_df['RR'] = induction_ablation_df['rank_of_correct_token'].apply(lambda x: 1/(x+1))
# filter out the rows in for which 1/rank_of_correct_token is < 1
#induction_ablation_df_f = induction_ablation_df[induction_ablation_df['1/rank_of_correct_token'] >=1]
induction_ablation_df_f = induction_ablation_df[induction_ablation_df.pos < 101]
#induction_ablation_df_f = induction_ablation_df_f[induction_ablation_df_f['31.7335_activation'] >0 ]
neuron_selection = '11.2378'
induction_ablation_df_f = induction_ablation_df_f[induction_ablation_df_f[f'{neuron_selection}_activation'] > mean_values_on_random_text[neuron_selection]]
fig = px.scatter(induction_ablation_df_f[induction_ablation_df_f['component_name'] == neuron_selection], x=f'loss', y='delta_loss_post_ablation', color='RR', hover_data=['str_tokens', 'loss'], opacity=1, color_continuous_scale='Picnic', title=f'Change in loss upon ablation of {neuron_selection} on induction.', labels={'loss': 'Loss', 'delta_loss_post_ablation': 'ΔLoss Post Ablation', 'RR': '1/RC'})

# add title
fig.update_layout(title=f"(a) 11.2378: Change in Loss Upon Abl.")
# set axis labels
#fig.update_xaxes(title_text='Loss')
# add label to y axis
#fig.update_yaxes(title_text='Delta Loss Post Ablation')

# remove padding
fig.update_layout(margin=dict(l=0, r=3, t=30, b=0))

# decrease the width of the plot
rescaling_factor = 1
fig.update_layout(width=350*rescaling_factor, height=275/rescaling_factor)

# decrease title font size
fig.update_layout(title_font_size=16)

# save the plot as a pdf
fig.write_image('../img/general_loss_change_2378.pdf')

fig.show()

# %% 

# %%
neuron_selection = '11.2378'
induction_ablation_df_f = induction_ablation_df[induction_ablation_df.component_name == neuron_selection]
induction_ablation_df_f = induction_ablation_df_f[induction_ablation_df_f[f'{neuron_selection}_activation'] > mean_values_on_random_text[neuron_selection]]
induction_ablation_df_f = induction_ablation_df_f[induction_ablation_df_f.pos > 100]
#induction_ablation_df_f = induction_ablation_df_f[induction_ablation_df_f['rank_of_correct_token'] == 0]
# make histogram of the delta loss, make it density
fig = px.histogram(induction_ablation_df_f, x='delta_loss_post_ablation', marginal='box', title=f'Change in loss upon ablation of {neuron_selection} on induction. Ablation type: {ablation_type}')
fig.show()
# %%
# check wheter delta_loss_post_ablation is significantly different from 0
print(ttest_rel(induction_ablation_df_f['delta_loss_post_ablation'], np.zeros(len(induction_ablation_df_f))))
print(induction_ablation_df_f['delta_loss_post_ablation'].mean())

# %%
neuron_selection = '11.2910'
induction_ablation_df_f = induction_ablation_df[(induction_ablation_df.component_name == neuron_selection) & (induction_ablation_df.pos > 100)]
fig = px.scatter(induction_ablation_df_f, x=f'{neuron_selection}_activation', y='delta_loss_post_ablation', color='1/rank_of_correct_token', hover_data=['str_tokens', 'loss'], opacity=1, color_continuous_scale=plotly.colors.diverging.Picnic, title=f'Change in loss upon ablation of {neuron_selection} on induction. Ablation type: {ablation_type}', marginal_y='histogram', ) #range_color=[-0.3, 0.3])
# add vertical line at mean value
fig.add_vline(x=mean_values_on_random_text[neuron_selection], line_dash="dash", line_color="black", annotation_text="mean value")
fig.show()

# %%
# for neuron 11.2748, plot both loss and loss_post_ablation
fig = go.Figure()
neuron_name = '11.2910'
neuron_df = agg_results[agg_results.component_name == neuron_name]
fig.add_trace(go.Scatter(x=neuron_df.pos, y=neuron_df['loss'], mode='lines', name=f'loss'))
fig.add_trace(go.Scatter(x=neuron_df.pos, y=neuron_df['loss_post_ablation'], mode='lines', name=f'loss_post_ablation'))
fig.add_trace(go.Scatter(x=neuron_df.pos, y=neuron_df['delta_loss_post_ablation'], mode='lines', name=f'delta_loss_post_ablation'))

# add title
fig.update_layout(title=f"Change in loss upon ablation of {neuron_name} on induction. Ablation type: {ablation_type}")
# add label to y axis
fig.update_yaxes(title_text='loss')
# add label to x axis
fig.update_xaxes(title_text='position in the sequence')
fig.add_vline(x=99, line_dash="dash", line_color="black", annotation_text="end of first occurrence of sequence")
fig.show()

# %%
