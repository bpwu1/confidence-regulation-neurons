# %% 
import os
import sys
sys.path.append('../')

import random 
import pandas as pd
from datasets import load_dataset
from utils import load_model_from_tl_name, get_potential_entropy_neurons_udark, get_entropy_activation_df, generate_induction_examples, generate_induction_df, get_induction_data_and_token_df, bos_ablate_attn_heads
import plotly.express as px
import plotly.graph_objects as go
import torch
import numpy as np
import transformer_lens.utils as tl_utils
import neel.utils as nutils

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
model, tokenizer = load_model_from_tl_name(model_name, device, transformers_cache_dir)
model = model.to(device)

# %%
data = load_dataset("stas/c4-en-10k", split='train')
first_1k = data.select([i for i in range(0, 1000)])

tokenized_data = tl_utils.tokenize_and_concatenate(first_1k, tokenizer, max_length=256, column_name='text')
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
seq_len = 100
use_natural_text = False # this is used for calculating induction score of attn heads

if model_name == "Llama-2-7b":
    batch_size = 2
else: 
    batch_size = 16

num_examples = 1024
num_seq = num_examples 
num_batches = num_examples // batch_size
threshold = 0.7 # for an induction head score
# %%

#calculating induction score
induction_tokens = generate_induction_examples(model, tokenizer, seq_length=seq_len, num_examples=num_examples, seed=SEED, device='cuda', use_natural_text=use_natural_text, use_separator=None, num_repetitions=1)

induction_df = generate_induction_df(model, induction_tokens, batch_size=batch_size, num_batches=num_batches, threshold=threshold)
induction_df = induction_df.reset_index()

induction_df['layer_head_index'] = induction_df['labels'].apply(lambda x: model.cfg.n_heads * int(x.split('L')[1].split('H')[0]) + int(x.split('H')[1]))
induction_df.set_index('layer_head_index', inplace=True, drop=False)
induction_df.sort_index(inplace=True)

induction_heads = induction_df[induction_df['is_induction'] == True].sort_values("induction_scores", ascending=False)['labels'].tolist()
print("Induction heads = ", ", ".join(induction_heads))

fig = px.bar(induction_df, x='labels', y='induction_scores', title=f"Induction scores for threshold {threshold}")
fig.show()

# %%
# =============================================================================
# Induction
# =============================================================================

# smaller dataset for speed
smaller_seq_len = 50
smaller_num_examples = 100
induction_tokenized_data, induction_token_df = get_induction_data_and_token_df(model, tokenizer, seq_length=smaller_seq_len, num_examples=smaller_num_examples, seed=SEED, device='cuda', use_natural_text=True, use_separator=None, num_repetitions=1)


# %%
n_of_baselines = 5
# sample n_of_baselines random lists of neurons to ablate
random_baselines = []
n_single_neuron_baselines = 5
random_neuron_indices = random.sample(possible_random_neuron_indices, n_single_neuron_baselines)
random_neurons = [f"{entropy_neuron_layer}.{i}" for i in random_neuron_indices]


# %%
# check if random_baseline is a defined variable
components_to_track = entropy_neurons + random_neurons
all_neuron_names = [f"{entropy_neuron_layer}.{i}" for i in range(model.cfg.d_mlp)]
all_neuron_names_with_activation = [f"{neuron_name}_activation" for neuron_name in all_neuron_names]
selected_neurons = entropy_neurons + random_neurons


unigram_distrib = None


entropy_dim_layer = model.cfg.n_layers - 1
component_output_to_cache = {'resid_post': []}
entropy_df, resid_dict = get_entropy_activation_df(all_neuron_names,
                                                   induction_tokenized_data,
                                                   induction_token_df,
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
plotting_num_seq = len(entropy_df['batch'].unique())
plotting_seq_len = entropy_df['pos'].max() /2 #max bos guranteed to be even number
# average neuron activations across sequences
neuron_activations_cache_dict = {}
for neuron_name in entropy_neurons +random_neurons:
    neuron_activations_cache_dict[neuron_name] = entropy_df[[f'{neuron_name}_activation']].values.reshape((plotting_num_seq, -1)).mean(axis=0)

# average entropy across sequences
entropy = entropy_df['entropy'].values.reshape((plotting_num_seq, -1)).mean(axis=0)

# average ln_final_scale across sequences
ln_final_scale = entropy_df['ln_final_scale'].values.reshape((plotting_num_seq, -1)).mean(axis=0)

#kl_from_unigram = entropy_df['kl_from_unigram'].values.reshape((num_seq, -1)).mean(axis=0)

# average loss across sequences
loss = entropy_df['loss'].values.reshape((plotting_num_seq, -1)).mean(axis=0)
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
fig.add_vline(x=plotting_seq_len, line_dash="dash", line_color="black", annotation_text="end of first occurrence of sequence")
# add title
fig.update_layout(title=f"Average activations, entropy, and loss across {plotting_num_seq} sequences. Model: {model_name}")
# set axis labels
fig.update_xaxes(title_text='Position in sequence')
activation_fig = fig
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
    fig.add_trace(go.Scatter(x=x_axis, y=neuron_activations, mode='lines', name=f'{neuron_name}'))

#fig.add_trace(go.Scatter(x=x_axis, y=ln_final_scale, mode='lines', name='ln_final_scale'))
#fig.add_trace(go.Scatter(x=x_axis, y=kl_from_unigram, mode='lines', name='kl_from_unigram'))
# add vertical lines to indicate the end of the first sequence
fig.add_vline(x=plotting_seq_len, line_dash="dash", line_color="black", annotation_text="start of induction", line_width=2)
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
mean_values_on_random_text = {neuron_name : entropy_df[entropy_df.pos < plotting_seq_len+1][f'{neuron_name}_activation'].mean() for neuron_name in entropy_neurons + random_neurons}
mean_values_on_induction = {neuron_name : entropy_df[entropy_df.pos >= plotting_seq_len +1][f'{neuron_name}_activation'].mean() for neuron_name in entropy_neurons + random_neurons}

# seq_len + 1 because of bos token
# %%

# %%
# =============================================================================
# Induction: BOS Ablations
# =============================================================================
def bos_ablate_components(
    list_of_components_to_ablate, 
    tokenized_data, 
    entropy_df, 
    model,              
    select="all",
    k=50,
    device=device,
    cache_pre_activations=False,
    compute_resid_norm_change=False, # requires entropy_df to have cached pre-ablation norm. currently hard-coded to do "final_layer".resid_post_norm 
    subtract_b_U=False,
    seed = 42,
    compute_kl = False,
    save_single_df = False):

    final_df = None 

    for components_to_ablate in list_of_components_to_ablate: 
        print(f"ablate {components_to_ablate}")
        ablation_df = bos_ablate_attn_heads(
            attn_head_names=components_to_ablate,
            tokenized_data=tokenized_data,
            entropy_df=entropy_df.copy(),
            model=model,
            select=select,
            k=k,
            device=device,
            cache_pre_activations=cache_pre_activations,
            compute_resid_norm_change=compute_resid_norm_change,
            subtract_b_U=subtract_b_U,
            seed=seed,
            compute_kl=compute_kl
            )
        
        ablation_df['component_name'] = '-'.join(components_to_ablate)

        if save_single_df: 
            single_df = ablation_df.reset_index()
            single_df.to_feather(f"../large_scale_exp/results/{model_name}/{model_name}_bos_ablation_df_seq{smaller_seq_len}_k{smaller_num_examples}_{'-'.join(components_to_ablate)}.feather")
        # stack the df_to_append to final_df
        if final_df is None:
            final_df = ablation_df
        else:
            final_df = pd.concat([final_df, ablation_df])
        
    return final_df
    

# %%
k = 1000
#whether to ablate k rows of induction_df, or all rows
select_type= "all" 
# select_type = 'fraction'

list_of_components_to_ablate = [induction_heads[:1], induction_heads[:2], induction_heads[:3]] 
list_of_components_to_ablate += [induction_heads[1:2], induction_heads[2:3]]

# gpt2-small heads
single_head_baselines =  ["L6H1", "L5H7", "L5H6", "L5H8", "L6H10", "L6H3"] 
multi_head_baselines = ["L5H7-L5H6", "L6H1-L5H7", "L5H8-L6H3", "L5H7-L5H6-L5H8", "L6H1-L5H6-L6H10", "L5H7-L6H10-L6H3"]

list_of_components_to_ablate +=  single_head_baselines #baselines 
list_of_components_to_ablate += multi_head_baselines #baselines

list_of_components_to_ablate = [induction_heads[:1]] + [induction_heads[1:2], induction_heads[2:3]]

bos_ablation_df = bos_ablate_components(
    list_of_components_to_ablate=list_of_components_to_ablate,
    tokenized_data=induction_tokenized_data,
    entropy_df=induction_token_df,
    model=model,
    select=select_type,
    k=k,
    device=device,
    cache_pre_activations=False,
    compute_resid_norm_change=False, 
    subtract_b_U=False,
    seed = SEED,
    compute_kl = False,
    save_single_df = True
    )
    # add is_entropy column
# %%
tmp_df = bos_ablation_df.reset_index()
tmp_df.to_feather(f"../large_scale_exp/results/gpt2-small/{model_name}_bos_ablation_df_seq{smaller_seq_len}_k{smaller_num_examples}.feather")
# %%