# %%
import sys
sys.path.append('../')
import tqdm
import random
import numpy as np
import pandas as pd
import torch
import plotly.express as px
import plotly.graph_objects as go
from plotly.express.colors import qualitative
from utils import load_model_from_tl_name, get_pile_unigram_distribution
# %%
output_dir = 'results'
model_name = 'gpt2-small'
dataset = 'stas/c4-en-10k'
data_range_start = 0
data_range_end = 1000
# k = 100 
k = 10

save_path = f'./{output_dir}/{model_name}/unigram/{dataset.replace("/","_")}_{data_range_start}-{data_range_end}/k{k}.feather'
final_df = pd.read_feather(save_path)
# %%
final_df['delta_loss'] = final_df['loss_post_ablation'] - final_df['loss']
final_df['delta_loss_with_frozen_unigram'] = final_df['loss_post_ablation_with_frozen_unigram'] - final_df['loss']
final_df['abs_delta_loss_post_ablation'] = np.abs(final_df['loss_post_ablation'] - final_df['loss'])
final_df['abs_delta_loss_post_ablation_with_frozen_unigram'] = np.abs(final_df['loss_post_ablation_with_frozen_unigram'] - final_df['loss'])
final_df['delta_entropy'] = final_df['entropy_post_ablation'] - final_df['entropy']
if 'kl_divergence_before' in final_df.columns:
    print('kl_divergence_before found')
    final_df['kl_from_unigram_diff'] = final_df['kl_divergence_after'] - final_df['kl_divergence_before']
    final_df['kl_from_unigram_diff_with_frozen_unigram'] = final_df['kl_divergence_after_frozen_unigram'] - final_df['kl_divergence_before']
    final_df['abs_kl_from_unigram_diff'] = final_df['kl_from_unigram_diff'].abs()
final_df['abs_kl_from_unigram_diff'] = final_df['kl_from_unigram_diff'].abs()

unigram_neurons_dict = {
    'pythia-410m':  ['23.87', '23.417', '23.730', '23.2017','23.3412'],
}
entropy_neurons_dict = {
    'pythia-410m': ['23.1829', '23.1332', '23.2455', '23.3562', '23.2756', '23.2362'],
}

unigram_neurons = unigram_neurons_dict.get(model_name, [])
entropy_neurons = entropy_neurons_dict.get(model_name, [])

final_df['is_unigram'] = final_df['component_name'].isin(unigram_neurons).astype(bool)
final_df['is_entropy'] = final_df['component_name'].isin(entropy_neurons).astype(bool)


columns_to_aggregate =list(final_df.columns[8:]) + ['loss']
print(columns_to_aggregate)
agg_results = final_df[columns_to_aggregate].groupby('component_name').mean().reset_index()

# make scatter plot of delta_loss and delta_loss_with_frozen_unigram for each neuron
agg_results['delta_loss-delta_loss_with_frozen_unigram'] = agg_results['delta_loss'] - agg_results['delta_loss_with_frozen_unigram']
agg_results['abs_delta_loss-abs_delta_loss_with_frozen_unigram'] = agg_results['abs_delta_loss_post_ablation'] - agg_results['abs_delta_loss_post_ablation_with_frozen_unigram']
# %%
# make scatter plot of delta_loss and delta_loss_with_frozen_unigram for each neuron
agg_results['delta_loss-delta_loss_with_frozen_unigram'] = agg_results['delta_loss'] - agg_results['delta_loss_with_frozen_unigram']
agg_results['abs_delta_loss-abs_delta_loss_with_frozen_unigram'] = agg_results['abs_delta_loss_post_ablation'] - agg_results['abs_delta_loss_post_ablation_with_frozen_unigram']
agg_results['1-abs_delta_loss_with_frozen_unigram/abs_delta_loss'] = 1 - agg_results['abs_delta_loss_post_ablation_with_frozen_unigram'] / agg_results['abs_delta_loss_post_ablation']


# %%
# =============================================================================
# plot for paper
# =============================================================================

device='cuda'

model, tokenizer = load_model_from_tl_name(model_name, device)
model = model.to(device)

# turn off gradient computation
model.eval()
for param in model.parameters():
    param.requires_grad = False

if 'pythia' in model_name:
    unigram_distrib = get_pile_unigram_distribution(device=device, file_path='../datasets/pythia-unigrams.npy')
elif 'gpt' in model_name:
    unigram_distrib = get_pile_unigram_distribution(device=device, file_path='../datasets/gpt2-small-unigrams_openwebtext-2M_rows_500000.npy', pad_to_match_W_U=False)
else:
    raise Exception(f'No unigram distribution for {model_name}')

unigram_logits = unigram_distrib.log() - unigram_distrib.log().mean()

# %%
cosine_sims_in_vocab = torch.zeros((model.cfg.n_layers, model.cfg.d_mlp))
for layer_idx in tqdm.tqdm(range(0,model.cfg.n_layers)):
    neurons = model.W_out[layer_idx, :].to(device)
    cosine_sim = (neurons @ model.W_U) @ unigram_logits / ((neurons @ model.W_U).norm(dim=-1) * unigram_logits.norm())
    cosine_sims_in_vocab[layer_idx, :] = cosine_sim
# %%
# make scatter plot of cosine sim with unigram dir and delta_loss-delta_loss_with_frozen_unigram
agg_results['cos_sim_with_unigram'] = [cosine_sims_in_vocab[int(name.split('.')[0]), int(name.split('.')[1])].item() for name in agg_results['component_name']]
# %%
# add a new column 'Neuron Type' that is 'Unigram' if 'is_unigram' is True, and 'Entropy' if 'is_entropy' is True
conditions = [
    (agg_results['is_unigram'] == True),
    (agg_results['is_entropy'] == True)
]

choices = ['Token Frequency', 'Entropy']

agg_results['Neuron Type'] = np.select(conditions, choices, default='Normal')

tf_color = qualitative.Plotly[2]

x_axis = '1-abs_delta_loss_with_frozen_unigram/abs_delta_loss'
y_axis = 'abs_kl_from_unigram_diff'
agg_results['Total Eff.'] = agg_results['abs_delta_loss_post_ablation']
fig = px.scatter(agg_results, y=y_axis, x=x_axis, hover_data=['component_name'], color='Neuron Type', color_discrete_map={'Normal': qualitative.Plotly[0], 'Entropy': qualitative.Plotly[1], 'Token Frequency': tf_color})

# Add text labels
entropy_neuron_indices = [int(neuron.split('.')[1]) for neuron in unigram_neurons]
# Add a new column 'neuron_index' to the DataFrame
for neuron_index in entropy_neuron_indices:
    entropy_df = agg_results[agg_results['component_name'] == f'23.{neuron_index}']
    fig.add_trace(go.Scatter(x=entropy_df[x_axis]-0.01, y=entropy_df[y_axis], mode='text', text=str(neuron_index), textposition='bottom left', showlegend=False, textfont=dict(color=tf_color)))

# move the legend to the bottom
fig.update_layout(legend=dict(
    orientation='h',
    yanchor='bottom',
    y=-0.6,
    xanchor='right',
    x=0.9
))

# set labels
fig.update_yaxes(title_text='Avg. |Δ D<sub>KL</sub>(P<sub>model</sub>||P<sub>freq</sub>)|')
fig.update_xaxes(title_text='1 - DE<sub>freq</sub>/TE')

# set x and y limits
fig.update_xaxes(range=[-0.25, 0.62])

# set title
fig.update_layout(title=f'(c) Token Frequency Neurons')

# remove padding
fig.update_layout(margin=dict(l=0, r=3, t=30, b=0))

# decrease the width of the plot
fig.update_layout(width=350, height=275)

# decrease title font size
fig.update_layout(title_font_size=16)

fig.show()
# %%
# =============================================================================
# plot for paper: box plot of KL from unigram
# =============================================================================
output_dir = 'results'
model_name = 'pythia-410m'
dataset = 'stas/c4-en-10k'
data_range_start = 0
data_range_end = 1000
# k = 50
k = 10

save_path = f'./{output_dir}/{model_name}/unigram/{dataset.replace("/","_")}_{data_range_start}-{data_range_end}/k{k}.feather'
final_df = pd.read_feather(save_path)

final_df['delta_loss'] = final_df['loss_post_ablation'] - final_df['loss']
final_df['delta_loss_with_frozen_unigram'] = final_df['loss_post_ablation_with_frozen_unigram'] - final_df['loss']
final_df['abs_delta_loss_post_ablation'] = np.abs(final_df['loss_post_ablation'] - final_df['loss'])
final_df['abs_delta_loss_post_ablation_with_frozen_unigram'] = np.abs(final_df['loss_post_ablation_with_frozen_unigram'] - final_df['loss'])
final_df['delta_entropy'] = final_df['entropy_post_ablation'] - final_df['entropy']

if 'kl_divergence_before' in final_df.columns:
    print('kl_divergence_before found')
    final_df['kl_from_unigram_diff'] = final_df['kl_divergence_after'] - final_df['kl_divergence_before']
    final_df['kl_from_unigram_diff_with_frozen_unigram'] = final_df['kl_divergence_after_frozen_unigram'] - final_df['kl_divergence_before']
    final_df['abs_kl_from_unigram_diff'] = final_df['kl_from_unigram_diff'].abs()
columns_to_aggregate =list(final_df.columns[-17:]) + ['loss']
agg_results = final_df[columns_to_aggregate].groupby('component_name').mean().reset_index()

neurons_to_highlight_dict = {
    'pythia-410m':  ['23.87', '23.417', '23.730', '23.2017', '23.2952','23.3412'],
}

neurons_to_highlight = neurons_to_highlight_dict[model_name]

final_df['is_unigram'] = final_df['component_name'].isin(entropy_neurons)
columns_to_aggregate =list(final_df.columns[-17:]) + ['loss']
agg_results = final_df[columns_to_aggregate].groupby('component_name').mean().reset_index()


# %%
final_df['Total Effect'] = final_df['abs_delta_loss_post_ablation']
final_df['Direct Effect'] = final_df['abs_delta_loss_post_ablation_with_frozen_unigram']

# sample 10 random neurons
random_neurons = random.sample(list(final_df[~final_df['is_unigram']]['component_name'].unique()), 10)

neurons_to_keep = entropy_neurons + random_neurons

final_df_filtered = final_df[final_df['component_name'].isin(neurons_to_keep)]
final_df_filtered['component_or_baseline'] = np.where(final_df_filtered['is_unigram'], final_df_filtered['component_name'], 'Random')
# %%
# Reshape DataFrame from wide format to long format
melted_results = pd.melt(final_df_filtered, id_vars='component_or_baseline', value_vars=['Total Effect', 'Direct Effect'], var_name='Effect Type', value_name='Average Absolute Loss Difference')

# %%
new_df = melted_results.copy()
# Define a function to remove outliers
def remove_outliers(df, column, component_or_baseline):
    for eff in ['Total Effect', 'Direct Effect']:
        for component in component_or_baseline:
            Q1 = df[(df['component_or_baseline'] == component) & (df['Effect Type'] == eff)][column].quantile(0.25)
            Q3 = df[(df['component_or_baseline'] == component) & (df['Effect Type'] == eff)][column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df[(df['component_or_baseline'] == component) & (df['Effect Type'] == eff)] = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return new_df

# Remove outliers from 'Average Absolute Loss Difference'
melted_results_no_outliers = remove_outliers(new_df, 'Average Absolute Loss Difference', melted_results['component_or_baseline'].unique())

# %%
# Create box plot
melted_results_no_outliers = melted_results_no_outliers.sort_values(['component_or_baseline', 'Effect Type'], ascending=[True, False])
fig = px.box(melted_results_no_outliers, x='component_or_baseline', y='Average Absolute Loss Difference', color='Effect Type', title='(d) Change in Loss Upon Ablation', labels={'component_or_baseline': 'Neuron Name', 'Average Absolute Loss Difference': 'Avg. Abs. Loss Diff.'}, points=False, color_discrete_map={'Total Effect': px.colors.qualitative.Plotly[0], 'Direct Effect': px.colors.qualitative.Plotly[2]})

# add vertical line at agg_results.is_entropy.sum()
fig.add_vline(x=agg_results.is_unigram.sum()-0.5, line_dash="dash", line_color="black")

# move the legend to the bottom
fig.update_layout(legend=dict(
    orientation='h',
    yanchor='bottom',
    y=-0.7,
    xanchor='right',
    x=0.9
))

# remove padding
fig.update_layout(margin=dict(l=0, r=20, t=30, b=0))

# decrease the width of the plot
fig.update_layout(width=350, height=275)

# decrease title font size
fig.update_layout(title_font_size=16)

fig.show()
# %%
