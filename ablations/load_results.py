# %%
import sys
sys.path.append('../')
import pandas as pd
import plotly.express as px
from utils import *
from scipy.stats import spearmanr
import plotly.graph_objects as go
from plotly.express.colors import qualitative
import random
# %%
output_dir = './results'
model = 'gpt2-small'
dataset = 'stas/c4-en-10k'
data_range_start = 0
data_range_end = 1000
# k = 50
k = 10

save_path = f'./{output_dir}/{model}/ln_scale/{dataset.replace("/","_")}_{data_range_start}-{data_range_end}/k{k}.feather'
final_df = pd.read_feather(save_path)
# %%
final_df['delta_loss'] = final_df['loss_post_ablation'] - final_df['loss']
final_df['delta_loss_with_frozen_ln'] = final_df['loss_post_ablation_with_frozen_ln'] - final_df['loss']
final_df['abs_delta_loss_post_ablation'] = np.abs(final_df['loss_post_ablation'] - final_df['loss'])
final_df['abs_delta_loss_post_ablation_with_frozen_ln'] = np.abs(final_df['loss_post_ablation_with_frozen_ln'] - final_df['loss'])
final_df['delta_entropy'] = final_df['entropy_post_ablation'] - final_df['entropy']
final_df['delta_entropy_with_frozen_ln'] = final_df['entropy_post_ablation_with_frozen_ln'] - final_df['entropy']
columns_to_aggregate =list(final_df.columns[-17:]) + ['loss']
agg_results = final_df[columns_to_aggregate].groupby('component_name').mean().reset_index()
# %%
# make scatter plot of delta_loss and delta_loss_with_frozen_ln for each neuron
agg_results['delta_loss-delta_loss_with_frozen_ln'] = agg_results['delta_loss'] - agg_results['delta_loss_with_frozen_ln']
agg_results['abs_delta_loss-abs_delta_loss_with_frozen_ln'] = agg_results['abs_delta_loss_post_ablation'] - agg_results['abs_delta_loss_post_ablation_with_frozen_ln']
fig = px.scatter(agg_results, x='abs_delta_loss_post_ablation', y='abs_delta_loss-abs_delta_loss_with_frozen_ln', title=f'Model: {model}. Mean ablation on k={k} sequences', color='delta_entropy', color_continuous_scale='viridis', hover_data=['component_name'])
fig.show()
# %%
# make scatter plot of delta_loss and delta_loss_with_frozen_ln for each neuron
agg_results['delta_loss_with_frozen_ln/delta_loss'] = agg_results['delta_loss_with_frozen_ln'] / agg_results['delta_loss']
agg_results['abs_delta_loss_post_ablation_with_frozen_ln/abs_delta_loss_post_ablation'] = agg_results['abs_delta_loss_post_ablation_with_frozen_ln'] / agg_results['abs_delta_loss_post_ablation']
fig = px.scatter(agg_results, x='abs_delta_loss_post_ablation', y='abs_delta_loss_post_ablation_with_frozen_ln/abs_delta_loss_post_ablation', title=f'Model: {model}. Mean ablation on k={k} sequences', color='delta_entropy', color_continuous_scale='viridis', hover_data=['component_name'])
fig.show()
# %%
# make a scatter plot of loss and loss_post_ablation
fig = px.scatter(final_df, x='loss', y='loss_post_ablation', title=f'Model: {model}. Mean ablation on k={k} sequences', color='component_name', color_continuous_scale='viridis', hover_data=['component_name'])
fig.show()

# %%
lowest_composing_neurons_dict = {
    'stanford-gpt2-small-a': ['11.3030', '11.2859', '11.2546',  '11.2748',],
    'gpt2-small': ['11.584', '11.2378', '11.2870', '11.2123', '11.1611', '11.2910'],
}

entropy_neurons = lowest_composing_neurons_dict[model]

final_df['is_entropy'] = final_df['component_name'].isin(entropy_neurons)
columns_to_aggregate =list(final_df.columns[-17:]) + ['loss']
agg_results = final_df[columns_to_aggregate].groupby('component_name').mean().reset_index()

# %%
# make scatter plot of delta_loss and delta_loss_with_frozen_ln for each neuron
agg_results['delta_loss-delta_loss_with_frozen_ln'] = agg_results['delta_loss'] - agg_results['delta_loss_with_frozen_ln']
agg_results['abs_delta_loss-abs_delta_loss_with_frozen_ln'] = agg_results['abs_delta_loss_post_ablation'] - agg_results['abs_delta_loss_post_ablation_with_frozen_ln']
fig = px.scatter(agg_results, x='abs_delta_loss_post_ablation', y='abs_delta_loss-abs_delta_loss_with_frozen_ln', title=f'Model: {model}. Mean ablation on k={k} sequences', color='is_entropy', color_continuous_scale='Picnic', hover_data=['component_name'])
fig.show()
# %%
# =============================================================================
# box plot of effects
# =============================================================================

# sample 10 random neurons
random_neurons = random.sample(list(final_df[~final_df['is_entropy']]['component_name'].unique()), 100)

neurons_to_keep = entropy_neurons + random_neurons

final_df_filtered = final_df[final_df['component_name'].isin(neurons_to_keep)]
final_df_filtered['component_or_baseline'] = np.where(final_df_filtered['is_entropy'], final_df_filtered['component_name'], 'Random')
# %%

final_df_filtered['Total'] = final_df_filtered['abs_delta_loss_post_ablation']
final_df_filtered['Direct'] = final_df_filtered['abs_delta_loss_post_ablation_with_frozen_ln']
# Add a new column 'component_or_baseline' that combines the components for which 'is_entropy' is false
final_df_filtered['component_or_baseline'] = np.where(final_df_filtered['is_entropy'], final_df_filtered['component_name'], 'Random')

# Reshape DataFrame from wide format to long format
melted_results = pd.melt(final_df_filtered, id_vars='component_or_baseline', value_vars=['Total', 'Direct'], var_name='Effect Type', value_name='Average Absolute Loss Difference')

# %%
new_df = melted_results.copy()

# Define a function to remove outliers
def remove_outliers(df, column, component_or_baseline):
    for eff in ['Total', 'Direct']:
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
melted_results_no_outliers = melted_results_no_outliers.sort_values(['component_or_baseline', 'Effect Type'], ascending=[True, False])
# Create box plot
fig = px.box(melted_results_no_outliers, x='component_or_baseline', y='Average Absolute Loss Difference', color='Effect Type', title='(c) Total & Direct Effects', labels={'component_or_baseline': 'Neuron Name', 'Average Absolute Loss Difference' : 'Average Effect', 'Total' : 'Total'}, points=False, color_discrete_map={'Total': qualitative.Plotly[0], 'Direct':  qualitative.Plotly[2]})

# add vertical line at agg_results.is_entropy.sum()
fig.add_vline(x=agg_results.is_entropy.sum()-0.5, line_dash="dash", line_color="black")

# incline the x-axis labels
fig.update_layout(xaxis_tickangle=-45)

# remove padding
fig.update_layout(margin=dict(l=0, r=3, t=30, b=0))

# decrease the width of the plot
fig.update_layout(width=350, height=275)

# decrease title font size
fig.update_layout(title_font_size=16)

# move the legend to the bottom
fig.update_layout(legend=dict(orientation="h", yanchor="bottom", x=0.82, y=-0.77, xanchor="right"))

fig.show()


# %%
# =============================================================================
# LLaMA: behavior of neurons on full distribution
# =============================================================================
output_dir = './results'
model = 'Llama-2-7b'
dataset = 'stas/c4-en-10k'
data_range_start = 0
data_range_end = 1000
k = 30

save_path = f'./{output_dir}/{model}/ln_scale/{dataset.replace("/","_")}_{data_range_start}-{data_range_end}/k{k}.feather'
final_df = pd.read_feather(save_path)
# %%
final_df['delta_loss'] = final_df['loss_post_ablation'] - final_df['loss']
final_df['delta_loss_with_frozen_ln'] = final_df['loss_post_ablation_with_frozen_ln'] - final_df['loss']
final_df['abs_delta_loss_post_ablation'] = np.abs(final_df['loss_post_ablation'] - final_df['loss'])
final_df['abs_delta_loss_post_ablation_with_frozen_ln'] = np.abs(final_df['loss_post_ablation_with_frozen_ln'] - final_df['loss'])
final_df['delta_entropy'] = final_df['entropy_post_ablation'] - final_df['entropy']
final_df['entropy_post_ablation/entropy'] = final_df['entropy_post_ablation'] / final_df['entropy']
final_df['loss_post_ablation/loss'] = final_df['loss_post_ablation'] / final_df['loss']
final_df['abs_delta_entropy'] = np.abs(final_df['entropy_post_ablation'] - final_df['entropy'])
final_df['delta_entropy_with_frozen_ln'] = final_df['entropy_post_ablation_with_frozen_ln'] - final_df['entropy']
final_df['1/rank_of_correct_token'] = 1 / (final_df['rank_of_correct_token'] + 1)
if 'top_logits_post_ablation' in final_df.columns:
    final_df['pred_change'] = (final_df['top_logits_post_ablation'] != final_df['pred']).astype(int)
if 'rank_of_correct_token_post_ablation' in final_df.columns:
    print('Rank of correct token post ablation')
    final_df['1/rank_of_correct_token_post_ablation'] = 1 / (final_df['rank_of_correct_token_post_ablation'] + 1)
    final_df['1/rank_of_correct_token_before_ablation'] = 1 / (final_df['rank_of_correct_token_before_ablation'] + 1)
    final_df['change_in_rank'] = np.abs(final_df['1/rank_of_correct_token_post_ablation'] - final_df['1/rank_of_correct_token_before_ablation'])

columns_to_aggregate =list(final_df.columns[8:]) + ['loss']

agg_results = final_df[columns_to_aggregate].groupby('component_name').mean().reset_index()
agg_results['1-abs_delta_loss_post_ablation_with_frozen_ln/abs_delta_loss_post_ablation'] = 1 - agg_results['abs_delta_loss_post_ablation_with_frozen_ln'] / agg_results['abs_delta_loss_post_ablation']

# %%
neuron_name = '11.2378'
neuron_df = final_df[final_df['component_name'] == neuron_name]
# get mean activation values for the neuron
mean_activation = neuron_df['activation'].mean()
print(f'Mean activation: {mean_activation}')
# %%
# filter out low activation
filtered_df = neuron_df[neuron_df['activation'] > mean_activation]

# make scatter plot of delta_loss and loss, color by activation
fig = px.scatter(filtered_df, x='delta_entropy', y='delta_loss', title=f'Model: {model}. Mean ablation on k={k} sequences', color='1/rank_of_correct_token_post_ablation', color_continuous_scale='Picnic', hover_data=['str_tokens'])
fig.show()
# %%
# =============================================================================
# plots for all neurons
# =============================================================================
if 'rank_of_correct_token_post_ablation' in final_df.columns:
    # make scatter plot of change in rank and delta entropy
    fig = px.scatter(agg_results, x='change_in_rank', y='abs_delta_entropy', title=f'Model: {model}. Mean ablation on k={k} sequences', hover_data=['component_name'], color='1-abs_delta_loss_post_ablation_with_frozen_ln/abs_delta_loss_post_ablation', marginal_y='histogram', marginal_x='histogram')
    fig.show()
else:
    print('Rank of correct token post ablation not available')
# %%