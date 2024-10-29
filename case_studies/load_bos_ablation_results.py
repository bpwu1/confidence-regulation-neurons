# script for plotting the results of the BOS ablation experiments for gpt2-small

# %% 
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import sys
sys.path.append('../')

import random 
import pandas as pd
from datasets import load_dataset
from utils import load_model_from_tl_name, get_potential_entropy_neurons_udark,generate_induction_examples, generate_induction_df, get_induction_data_and_token_df
import plotly.express as px
import plotly.graph_objects as go
import torch
import numpy as np
import transformer_lens.utils as tl_utils
import neel.utils as nutils
from ast import literal_eval

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
lowest_composing_neurons_dict = {
    'gpt2-small': ['11.584', '11.2378', '11.2870', '11.2123', '11.1611', '11.2910'],
}

# %%
model, tokenizer = load_model_from_tl_name(model_name, device, transformers_cache_dir)
model = model.to(device)

# %%
data = load_dataset("stas/c4-en-10k", split='train')
first_1k = data.select([i for i in range(0, 1000)])

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
seq_len = 100
use_natural_text = False

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

# induction_tokenized_data, induction_token_df = get_induction_data_and_token_df(model, tokenizer, seq_len, num_examples, seed=SEED, device='cuda', use_natural_text=True, use_separator=None, num_repetitions=1)

# smaller dataset for speed
induction_tokenized_data, induction_token_df = get_induction_data_and_token_df(model, tokenizer, seq_length=100, num_examples=100, seed=SEED, device='cuda', use_natural_text=True, use_separator=None, num_repetitions=1)


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

# %%
def load_and_combine_ablation_dfs(folder_path, string_match): 
    files_names = os.listdir(folder_path)
    valid_file_paths = [file_name for file_name in files_names if string_match in file_name and file_name+".feather" not in file_name]
    print(valid_file_paths)

    dfs = []
    for valid_file in valid_file_paths:
        path = os.path.join(folder_path, valid_file)
        dfs.append(pd.read_feather(path))
    
    combined_df = pd.concat(dfs)
    return combined_df

# %%
# plotting for gpt2-small begins here 
ablation_type = "BOS Ablation" # "Mean Ablation" or "BOS Ablation"

os.chdir("../")

if ablation_type == "Mean Ablation":
    bos_ablation_df = load_and_combine_ablation_dfs(f"./large_scale_exp/results/{model_name}", "mean_attn_ablation_df_seq50_k100")
    ablation_file_name = "mean_attn" #for generating save name
    plot_seq_len = 50
    plot_k = 16

elif ablation_type == "BOS Ablation":
    bos_ablation_df = load_and_combine_ablation_dfs(f"./large_scale_exp/results/{model_name}", "bos_ablation_df_seq50_k100")
    ablation_file_name = "bos_ablation" #for generating save name
    plot_seq_len = 50
    plot_k = 100


# %%
component_selection = "custom" #all or custom
induction_heads_type = "random" #random or natural 
#score used to determine order of induction heads. i.e. random means caculated from random repeated tokens, natural means caculated from (synthetic) natural text
baseline_type = "all" # or all
number_of_induction_heads = 3

single_head_baselines =  ["L6H1", "L5H7", "L5H6", "L5H8", "L6H10", "L6H3"] 
multi_head_baselines = ["L5H7-L5H6", "L6H1-L5H7", "L5H8-L6H3", "L5H7-L5H6-L5H8", "L6H1-L5H6-L6H10", "L5H7-L6H10-L6H3"]


baselines = single_head_baselines
if baseline_type == "all" or baseline_type == "multi":
    baselines += multi_head_baselines

if induction_heads_type == "random":
    induction_heads = ["L5H1", "L5H5", "L6H9", "L7H10", "L7H2"]
    joint_induction_heads = ["L5H1-L5H5", "L5H1-L5H5-L6H9"]
elif induction_heads_type == "natural":
    induction_heads = ["L5H1","L7H2", "L7H10","L6H9", "L5H5"]
    joint_induction_heads = ["L5H1-L7H2", "L5H1-L7H2-L7H10"]

if component_selection == "all":
    components_tracked = bos_ablation_df['component_name'].unique()
    induction_components = induction_heads[:number_of_induction_heads] + joint_induction_heads
elif component_selection == "custom":
    induction_components = induction_heads[:number_of_induction_heads] + joint_induction_heads
    components_tracked = baselines + induction_components


post_abl_columns = [f'{neuron_name}_activation_post_abl' for neuron_name in all_neuron_names]
agg_activations = bos_ablation_df.groupby(["pos", "component_name"])[all_neuron_names_with_activation+post_abl_columns].mean().reset_index()

agg_activations = agg_activations[agg_activations.component_name.isin(components_tracked)]

# %%
def get_color_for_val(val, vmin, vmax, pl_colors):
    
    if pl_colors[0][:3] != 'rgb':
        raise ValueError('This function works only with Plotly rgb-colorscales')
    if vmin >= vmax:
        raise ValueError('vmin should be < vmax')
    
    scale = [k/(len(pl_colors)-1) for k in range(len(pl_colors))] 


    colors_01 = np.array([literal_eval(color[3:]) for color in pl_colors])/255.  #color codes in [0,1]

    v= (val - vmin) / (vmax - vmin) # val is mapped to v in [0,1]
    #find two consecutive values in plotly_scale such that   v is in  the corresponding interval
    idx = 1

    while(v > scale[idx]): 
        idx += 1
    vv = (v - scale[idx-1]) / (scale[idx] -scale[idx-1] )

    #get   [0,1]-valued color code representing the rgb color corresponding to val
    val_color01 = colors_01[idx-1] + vv * (colors_01[idx ] - colors_01[idx-1])
    val_color_0255 = (255*val_color01+0.5).astype(int)
    return f'rgb{str(tuple(val_color_0255))}'
# %%
save_fig = True 
set_to_fixed_size = True
show_colorbar = False
relabel_legend = True
manual_color_adjust = True # used to distinguish colors for top 3 induction heads, otherwise they are all given the same greenish color since they have close-ish induction scores

# for single heads
max_color_addition = 0.3
max_color_subtraction = -0.2

# for joint heads
joint_head_manual_color_adjust_val = 0.5

neurons_to_plot = ['11.2378', '11.2870', '11.2123', '11.1611', '11.2910', "11.584"]
# neurons_to_plot = ['11.2378']


# getting sum of induction scores - used for setting the color scale
induction_component_scores  = {induction_component:induction_df[induction_df.labels==induction_component]['induction_scores'].values[0] for induction_component in induction_heads}
for joint_head_name in joint_induction_heads:
    sum_of_scores = 0
    head_components = joint_head_name.split('-')
    for component in head_components:
        sum_of_scores += induction_component_scores[component]
    induction_component_scores[joint_head_name] = sum_of_scores
        
manual_adjust_color_values = np.linspace(max_color_addition, max_color_subtraction, number_of_induction_heads)

#getting relabel names
relabel_names = {}
for i, induction_head_name in enumerate(induction_heads[:number_of_induction_heads]):
    relabel_names[induction_head_name] = f"Ind.head {i+1}"

for joint_head_name in joint_induction_heads:
    num_components = len(joint_head_name.split('-'))
    relabel_names[joint_head_name] = f"Ind.heads 1-{num_components}"

for neuron_selection in neurons_to_plot: 
    max_induction_score = 3
    min_induction_score = 0 
    color_scale = plotly.colors.diverging.Portland 

    fig = go.Figure()
    

    for induction_component in induction_components:

        #setting manual color adjustment, assumes induction heads are sorted by score
        if manual_color_adjust:
            if induction_component in induction_heads:
                manual_adjust_color_val = manual_adjust_color_values[induction_heads.index(induction_component)]
            elif induction_component in joint_induction_heads:
                manual_adjust_color_val = joint_head_manual_color_adjust_val
        else: 
            manual_adjust_color_val = 0.0

        adjusted_color_val = min(max_induction_score,induction_component_scores[induction_component]+manual_adjust_color_val) # to ensure that the color scale is not exceeded
        color_rgb = get_color_for_val(adjusted_color_val, min_induction_score, max_induction_score, color_scale)
        induction_agg_activations = agg_activations[agg_activations.component_name==induction_component]

        if relabel_legend:
            plot_name = relabel_names[induction_component]
        else:
            plot_name = induction_component
        fig.add_trace(go.Scatter(x=induction_agg_activations.pos, y=induction_agg_activations[f"{neuron_selection}_activation_post_abl"], mode='lines', name=f'{plot_name}', line=dict(dash='solid', color=color_rgb)))

    if show_colorbar:
        fig.add_trace(go.Scatter(x=[0, 0], y=[min_induction_score, max_induction_score], mode='markers', showlegend=False, marker=dict(color=[min_induction_score, max_induction_score], colorscale=color_scale, opacity=0, size=0, colorbar=dict(thickness=20, yanchor="top", lenmode="fraction", len=0.5, title="Sum of<br>Induction<br>Scores"))))

    # for legend ordering
    fig.add_trace(go.Scatter(x=induction_agg_activations.pos, y=induction_agg_activations[f"{neuron_selection}_activation"], mode='lines', name=f'Original Act.', line=dict(dash='solid', color="black", width=1.0)))

    baseline_opacity = None
    if len(baselines) > 6: 
        baseline_opacity = None
    for baseline_component in baselines:
        if baseline_component == baselines[0]:
            show_legend = True
        else:
            show_legend = False
        baseline_agg_activations = agg_activations[agg_activations.component_name==baseline_component]
        fig.add_trace(go.Scatter(x=baseline_agg_activations.pos, y=baseline_agg_activations[f"{neuron_selection}_activation_post_abl"], mode='lines', name="Baselines", showlegend=show_legend, opacity=baseline_opacity, line=dict(dash='dot', color=get_color_for_val(0,min_induction_score, max_induction_score,color_scale), width=1.0)))

    # for legend ordering so that orig activation shows on top of baselines
    fig.add_trace(go.Scatter(x=induction_agg_activations.pos, y=induction_agg_activations[f"{neuron_selection}_activation"], mode='lines', name=f'orig.', showlegend=False, line=dict(dash='solid', color="black", width=0.8)))


    fig.add_vline(x=plot_seq_len+1, line_dash="dash", line_color="black", annotation_text="start of ind.")
    # add title

    fig.update(layout_xaxis_range = [0, (2*plot_seq_len) + 1])

    if ablation_type == "Mean Ablation":
        fig.update_layout(title=f"{neuron_selection} Activation after Mean Attn. Ablation")
    else:
        fig.update_layout(title=f"{neuron_selection} Activation after {ablation_type}")
    
    
    fig.update_layout(margin=dict(l=0, r=3, t=30, b=0))
    fig.update_layout(yaxis_title=f"{neuron_selection} Activation")
    fig.update_layout(xaxis_title='Position in Sequence')
    fig.update_layout(title_font_size=16)
    fig.update_layout(legend_title_text='Attn heads')
    # decrease the width of the plot
    rescaling_factor = 1.0

    if set_to_fixed_size:
        size = "fixed_size" 
        fig.update_layout(width=350*rescaling_factor, height=275/rescaling_factor)
    else:
        size = "default_size"

    if save_fig: 
        save_path_prefix = "./"
        save_file_name = ""
        fig.write_image(save_path_prefix+f"model_graphs/{model_name}/{save_file_name}_induction_ablation_{ablation_file_name}_seq{plot_seq_len}_k{plot_k}_{neuron_selection}_induction_heads_{induction_heads_type}_components_selection_{component_selection}_{baseline_type}_new_colors_{manual_color_adjust}_{size}.pdf")
        fig.write_json(save_path_prefix+f"model_graphs/{model_name}/{save_file_name}_induction_ablation_{ablation_file_name}_seq{plot_seq_len}_k{plot_k}_{neuron_selection}_induction_heads_{induction_heads_type}_components_selection_{component_selection}_baseline_{baseline_type}_new_colors_{manual_color_adjust}_{size}.json")
    else:
        fig.show()