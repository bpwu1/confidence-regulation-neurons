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
print_summary_info = False
use_log2_entropy = False 

entropy_type = 'base e'
if use_log2_entropy:
    entropy_type = 'base 2'

# %%
os.chdir("../")

save_path_prefix = "./"


# %%
model, tokenizer = load_model_from_tl_name(model_name, device, transformers_cache_dir)
model = model.to(device)

if 'gpt2' in model_name: 
    model2, _ = load_model_from_tl_name("gpt2-xl", device, transformers_cache_dir)
else: 
    model2 = None

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

# %%
n_single_neuron_baselines = 5
random_neuron_indices = random.sample(possible_random_neuron_indices, n_single_neuron_baselines)
random_neurons = [f"{entropy_neuron_layer}.{i}" for i in random_neuron_indices]

# %%
#data = load_dataset("stas/openwebtext-10k", split='train')
data = load_dataset("stas/c4-en-10k", split='train')
first_1k = data.select([i for i in range(0, 7500)])


tokenized_data = utils.tokenize_and_concatenate(first_1k, tokenizer, max_length=256, column_name='text')

tokenized_data = tokenized_data.shuffle(SEED)
token_df = nutils.make_token_df(tokenized_data['tokens'])


# %%
k = 1000 # for ablation

# %%
# =============================================================================
# Natural Induction
# =============================================================================

# %%
induction_prefix_length=6
max_induction_ngram_count=2
min_distance_between_induction_ngrams=1

save_file_name = f'natural_induction_prefix{induction_prefix_length}_k{k}'

banned_tokens = get_banned_tokens_for_induction(model, tokenizer)
tokenized_natural_induction_data = get_natural_induction_data(tokenized_data=tokenized_data, tokenizer=tokenizer, induction_prefix_length=induction_prefix_length, max_induction_ngram_count=max_induction_ngram_count, min_distance_between_induction_ngrams=min_distance_between_induction_ngrams, banned_tokens=banned_tokens)

# %%
# checking how frequently induction is the 'right' answer
natural_induction_token_df = nutils.make_token_df(tokenized_natural_induction_data['tokens'])

# columns with induction info to add to natural_induction_token_df
induction_col_names = ['induction_ngram','induction_ngram_first_pos', 'induction_ngram_second_pos', 'induction_ngram_count','n_tokens_between_induction_ngrams']
extra_induction_info_df = pd.DataFrame()
for col_name in induction_col_names:
    extra_induction_info_df[col_name] = tokenized_natural_induction_data[col_name]

# repeat to match size of natural_induction_token_df
# assumes each sample is of the same length. To remove assumption, just pass in a list of indices to repeat to .repeat()
seq_length = natural_induction_token_df['pos'].max() + 1 # add one since pos is 0 indexed
extra_induction_info_df = extra_induction_info_df.loc[extra_induction_info_df.index.repeat(seq_length)].reset_index(drop=True)

natural_induction_token_df = pd.concat([natural_induction_token_df, extra_induction_info_df], axis=1)

# %%
# A .... A B2
def distance_from_b2(example, induction_prefix_length): 
    return example["pos"] - example["induction_ngram_second_pos"] - induction_prefix_length 
def distance_from_b1(example, induction_prefix_length): 
    return example["pos"] - example["induction_ngram_first_pos"] - induction_prefix_length 

natural_induction_token_df["distance_from_b2"] = natural_induction_token_df.apply(lambda x: distance_from_b2(x, induction_prefix_length), axis=1)

natural_induction_token_df["distance_from_b1"] = natural_induction_token_df.apply(lambda x: distance_from_b1(x, induction_prefix_length), axis=1)

# %%
b_df = natural_induction_token_df[(natural_induction_token_df["distance_from_b1"]==0) | (natural_induction_token_df["distance_from_b2"]==0)]
#filter out cases where there is no valid b2
b_df = b_df.groupby("batch").filter(lambda x: len(x) > 1)

b_df["matches_b1"] = b_df.str_tokens.eq(b_df.str_tokens.shift())
px.histogram(b_df["matches_b1"], title=f"AB1 ... AB2, Cases where b1 matches b2, {100 *len(b_df[b_df['matches_b1']])/len(b_df):.1f}% ({len(b_df[b_df['matches_b1']])} out of {len(b_df)})")

# %%
unigram_distrib = None

# %%
all_neuron_names = [f"{entropy_neuron_layer}.{i}" for i in range(model.cfg.d_mlp)]
all_neuron_names_with_activation = [f"{neuron_name}_activation" for neuron_name in all_neuron_names]

if model_name == "Llama-2-7b":
    batch_size = 2
else: 
    batch_size = 32
entropy_dim_layer = model.cfg.n_layers - 1
component_output_to_cache = {'resid_post': []}
entropy_df, resid_dict = get_entropy_activation_df(all_neuron_names,
                                                   tokenized_natural_induction_data,
                                                   natural_induction_token_df,
                                                   model,
                                                   batch_size=batch_size,
                                                   device=device,
                                                   cache_residuals=True,
                                                   cache_pre_activations=False,
                                                   compute_kl_from_bu=False,
                                                   residuals_layer=entropy_dim_layer,
                                                   residuals_dict=component_output_to_cache,
                                                   unigram_distrib=unigram_distrib,
                                                   model2_for_kl=model2)

# %%
save_entropy_df = True # memory intensive 
if save_entropy_df and model_name == "Llama-2-7b": 
    entropy_df.to_feather(save_path_prefix+f'large_scale_exp/results/{model_name}/{save_file_name}_entropy_df.feather')

# %%
# plotting activation diffs
first_sequence = entropy_df[(entropy_df['distance_from_b1']>=-induction_prefix_length) & (entropy_df['distance_from_b1']<=0)]

second_sequence = entropy_df[(entropy_df['distance_from_b2']>=-induction_prefix_length) & (entropy_df['distance_from_b2']<=0)]

selected_activations = [f"{neuron_name}_activation" for neuron_name in (entropy_neurons + random_neurons)]

first_sequence_activations = first_sequence.groupby("distance_from_b1")[selected_activations+['entropy', 'loss']].mean().reset_index()

second_sequence_activations = second_sequence.groupby("distance_from_b2")[selected_activations+['entropy', 'loss']].mean().reset_index()



# %%
one_off_plotting = False
if one_off_plotting and model_name == "gpt2-small":
    # plotting activation diffs
    first_sequence = entropy_df[(entropy_df['distance_from_b1']>=-induction_prefix_length) & (entropy_df['distance_from_b1']<=0)]

    second_sequence = entropy_df[(entropy_df['distance_from_b2']>=-induction_prefix_length) & (entropy_df['distance_from_b2']<=0)]

    gpt2_neurons = ["11.584", "11.2378", "11.2870", "11.2123", "11.1611", "11.2910"]

    selected_activations = [f"{neuron_name}_activation" for neuron_name in gpt2_neurons]

    first_sequence_activations = first_sequence.groupby("distance_from_b1")[selected_activations+['entropy', 'loss']].mean().reset_index()

    second_sequence_activations = second_sequence.groupby("distance_from_b2")[selected_activations+['entropy', 'loss']].mean().reset_index()


    #this is only for plotting
    second_sequence_activations["distance_from_b1"] = second_sequence_activations["distance_from_b2"].apply(lambda x: x + induction_prefix_length+1)

    dummy_row = pd.DataFrame({"distance_from_b1": [1], "distance_from_b2": [0]})
    dummy_row[selected_activations+['entropy', 'loss']] = None

    combined_activations = pd.concat([first_sequence_activations, dummy_row, second_sequence_activations], axis=0)

    x_tick_text = [f"A{i}" for i in range(induction_prefix_length)] + ["B"] + ["..."] + [f"A{i}" for i in range(induction_prefix_length)] + ["X"]

    if model_name == "gpt2-small": 

        gpt2_neurons = ["11.584", "11.2378", "11.2870", "11.2123", "11.1611", "11.2910"]
        gpt2_activations = [f"{neuron_name}_activation" for neuron_name in gpt2_neurons]


        combined_activations = combined_activations.rename(columns=dict(zip(gpt2_activations, gpt2_neurons)))


        color_map = discrete_map={
                 "entropy": "#636EFA",
                 "loss": "#EF553B",
                 "11.584": "#00CC96", 
                 "11.2378": "#AB63FA", 
                 "11.2870": "#FFA15A", 
                 "11.2123": "#19D3F3", 
                 "11.1611": "#FF6692", 
                 "11.2910": "#B6E880"
             }
        y_mode = "neurons"
        if y_mode == "all": 
            y_lines = ['entropy', 'loss']+gpt2_neurons
            activation_title = '''(a) Activations, Entropy, Loss'''
        elif y_mode == "neurons":
            y_lines = gpt2_neurons
            activation_title = '''(a) Induction in the wild: Activations'''
        elif y_mode == "loss_and_entropy": 
            y_lines = ["entropy", "loss"]
            activation_title = '''(a) Induction in the wild: Entropy, Loss'''
        elif y_mode == "2378_only": 
            y_lines = ['entropy', 'loss']+["11.2378"]
            activation_title = '''(a) 2378: Activations, Entropy, Loss'''
        
        fig = px.line(combined_activations, y=y_lines, x="distance_from_b1", title=activation_title, color_discrete_map=color_map)

        fig.update_traces(patch = {"line": {"width": 2.5, "dash": 'dot'}})
        if 'entropy'in y_lines and 'loss' in y_lines:
            fig.update_traces(patch = {"line": {"width": 2.5, "dash": 'solid'}}, selector ={"legendgroup": "entropy"})
            fig.update_traces(patch = {"line": {"width": 2.5, "dash": 'solid'}}, selector ={"legendgroup": "loss"})



    else:
        fig = px.line(combined_activations, y=['entropy', 'loss']+selected_activations, x="distance_from_b1")
    



    fig.add_vrect(
        x0=0,
        x1=1, 
        fillcolor="black",
        opacity=0.1,
        line_width=0,
        )
    fig.update_yaxes(title_text='')
    fig.update_xaxes(title_text='Relative Position')

    fig.add_annotation(dict(font=dict(color='black',size=11),
        x=1,
        y=4.5,
        showarrow=False,
        text="start of induction",
        xanchor='left',
        textangle=0))

    fig.update_xaxes(tickvals=combined_activations["distance_from_b1"], ticktext=x_tick_text, tickangle=60)

    # set axis labels
    # add label to y axis
    # remove padding
    fig.update_layout(margin=dict(l=0, r=3, t=30, b=0))

    # decrease the width of the plot
    rescaling_factor = 1.0

    set_to_fixed_size = True
    size="default_size"
    if set_to_fixed_size:
        size = "fixed_size" 
        fig.update_layout(width=350*rescaling_factor, height=275/rescaling_factor)

    # decrease title font size
    fig.update_layout(title_font_size=16)
    
    #remove title
    fig.update_layout(legend_title=None)

    fig.show()

    fig.write_image(save_path_prefix+f"model_graphs/{model_name}/{save_file_name}_activations_paper_{y_mode}_{size}.pdf")
    fig.write_json(save_path_prefix+f"model_graphs/{model_name}/{save_file_name}_activations_paper_{y_mode}_{size}.json")
# %%
mean_activation_df = pd.DataFrame({'neuron_name': all_neuron_names})
# # activations on tokens A to B2
mean_activation_df['mean_activation_on_first_occurence'] = [first_sequence[f"{neuron_name}_activation"].mean() for neuron_name in all_neuron_names]

mean_activation_df['mean_activation_on_second_occurence'] = [second_sequence[f"{neuron_name}_activation"].mean() for neuron_name in all_neuron_names]

# %%
print("getting mean activations from normal text (c4-10k)")
data = load_dataset("stas/c4-en-10k", split='train')
#originally 5000, 6000
first_1k = data.select([i for i in range(5000, 5500)])
tokenized_data = utils.tokenize_and_concatenate(first_1k, tokenizer, max_length=256, column_name='text')

tokenized_data = tokenized_data.shuffle(SEED)
token_df = nutils.make_token_df(tokenized_data['tokens'])

norm_acts_to_cache = {'resid_post': []}
compute_kl_from_bu=False

norm_text_entropy_df, norm_text_resid_dict = get_entropy_activation_df(all_neuron_names,
    tokenized_data,
    token_df,
    model,
    batch_size=batch_size,
    device=device,
    cache_residuals=True,
    cache_resid_norm=True,                                                   
    cache_pre_activations=False,
    compute_kl_from_bu=compute_kl_from_bu,
    residuals_layer=model.cfg.n_layers-1,
    residuals_dict=norm_acts_to_cache,
    apply_ln_location_layer=model.cfg.n_layers-1, 
    apply_ln_location_mlp_input=True,
    apply_ln_to_cache=False, 
    model2_for_kl=model2)


mean_activation_df['mean_activation_on_normal_text'] = [norm_text_entropy_df[f"{neuron_name}_activation"].mean() for neuron_name in all_neuron_names]


# %%
mean_activation_df['neuron_norm'] = model.W_out[entropy_dim_layer].norm(dim=-1).cpu()
mean_activation_df['diff_natural_induction'] = mean_activation_df['mean_activation_on_second_occurence'] - mean_activation_df['mean_activation_on_first_occurence']
mean_activation_df['weighted_diff_natural_induction'] = mean_activation_df['diff_natural_induction'] * mean_activation_df['neuron_norm']
mean_activation_df['abs_weighted_diff_natural_induction'] = mean_activation_df['weighted_diff_natural_induction'].abs()
mean_activation_df["neuron_name_with_activation"] = mean_activation_df["neuron_name"].apply(lambda x: f"{x}_activation")

# %%
mean_activation_df['is_entropy'] = mean_activation_df["neuron_name"].apply(lambda x: x in entropy_neurons)
mean_activation_df["neuron_idx"] = mean_activation_df["neuron_name"].apply(lambda x: int(x.split(".")[1]))
px.scatter(mean_activation_df, x="neuron_idx", y="weighted_diff_natural_induction", color="is_entropy")
# %%
mean_activation_df.to_feather(save_path_prefix+f'/large_scale_exp/results/{model_name}/{save_file_name}_mean_activation_df.feather')

# %%
select_top_n = 20
selected_neurons_idx = mean_activation_df['abs_weighted_diff_natural_induction'].sort_values(ascending=False).index[:select_top_n]
selected_neurons = mean_activation_df['neuron_name'][selected_neurons_idx].tolist()

# add in the top 5 entropy neurons if they aren't included already 
for entropy_neuron_name in entropy_neurons[:5]: 
    if entropy_neuron_name not in selected_neurons: 
        selected_neurons.append(entropy_neuron_name)

print("selected_neurons", selected_neurons)
#legacy to work with neuron hook
weighted_diffs = dict(zip(mean_activation_df['neuron_name'],mean_activation_df['weighted_diff_natural_induction']))

#choice of what to use as ablation_value
ablation_value_dict = dict(zip(mean_activation_df['neuron_name_with_activation'], mean_activation_df['mean_activation_on_normal_text']))
# %%
px.scatter(mean_activation_df, y='diff_natural_induction', x='neuron_name', title="Mean activation values on first occurence text vs induction text")

# %%
px.scatter(mean_activation_df, y='weighted_diff_natural_induction', x='neuron_name', title="Weighted differences between mean activation values on first occurence text vs induction text")


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
    # if for all abl_val in ablation_values value[0, :, neuron_index] > abl_val, set all neuron_indices to ablation_values
    # todo update this for neurons that "deactivate" on induction
    #mask = torch.all(value[0, :, neuron_indices] > ablation_values, dim=-1)
    # assign to value[0, mask, neuron_indices] the ablation_values
    #value[0, mask, :][:, neuron_indices] = ablation_values
    #print(f'neuron_indices = {neuron_indices}, ablation_values = {ablation_values}')
    #print(f'value[0, mask, :][:, neuron_indices] = {value[0, mask, :][:, neuron_indices]}')
    #print(f'mask = {mask}')
    #print(f'mask.sum() = {mask.sum()}')

    if len(neuron_indices) == 1:
        ablation_value = ablation_values[0]
        neuron_index = neuron_indices[0]
        #if ablation_value > 0.5: # pretty hacky
        #    value[0, :, neuron_index] = torch.max(ablation_value, value[0, :, neuron_index])
        #else:
        if weighted_diffs[f'{entropy_neuron_layer}.{neuron_index}'] < 0:
            value[0, :, neuron_index] = torch.max(ablation_value, value[0, :, neuron_index])
        else:
            value[0, :, neuron_index] = torch.min(ablation_value, value[0, :, neuron_index])
    else:
        #for i in range(value.shape[1]):
        #    if torch.all(value[0, i, neuron_indices] > ablation_values):
        #        value[0, i, neuron_indices] = ablation_values

        value[0, :, neuron_indices] = ablation_values
    return value

def neurons_no_clipping_hook(value, hook, neuron_indices, ablation_values):
    if len(neuron_indices) == 1:
        ablation_value = ablation_values[0]
        neuron_index = neuron_indices[0]
        value[0, :, neuron_index] = ablation_value
    else:
        value[0, :, neuron_indices] = ablation_values
    return value

def ln_final_scale_hook(value, hook, scale_values):
    #scale hook (batch, seq, 1)
    #only applies it to current in theory we could be efficient and do it on the whole sequence + pass in batches
    value[0, :, 0] = scale_values
    return value



def clip_ablate_neurons(list_of_components_to_ablate=None,
                      unigram_distrib=None,
                      tokenized_data=None,
                      entropy_df=None,
                      model=None,
                      k=10,
                      device='mps',
                      ablation_type='mean',
                      ablation_value_dict=None,
                      model2_for_kl_div=None,
                      use_clipping=False):
    
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

                # manually pass in mean value for each neuron
                # to be even more precise we should do mean value for each pos
                if ablation_value_dict is not None: 
                    ablation_values = torch.tensor([ablation_value_dict[f"{neuron_layer}.{neuron_idx}_activation"] for neuron_idx in neuron_indices])
                else: 
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
    
            if use_clipping:
                hooks = [(utils.get_act_name("post", neuron_layer), partial(strict_neurons_clipped_ablation_hook, neuron_indices=neuron_indices, ablation_values=ablation_values.to(device)))]
            else: 
                hooks = [(utils.get_act_name("post", neuron_layer), partial(neurons_no_clipping_hook, neuron_indices=neuron_indices, ablation_values=ablation_values.to(device)))]

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


            if model2_for_kl_div is not None:
                model2_log_probs = model2_for_kl_div(inp).log_softmax(dim=-1)
                ablated_model_log_probs = ablated_logits.log_softmax(dim=-1)
                kl_divergence = torch.nn.functional.kl_div(ablated_model_log_probs, model2_log_probs, log_target=True, reduction='none').sum(dim=-1)
                df_to_append.loc[df_to_append.batch == batch_n, 'kl_from_xl_post_ablation'] = kl_divergence.squeeze().cpu().numpy()
                

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


# k = 1000

ablation_type = 'mean'
#components_to_ablate = [entropy_neurons] + [[neuron] for neuron in entropy_neurons] #+ random_baselines
# %%
#
# %%
# n_of_baselines = 5
# # sample n_of_baselines random lists of neurons to ablate
# random_baselines = []
# for i in range(n_of_baselines):
#     random_baseline = random.sample(possible_random_neuron_indices, len(entropy_neurons))
#     random_baseline = [f"{entropy_neuron_layer}.{i}" for i in random_baseline]
#     random_baselines.append(random_baseline)
# %%
# components_to_ablate = [selected_neurons] + random_baselines #+ [[neuron] for neuron in selected_neurons] #+ random_baselines

if model_name =="gpt2-small":
    for neuron_name in gpt2_neurons: 
        if neuron_name not in selected_neurons: 
            selected_neurons += [neuron_name]


components_to_ablate = [[neuron] for neuron in selected_neurons] + [[neuron] for neuron in random_neurons]

#dropping unneeded columns to save space 
columns_to_drop = [f"{entropy_neuron_layer}.{i}_activation" for i in range(model.cfg.d_mlp) if f"{entropy_neuron_layer}.{i}" not in selected_neurons and f"{entropy_neuron_layer}.{i}" not in random_neurons]
entropy_df = entropy_df.drop(columns=columns_to_drop)



# %%

# %%
induction_multiple_ablation_df = clip_ablate_neurons(list_of_components_to_ablate=components_to_ablate,
                                tokenized_data=tokenized_natural_induction_data,
                                entropy_df=entropy_df,
                                model=model,
                                k=k,
                                device=device,
                                ablation_type=ablation_type,
                                ablation_value_dict=ablation_value_dict, 
                                model2_for_kl_div=model2, 
                                use_clipping=False)
# add is_entropy column
induction_multiple_ablation_df['is_entropy'] = induction_multiple_ablation_df['component_name'].isin(entropy_neurons)
induction_multiple_ablation_df['delta_loss_post_ablation'] = induction_multiple_ablation_df['loss_post_ablation'] - induction_multiple_ablation_df['loss']
induction_multiple_ablation_df['loss_post_ablation/loss'] = induction_multiple_ablation_df['loss_post_ablation'] / induction_multiple_ablation_df['loss']
induction_multiple_ablation_df['entropy_post_ablation/entropy'] = induction_multiple_ablation_df['entropy_post_ablation'] / induction_multiple_ablation_df['entropy']
induction_multiple_ablation_df['abs_delta_loss_post_ablation'] = np.abs(induction_multiple_ablation_df['loss_post_ablation'] - induction_multiple_ablation_df['loss'])
induction_multiple_ablation_df['abs_delta_loss_post_ablation_with_frozen_ln'] = np.abs(induction_multiple_ablation_df['loss_post_ablation_with_frozen_ln'] - induction_multiple_ablation_df['loss'])
induction_multiple_ablation_df['delta_entropy'] = induction_multiple_ablation_df['entropy_post_ablation'] - induction_multiple_ablation_df['entropy']
induction_multiple_ablation_df['delta_entropy_with_frozen_ln'] = induction_multiple_ablation_df['entropy_post_ablation_with_frozen_ln'] - induction_multiple_ablation_df['entropy']
induction_multiple_ablation_df['abs_delta_entropy'] = np.abs(induction_multiple_ablation_df['delta_entropy'])
induction_multiple_ablation_df['abs_delta_entropy_with_frozen_ln'] = np.abs(induction_multiple_ablation_df['delta_entropy_with_frozen_ln'])

if model2 is not None:
    induction_multiple_ablation_df['delta_kl_from_xl'] = induction_multiple_ablation_df['kl_from_xl_post_ablation'] - induction_multiple_ablation_df['kl_from_xl']
    induction_multiple_ablation_df["kl_from_xl_post_ablation/kl_from_xl"] = induction_multiple_ablation_df["kl_from_xl_post_ablation"] / induction_multiple_ablation_df["kl_from_xl"]


columns_to_aggregate =list(induction_multiple_ablation_df.columns[-17:]) + ['loss', 'pos', 'rank_of_correct_token', 'entropy','distance_from_b2']
agg_results = induction_multiple_ablation_df[columns_to_aggregate].groupby(['distance_from_b2', 'component_name']).mean().reset_index()
agg_results_std = induction_multiple_ablation_df[columns_to_aggregate].groupby(['distance_from_b2', 'component_name']).std().reset_index()

induction_multiple_ablation_df['1/rank_of_correct_token'] = induction_multiple_ablation_df['rank_of_correct_token'].apply(lambda x: 1/(x+1))

# %%
save_ablation_results = True
if save_ablation_results: 
    # save_path = save_path_prefix + 

    save_path = save_path_prefix + f'large_scale_exp/results/{model_name}/{save_file_name}.feather'
    # neuron_activation_columns = [f"{neuron_name}_activation" for neuron_name in selected_neurons+random_neurons]
    # # neuron_pre_activation_columns = [f"{neuron_name}_pre_activation" for neuron_name in selected_neurons]
    # neuron_pre_activation_columns = []
    # columns_to_keep = list(set(list(induction_multiple_ablation_df.columns[:18]) + list(induction_multiple_ablation_df.columns[-20:]) + neuron_activation_columns + neuron_pre_activation_columns))
    columns_to_keep = induction_multiple_ablation_df.columns
    df_to_save = induction_multiple_ablation_df[columns_to_keep]
    df_to_save = df_to_save.reset_index()
    df_to_save.to_feather(save_path)

# %%
# change in entropy upon ablation 
if one_off_plotting: 
    # same plot, but delta in entropy

    # neuron_plot_mode = "with_non_entropy_induction_baselines"
    neuron_plot_mode = "set_of_6"
    if neuron_plot_mode == "with_non_entropy_induction_baselines": 
        neurons_to_plot = selected_neurons + random_neurons
    elif neuron_plot_mode == "set_of_6": 
        neurons_to_plot = gpt2_neurons + random_neurons

    fig = go.Figure()
    for neuron_name in neurons_to_plot:
    # for neuron_name in gpt2_neurons + random_neurons:
    # for neuron_name in ["11.2378"]:
        neuron_df = induction_multiple_ablation_df[induction_multiple_ablation_df["component_name"]==neuron_name]


        #filter to simulate clipping
        # a hacky way to achieve this would just be to filter out all cases where its less than the baseline activation


        neuron_activation_info = mean_activation_df[mean_activation_df["neuron_name"]==neuron_name]

        neuron_on_random_text = neuron_activation_info["mean_activation_on_normal_text"].values[0]
        neuron_on_induction = neuron_activation_info["mean_activation_on_second_occurence"].values[0]

        if neuron_on_induction > neuron_on_random_text: 
            goes_up_on_induction = True 
        else: 
            goes_up_on_induction = False

        # # principled way
        if True: 
            if goes_up_on_induction: 
                neuron_df["with_clipping_entropy_post_ablation/entropy"] = neuron_df.apply(lambda x: x["entropy_post_ablation/entropy"] if x[f"{neuron_name}_activation"] > neuron_on_random_text  else 1, axis=1)   
            else: 
                neuron_df["with_clipping_entropy_post_ablation/entropy"] = neuron_df.apply(lambda x: x["entropy_post_ablation/entropy"] if x[f"{neuron_name}_activation"] < neuron_on_random_text  else 1, axis=1)
        # hacky way -> this isn't being used
        if goes_up_on_induction: 
            neuron_df_post_clipping = neuron_df[neuron_df[f"{neuron_name}_activation"] > neuron_on_random_text]
        else: 
            neuron_df_post_clipping = neuron_df[neuron_df[f"{neuron_name}_activation"] < neuron_on_random_text]
        
        first_sequence = neuron_df[(neuron_df['distance_from_b1']>=-induction_prefix_length) & (neuron_df['distance_from_b1']<=0)]

        second_sequence = neuron_df[(neuron_df['distance_from_b2']>=-induction_prefix_length) & (neuron_df['distance_from_b2']<=0)]

        first_sequence_with_clipping = neuron_df_post_clipping[(neuron_df_post_clipping['distance_from_b1']>=-induction_prefix_length) & (neuron_df_post_clipping['distance_from_b1']<=0)]

        second_sequence_with_clipping = neuron_df_post_clipping[(neuron_df_post_clipping['distance_from_b2']>=-induction_prefix_length) & (neuron_df_post_clipping['distance_from_b2']<=0)]

        # clip_simulation_method = "filter"
        clip_simulation_method = "principled_clip"
        if clip_simulation_method == "filter": 
            y_metric = "entropy_post_ablation/entropy"
        else: 
            y_metric = "with_clipping_entropy_post_ablation/entropy"


        first_sequence_entropy = first_sequence.groupby("distance_from_b1")[y_metric].mean().reset_index()

        second_sequence_entropy = second_sequence.groupby("distance_from_b2")[y_metric].mean().reset_index()


        second_sequence_entropy["distance_from_b1"] = second_sequence_entropy["distance_from_b2"].apply(lambda x: x + induction_prefix_length+1)

        dummy_row = pd.DataFrame({"distance_from_b1": [1], "distance_from_b2": [0]})
        dummy_row[y_metric] = None

        combined_df = pd.concat([first_sequence_entropy, dummy_row, second_sequence_entropy], axis=0)


        color_map = discrete_map={
                 "entropy": "#636EFA",
                 "loss": "#EF553B",
                 "11.584": "#00CC96", 
                 "11.2378": "#AB63FA", 
                 "11.2870": "#FFA15A", 
                 "11.2123": "#19D3F3", 
                 "11.1611": "#FF6692", 
                 "11.2910": "#B6E880"
             }
        
        line_dash = None
        if neuron_name in gpt2_neurons:
            neuron_label = neuron_name
            line_color = color_map[neuron_name]
            show_legend = True
            line_dash = "dot"
        elif neuron_name in selected_neurons: 
            if neuron_name in entropy_neurons:
                neuron_label = "entropy"
                line_color = "red"
                show_legend = True
                line_dash = "dot"

            else: 
                neuron_label = "not entropy"
                line_color = "black"
                show_legend = True
        elif neuron_name in random_neurons:
            neuron_label = "Baselines"  
            line_color = "blue"
            if neuron_name == random_neurons[0]:
                show_legend = True 
            else: 
                show_legend = False

        fig.add_trace(go.Scatter(x=combined_df["distance_from_b1"], y=combined_df[y_metric], mode='lines', name=neuron_label, showlegend=show_legend, line =dict(color=line_color, dash=line_dash)))


    fig.add_vrect(
        x0=0,
        x1=1, 
        fillcolor="black",
        opacity=0.1,
        line_width=0,
        # annotation_text="Start of repeated sequence",
        # annotation_position="top right"
        )

    fig.add_annotation(dict(font=dict(color='black'),
        x=1,
        y=1.07,
        showarrow=False,
        text="start of induct.",
        xanchor='left',
        textangle=0))

    fig.update_yaxes(title_text='Entropy Post Ablation / Entropy')
    fig.update_xaxes(title_text='Relative Position')


    # add title
    fig.update_layout(title=f"(b) Change in Entropy Upon Ablation")
    # set axis labels
    # add label to y axis
    # remove padding
    fig.update_layout(margin=dict(l=0, r=3, t=30, b=0))

    # decrease the width of the plot
    rescaling_factor = 1.0

    set_to_fixed_size = True
    size="default_size"
    if set_to_fixed_size:
        size = "fixed_size" 
        fig.update_layout(width=350*rescaling_factor, height=275/rescaling_factor)

    # decrease title font size
    fig.update_layout(title_font_size=16)



    x_tick_text = [f"A{i}" for i in range(induction_prefix_length)] + ["B"] + ["..."] + [f"A{i}" for i in range(induction_prefix_length)] + ["X"]

    fig.update_xaxes(tickvals=combined_activations["distance_from_b1"], ticktext=x_tick_text, tickangle=60)

    fig.show()

    fig.write_image(save_path_prefix+f"model_graphs/{model_name}/{save_file_name}_change_in_entropy_paper_{neuron_plot_mode}_{size}.pdf")
    fig.write_json(save_path_prefix+f"model_graphs/{model_name}/{save_file_name}_change_in_entropy_paper_{neuron_plot_mode}_{size}.json")


# %%
# plotting 2378 delta loss
if one_off_plotting: 
    # same plot, but delta in entropy

    neuron_name = "11.2378"


    neuron_df = induction_multiple_ablation_df[induction_multiple_ablation_df["component_name"]==neuron_name]
    neuron_df = neuron_df[(neuron_df['distance_from_b2']>=-induction_prefix_length) & (neuron_df['distance_from_b2']<=0)]


    #filter to simulate clipping
    # a hacky way to achieve this would just be to filter out all cases where its less than the baseline activation

    neuron_activation_info = mean_activation_df[mean_activation_df["neuron_name"]==neuron_name]

    neuron_on_random_text = neuron_activation_info["mean_activation_on_normal_text"].values[0]
    neuron_on_induction = neuron_activation_info["mean_activation_on_second_occurence"].values[0]

    if neuron_on_induction > neuron_on_random_text: 
        goes_up_on_induction = True 
    else: 
        goes_up_on_induction = False

    # # principled way
    if True: 
        if goes_up_on_induction: 
            neuron_df["delta_loss_post_ablation"] = neuron_df.apply(lambda x: x["delta_loss_post_ablation"] if x[f"{neuron_name}_activation"] > neuron_on_random_text  else 0, axis=1)   
        else: 
            neuron_df["delta_loss_post_ablation"] = neuron_df.apply(lambda x: x["delta_loss_post_ablation"] if x[f"{neuron_name}_activation"] < neuron_on_random_text  else 0, axis=1)

    filter_clipped = True
    if filter_clipped: 
        if goes_up_on_induction: 
            neuron_df = neuron_df[neuron_df[f"{neuron_name}_activation"] > neuron_on_random_text]
        else: 
            neuron_df = neuron_df[neuron_df[f"{neuron_name}_activation"] < neuron_on_random_text]
    
    log_x = False


    fig = px.scatter(neuron_df, x="loss", y="delta_loss_post_ablation", color="1/rank_of_correct_token", hover_data=["context"], color_continuous_scale=plotly.colors.diverging.Picnic, title="(c) 11.2378: Change in Loss Upon Abl.", log_x=log_x)

    fig.update_layout(margin=dict(l=0, r=3, t=30, b=0))

    # decrease the width of the plot
    rescaling_factor = 1.0

    set_to_fixed_size = True
    size="default_size"
    if set_to_fixed_size:
        size = "fixed_size" 
        fig.update_layout(width=350*rescaling_factor, height=275/rescaling_factor)

    fig.update_layout(coloraxis_colorbar_title_text ="1/RC")
    # decrease title font size
    fig.update_layout(title_font_size=16)

    fig.update_layout(
        yaxis_title="Delta Loss Post Ablation"
    )
    fig.show()


    fig.write_image(save_path_prefix+f"model_graphs/{model_name}/{save_file_name}_{neuron_name}_delta_loss_scatter_paper_log_x_{log_x}_{size}.pdf")
    fig.write_json(save_path_prefix+f"model_graphs/{model_name}/{save_file_name}_delta_loss_scatter_paper_{neuron_plot_mode}_log_x_{log_x}_{size}.json")

# end of paper one-off plotting
