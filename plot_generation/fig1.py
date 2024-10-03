# %% 
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import sys
sys.path.append('../')
from transformer_lens import HookedTransformer
from scipy.stats import ttest_rel, ttest_ind
from neel.imports import *
from neel_plotly import *
import neel 
import tqdm
import math 
from datasets import Dataset
import json
import pandas as pd
import plotly.express as px
from utils import *
import math
from scipy.stats import spearmanr
import plotly.graph_objects as go
from scipy.special import kl_div
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
os.chdir('../')

# %%
model_name = "gpt2-small"
#model_name = "google/gemma-2b"
#model_name = 'pythia-410m'
#model_name = 'phi-2'
#model_name = 'Llama-2-7B'

lowest_composing_neurons_dict = {
    'stanford-gpt2-small-a': ['11.3030', '11.2859', '11.995', '11.2546', '11.823', '11.2748'],
    'gpt2-small': ['11.584', '11.2378', '11.2870', '11.2123', '11.1611', '11.2910'],
    'pythia-410m':  [],
    'pythia-1.4b':  ['23.5104', '23.435', '23.6368', '23.7756', '23.3636'],
    'pythia-2.8b':  ['31.8255', '31.6326', '31.6955', '31.5251', '31.1676'],
    'pythia-6.9b':  [],
}

print_summary_info = False
use_log2_entropy = False 

entropy_type = 'base e'
if use_log2_entropy:
    entropy_type = 'base 2'

# %%
model, tokenizer = load_model_from_tl_name(model_name, device, transformers_cache_dir, hf_token=None)
model = model.to(device)
# %%
last_layer_neurons = model.W_out[-1]
norm = torch.norm(last_layer_neurons, dim=1)
comp_with_unemb = last_layer_neurons @ model.W_U
normalized_composition = last_layer_neurons / last_layer_neurons.norm(dim=1).unsqueeze(1) @ model.W_U / model.W_U.norm(dim=0)
comp_var = torch.var(comp_with_unemb, dim=1)
cos_var = normalized_composition.var(dim=1)
# %%
# make scatter plot of norm and comp_with_unemb
entropy_neurons = lowest_composing_neurons_dict[model_name]
entropy_neuron_indices = [int(name.split('.')[1]) for name in entropy_neurons]
# make entropy neurons red
is_entropy = [ind in entropy_neuron_indices for ind in range(last_layer_neurons.shape[0])]
# make dataframe
df = pd.DataFrame({'norm': norm.cpu(),'cos_var': cos_var.cpu(), 'is_entropy': is_entropy})

# Add a new column 'neuron_index' to the DataFrame
df['neuron_index'] = df.index

df['is_entropy'] = df['is_entropy'].map({True: 'Entropy', False: 'Normal'})

# Create a new column 'label' that contains the neuron index for entropy neurons and is empty for other neurons
df['label'] = df.apply(lambda row: str(row['neuron_index']) if row['is_entropy'] == 'Entropy' else '', axis=1)

#ylabel = r'$(\mathrm{Var}(W_{O}W_{U})$'}'
ylabel = 'LogitVar(w<sub>out</sub>)'
xlabel = '||w<sub>out</sub>||'
fig = px.scatter(df, x='norm', y='cos_var', title='(a) Norm vs. LogitVar', color_discrete_sequence=['#636EFA', '#EF553B'], labels={'norm':xlabel, 'cos_var': ylabel, 'is_entropy': 'Neuron'}, log_y=True, marginal_x='histogram', marginal_y='box', color='is_entropy')

# Add text labels
for neuron_index in entropy_neuron_indices:
    entropy_df = df[df['neuron_index'] == neuron_index]
    if neuron_index == 584:
        fig.add_trace(go.Scatter(x=entropy_df['norm']+0.5, y=entropy_df['cos_var'], mode='text', text=entropy_df['label'], textposition='middle right', showlegend=False, textfont=dict(color='#EF553B')))
    elif neuron_index == 2870:
        fig.add_trace(go.Scatter(x=entropy_df['norm'], y=entropy_df['cos_var']*1.2, mode='text', text=entropy_df['label'], textposition='top center', showlegend=False, textfont=dict(color='#EF553B')))
    elif neuron_index == 2910:
        fig.add_trace(go.Scatter(x=entropy_df['norm']-0.5, y=entropy_df['cos_var'], mode='text', text=entropy_df['label'], textposition='bottom left', showlegend=False, textfont=dict(color='#EF553B')))
    else:
        fig.add_trace(go.Scatter(x=entropy_df['norm'], y=entropy_df['cos_var']*0.77, mode='text', text=entropy_df['label'], textposition='bottom center', showlegend=False, textfont=dict(color='#EF553B')))

# remove the legend
#fig.update_layout(showlegend=False)
# remove padding
fig.update_layout(margin=dict(l=0, r=3, t=30, b=0))

# decrease the width of the plot
fig.update_layout(width=350, height=275)

# decrease title font size
fig.update_layout(title_font_size=16)

fig.update_layout(legend=dict(
        x=0.5,
        y=-0.29,
        xanchor="center",
        yanchor="top",
        orientation="h",
        title_text='Neuron Type:',
    ),)

fig.show()
# store the plot as pdf
#fig.write_image('./img/norm_vs_var.pdf')