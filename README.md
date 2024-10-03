# Installation and setup: 
Python version=3.9

```
pip install -r requirements.txt
pip install git+https://github.com/neelnanda-io/neelutils.git 
pip install git+https://github.com/neelnanda-io/neel-plotly.git
```

To load certain models from huggingface: 
Create the file ablations/hf_token.txt and paste your huggingface access token

# Description of files  

ablations/
- run_and_store_ablation_results.py: runs mean ablations to quantify the total vs direct effect for Entropy Neurons (Section 3)
- run_and_store_unigram_results.py: runs mean ablations to quantify the total vs direct effect for Token Frequency Neurons (Section 4).
- load_results and load_unigram_results.py: visualisation of mean ablation experiment results

plot_generation/
- fig1.py: generates the neuron W_out norm vs LogitVar scatter plot (Figure 1(a))

case_studies/
- induction.py: runs ablation experiments for synthetic induction (Section 6)
- natural_induction_true.py: runs ablation experiments for naturally occuring induction sequences (Section 6 and Appendix)
- induction_with_bos_ablation.py: runs BOS ablation experiments on attention heads (Section 6.2)
- load_bos_ablation_results.py: produces activation line graph
  
datasets/ contains .npy files for the token counts of OpenWebText and The Pile. These are used to calculate the token frequency distribution.



