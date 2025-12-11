# QLens
This repository contains the code for the developing quantum-inspired interpretability framework: "QLens: Towards A Quantum Perspective of Transformers."

**Abstract:** Current methods for understanding Transformers are successful at extracting intermediate output probability distributions during inference. However, these approaches function as limited diagnostic checkpoints, lacking a mathematical framework for modeling how each layer facilitates transitions between these distributions. This gap inspires us to turn to quantum mechanics, a field possessing a pre-built mathematical toolkit for describing the evolution of probability distributions from its study of stochastic particle measurements. We propose QLens a novel attempt to develop a physics-based perspective on the Transformer generation process. Under QLens, these neural networks are studied by converting their latent activations into a state vector in a Hilbert space derived from the model's output units. The evolution of this state through subsequent hidden layers is modeled with unitary operators and analogously defined Hamiltonians. To demonstrate QLens' potential, we conduct a proof-of-concept by probing three one-layer Transformers on common deployment tasks to investigate the influence of individual layers in model prediction trajectories. We present our work as a foundation for interdisciplinary methods to be leveraged towards a broader understanding of Transformers.

## Repository Layout
The contents of this repositiory are divided into three overarching folders, each of which contains code for one of the three datasets used to test QLens:
- `Sentihood/`
- `Amazon_Books/`
- `Tiny_Stories/`

### Sentihood Folder
The `Sentihood/` folder contains experiments conducted on the [Sentihood](https://aclanthology.org/C16-1146.pdf) sentiment classification dataset. Its substructure breaks down as follows:

**Jupyter Notebooks**: This sequence of notebooks split the code into multiple thematic segments. 
- `001_Sentihood_Data_Preprocessing.iypnb`: Simplifies task to simple positive or negative sentiment prediction.
- `002_Sentihood_Model_Training.iypnb`: Trains a base Transformer comprising of one encoder block on the preprocessed Sentihood dataset.
- `003_Sentihood_Lens_Training.iypnb`: Prepares two Tuned Lenses for extracting intermediate output probability distributions from the Sentihood Transformer.
- `004_Sentihood_Quantum_Analysis.iypnb`: Employs QLens' mathematical framework to derive and analyze analogous quantum vectors and operators extracted from the Sentihood Transformer.

**Python Files**: These contain modules or helpful functions that are imported by the notebooks above when needed.
- `transformer_model.py`: Contains Sentihood Transformer class.
- `transformer_dataset.py`: Defines the dataset class used to train the Sentihood Transformer.
- `tuned_lens.py`: Prepares the class for the Tuned Lenses trained to extract intermediate output probability distributions.
- `lens_dataset.py`: Houses the dataset used to train the Tuned Lenes.
- `sub_models.py`: Contains classes for parital models of the Sentihood Transformer that used to obtain intermediate hidden states.
- `utils.py`: Includes various helper functions

**Figures Subfolder**
- `Figures/`: Contains visualizations produced by the Sentihood notebooks and is split into two subfolders
    - `Attention_Layer/`: Houses the figures associated with the attention layer of the trained Sentihood Transformer.
    - `MLP_Layer/`: Houses the figures associated with the MLP layer of the trained Sentihood Transformer.
 
### Amazon Books Folder
The `Amazon_Books/` folder contains experiments conducted on the [Amazon Books](https://aclanthology.org/D19-1018.pdf) recommendation dataset, a subset of the larger Amazon Review dataset. Substructure:

**Jupyter Notebooks**:
- `001_Books_Model_Training.iypnb`: Prepocesses a subset of the 50,000 most active users to 5-core and trains a Transformer (self-attentive sequential recommender) with one encoder block on the resulting corupus.
- `002_Books_Lens_Training.iypnb`: Prepares two Tuned Lenses for extracting intermediate probability distributions from the Amazon Books model.
- `003_Books_Quantum_Analysis.iypnb`: Implements QLens' mathematical framework on the Amazon Books model.
  
**Figures Subfolder**
- `Figures/`: Contains visualizations obtained by applying QLens to the Amazon Books Transformer.
    - `Attention_Layer/`: Houses figures associated with the attention layer.
    - `MLP_Layer/`: Houses figures associated with the MLP layer.

### Tiny Stories Folder
The `Tiny_Stories/` folder holds the experiments conducted using the [Tiny Stories](https://arxiv.org/pdf/2305.07759) text generation dataset and the pre-trained [Tiny Stories one enocder block Transformer](https://huggingface.co/roneneldan/TinyStories-1Layer-21M). Substructure:

**Jupyter Notebooks**:
- `001_Stories_Lens_Training.iypnb`: Loads the pre-trained Tiny Stories model and trains two Tuned Lenses on it.
- `002_Stories_Quantum_Analysis.iypnb`: Performs QLens' analysis on the Tiny Stories model.
  
**Figures Subfolder**
- `Figures/`: Contains QLens-derived visualizations.
    - `Attention_Layer/`: Houses figures associated with the attention layer.
    - `MLP_Layer/`: Houses figures associated with the MLP layer.
