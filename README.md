Reproduced the results for Burger's and Schrodinger partial differential equations (continuous inference) mentioned in the Raissi et al., 2019.
For more information, please reach out to gh2546@columbia.edu.

### References
Raissi, Maziar, Paris Perdikaris, and George E. Karniadakis. "Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations." Journal of Computational Physics 378 (2019): 686-707.

# Organization of this directory
To be populated by students, as shown in previous assignments
```
.
├── E4040.2020Fall.PINN.report.gh2546.mk4121.pdf
├── README.md
├── burgers_Layer_Neuron.ipynb
├── continuous_time_inference_burgers.ipynb
├── continuous_time_inference_burgers_hyperparameters.ipynb
├── continuous_time_inference_burgers_sigmoid.ipynb
├── continuous_time_inference_schrodinger.ipynb
├── continuous_time_inference_schrodinger_hyperparameters.ipynb
├── data
│   ├── NLS.mat
│   └── burgers_shock.mat
├── figures
│   ├── Burgers
│   │   ├── burgers_3.png
│   │   ├── burgers_loss.png
│   │   ├── error_heatmap.png
│   │   └── solution_heatmap.png
│   ├── Burgers_sigmoid
│   │   ├── burgers_3.png
│   │   ├── burgers_loss.png
│   │   ├── error_heatmap.png
│   │   └── solution_heatmap.png
│   └── Schrodinger
│       ├── error_heatmap.png
│       ├── schrod_3.png
│       ├── schrod_loss.png
│       ├── schrod_loss_alone.png
│       └── solution_heatmap.png
├── hyperparams_analysis
│   ├── hyper_Burgers
│   │   ├── best_dict
│   │   ├── dict_burgers
│   │   └── track_dict
│   └── hyper_Schrod
│       ├── best_dict
│       ├── dict_schrod
│       └── track_dict
├── installation.md
├── models
│   ├── Burgers
│   │   ├── Burgers_shared_model
│   │   │   ├── saved_model.pb
│   │   │   └── variables
│   │   │       ├── variables.data-00000-of-00002
│   │   │       ├── variables.data-00001-of-00002
│   │   │       └── variables.index
│   │   ├── checkpoint
│   │   ├── weights.data-00000-of-00002
│   │   ├── weights.data-00001-of-00002
│   │   └── weights.index
│   └── Schrodinger
│       ├── Schrodinger_shared_model
│       │   ├── saved_model.pb
│       │   └── variables
│       │       ├── variables.data-00000-of-00002
│       │       ├── variables.data-00001-of-00002
│       │       └── variables.index
│       ├── checkpoint
│       ├── weights.data-00000-of-00002
│       ├── weights.data-00001-of-00002
│       └── weights.index
├── requirements.txt
├── series_burgers
│   ├── LayerNeuronBurgers
│   ├── series.py
│   └── series_burgers_Layer_Neuron.py
└── utils
    ├── burgers_utils.py
    ├── mlp_network.py
    ├── plotting.py
    └── schroding_utils.py

17 directories, 54 files
