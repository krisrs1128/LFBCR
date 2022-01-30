First download the dataset at this [link](https://drive.google.com/file/d/129LgU7_YqV8XSdtC6hHnsTcJao0UfGR2/view?usp=sharing) into this directory. Then, unzip it,

```
tar -zxvf raw_data.tar.gz
```

You should see the following files / folders,

* `spatial_process_models`: This contains features from all the trained CNN, VAE, and RCF models across dataset and model parameters in the LCMP spatial process simulation. This is used as input for the `bootstrap.Rmd` script in `analysis/simulation/spatial_process_simulation`. Each file is named using the identifiers `{model_name}-k{model complexity}-{data fraction}.yaml_{bootstrap number}.tar.gz`
* `tnbc_models`: These are the analogous features for the TNBC spatial proteomics data analysis. Note that only the model type and complexity is varied in this experiment.
* `stability_data_sim.tar.gz`: This contains the simulated tiles and metadata used to train the models in `spatial_process_models`. It is not needed in any of the bootstrap scripts in this compendium, but it is used by the python notebooks used to train feature extractors. It was generated using the `generate.Rmd` file in the spatial process simulation folder.
* `stability_data_tnbc.tar.gz`: These are the analogous data used for feature learning in the TNBC spatial proteomics dataset. The input tiles for model training were generated using the `prepare_mibi.Rmd` script in the data analysis folder.
