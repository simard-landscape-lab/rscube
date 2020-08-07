# RSCube 

These general tutorials are meant to concisely demonstrate how to apply numerical and GIS python to analyze SAR and other remotely sensed data for basic environmental monitoring and land use classification.
We have some simple functions which we include under `rscube`, though they are basic wrappers around the powerful GIS libraries `rasterio`, `geopandas`, etc.

# Installation

1. Download the repository.
2. Open the [terminal](https://support.apple.com/guide/terminal/welcome/mac).
3. Change the working directory of the terminal session to the downloaded repository.
4. Create a virtual environment using conda via: 

	`conda create --name rscube python=3.7 --yes`
	
	Make sure to hit `y` to confirm that the listed packages can be downloaded for this environment.

5. Activate the virtual environment: 

	`conda activate rscube`.

6. Install requirements: 

	`conda install -c conda-forge --yes --file requirements.txt`

7. Create a new jupyter kernel: 

	`python -m ipykernel install --user --name rscube`.