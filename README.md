# Street network indices for capital cities from around the world

## Characterising active mobility using street network indices

Supplementary code and notebooks for the research on characterisation of active mobility using street network indices.

## Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/Diuke/active-mobility-indices.git
    ```
2. Navigate to the project directory:
    ```bash
    cd active-mobility-indices
    ```

3. Create and activate virtual environment:
    ```bash
    python -m venv .venv
    source .venv/bin/activate 
    ```

4. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Project Structure
- data folder:

    Main data folder for the notebooks. Contains test data for the city of Reykjavik, including the downloaded street networks and shapefiles.

    This folder also contains the DEM data for the cities to download and the GHS-UCDB for the socioeconomic data.

- modules folder:

    This folder contains all the necessary processing scripts and modules for the download of street networks and the calculation of indices.

- download_street_networks.ipynb:

    This notebook contains the code for downloading street network indices for 176 capital cities from around the world using the GHS-UCDB dataset for their areas of interest and DEMs for elevation data.

    Uses the Ray library for paralellizing the download of pedestrian, cycling, driving, and public transport street networks.

- load_street_networks.ipynb:

    This notebook contains the code for loading a graph in networkx. It can be used for further processing of the street network graphs.

- calculate_indices.ipynb

    This notebook contains the code for the calculation of the street networks.
