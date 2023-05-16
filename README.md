# End-to-End ML Project

This repository contains code for an end-to-end machine learning project that predicts math scores based on input fields.

## Prerequisites

Before running the project, ensure that you have the following prerequisites installed on your system:

- Anaconda or Miniconda (https://www.anaconda.com/products/individual)
- Python 3.9

## Setup

Follow the steps below to set up and run the project:

1. Clone the repository:

```shell
git clone https://github.com/MariosKadriu/end-to-end-ml-project.git
```

2. Change into the project directory:

```shell
cd end-to-end-ml-project
```

3. Create a new virtual environment using Conda:

```shell
conda create -p venv python==3.9 -y
```

4. Activate the virtual environment:

```shell
conda activate venv/
```

5. Install the required dependencies:

```shell
pip install -r requirements.txt
```

## Running the Project

1. Execute the data ingestion script to prepare the data:
```shell
python src/components/data_ingestion.py
```

This script will process the data and prepare it for the machine learning model.

2. Start the application:
```shell
python application.py
```

3. Open your web browser and go to http://127.0.0.1:5000/predictdata

4. Fill in the required fields in the web form and submit the form to predict the math scores based on the provided input.
