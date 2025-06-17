import os
import torch
import logging
import hashlib
import tempfile
import zipfile
import requests
from tqdm import tqdm
from fnn.microns.build import network
from fnn.microns.load import params, units, unit_ids

URL = "https://bossdb-open-data.s3.amazonaws.com/iarpa_microns/minnie/functional_data/foundation_model/foundational_model_weights_and_metadata_v1.zip"
MD5 = "58fcac4b31ad2902c81e339432cec787" #verified downloaded model matches

logger = logging.getLogger("fnn.microns")
logger.addHandler(logging.StreamHandler())


def download(url, file_path, chunk_size=8192, verbose=True):
    """
    Parameters
    ----------
    url : str
        source url
    file_path : os.PathLike
        destination file path
    chunk_size : int
        size of the chunks (in bytes) to read
    verbose : bool
        display download progress

    Returns
    -------
    str
        MD5 checksum of downloaded file as a hexadecimal string
    """
    response = requests.get(url, stream=True)
    response.raise_for_status()
    size = int(response.headers.get("content-length", 0))
    md5 = hashlib.md5()

    with open(file_path, "wb") as file:
        with tqdm(total=size, unit="B", unit_scale=True, disable=not verbose) as bar:
            for chunk in response.iter_content(chunk_size):

                file.write(chunk)
                md5.update(chunk)
                bar.update(len(chunk))

    return md5.hexdigest()


def scan(session, scan_idx, cuda=True, directory=None, verbose=True):
    """
    Parameters
    ----------
    session : int
        scan session
    scan_idx : int
        scan index
    directory : os.PathLike | None
        directory for model parameters and metadata. defaults to current working directory
    verbose : bool
        display download progress
    cuda : bool
        use cuda if available

    Returns
    -------
    fnn.model.networks.Visual
        predictive model of the experimental scan
    pd.DataFrame
        dataframe mapping readout ids to unit ids
    """
    logger.setLevel(logging.INFO if verbose else logging.WARNING)

    directory = os.path.join(directory or os.getcwd(), "fnn", "data", "microns")
    load = lambda f: f(session, scan_idx, directory)

    if not os.path.exists(directory):
        logger.info(f"Downloading model parameters and metadata to `{directory}`")

        with tempfile.TemporaryDirectory() as tmpdir:
            zip_path = os.path.join(tmpdir, "microns.zip")

            md5 = download(URL, zip_path, verbose=verbose)
            assert md5 == MD5, f"md5 for downloaded file is {md5}, expected {MD5}"

            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(directory)

    model = network(load(units))
    model.load_state_dict(load(params))

    if cuda and torch.cuda.is_available():
        model.to(device="cuda")

    return model, load(unit_ids)


# Download instruction from josh
pred_model, table = scan(4,7, directory = os.path.join(os.getcwd(), "data","microns")) # look at data/microns/scans.csv for numbers that work

print("Scan Information")
try:
    print(f"Shape: {table.shape}, Columns: {table.columns}")
except Exception as e: print("Error: ",e)


# test prediction
from numpy import concatenate, full

frames = concatenate([
    full(shape=[30, 144, 256], dtype="uint8", fill_value=0),   # 1 second of black
    full(shape=[30, 144, 256], dtype="uint8", fill_value=128), # 1 second of gray
    full(shape=[30, 144, 256], dtype="uint8", fill_value=255), # 1 second of white
])

results = pred_model.predict(stimuli = frames)
print(f"Results: ",results)