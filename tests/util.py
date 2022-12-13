import requests
import tarfile
from pathlib import Path
import sys, os
from typing import Union
import numpy as np
from numpy.typing import NDArray, DTypeLike
import plyfile

__all__ = ['load_standford_bunny']


def load_standford_bunny(directory: Union[Path, os.PathLike, str, bytes] = "test_data",
                         dtype: DTypeLike = np.float32,
                         chunk_size: int = 8192) -> NDArray:
    """
    Downloads the stanford bunny into the specified directory and returns the PointCloud

    Args:
        directory: The directory to load.
    Returns:
        A numpy array containing the stanford bunny position data.
    """
    directory = Path(directory)
    directory.mkdir(exist_ok=True)
    bunny_path = directory / "bunny.tar.gz"

    # download the file if it does not exist
    if not bunny_path.exists():
        url = 'http://graphics.stanford.edu/pub/3Dscanrep/bunny.tar.gz'
        with open(bunny_path, 'wb') as f:

            response = requests.get(url, stream=True)
            total_length = response.headers.get('content-length')

            if total_length is None:  # no content length header
                f.write(response.content)
            else:
                for data in response.iter_content(chunk_size=chunk_size):
                    f.write(data)

    # Extract the data
    bunny_tar_file = tarfile.open(bunny_path)

    data_dir = bunny_path.parent / "bunny_data"
    data_dir.mkdir(exist_ok=True)
    bunny_tar_file.extractall(data_dir)
    bunny_tar_file.close()

    # Load the ply file from the data path

    with open(data_dir / "bunny" / "reconstruction" / "bun_zipper.ply", 'rb') as f:
        plydata = plyfile.PlyData.read(f)
        x = plydata.elements[0].data['x']
        y = plydata.elements[0].data['y']
        z = plydata.elements[0].data['z']
        positions = np.stack((x, y, z), axis=1)

    positions = positions.astype(dtype, copy=False)

    return positions
