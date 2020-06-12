import os.path
from datetime import datetime


def get_expm_folder(script_path, output_folder='out', expm_id=''):
    """
    Get automatically generated folder name.

    Args:
        script_path: Path of the training file.
        output_folder: Output folder.
        expm_id: Memorable id for experiment.

    Returns:
        Creates and return directory path output_folder/<script_name>/<expm_id>_<date>.
    """
    # remove extension
    script_name = os.path.basename(script_path)
    folder_path = os.path.splitext(script_name)[0]
    now = datetime.today().strftime('%Y-%m-%d_%H%M')
    bottom_folder = now + '_' + expm_id if expm_id else now
    return os.path.join(output_folder, folder_path, bottom_folder)
