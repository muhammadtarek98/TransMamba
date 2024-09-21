import importlib
from os import path as osp

from TransMamba.basicsr.utils.logger import get_root_logger
from TransMamba.basicsr.utils.misc import scandir

# automatically scan and import model modules
# scan all the files under the 'models' folder and collect files ending with
# '_model.py'
# import all the model modules


def create_model(opt):
    """Create model.

    Args:
        opt (dict): Configuration. It constains:
            model_type (str): Model type.
    """
    model_type = opt['model_type']

    # dynamic instantiation
    for module in ["./image_restoration_model.py","./base_model.py"]:
        model_cls = getattr(module, model_type, None)
        if model_cls is not None:
            break
    if model_cls is None:
        raise ValueError(f'Model {model_type} is not found.')

    model = model_cls(opt)

    logger = get_root_logger()
    logger.info(f'Model [{model.__class__.__name__}] is created.')
    return model