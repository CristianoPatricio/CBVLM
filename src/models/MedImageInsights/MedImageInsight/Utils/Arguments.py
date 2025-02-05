import argparse
import json
import logging
import os
import re
import yaml

logger = logging.getLogger(__name__)


def add_env_parser_to_yaml():
    """
    Adding ability of resolving environment variables to the yaml SafeLoader.
    Environment variables in the form of "${<env_var_name>}" can be resolved as strings.
    If the <env_var_name> is not in the env, <env_var_name> itself would be used.

    E.g.:
    config:
      username: admin
      password: ${SERVICE_PASSWORD}
      service: https://${SERVICE_HOST}/service
    """
    loader = yaml.SafeLoader
    env_pattern = re.compile(r".*?\${(.*?)}.*?")

    def env_constructor(loader, node):
        value = loader.construct_scalar(node)
        for group in env_pattern.findall(value):
            value = value.replace(f"${{{group}}}", os.environ.get(group, group))
        return value

    yaml.add_implicit_resolver("!ENV", env_pattern, Loader=loader)
    yaml.add_constructor("!ENV", env_constructor, Loader=loader)


def load_config_dict_to_opt(opt, config_dict, splitter='.', log_new=False):
    """
    Load the key, value pairs from config_dict to opt, overriding existing values in opt
    if there is any.
    """
    if not isinstance(config_dict, dict):
        raise TypeError("Config must be a Python dictionary")
    for k, v in config_dict.items():
        k_parts = k.split(splitter)
        pointer = opt
        for k_part in k_parts[:-1]:
            if '[' in k_part and ']' in k_part:
                # for the format "a.b[0][1].c: d"
                k_part_splits = k_part.split('[')
                k_part = k_part_splits.pop(0)
                pointer = pointer[k_part]
                for i in k_part_splits:
                    assert i[-1] == ']'
                    pointer = pointer[int(i[:-1])]
            else:
                if k_part not in pointer:
                    pointer[k_part] = {}
                pointer = pointer[k_part]
            assert isinstance(pointer, dict), "Overriding key needs to be inside a Python dict."
        if '[' in k_parts[-1] and ']' in k_parts[-1]:
            k_part_splits = k_parts[-1].split('[')
            k_part = k_part_splits.pop(0)
            pointer = pointer[k_part]
            for i in k_part_splits[:-1]:
                assert i[-1] == ']'
                pointer = pointer[int(i[:-1])]
            assert k_part_splits[-1][-1] == ']'
            ori_value = pointer[int(k_part_splits[-1][:-1])]
            pointer[int(k_part_splits[-1][:-1])] = v
        else:
            ori_value = pointer.get(k_parts[-1])
            pointer[k_parts[-1]] = v
        if ori_value:
            logger.warning(f"Overrided {k} from {ori_value} to {v}")
        elif log_new:
            logger.warning(f"Added {k}: {v}")


def load_opt_from_config_files(conf_files):
    """
    Load opt from the config files, settings in later files can override those in previous files.

    Args:
        conf_files (list): a list of config file paths

    Returns:
        dict: a dictionary of opt settings
    """
    opt = {}
    for conf_file in conf_files:
        with open(conf_file, encoding='utf-8') as f:
            # config_dict = yaml.safe_load(f)
            config_dict = yaml.unsafe_load(f)

        load_config_dict_to_opt(opt, config_dict)

    return opt


def load_opt_command(args):
    parser = argparse.ArgumentParser(description='MainzTrain: Pretrain or fine-tune models for NLP tasks.')
    parser.add_argument('command', help='Command: train/evaluate/train-and-evaluate')
    parser.add_argument('--conf_files', nargs='+', required=True, help='Path(s) to the MainzTrain config file(s).')
    parser.add_argument('--user_dir', help='Path to the user defined module for tasks (models, criteria), optimizers, and lr schedulers.')
    parser.add_argument('--config_overrides', nargs='*', help='Override parameters on config with a json style string, e.g. {"<PARAM_NAME_1>": <PARAM_VALUE_1>, "<PARAM_GROUP_2>.<PARAM_SUBGROUP_2>.<PARAM_2>": <PARAM_VALUE_2>}. A key with "." updates the object in the corresponding nested dict. Remember to escape " in command line.')

    cmdline_args = parser.parse_args() if not args else parser.parse_args(args)

    add_env_parser_to_yaml()
    opt = load_opt_from_config_files(cmdline_args.conf_files)

    if cmdline_args.config_overrides:
        config_overrides_string = ' '.join(cmdline_args.config_overrides)
        config_overrides_string = os.path.expandvars(config_overrides_string)
        logger.warning(f"Command line config overrides: {config_overrides_string}")
        config_dict = yaml.safe_load(config_overrides_string)
        load_config_dict_to_opt(opt, config_dict)

    # combine cmdline_args into opt dictionary
    for key, val in cmdline_args.__dict__.items():
        if val is not None:
            opt[key] = val

    return opt, cmdline_args


def save_opt_to_json(opt, conf_file):
    with open(conf_file, 'w', encoding='utf-8') as f:
        json.dump(opt, f, indent=4)


def save_opt_to_yaml(opt, conf_file):
    with open(conf_file, 'w', encoding='utf-8') as f:
        yaml.dump(opt, f)
