import os
import time
import shutil
import logging
import subprocess
import os.path as op
from typing import List
from collections import OrderedDict

import torch.distributed as distributed

logger = logging.getLogger(__name__)

DEFAULT_AZCOPY_PATH = 'azcopy/azcopy'


def disk_usage(path: str) -> float:
    stat = shutil.disk_usage(path)
    return stat.used / stat.total


def is_download_successful(stdout: str) -> bool:
    for line in stdout.split('\n'):
        if line == "Number of Transfers Failed: 0":
            return True
    logger.info("Azcopy message:\n %s" % stdout)
    return False


def ensure_directory(path):
    """Check existence of the given directory path. If not, create a new directory.

    Args:
        path (str): path of a given directory.
    """
    if path == '' or path == '.':
        return
    if path is not None and len(path) > 0:
        assert not op.isfile(path), '{} is a file'.format(path)
        if not op.exists(path) and not op.islink(path):
            os.makedirs(path, exist_ok=True)
        # we should always check if it succeeds.
        assert op.isdir(op.abspath(path)), path


class LRU(OrderedDict):
    def __init__(self, maxsize=3):
        self.maxsize = maxsize

    def __getitem__(self, key):
        value = super().__getitem__(key)
        self.move_to_end(key)
        return value

    def __setitem__(self, key, value):
        if key in self:
            if self[key] is not None:
                self[key].close()
                self.move_to_end(key)

        logger.debug('=> Cache {}'.format(key))
        super().__setitem__(key, value)

        if len(self) > self.maxsize:
            oldest = next(iter(self))
            if self[oldest] is not None:
                self[oldest].close()
            logger.debug('=> Purged {}'.format(oldest))
            del self[oldest]


class BlobStorage(OrderedDict):
    """ Pseudo Blob Storage manager

    The registered blobs are maintained in a LRU cache.
    Limit size, evicting the least recently looked-up key when full.
    https://docs.python.org/3/library/collections.html#collections.OrderedDict

    Input argument:
        sas_token (str): path to SAS token.
    """
    def __init__(self,
                 is_train: bool,
                 sas_token_path: str = None,
                 azcopy_path: str = None,
                 *args, **kwds):
        super().__init__(*args, **kwds)
        self.maxsize = 2 if is_train else 10    # Set maxsize to large number such val data never get purged.
        self.is_train = is_train

        if sas_token_path:
            self.sas_token = BlobStorage.read_sas_token(sas_token_path)
            self.base_url = self.sas_token[:self.sas_token.index("?")]
            self.query_string = self.sas_token[self.sas_token.index("?"):]
            self.container = BlobStorage.extract_container(self.sas_token)
        else:
            self.sas_token = None
            self.base_url = None
            self.query_string = None
            self.container = None

        logger.debug(
            f"=> [BlobStorage] Base url: {self.base_url}"
            f"=> [BlobStorage] Query string: {self.query_string}"
            f"=> [BlobStorage] Container name: {self.container}"
        )

        self.azcopy_path = azcopy_path if azcopy_path else DEFAULT_AZCOPY_PATH
        self._cached_files = LRU(3)

    def __getitem__(self, key):
        value = super().__getitem__(key)
        self.move_to_end(key)
        return value

    def __setitem__(self, key, value):
        if key in self:
            self.move_to_end(key)
        super().__setitem__(key, value)
        # NOTE: purge the least recently used data if the disk usage is high.
        # ITP restarts GPU clusters when disk usage reaches 80%.
        if len(self) > self.maxsize:
            oldest = next(iter(self))
            del self[oldest]

    @staticmethod
    def read_sas_token(path: str) -> str:
        with open(path, 'r') as f:
            token = f.readline().strip()
        return token

    @staticmethod
    def extract_container(token: str) -> str:
        """
        Input argument:
            token (str): the full URI of Shared Access Signature (SAS) in the following format.
            https://[storage_account].blob.core.windows.net/[container_name][SAS_token]
        """
        return os.path.basename(token.split('?')[0])

    def _convert_to_blob_url(self, local_path: str):
        return self.base_url + local_path.split("azcopy")[1] + self.query_string

    def _convert_to_blob_folder_url(self, local_path: str):
        return self.base_url + local_path.split("azcopy")[1] + "/*" + self.query_string

    def fetch_blob(self, local_path: str) -> None:
        if op.exists(local_path):
            logger.info('=> Try to open {}'.format(local_path))
            fp = open(local_path, 'r')
            self._cached_files[local_path] = fp
            logger.debug("=> %s downloaded. Skip." % local_path)
            return
        blob_url = self._convert_to_blob_url(local_path)
        rank = '0' if 'RANK' not in os.environ else os.environ['RANK']
        cmd = [self.azcopy_path, "copy", blob_url, local_path + rank]
        curr_usage = disk_usage('/')
        logger.info(
            "=> Downloading %s with azcopy ... (disk usage: %.2f%%)"
            % (local_path, curr_usage * 100)
        )
        proc = subprocess.run(cmd, stdout=subprocess.PIPE)
        while not is_download_successful(proc.stdout.decode()):
            logger.info("=> Azcopy failed to download {}. Retrying ...".format(blob_url))
            proc = subprocess.run(cmd, stdout=subprocess.PIPE)
        if not op.exists(local_path):
            os.rename(local_path + rank, local_path)
        else:
            os.remove(local_path + rank)
        logger.info(
            "=> Downloaded %s with azcopy ... (disk usage: %.2f%% => %.2f%%)" %
            (local_path, curr_usage * 100, disk_usage('/') * 100)
        )

    def fetch_blob_folder(self, local_path: str, azcopy_args: list=[]) -> None:
        blob_url = self._convert_to_blob_folder_url(local_path)
        cmd = [self.azcopy_path, "copy", blob_url, local_path] + azcopy_args
        curr_usage = disk_usage('/')
        logger.info(
            "=> Downloading %s with azcopy args %s ... (disk usage: %.2f%%)"
            % (local_path, ' '.join(azcopy_args), curr_usage * 100)
        )
        proc = subprocess.run(cmd, stdout=subprocess.PIPE)
        while not is_download_successful(proc.stdout.decode()):
            logger.info("=> Azcopy failed to download {} with args {}. Retrying ...".format(blob_url, ' '.join(azcopy_args)))
            proc = subprocess.run(cmd, stdout=subprocess.PIPE)
        logger.info(
            "=> Downloaded %s with azcopy args %s ... (disk usage: %.2f%% => %.2f%%)" %
            (local_path, ' '.join(azcopy_args), curr_usage * 100, disk_usage('/') * 100)
        )

    def register_local_tsv_paths(self, local_paths: List[str]) -> List[str]:
        if self.sas_token:
            tsv_paths_new = []
            lineidx_paths = set()
            linelist_paths = set()
            for path in local_paths:
                tsv_path_az = path.replace(self.container, 'azcopy')
                tsv_paths_new.append(tsv_path_az)
                logger.debug("=> Registering {}".format(tsv_path_az))

                if not self.is_train:
                    logger.info('=> Downloading {}...'.format(tsv_path_az))
                    self.fetch_blob(tsv_path_az)
                    logger.info('=> Downloaded {}'.format(tsv_path_az))

                lineidx = op.splitext(path)[0] + '.lineidx'
                lineidx_ = lineidx.replace(self.container, 'azcopy')
                if self.is_train:
                    if not op.isfile(lineidx_) and op.dirname(lineidx_) not in lineidx_paths:
                        lineidx_paths.add(op.dirname(lineidx_))
                else:
                    if not op.isfile(lineidx_):
                        ensure_directory(op.dirname(lineidx_))
                        self.fetch_blob(lineidx_)

                linelist = op.splitext(path)[0] + '.linelist'
                linelist_ = linelist.replace(self.container, 'azcopy')
                # .linelist does not always exist. Check existence before fetch
                if self.is_train:
                    if op.isfile(linelist) and not op.isfile(linelist_) and op.dirname(linelist_) not in linelist_paths:
                        linelist_paths.add(op.dirname(linelist_))
                else:
                    if op.isfile(linelist) and not op.isfile(linelist_):
                        ensure_directory(op.dirname(linelist_))
                        self.fetch_blob(linelist_)

            if self.is_train:
                for path in lineidx_paths:
                    self.fetch_blob_folder(path, azcopy_args=['--include-pattern', '*.lineidx'])

                for path in linelist_paths:
                    self.fetch_blob_folder(path, azcopy_args=['--include-pattern', '*.linelist'])

            return tsv_paths_new
        else:
            return local_paths

    def open(self, local_path: str):
        if self.sas_token and 'azcopy' in local_path:
            while not op.exists(local_path):
                time.sleep(1)
        fid = open(local_path, 'r')
        return fid
