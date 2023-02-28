import urllib.request
import os
import zipfile
import logging

log = logging.getLogger("21cmEMU")

class Download_EMU:
    """
    Class that downloads a version of the 21cmEMU emulator.
    
    Parameters
    ----------
    url : str, optional
        21cmEMU download link. Default is dropbox link.
    destination_dir : str, optional
        path where to download 21cmEMU. Default is in the current directory.
    version : str, optional
        When multiple versions will be available, one will be able to specify 
        the version number instead of the link.
    """
    
    def __init__(self, url = None, 
                 destination_dir = None,
                 version = 'v1'):
        if url is None:
            if version == 'v1' or version is None:
                url = ''
            else:
                raise ValueError('The only version currently available is v1')
        self.url = url
        
        if destination_dir is None:
            destination_dir = os.path.dirname(os.path.realpath(__file__)) + '/'
        else:
            if destination_dir[-1] != '/':
                destination_dir = destination_dir + '/'
        self.destination_dir = destination_dir
           
        
    def download_and_extract(self):
        # download and extract the emulator
        try:
            urllib.request.urlretrieve(self.url, self.destination_dir + 'zipped_emulator')
            if os.path.isfile(self.destination_dir + 'zipped_emulator'):
                log.info('Downloaded the emulator successfully')
        except Exception as e:
            raise ValueError('Download failed, wrong link provided', e)
        try:
            with zipfile.ZipFile(self.destination_dir + 'zipped_emulator', 'r') as zip_ref:
                zip_ref.extractall(self.destination_dir)
            if os.path.isdir(self.destination_dir + '21cmEMU'):
                log.info('Extracted the emulator successfully')
        except Exception as e:
            raise TypeError('The downloaded file is not a zipped file. Check that the download link is correct!', e)
        return 1
