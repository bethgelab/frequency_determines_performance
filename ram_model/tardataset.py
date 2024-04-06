
import tarfile
from io import BytesIO
from PIL import Image, ImageFile
import json

from torch.utils.data import Dataset, get_worker_info

try:  # make torchvision optional
  from torchvision.transforms.functional import to_tensor
except:
  to_tensor = None

ImageFile.LOAD_TRUNCATED_IMAGES = True


class UnexpectedEOFTarFile(tarfile.TarFile):
  def _load(self):
    """Read through the entire archive file and look for readable
       members.
    """
    try:
      while True:
        tarinfo = self.next()
        if tarinfo is None:
          break
    except tarfile.ReadError as e:
      assert e.args[0] == "unexpected end of data"
    self._loaded = True


class TarDataset(Dataset):
  """Dataset that supports Tar archives (uncompressed).

  Args:
    archive (string or TarDataset): Path to the Tar file containing the dataset.
      Alternatively, pass in a TarDataset object to reuse its cached information;
      this is useful for loading different subsets within the same archive.
    extensions (tuple): Extensions (strings starting with a dot), only files
      with these extensions will be iterated. Default: png/jpg/jpeg.
    is_valid_file (callable): Optional function that takes file information as
      input (tarfile.TarInfo) and outputs True for files that need to be
      iterated; overrides extensions argument.
      Example: lambda m: m.isfile() and m.name.endswith('.png')
    transform (callable): Function applied to each image by __getitem__ (see
      torchvision.transforms). Default: ToTensor (convert PIL image to tensor).
    ignore_unexpected_eof (bool): ignore Unexpected EOF when iterating the tar file
      allows working with Datasets cut to a smaller size with dd

  Attributes:
    members_by_name (dict): Members (files and folders) found in the Tar archive,
      with their names as keys and their tarfile.TarInfo structures as values.
    samples (list): Items to iterate (can be ignored by overriding __getitem__
      and __len__).

  Author: Joao F. Henriques
  """
  def __init__(self, archive, transform=to_tensor, tagset=None, extensions=('.png', '.jpg', '.jpeg'),
    is_valid_file=None, ignore_unexpected_eof=False, assume_jsons=True):
    if not isinstance(archive, TarDataset):
      # open tar file. in a multiprocessing setting (e.g. DataLoader workers), we
      # have to open one file handle per worker (stored as the tar_obj dict), since
      # when the multiprocessing method is 'fork', the workers share this TarDataset.
      # we want one file handle per worker because TarFile is not thread-safe.
      worker = get_worker_info()
      worker = worker.id if worker else None
      self.tar_obj = {worker: tarfile.open(archive) if ignore_unexpected_eof is False else UnexpectedEOFTarFile.open(archive)}
      self.archive = archive
      self.assume_jsons = assume_jsons

      # store headers of all files and folders by name
      members = sorted(self.tar_obj[worker].getmembers(), key=lambda m: m.name)
      self.members_by_name = {m.name: m for m in members}
    else:
      # passed a TarDataset into the constructor, reuse the same tar contents.
      # no need to copy explicitly since this dict will not be modified again.
      self.members_by_name = archive.members_by_name
      self.archive = archive.archive  # the original path to the Tar file
      self.tar_obj = {}  # will get filled by get_file on first access

    # also store references to the iterated samples (a subset of the above)
    self.filter_samples(is_valid_file, extensions)
    self.tagset = tagset
    self.transform = transform


  def filter_samples(self, is_valid_file=None, extensions=('.png', '.jpg', '.jpeg')):
    """Filter the Tar archive's files/folders to obtain the list of samples.
    
    Args:
      extensions (tuple): Extensions (strings starting with a dot), only files
        with these extensions will be iterated. Default: png/jpg/jpeg.
      is_valid_file (callable): Optional function that takes file information as
        input (tarfile.TarInfo) and outputs True for files that need to be
        iterated; overrides extensions argument.
        Example: lambda m: m.isfile() and m.name.endswith('.png')
    """
    # by default, filter files by extension
    # Assume that all jpgs have associated json file
    if is_valid_file is None:
      def is_valid_file(m):
        return (m.isfile() and m.name.lower().endswith(extensions))

    # filter the files to create the samples list
    self.samples = [m.name for m in self.members_by_name.values() if is_valid_file(m)]

  def __getitem__(self, index):
    """Return a single sample.
    
    Should be overriden by a subclass to support custom data other than images (e.g.
    class labels). The methods get_image/get_file can be used to read from the Tar
    archive, and a dict of files/folders is held in the property members_by_name.

    By default, this simply applies the given transforms or converts the image to
    a tensor if none are specified.

    Args:
      index (int): Index of item.
    
    Returns:
      Tensor: The image.
    """
    jpg_name = self.samples[index]
    # Assume that all jpgs have associated json file
    if self.assume_jsons:
      json_name = jpg_name.rsplit('.', 1)[0] + '.json'

    image = self.get_image(jpg_name)
    
    if self.tagset is not None:
      for tags in self.tagset[self.samples[index]]:
        texts = ['a photo of '+tag for tag in tags]
        image = self.transform(text=[texts], images=image, return_tensors="pt")
    elif self.transform:  # apply any custom transforms
      image = self.transform(image)
    if self.assume_jsons:
      json_data = json.loads(self.get_text_file(json_name))
    else:
      json_data = {}
    
    assert image is not None, 'image {} is None'.format(jpg_name)
    assert json_data is not None, 'json {} is None'.format(json_name)
    assert self.samples[index] is not None, 'sample {} is None'.format(index)

    return image, json_data, self.samples[index]

  def __len__(self):
    """Return the length of the dataset (length of self.samples)

    Returns:
      int: Number of samples.
    """
    return len(self.samples)


  def get_image(self, name):
    """Read an image from the Tar archive, returned as a PIL image or PyTorch tensor.

    Args:
      name (str): File name to retrieve.
      pil (bool): If true, a PIL image is returned (default is a PyTorch tensor).

    Returns:
      Image or Tensor: The image, possibly in PIL format.
    """
    image = Image.open(BytesIO(self.get_file(name).read()))
    return image



  def get_text_file(self, name, encoding='utf-8'):
    """Read a text file from the Tar archive, returned as a string.

    Args:
      name (str): File name to retrieve.
      encoding (str): Encoding of file, default is utf-8.

    Returns:
      str: Content of text file.
    """
    return self.get_file(name).read().decode(encoding)


  def get_file(self, name):
    """Read an arbitrary file from the Tar archive.

    Args:
      name (str): File name to retrieve.

    Returns:
      io.BufferedReader: Object used to read the file's content.
    """
    # ensure a unique file handle per worker, in multiprocessing settings
    worker = get_worker_info()
    worker = worker.id if worker else None

    if worker not in self.tar_obj:
      self.tar_obj[worker] = tarfile.open(self.archive)

    return self.tar_obj[worker].extractfile(self.members_by_name[name])


  def __del__(self):
    """Close the TarFile file handles on exit."""
    for o in self.tar_obj.values():
      o.close()


  def __getstate__(self):
    """Serialize without the TarFile references, for multiprocessing compatibility."""
    state = dict(self.__dict__)
    state['tar_obj'] = {}
    return state
