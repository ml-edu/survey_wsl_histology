from PIL import Image
from torch.utils.data import Dataset
from pathlib import Path
from os.path import join
from torchvision import transforms
import csv

class PhotoDataset(Dataset):
    def __init__(self, data_path, files, transform, preload=False, resize=None, with_mask=False):
        from .utils import check_files
        from utils.data.utils import load_data

        self.transform = transform
        self.resize = resize
        self.with_mask = with_mask

        self.samples = check_files(data_path, files)
        self.n = len(self.samples)

        self.preloaded = False
        if preload:
            self.images = load_data([image_path for image_path, _, _ in self.samples], resize=resize)
            if with_mask:
                self.masks = load_data([mask_path for _, mask_path, _ in self.samples if mask_path != ''],
                                       resize=resize)
            self.preloaded = True
            print(self.__class__.__name__ + ' loaded with {} images'.format(len(self.images.keys())))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, mask_path, label = self.samples[index]

        if self.preloaded:
            image = self.images[image_path].convert('RGB')
        else:
            image = Image.open(image_path).convert('RGB')
            if self.resize is not None:
                image = image.resize(self.resize, resample=Image.LANCZOS)
        image_size = image.size # to generate the mask if there is no file

        if self.transform is not None:
            image = self.transform(image)

        if self.with_mask:
            if mask_path == '':
                mask = Image.new('L', image_size)
            else:
                if self.preloaded:
                    mask = self.masks[mask_path].convert('L')
                else:
                    mask = Image.open(mask_path).convert('L')
                    if self.resize is not None:
                        mask = mask.resize(self.resize, resample=Image.LANCZOS)

            if self.transform is not None:
                mask = self.transform(mask)

            return image, mask, label

        return image, label


def get_files(dir_path, ext):
    return [str(path.name) for path in Path(dir_path).rglob('*.{}'.format(ext))]


def get_paths(dir_path, ext):
    return [str(path) for path in Path(dir_path).rglob('*.{}'.format(ext))]

def csv_reader(fname):
    with open(fname, 'r') as f:
        out = list(csv.reader(f))
    return out

class CamsDataset(Dataset):

    def __init__(self, data_dir):

        file_paths = get_paths(data_dir, 'bmp')
        file_names = get_files(data_dir, 'bmp')

        img_file_paths = list(filter(lambda f: 'anno' not in f, file_paths))
        img_file_names = list(filter(lambda f: 'anno' not in f, file_names))

        csv = csv_reader(get_paths(data_dir, 'csv')[0])

        self.img_file_names = img_file_names
        self.images = self.load_images(img_file_paths)
        self.labels = [self.get_label(f, csv) for f in img_file_names]
        self.masks = self.get_masks(img_file_paths)
        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index]
        mask = self.masks[index]
        label = self.labels[index]
        name = self.img_file_names[index]

        image = self.transform(image)
        mask = self.transform(mask)
        name = name.replace('.bmp', '')

        return image, mask, label, name

    @staticmethod
    def load_images(file_paths):
        images = []
        for f in file_paths:
            images.append(Image.open(f))

        return images

    @staticmethod
    def get_label(file, csv):
        name = file.replace('.bmp', '')
        lbl = None
        for line in csv:
            if line[0] == name:
                lbl = line[2].strip()
                break
        return 0 if lbl == 'benign' else 1

    @staticmethod
    def get_masks(img_files):
        masks = []
        for f in img_files:
            mask_file = f.replace('.bmp', '_anno.bmp')
            masks.append(Image.open(mask_file))
        return masks

if __name__ == "__main__":
    ds = CamsDataset('/home/victor/PycharmProjects/survey_wsl_histology/data/GlaS/')

    for i in range(ds.__len__()):
        image, mask, label, file_name = ds.__getitem__(i)
        print(file_name)