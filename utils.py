from PIL import Image
import torch
import torch.utils.data as data

class DatasetCreator(data.Dataset):
	def __init__(self, img_paths, transform=None):
		self.img_paths = img_paths
		self.transform = transform

	def __getitem__(self,index):
		img_path = self.img_paths[index]
		img = Image.open(img_path)
		img = img.convert('RGB')
		if self.transform is not None:
			img = self.transform(img)
		return img

	def __len__(self):
		return len(self.img_paths)
