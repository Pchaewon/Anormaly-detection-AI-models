from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms

class Dataloader():
    def __init__(self,path_data='data/resized/'):
        self.train_path=path_data+'train/'
        self.test_path=path_data+'test/'
        #transform
        self.train_transformation = transform = transforms.Compose([transforms.Resize((256, 256)),
                                transforms.CenterCrop(256),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.test_transformation = transform = transforms.Compose([transforms.Resize((256, 256)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        
        #AE transform
        self.AE_train_transformation = transform = transforms.Compose([transforms.Resize((100, 100)),
                                transforms.CenterCrop(100),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.AE_test_transformation = transform = transforms.Compose([transforms.Resize((100, 100)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        
    def total_dataset(self): #total dataset loading
        train_imgs=ImageFolder(self.train_path,self.train_transformation)
        train_loader=DataLoader(train_imgs,batch_size=128)
        test_imgs=test_imgs=ImageFolder(self.test_path,self.test_transformation)
        test_loader=DataLoader(test_imgs,batch_size=128)
        return train_loader, test_loader

    def AE_total_dataset(self): #total dataset loading
        train_imgs=ImageFolder(self.train_path,self.AE_train_transformation)
        train_loader=DataLoader(train_imgs,batch_size=128)
        test_imgs=test_imgs=ImageFolder(self.test_path,self.AE_test_transformation)
        test_loader=DataLoader(test_imgs,batch_size=128)
        return train_loader, test_loader

    def normal_dataset(self):
        pass
    def abnormal_dataset(self):
        pass