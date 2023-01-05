import torch
import torch.nn as nn
import torch.nn.functional as F

class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=2, stride=2),#(32,50,50)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=2, stride=2),#(64,25,25)
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.flatten = nn.Flatten(start_dim=1)
        self.encoder_lin = nn.Sequential(
            nn.Linear(16384,256),
            nn.ReLU(),
            nn.Linear(in_features=256,out_features = 128)
        )
        
        self.decoder_lin = nn.Sequential(
            nn.Linear(128,256),
            nn.ReLU(),
            nn.Linear(256,254016),
            nn.ReLU()
        )
        self.unflatten = nn.Unflatten(dim=1,unflattened_size=(64,63,63))

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64,out_channels=32,kernel_size=4,stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=32,out_channels=3,kernel_size=2,stride=2)
        )
        
    
    
    def train(self,model, params):
        num_epochs=params["num_epochs"]
        loss_function=params["loss_function"]
        optimizer=params["optimizer"]
        train_dataloader=params["train_dataloader"]
        test_dataloader=params["test_dataloader"]
        device=params["device"]
        
        for epoch in range(0, num_epochs):
            for i, data in enumerate(train_dataloader, 0):
                # train dataloader 로 불러온 데이터에서 이미지와 라벨을 분리
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                optimizer.zero_grad() 
                
                outputs=model(inputs)
                print(inputs.shape)
                print(outputs.shape)

                # forward + back propagation 연산
                train_loss = loss_function(outputs, inputs)
                train_loss.backward()
                
                optimizer.step()

            # test accuracy 계산
            total = 0
            correct = 0
            accuracy = []
            for i, data in enumerate(test_dataloader, 0):
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                outputs=model(inputs)

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                test_loss = loss_function(outputs, labels).item()
                accuracy.append(100 * correct/total)
        

    # 학습 결과 출력
        print('Epoch: %d/%d, Train loss: %.6f, Test loss: %.6f, Accuracy: %.2f' %(epoch+1, num_epochs, train_loss.item(), test_loss, 100*correct/total))
    
    def forward(self, x):
        x = self.encoder(x)
        x = F.max_pool2d(x,4)
        x = self.flatten(x)
        x = self.encoder_lin(x)
        x = self.decoder_lin(x)
        x = self.unflatten(x)
        x = self.decoder(x)
        return x     
                      

