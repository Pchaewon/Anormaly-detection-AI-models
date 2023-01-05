import torch
from torchvision.utils import save_image
import torch.optim as optim
from model.core.solver import *
from model.core.utils import *
import model.UNet as UNet
from model.diffusion import DiffusionModel
import model.core.evaluation as evaluation
import matplotlib.pyplot as plt
import os
import time
import numpy as np
import copy
import collections

def training_outputs(diffusion, x, est, noisy, epoch, row_size, ema, params):
    ARG_NUM=params["ARG_NUM"]
    SAMPLE_DISTANCE=params["SAMPLE_DISTANCE"]
    save_imgs=params["SAVE_IMGS"]
    save_vids=params["SAVE_VIDS"]
    
    try:
        os.makedirs(f'./diffusion-videos/ARGS={ARG_NUM}')
        os.makedirs(f'./diffusion-training-images/ARGS={ARG_NUM}')
    except OSError:
        pass
    if save_imgs:
        if epoch % 100 == 0:
            # for a given t, output x_0, & prediction of x_(t-1), and x_0
            noise = torch.rand_like(x)
            t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=x.device)
            x_t = diffusion.sample_q(x, t, noise)
            temp = diffusion.sample_p(ema, x_t, t)
            out = torch.cat(
                    (x[:row_size, ...].cpu(), temp["sample"][:row_size, ...].cpu(),
                     temp["pred_x_0"][:row_size, ...].cpu())
                    )
            plt.title(f'real,sample,prediction x_0-{epoch}epoch')
        else:
            # for a given t, output x_0, x_t, & prediction of noise in x_t & MSE
            out = torch.cat(
                    (x[:row_size, ...].cpu(), noisy[:row_size, ...].cpu(), est[:row_size, ...].cpu(),
                     (est - noisy).square().cpu()[:row_size, ...])
                    )
            plt.title(f'real,noisy,noise prediction,mse-{epoch}epoch')
        plt.rcParams['figure.dpi'] = 150
        plt.grid(False)
        plt.imshow(gridify_output(out, row_size), cmap='gray')

        plt.savefig(f'./diffusion-training-images/ARGS={ARG_NUM}/EPOCH={epoch}.png')
        plt.clf()
    if save_vids:
        fig, ax = plt.subplots()
        if epoch % 500 == 0:
            plt.rcParams['figure.dpi'] = 200
            if epoch % 1000 == 0:
                #out = diffusion.forward_backward(ema, x, "half", SAMPLE_DISTANCE // 2, denoise_fn="noise_fn")
                print("ep 1000, check pls")
            else:
                out = diffusion.forward_backward(ema, x, "half", SAMPLE_DISTANCE // 4, denoise_fn="noise_fn")
            #imgs = [[ax.imshow(gridify_output(x, row_size), animated=True)] for x in out]
            #ani = animation.ArtistAnimation(fig, imgs, interval=50, blit=True,repeat_delay=1000)

            #ani.save(f'{ROOT_DIR}diffusion-videos/ARGS={ARG_NUM}/sample-EPOCH={epoch}.mp4')

    plt.close('all')
    
class trainer():
    def __init__(self,model):
        super(trainer, self).__init__()
        self.model=model
        
    def trainer(self, params):
            num_epochs=params["num_epochs"]
            loss_function=params["loss_function"]
            train_dataloader=params["train_dataloader"]
            test_dataloader=params["test_dataloader"]
            optimizer=params["optimizer"]
            device=params["device"]

            for epoch in range(0, num_epochs):
                for i, data in enumerate(train_dataloader, 0):
                    # train dataloader 로 불러온 데이터에서 이미지와 라벨을 분리
                    inputs, labels = data
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # 이전 batch에서 계산된 가중치를 초기화
                    optimizer.zero_grad() 

                    # forward + back propagation 연산
                    outputs = self.model(inputs)
                    train_loss = loss_function(outputs, labels)
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

                    # 결과값 연산
                    outputs = self.model(inputs)

                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    test_loss = loss_function(outputs, labels).item()
                    accuracy.append(100 * correct/total)
                

                # 학습 결과 출력
                print('Epoch: %d/%d, Train loss: %.6f, Test loss: %.6f, Accuracy: %.2f' %(epoch+1, num_epochs, train_loss.item(), test_loss, 100*correct/total))
    
    def AE_trainer(self, params):
        num_epochs=params["num_epochs"]
        loss_function=params["loss_function"]
        train_dataloader=params["train_dataloader"]
        test_dataloader=params["test_dataloader"]
        optimizer=params["optimizer"]
        device=params["device"]

        for epoch in range(num_epochs):
            for index, (data,_) in enumerate(train_dataloader):
                data = data.to(device)
                
                output = self.model(data)

                loss = loss_function(output,data)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            print('epoch [{}/{}], loss:{:.4f}' .format(epoch + 1, num_epochs, float(loss.data) ))
            if epoch % 10 == 0:
                    # pic = to_img(output.cpu().data)
                    pic = output.cpu().data
                    pic = pic.view(pic.size(0), 3, 256, 256)
            
                    save_image(pic, './output/output_image_{}.png'.format(epoch))

    def anoddpm_trainer(self, params):
        # params
        num_epochs=params["EPOCHS"]
        train_dataloader=params["train_dataloader"]
        test_dataloader=params["test_dataloader"]
        optimizer=params["optimizer"]
        device=params["device"]
        model=params['model']
        diffusion=params['diffusion']
        ema=params['ema']
        
        batch_size=1
        start_epoch=0
        start_time = time.time()
        losses = []
        vlb = collections.deque([], maxlen=10)
        
        for epoch in range(0, num_epochs):
            mean_loss=[]
            
            for i in range(200):
                data=next(iter(train_dataloader))
                x=data[0].to(device)
                
                loss, estimates = diffusion.p_loss(model, x)
                noisy,est = estimates[1],estimates[2]
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),1)
                optimizer.step()

                update_ema_params(ema,model)
                mean_loss.append(loss.data.cpu())
                if epoch % 50 == 0 and i == 0:
                    row_size = min(8, batch_size)
                    training_outputs(
                            diffusion, x, est, noisy, epoch, row_size, save_imgs=False,save_vids=True, ema=ema)


            losses.append(np.mean(mean_loss))
            if epoch % 200 == 0:
                time_taken = time.time() - start_time
                remaining_epochs = num_epochs - epoch
                time_per_epoch = time_taken / (epoch + 1 - start_epoch)
                hours = remaining_epochs * time_per_epoch / 3600
                mins = (hours % 1) * 60
                hours = int(hours)

                vlb_terms = diffusion.calc_total_vlb(x, model)
                vlb.append(vlb_terms["total_vlb"].mean(dim=-1).cpu().item())
                print(
                        f"epoch: {epoch}, most recent total VLB: {vlb[-1]} mean total VLB:"
                        f" {np.mean(vlb):.4f}, "
                        f"prior vlb: {vlb_terms['prior_vlb'].mean(dim=-1).cpu().item():.2f}, vb: "
                        f"{torch.mean(vlb_terms['vb'], dim=list(range(2))).cpu().item():.2f}, x_0_mse: "
                        f"{torch.mean(vlb_terms['x_0_mse'], dim=list(range(2))).cpu().item():.2f}, mse: "
                        f"{torch.mean(vlb_terms['mse'], dim=list(range(2))).cpu().item():.2f}"
                        f" time elapsed {int(time_taken / 3600)}:{((time_taken / 3600) % 1) * 60:02.0f}, "
                        f"est time remaining: {hours}:{mins:02.0f}\r"
                        )
            
            if epoch % 1000 == 0 and epoch >= 0:
                save(unet=model, optimizer=optimizer, final=False, ema=ema, epoch=epoch)

        save(unet=model, optimizer=optimizer, final=True, ema=ema)
        evaluation.testing(test_dataloader, diffusion, ema=ema,model=model)
