from tqdm import tqdm
import torch

from evaluation import calc_accuracy, get_confusion_matrix_image, get_mean_squared_error
from evaluation import get_sample_dict, update_hardsample_indice, draw_cam
import torchvision.utils as vutils
from utils import tensor_rgb2bgr



def trainer(
    max_epoch, 
    model, 
    train_loader, 
    test_loader, 
    loss_fn,
    optimizer,
    scheduler,
    meta, 
    writer = None
):

    save_every = meta['save_every']
    print_every = meta['print_every']
    test_every = meta['test_every']


    for ep in range(1, max_epoch+1):
        train(ep, max_epoch, model, train_loader, loss_fn, optimizer, writer, print_every)
        if scheduler is not None:
            scheduler.step()


        if ep % test_every == 0:
            loss = test(ep, max_epoch, model, test_loader, writer, loss_fn = loss_fn)
            loss *= -1

            
            writer.update(model, loss)
        
        if ep == 1 or ep % save_every == 0:
            writer.save(model, ep)
            
    writer.close()
    

def tester(
    model,
    test_loader,
    writer,
    hard_sample_mining,
    confusion_matrix,
    n_class,
    task_type
):
    pbar=tqdm(total=len(test_loader))
    acc = test(
        None,None,
        model, test_loader, writer,
        pbar = pbar,
        hard_sample_mining = hard_sample_mining,
        confusion_matrix = confusion_matrix,
        n_class = n_class,
        task_type = task_type
    )
    
    writer.close()




def train(ep, max_epoch, model, train_loader, loss_fn, optimizer, writer, _print_every):
    model.train()

    epoch_loss = 0.0
    mean_loss = 0.0

    print_every = len(train_loader) // _print_every     
    if print_every == 0:
        print_every = 1

    recon = []
    gts = []

    step = 0
    step_cnt = 1

    global_step = (ep - 1) * len(train_loader)
    local_step = 0

    for i, batch in enumerate(train_loader):
        x, y = batch
        x = x.cuda()


        x_hat,latent_code = model(x)
        loss = loss_fn(x_hat, x)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        mean_loss += loss.item()
        epoch_loss += loss.item()
        step += 1
        global_step += 1
        local_step += 1


        if (i+1) % print_every == 0:
            mean_loss /= step
            writer.add_scalar('train/loss', mean_loss, global_step)
            print('Epoch [{}/{}] Step[{}/{}] Loss: {:.4f}'.format(
                ep, max_epoch, step_cnt, _print_every, mean_loss))
            
            mean_loss = 0.0
            step = 0
            step_cnt += 1

            # add_img
            recon_image = vutils.make_grid(x_hat.detach().cpu().clamp(0.0,1.0), normalize=True, scale_each=True)
            gt_image    = vutils.make_grid(x.detach().cpu().clamp(0.0,1.0), normalize=True, scale_each=True)

            if writer is not None:
                writer.add_image('train/recon_img', recon_image, global_step)
                writer.add_image('train/gt_img', gt_image, global_step)

    print ('Train Summary[{},{}] : Loss: {:.4f}'.format(ep, max_epoch, epoch_loss/local_step))

@torch.no_grad() # stop calculating gradient
def test(ep, max_epoch, model, test_loader, writer, loss_fn=None, pbar=None):
    model.eval()

    epoch_loss = 0.0
    local_step = 0
    global_step = (ep - 1) * len(test_loader)

    
    for idx, batch in enumerate(test_loader):
        x, y = batch
        x = x.cuda()

        x_hat, latent_code = model(x)
        loss = loss_fn(x_hat, x)
        epoch_loss += loss.item()

        local_step += 1
        
        if idx % 10 ==0:
            recon_image = vutils.make_grid(x_hat.detach().cpu().clamp(0.0,1.0), normalize=True, scale_each=True)
            gt_image    = vutils.make_grid(x.cpu().clamp(0.0,1.0), normalize=True, scale_each=True)

            if writer is not None:
                writer.add_image('test/recon_img', recon_image, global_step)
                writer.add_image('test/gt_img', gt_image, global_step)


        if pbar is not None:
            pbar.update()

    epoch_loss /=local_step
    print ('Test Summary[{}/{}] : Loss {:.4f}'.format(ep, max_epoch, epoch_loss))
    writer.add_scalar('test/loss', epoch_loss, (ep-1) * len(test_loader))

        
       
    return epoch_loss

def grad_cam(model, data_loader, writer, cam, export_csv, n_class, task_type):
    model.eval()
    pbar = tqdm(total=len(data_loader))

    print ('Dataset length: {}'.format(len(data_loader)))

    for idx, batch in enumerate(data_loader):
        x = batch['x']
        y = batch['y']
        f_name = batch['f_name']

        x = x.cuda()
        
        if task_type == 'classification':
            draw_cam(cam, x, y, n_class, writer)

        else:
            raise NotImplementedError

        if export_csv: # csv
            pred = model(x)
            writer.export_csv(f_name, y.cpu().item(), pred.argmax(1).cpu().item())

        pbar.update()

    writer.close()