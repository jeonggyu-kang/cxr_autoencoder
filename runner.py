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
    writer = None,
    task_type = 'classification'
):

    save_every = meta['save_every']
    print_every = meta['print_every']
    test_every = meta['test_every']


    for ep in range(1, max_epoch+1):
        train(ep, max_epoch, model, train_loader, loss_fn, optimizer, writer, print_every, task_type)
        if scheduler is not None:
            scheduler.step()


        if ep % test_every == 0:
            acc = test(ep, max_epoch, model, test_loader, writer, task_type = task_type)

            if task_type == 'regression':
                acc *= -1
            
            writer.update(model, acc)
        
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




def train(ep, max_epoch, model, train_loader, loss_fn, optimizer, writer, _print_every, task_type):
    model.train()
    mean_loss = 0.0

    print_every = len(train_loader) // _print_every     

    preds = []
    gts = []

    step = 0
    step_cnt = 1

    global_step = (ep - 1) * len(train_loader)

    for i, batch in enumerate(train_loader):
        x, y = batch
        x = x.cuda()
        y = y.cuda()

        out = model(x)

        loss = loss_fn(out, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        mean_loss += loss.item()
        step += 1
        global_step += 1

        preds.append(out.data.cpu())
        gts.append(y.data.cpu())

        if (i+1) % print_every == 0:
            mean_loss /= step
            writer.add_scalar('train/loss', mean_loss, global_step)
            print('Epoch [{}/{}] Step[{}/{}] Loss: {:.4f}'.format(
                ep, max_epoch, step_cnt, _print_every, mean_loss))
            
            mean_loss = 0.0
            step = 0
            step_cnt += 1
        
    preds = torch.cat(preds)
    gts = torch.cat(gts)

    if task_type == 'classification':
        acc = calc_accuracy(preds, gts)

        print('Epoch[{}/{}] Train Acc: {:.4f}'.format(ep,max_epoch, acc))
        print('='*40)

        writer.add_scalar('train/acc', acc, ep)
    
    elif task_type == 'regression':
        err = get_mean_squared_error(preds, gts)
        print('Epoch[{}/{}] Train Err: {:.4f}'.format(ep,max_epoch, err))
        print ('='*40)

        writer.add_scalar('train/err', err, ep)

@torch.no_grad() # stop calculating gradient
def test(ep, max_epoch, model, test_loader, writer, pbar=None, hard_sample_mining = False, confusion_matrix=False, n_class = 5, task_type = 'classification'):
    model.eval()

    preds = []
    gts = []

    hardsample_dict = get_sample_dict()

    for idx, batch in enumerate(test_loader):
        x, y = batch
        x = x.cuda()
        y = y.cuda()

        out = model(x)
        preds.append(out.data.cpu())
        gts.append(y.data.cpu())

        if hard_sample_mining:
            hardsample_dict = update_hardsample_indice(out.data.cpu(), y.data.cpu(), hardsample_dict,x )

        if pbar is not None:
            pbar.update()

    preds = torch.cat(preds)
    gts = torch.cat(gts)

    if task_type == 'classification':

        acc = calc_accuracy(preds, gts)
        print ('Epoch[{}/{}] TestAcc: {:.4f}'.format(ep, max_epoch, acc))
        print ('+'*40)

        writer.add_scalar('test/acc', acc, ep)
        ret = acc
    
    elif task_type == 'regression':
        err = get_mean_squared_error(preds, gts)
        print ('Epoch[{}/{}] Test Err: {:.4f}'.format(ep, max_epoch, err))
        print ('='*40)

        writer.add_scalar('test/err', err, ep)
        ret = err

    if hard_sample_mining:
        index2cls_name = {
            0: 0,
            1: 1,
            2: 2,
            3: 3,
            4: 4,
        }

        for gt_k in hardsample_dict:
            for pred_k, samples in hardsample_dict[gt_k].items():

                if samples:
                    num_sample = len(samples)
                    text = str(index2cls_name[gt_k]) + '/' + str(index2cls_name[pred_k])

                    samples = torch.cat(samples)

                    grid_samples = vutils.make_grid(tensor_rgb2bgr(samples), normalize = True, scale_each = True)
                    writer.add_image(text, grid_samples, 0)

                    print('{} : {}'.format(text, num_sample))

    if confusion_matrix:
        cm_image = get_confusion_matrix_image(preds, gts)
        writer.add_image('test/confusion_matrix', cm_image, 0)

        
    return ret

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