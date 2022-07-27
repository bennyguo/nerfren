import os
import time
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from utils.visualizer import save_visuals, create_writer
import torch


def main():
    opt = TrainOptions().parse()   # get training options

    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed_all(opt.seed)

    dataset = create_dataset(opt, mode=opt.train_split, shuffle=True)  # create a dataset given opt.dataset_mode and other options
    dataset_val = create_dataset(opt, mode=opt.val_epoch_split, shuffle=False)
    dataset_iterval = create_dataset(opt, mode=opt.val_split, shuffle=False)
    iter_val = iter(dataset_iterval)
    dataset_test = create_dataset(opt, mode=opt.test_split, shuffle=False)

    print(f'The number of training data = {len(dataset) if opt.keep_last else len(dataset) - len(dataset) % opt.batch_size}')
    print(f'The number of validation data = {len(dataset_val)}')
    print(f'The number of test data = {len(dataset_test)}')

    model = create_model(opt)      # create a model given opt.model and other options
    current_epoch = model.setup(opt)               # regular setup: load and print networks; create schedulers
    total_iters = current_epoch * len(dataset.dataloader)      # the total number of training iterations
    writer = create_writer(opt)
    for epoch in range(current_epoch + 1, opt.n_epochs + 1):    # outer loop for different epochs
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        writer.add_scalar('lr', model.get_learning_rate(), global_step=epoch)
        print('Learning rate:', f"{model.get_learning_rate():.3e}")

        model.update_hyperparameters(epoch)
        for i, data in enumerate(dataset):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration
            total_iters += 1
            epoch_iter += 1

            model.update_hyperparameters_step(total_iters)

            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            model.train()
            model.set_input(data)         # unpack data from dataset and apply preprocessing
            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights

            if total_iters % opt.print_freq == 0:    # print training losses
                losses = model.get_current_losses('train')
                t_comp = time.time() - iter_start_time
                for loss_name, loss_val in losses.items():
                    writer.add_scalars(f"{loss_name}", {'train': loss_val}, global_step=total_iters)
                print(f"Epoch {epoch} - Iteration {epoch_iter}/{len(dataset.dataloader)} (comp time {t_comp:.3f}, data time {t_data:.3f})")
                print("Training losses |", ' '.join([f"{k}: {v:.3e}" for k, v in losses.items()]))
            
            if total_iters % opt.val_freq == 0:
                model.eval()
                try:
                    val_data = next(iter_val)
                except StopIteration:
                    iter_val = iter(dataset_iterval)
                    val_data = next(iter_val)
                with torch.no_grad():
                    model.set_input(val_data)
                    model.validate_iter()
                val_losses = model.get_current_losses('val_iter')
                for loss_name, loss_val in val_losses.items():
                    writer.add_scalars(f"{loss_name}", {'val': loss_val}, global_step=total_iters)
                if total_iters % opt.vis_freq == 0:
                    save_visuals(os.path.join(model.save_dir, 'vis'), model.get_current_visuals('val_iter'), total_iters)
                print("Validation iter losses |", ' '.join([f"{k}: {v:.3e}" for k, v in val_losses.items()]))

            iter_data_time = time.time()

        if epoch % opt.val_epoch_freq == 0:
            model.eval()
            with torch.no_grad():
                model.validate(dataset_val)
            val_losses = model.get_current_losses('val')
            for loss_name, loss_val in val_losses.items():
                writer.add_scalars(f"{loss_name}", {'val_full': loss_val}, global_step=total_iters)
            save_visuals(os.path.join(model.save_dir, f"{epoch}_val_vis"), model.get_current_visuals('val'))
            print("Validation losses |", ' '.join([f"{k}: {v:.3e}" for k, v in val_losses.items()]))

        if epoch % opt.vis_epoch_freq == 0:
            model.eval()
            with torch.no_grad():
                model.test(dataset_test)
            save_visuals(os.path.join(model.save_dir, f"{epoch}_test_vis"), model.get_current_visuals('test'))

        if epoch % opt.save_epoch_freq == 0:
            print('Saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks(epoch)
            model.save_networks('latest')
        
        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs, time.time() - epoch_start_time))

        model.update_learning_rate()    # update learning rates at the end of every epoch.


if __name__ == '__main__':
    main()
