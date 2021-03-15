import os
import csv
import random
import PIL.Image
import numpy as np
import torch
import torchvision
from tqdm import tqdm
from model import *
from dataset import *
from session import *
from utils.image import *
from utils.option import parse
from torch.utils.tensorboard import SummaryWriter

MINIBATCH_SIZE = {0: 3, 1: 6, 2: 8, 3: 16, 4: 32, 5: 64}


def main():
    args = parse()

    # set seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)

    # set device
    device = 'cpu' if not torch.cuda.is_available() else 'cuda'
    num_device = torch.cuda.device_count()

    if not args.skip_train:
        # set model
        model = PyNetCA()
        if args.train_from_level is not None:
            # from specified level
            print(f'restarting from level {args.train_from_level}')
            assert os.path.isfile(os.path.join(args.target_dir, 'model', str(args.train_from_level+1), 'model.pth')), f'level {args.train_from_level+1} trained model not found'
            model.load_state_dict(torch.load(os.path.join(args.target_dir, 'model', str(args.train_from_level+1), 'model.pth')))
            optim_state_dict = None
            sched_state_dict = None
            current_level = args.train_from_level
            current_epoch = 0
            early_psnr = 0
            early_lpips = 1e9
            images_seen = 0
        elif os.path.isfile(os.path.join(args.target_dir, 'checkpoint.pt')):
            # from checkpoint
            print('resuming checkpoint experiment')
            checkpoint = torch.load(os.path.join(args.target_dir, 'checkpoint.pt'))
            model.load_state_dict(checkpoint['model'])
            optim_state_dict = checkpoint['optim']
            sched_state_dict = checkpoint['scheduler']
            current_level = checkpoint['level']
            current_epoch = checkpoint['epoch']
            early_psnr = checkpoint['psnr']
            early_lpips = checkpoint['lpips']
            images_seen = checkpoint['seen']
        else:
            # from scratch
            print('initializing experiment')
            optim_state_dict = None
            sched_state_dict = None
            current_level = 5
            current_epoch = 0
            early_psnr = 0
            early_lpips = 1e9
            images_seen = 0
            os.makedirs(os.path.join(args.target_dir, 'model'), exist_ok=True)
            os.makedirs(os.path.join(args.target_dir, 'summary'), exist_ok=True)
            with open(os.path.join(args.target_dir, 'argv.csv'), 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(vars(args).items())

        # set dataset
        train_dataset = DatasetZRR(args.source_dir)
        validation_dataset = DatasetZRR(args.source_dir, split='test')

        # set session
        session = PyNetSession(model, args.perceptual)
        session.to(device)
        session.parallel()

        # run experiment
        for level in range(current_level, -1, -1):
            os.makedirs(os.path.join(args.target_dir, 'model', str(level)), exist_ok=True)
            os.makedirs(os.path.join(args.target_dir, 'summary', str(level)), exist_ok=True)
            train_dataset.set_rgb_transforms([torchvision.transforms.Resize((train_dataset.shape[0]//(2**level), train_dataset.shape[1]//(2**level)))])
            validation_dataset.set_rgb_transforms([torchvision.transforms.Resize((validation_dataset.shape[0]//(2**level), validation_dataset.shape[1]//(2**level)))])
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=MINIBATCH_SIZE[level]*num_device, shuffle=True, num_workers=4*num_device, pin_memory=True)
            validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=MINIBATCH_SIZE[level]*num_device, shuffle=False, num_workers=4*num_device, pin_memory=True)

            train_summary_writer = SummaryWriter(os.path.join(args.target_dir, 'summary', str(level), 'train'), flush_secs=1, max_queue=1)
            validation_summary_writer = SummaryWriter(os.path.join(args.target_dir, 'summary', str(level), 'validation'), flush_secs=1, max_queue=1)

            session.set_optimizer(args.lr)
            session.set_scheduler(args.lr, 2*args.num_epochs if args.perceptual and level==0 else args.num_epochs, len(train_dataset)//(MINIBATCH_SIZE[level]*num_device)+1)
            session.set_criterion(level)
            if optim_state_dict is not None:
                session.optimizer.load_state_dict(optim_state_dict)
                session.scheduler.load_state_dict(sched_state_dict)
                optim_state_dict = None
                sched_state_dict = None

            for epoch in range(current_epoch, 2*args.num_epochs if args.perceptual and level==0 else args.num_epochs):
                # save checkpoint
                torch.save({'model': model.state_dict(), 'optim': session.optimizer.state_dict(), 'scheduler': session.scheduler.state_dict(), 'level': level, 'epoch': epoch, 'psnr': early_psnr, 'lpips': early_lpips, 'seen': images_seen}, os.path.join(args.target_dir, 'checkpoint.pt'))

                # train loop
                for step, data in enumerate(tqdm(train_loader, ncols=60, desc=f'l:{level} e:{epoch:02d}')):
                    session.step(data, level, train=True, augmentation=True)

                    # get summary
                    for p in session.optimizer.param_groups:
                        cur_lr = p['lr']
                    images_seen += MINIBATCH_SIZE[level] * num_device
                    train_loss = session.get_loss()
                    train_metrics = session.get_metrics()
                    train_metrics['lr'] = cur_lr

                    # write summary
                    for k, v in train_loss.items():
                        if not v is None:
                            train_summary_writer.add_scalar(f'loss/{k}', v, images_seen)
                    for k, v in train_metrics.items():
                        if not v is None:
                            train_summary_writer.add_scalar(f'metric/{k}', v, images_seen)

                    # write image summary
                    if step%args.checkpoint_step==0:
                        images = session.get_images()
                        train_summary_writer.add_image('enhanced', torchvision.utils.make_grid(images['enhanced']), images_seen)
                        train_summary_writer.add_image('rgb', torchvision.utils.make_grid(images['rgb']), images_seen)
                        # save checkpoint
                        if not step==0:
                            torch.save({'model': model.state_dict(), 'optim': session.optimizer.state_dict(), 'scheduler': session.scheduler.state_dict(), 'level': level, 'epoch': epoch, 'psnr': early_psnr, 'lpips': early_lpips, 'seen': images_seen}, os.path.join(args.target_dir, 'checkpoint.pt'))

                # validation loop
                for data in validation_loader:
                    session.step(data, level, train=False, augmentation=False)

                # get summary - epoch aggregated
                validation_loss = session.get_loss()
                validation_metrics = session.get_metrics()

                # write scalar summary
                for k, v in validation_loss.items():
                    if not v is None:
                        validation_summary_writer.add_scalar(f'loss/{k}', v, images_seen)
                for k, v in validation_metrics.items():
                    if not v is None:
                        validation_summary_writer.add_scalar(f'metric/{k}', v, images_seen)

                # write image summary
                images = session.get_images()
                validation_summary_writer.add_image('enhanced', torchvision.utils.make_grid(images['enhanced']), epoch)
                if epoch==0:
                    validation_summary_writer.add_image('raw', torchvision.utils.make_grid(images['raw']), epoch)
                    validation_summary_writer.add_image('rgb', torchvision.utils.make_grid(images['rgb']), epoch)

                # save early stop models
                if level==0:
                    if early_psnr < validation_metrics['psnr']:
                        early_psnr = validation_metrics['psnr']
                        torch.save(model.state_dict(), os.path.join(args.target_dir, 'model', 'model_early_fidelity.pth'))

                    if early_lpips > validation_metrics['lpips']:
                        early_lpips = validation_metrics['lpips']
                        torch.save(model.state_dict(), os.path.join(args.target_dir, 'model', 'model_early_perceptual.pth'))

            # save each level models
            if not level==0:
                torch.save(model.state_dict(), os.path.join(args.target_dir, 'model', str(level), 'model.pth'))
                current_epoch = 0
                images_seen = 0

        # save final model
        torch.save(model.state_dict(), os.path.join(args.target_dir, 'model', 'model_final.pth'))

    if args.test_dir is not None:
        assert os.path.isdir(args.test_dir), 'path to the test images not found'
        # set path
        test_target_dir = os.path.abspath(args.test_dir)+'_enhanced'
        os.makedirs(test_target_dir)
        raw_list = os.listdir(args.test_dir)
        raw_list.sort()

        # set transforms
        raw_to_tensor = BayerToTensor()
        tensor_to_pil = torchvision.transforms.ToPILImage()

        # restore model
        model = PyNetCA()
        model.load_state_dict(torch.load(os.path.join(args.target_dir, 'model', 'model_early_perceptual.pth' if args.perceptual else 'model_early_fidelity.pth')))
        model.eval()
        model.to(device)

        # run model
        with torch.no_grad():
            for raw in tqdm(raw_list, ncols=50):
                input = PIL.Image.open(os.path.join(args.test_dir, str(raw)))
                if input.size==(448,448):
                    enhanced = shrink(torch.mean(torch.cat([augment(model(augment(expand(raw_to_tensor(input)).unsqueeze(0).to(device), k), level=0).cpu(), k, inverse=True) for k in range(8)], dim=0), dim=0))
                else:
                    input = np.asarray(input)
                    original_shape = input.shape
                    patches = extract_patches(input[...,np.newaxis], patch_size=448, stride=224)
                    enhanced_patches = [shrink(torch.mean(torch.cat([augment(model(augment(expand(raw_to_tensor(patch[:,:,0])).unsqueeze(0).to(device), k), level=0).cpu(), k, inverse=True) for k in range(8)], dim=0), dim=0).permute(1, 2, 0).numpy()) for patch in patches]
                    enhanced = torch.from_numpy(reconstruct_patches(enhanced_patches, original_shape, stride=224, weighting='cos')).permute(2,0,1)
                tensor_to_pil(enhanced).save(os.path.join(test_target_dir, raw), compress_level=0)


if __name__=='__main__':
    main()
    exit(1)
