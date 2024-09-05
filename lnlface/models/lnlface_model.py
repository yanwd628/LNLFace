
import os.path as osp
import torch
from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.metrics import calculate_metric
from basicsr.models.base_model import BaseModel
from basicsr.utils import get_root_logger, imwrite, tensor2img
from basicsr.utils.registry import MODEL_REGISTRY
from collections import OrderedDict
from tqdm import tqdm


@MODEL_REGISTRY.register()
class LNLFaceModel(BaseModel):
    """Base FaceGCN model for real-world blind face restoratin """
    def __init__(self, opt):
        super(LNLFaceModel, self).__init__(opt)
        self.build_model()

        if self.is_train:
            self.init_training_settings()

    def build_model(self):
        # define network
        self.net_g = build_network(self.opt['network_g'])
        self.net_g = self.model_to_device(self.net_g)

        if self.opt.get('network_d'):
            self.net_d = build_network(self.opt['network_d'])
            self.net_d = self.model_to_device(self.net_d)
        else:
            self.net_d = None

        # load pretrained models
        load_path_g = self.opt['path'].get('pretrain_network_g', None)
        pure_load = self.opt['path'].get('g_pure_load', False)
        if load_path_g is not None:
            if pure_load:
                self.net_g.load_state_dict(torch.load(load_path_g),
                                           strict=self.opt['path'].get('strict_load_g', True))
            else:
                self.load_network(self.net_g, load_path_g,
                                  self.opt['path'].get('strict_load_g', True),
                                  param_key=self.opt['path'].get('param_key', 'params'))

        if self.opt.get('network_d'):
            load_path_d = self.opt['path'].get('pretrain_network_d', None)
            if load_path_d is not None:
                self.load_network(self.net_d, load_path_d,
                                  self.opt['path'].get('strict_load_d', True),
                                  param_key=self.opt['path'].get('param_key', 'params'))


    def init_training_settings(self):
        self.net_g.train()
        if self.net_d is not None:
            self.net_d.train()

        self.setup_losses()
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_losses(self):
        train_opt = self.opt['train']
        # pixel loss
        if train_opt.get('pixel_opt'):
            self.cri_pix = build_loss(train_opt['pixel_opt']).to(self.device)
        else:
            self.cri_pix = None

        # perceptual loss
        if train_opt.get('perceptual_opt'):
            self.cri_perceptual = build_loss(train_opt['perceptual_opt']).to(self.device)
        else:
            self.cri_perceptual = None

        # gan loss
        if train_opt.get('gan_opt'):
            self.cri_gan = build_loss(train_opt['gan_opt']).to(self.device)
        else:
            self.cri_gan = None

        if self.cri_pix is None and self.cri_perceptual is None:
            raise ValueError('Both pixel and perceptual losses are None.')

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []

        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                optim_params.append(v)

        optim_type = train_opt['optim_g'].pop('type')
        self.optimizer_g = self.get_optimizer(optim_type, optim_params, **train_opt['optim_g'])
        self.optimizers.append(self.optimizer_g)

        if train_opt.get('optim_d'):
            optim_type = train_opt['optim_d'].pop('type')
            self.optimizer_d = self.get_optimizer(optim_type, self.net_d.parameters(), **train_opt['optim_d'])
            self.optimizers.append(self.optimizer_d)

    def feed_data(self, data):
        self.lq = data['lq'].to(self.device)
        self.loc = data['loc']
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)
            self.gt_path = data['gt_path']

    def optimize_parameters(self, current_iter):
        # optimize net_g
        if self.net_d is not None:
            for p in self.net_d.parameters():
                p.requires_grad = False
        self.optimizer_g.zero_grad()

        try:
            self.output = self.net_g(self.lq, self.loc)
        except:
            print(self.gt_path)
            return


        l_g_total = 0.
        loss_dict = OrderedDict()

        if self.cri_pix:
            l_g_pix = self.cri_pix(self.output, self.gt)
            l_g_total += l_g_pix
            loss_dict['l_g_pix'] = l_g_pix

        if self.cri_perceptual:
            l_g_percep, l_g_style = self.cri_perceptual(self.output, self.gt)
            if l_g_percep is not None:
                l_g_total += l_g_percep
                loss_dict['l_g_percep'] = l_g_percep
            if l_g_style is not None:
                l_g_total += l_g_style
                loss_dict['l_g_style'] = l_g_style

        if self.cri_gan:
            fake_g_pred = self.net_d(self.output)
            l_g_gan = self.cri_gan(fake_g_pred, True, is_disc=False)
            l_g_total += l_g_gan
            loss_dict['l_g_gan'] = l_g_gan

        loss_dict['l_g_total'] = l_g_total
        l_g_total.backward()
        self.optimizer_g.step()

        # optimize net_d
        if self.net_d is not None:
            for p in self.net_d.parameters():
                p.requires_grad = True
            self.optimizer_d.zero_grad()

            fake_d_pred = self.net_d(self.output.detach())
            real_d_pred = self.net_d(self.gt)

            l_d = self.cri_gan(real_d_pred, True, is_disc=True) + self.cri_gan(fake_d_pred, False, is_disc=True)
            loss_dict['l_d'] = l_d
            # In WGAN, real_score should be positive and fake_score should be negative
            loss_dict['real_score'] = real_d_pred.detach().mean()
            loss_dict['fake_score'] = fake_d_pred.detach().mean()
            l_d.backward()
            self.optimizer_d.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

    def test(self):
        with torch.no_grad():
            self.net_g.eval()

            try:
                self.output = self.net_g(self.lq, self.loc)
            except:
                self.output = self.lq
                print(self.gt_path)
                return

            self.net_g.train()

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        if self.opt['rank'] == 0:
            self.nondist_validation(dataloader, current_iter, tb_logger, save_img)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        use_pbar = self.opt['val'].get('pbar', True)

        if with_metrics:
            if not hasattr(self, 'metric_results'):  # only execute in the first run
                self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
            self._initialize_best_metric_results(dataset_name)
            self.metric_results = {metric: 0 for metric in self.metric_results}

        metric_data = dict()
        if use_pbar:
            pbar = tqdm(total=len(dataloader), unit='image')

        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            self.feed_data(val_data)
            self.test()

            sr_img = tensor2img(self.output.detach().cpu(), min_max=(-1, 1))
            metric_data['img'] = sr_img
            if hasattr(self, 'gt'):
                gt_img = tensor2img(self.gt.detach().cpu(), min_max=(-1, 1))
                metric_data['img2'] = gt_img
                del self.gt

            # tentative for out of GPU memory
            del self.lq
            del self.output
            del self.loc
            torch.cuda.empty_cache()

            if save_img:
                if self.opt['is_train']:
                    save_img_path = osp.join(self.opt['path']['visualization'], str(current_iter),
                                             f'{img_name}.png')
                else:
                    if self.opt['val']['suffix']:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["val"]["suffix"]}.png')
                    else:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}.png')
                imwrite(sr_img, save_img_path)


            if with_metrics:
                # calculate metrics
                for name, opt_ in self.opt['val']['metrics'].items():
                    self.metric_results[name] += calculate_metric(metric_data, opt_)
            if use_pbar:
                pbar.update(1)
                pbar.set_description(f'Test {img_name}')
        if use_pbar:
            pbar.close()

        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= (idx + 1)
                # update the best metric result
                self._update_best_metric_result(dataset_name, metric, self.metric_results[metric], current_iter)

            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)




    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
        log_str = f'Validation {dataset_name}\n'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}'
            if hasattr(self, 'best_metric_results'):
                log_str += (f'\tBest: {self.best_metric_results[dataset_name][metric]["val"]:.4f} @ '
                            f'{self.best_metric_results[dataset_name][metric]["iter"]} iter')
            log_str += '\n'

        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{dataset_name}/{metric}', value, current_iter)


    def save(self, epoch, current_iter):
        # save net_g and net_d
        self.save_network(self.net_g, 'net_g', current_iter, param_key='params')
        if self.net_d is not None:
            self.save_network(self.net_d, 'net_d', current_iter)
        # save training state
        self.save_training_state(epoch, current_iter)










