import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from cubdl.das_torch import DAS_PW, DAS_PW_DL
from datasets.PWDataLoaders import load_data, get_filelist
from cubdl.PixelGrid import make_pixel_grid
from model_unet_supervised import *
from torch.utils.tensorboard import SummaryWriter
from losses import *
from metrics import *
from scipy.ndimage import gaussian_filter


import time
from tqdm import tqdm


cached_dict = {}


class CorrelationLoss(nn.Module):
    def __init__(self):
        super(CorrelationLoss, self).__init__()

    def forward(self, x, y):
        C = torch.sum(x * y, dim=(-1, -2), keepdim=True)
        A1 = torch.sum(x * x, dim=(-1, -2), keepdim=True)
        A2 = torch.sum(y * y, dim=(-1, -2), keepdim=True)

        correlation = C / (torch.sqrt(A1) * torch.sqrt(A2) + 1e-8)

        return -correlation.mean()


class PWTrainer:
    def __init__(self, config):
        self.config = config
        self.device = config["device"]

        self.model = BF_Model(config)
        self.model.build()

        self.model.to(self.device)
        self.global_step = 0

        self.config["lr"] = 1e-4
        # lr scheduler
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.config["lr"], weight_decay=0.01
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=5000, eta_min=1e-7
        )
        self.criterion = CorrelationLoss()
        self.mse_criterion = nn.MSELoss()
        self.l1_criterion = nn.L1Loss()

        self.init_flag = False

        self.losses = []
        self.val_losses = []
        self.best_val_loss = 1e10
        self.best_model = None

        self.filelist = None
        self.iqN = None

        self.top_dir = "/media/hyunwoo/External_HDD/DCL_Revision_Exps/"

        if not os.path.exists(
            os.path.join(self.top_dir, "revision_dcl_full_angles_exp")
        ):
            os.mkdir(os.path.join(self.top_dir, "revision_dcl_full_angles_exp"))

        self.exp_dir = os.path.join(self.top_dir, "revision_dcl_full_angles_exp")

        if not os.path.exists(
            os.path.join(self.exp_dir, "revision_dcl_full_angles_weights")
        ):
            os.mkdir(os.path.join(self.exp_dir, "revision_dcl_full_angles_weights"))
        if not os.path.exists(
            os.path.join(self.exp_dir, "revision_dcl_full_angles_tmp")
        ):
            os.mkdir(os.path.join(self.exp_dir, "revision_dcl_full_angles_tmp"))
        if not os.path.exists(
            os.path.join(self.exp_dir, "revision_dcl_full_angles_log")
        ):
            os.mkdir(os.path.join(self.exp_dir, "revision_dcl_full_angles_log"))
        if not os.path.exists(
            os.path.join(self.exp_dir, "revision_dcl_full_angles_validation_results")
        ):
            os.mkdir(
                os.path.join(
                    self.exp_dir, "revision_dcl_full_angles_validation_results"
                )
            )

        if not os.path.exists("val_npy_wvln3"):
            os.mkdir("val_npy_wvln3")

        self.picmus_database_path = os.path.join("datasets", "data", "picmus")
        self.picmus_dtype = "iq"

        # load weight and resume traininig
        self.model.load_state_dict(
            torch.load(
                os.path.join(
                    self.exp_dir, "revision_dcl_full_angles_weights", "model_133000.pt"
                )
            )
        )
        self.global_step = 133000
        self.scheduler.step(self.global_step)

        self.tb_writer = SummaryWriter(
            log_dir=os.path.join(self.exp_dir, "revision_dcl_full_angles_log")
        )

    def prepare_datalist(self):
        self.filelist = get_filelist(data_type="all")
        self.validation_list = get_filelist(data_type="val")

        self.filelist = {
            k: v
            for k, v in sorted(
                self.filelist.items(), key=lambda item: np.random.random()
            )
        }
        for key in self.filelist.keys():
            self.filelist[key] = np.random.permutation(self.filelist[key])

    def shuffle_datalist(self):
        self.filelist = {
            k: v
            for k, v in sorted(
                self.filelist.items(), key=lambda item: np.random.random()
            )
        }
        for key in self.filelist.keys():
            self.filelist[key] = np.random.permutation(self.filelist[key])

    def train(self):
        self.prepare_datalist()
        for epoch in range(self.config["num_epochs"]):
            self.train_epoch()

    def train_epoch(self):
        self.shuffle_datalist()
        self.model.train()

        total_files = 0
        current_file = 0
        for each in self.filelist.values():
            total_files += len(each)

        shuffled = []
        for data_source in self.filelist:
            for acq in self.filelist[data_source]:
                shuffled.append((data_source, acq))
        shuffled = np.random.permutation(shuffled)

        for data_source, acq in shuffled:
            self.model.train()
            if data_source != "PICMUS":
                acq = int(acq)
                P, xlims, zlims = load_data(data_source, acq)
                wvln = P.c / P.fc
                dx = wvln / 3
                dz = dx
                grid = make_pixel_grid(xlims, zlims, dx, dz, 4)
                fnum = 1
            else:
                type = acq.split("-")[0]
                target = acq.split("-")[1]
                if not type == "in_vivo":
                    P = PICMUSData(
                        self.picmus_database_path, type, target, self.picmus_dtype
                    )
                else:
                    P = PICMUSInvivo(
                        self.picmus_database_path, type, target, self.picmus_dtype
                    )

                xlims = [P.ele_pos[0, 0], P.ele_pos[-1, 0]]
                zlims = [5e-3, 55e-3]
                wvln = P.c / P.fc
                dx = wvln / 3
                dz = dx  # Use square pixels
                grid = make_pixel_grid(xlims, zlims, dx, dz, 4)
                fnum = 1

            current_file += 1

            self.prepare_data(P, grid, fnum, data_source, acq)

            for each_step in range(self.config["step_per_file"]):

                if self.global_step % 1000 == 0 and not self.init_flag:
                    self.validation()
                    self.save_model()
                    torch.cuda.empty_cache()
                else:
                    self.init_flag = False

                self.optimizer.zero_grad()
                patches, gt, cpwc = self.get_data(
                    self.config["batch_size"], self.config["angle_num"]
                )
                patches = torch.from_numpy(patches).float().to(self.device)
                gt = torch.from_numpy(gt).float().to(self.device)
                cpwc = torch.from_numpy(cpwc).float().to(self.device)

                # random lr flip
                if np.random.random() > 0.5:
                    patches = torch.flip(patches, [3])
                    gt = torch.flip(gt, [4])
                    cpwc = torch.flip(cpwc, [3])

                pred = self.model(patches)

                loss = 0
                for i in range(gt.shape[0]):
                    if i == self.input_angle:
                        continue
                    if i == self.validation_angle:
                        continue
                    each_gt = gt[i, :, :, :]
                    loss += self.criterion(pred, each_gt)

                loss /= gt.shape[0] - 2

                if not torch.isnan(loss):
                    loss.backward()
                else:
                    print("NAN LOSS")
                    continue

                self.optimizer.step()
                self.losses.append(loss.item())

                self.tb_writer.add_scalar("Loss", loss.item(), self.global_step)
                self.tb_writer.add_scalar(
                    "LR", self.optimizer.param_groups[0]["lr"], self.global_step
                )

                if self.global_step % 50 == 0:

                    print(
                        "File : {} / {}, Step : {}, LOSS : {}, LR : {}".format(
                            current_file,
                            total_files,
                            self.global_step,
                            loss.item(),
                            self.optimizer.param_groups[0]["lr"],
                        )
                    )

                    log_image = (
                        pred[0, 0, :, :].detach().to("cpu").numpy()
                        + 1j * pred[0, 1, :, :].detach().to("cpu").numpy()
                    )
                    log_image = np.abs(log_image)
                    log_image = 20 * np.log10(log_image)
                    log_image[log_image < np.max(log_image) - 60] = (
                        np.max(log_image) - 60
                    )
                    log_image = (log_image + np.abs(np.min(log_image))) / 60
                    cv2.imwrite(
                        os.path.join(
                            self.exp_dir,
                            "revision_dcl_full_angles_tmp",
                            str(self.global_step) + "_log_pred.png",
                        ),
                        log_image * 255,
                    )

                    log_gt = (
                        gt[0, 0, 0, :, :].detach().to("cpu").numpy()
                        + 1j * gt[0, 0, 1, :, :].detach().to("cpu").numpy()
                    )
                    log_gt = np.abs(log_gt)
                    log_gt = 20 * np.log10(log_gt)
                    log_gt[log_gt < np.max(log_gt) - 60] = np.max(log_gt) - 60
                    log_gt = (log_gt + np.abs(np.min(log_gt))) / 60
                    cv2.imwrite(
                        os.path.join(
                            self.exp_dir,
                            "revision_dcl_full_angles_tmp",
                            str(self.global_step) + "_log_gt.png",
                        ),
                        log_gt * 255,
                    )

                    log_cpwc = (
                        cpwc[0, 0, :, :].detach().to("cpu").numpy()
                        + 1j * cpwc[0, 1, :, :].detach().to("cpu").numpy()
                    )
                    log_cpwc = np.abs(log_cpwc)
                    log_cpwc = 20 * np.log10(log_cpwc)
                    log_cpwc[log_cpwc < np.max(log_cpwc) - 60] = np.max(log_cpwc) - 60
                    log_cpwc = (log_cpwc + np.abs(np.min(log_cpwc))) / 60
                    cv2.imwrite(
                        os.path.join(
                            self.exp_dir,
                            "revision_dcl_full_angles_tmp",
                            str(self.global_step) + "_log_cpwc.png",
                        ),
                        log_cpwc * 255,
                    )

                    log_input = (
                        patches[0, 0, :, :].detach().to("cpu").numpy()
                        + 1j * patches[0, 1, :, :].detach().to("cpu").numpy()
                    )
                    log_input = np.abs(log_input)
                    log_input = 20 * np.log10(log_input)
                    log_input[log_input < np.max(log_input) - 60] = (
                        np.max(log_input) - 60
                    )
                    log_input = (log_input + np.abs(np.min(log_input))) / 60
                    cv2.imwrite(
                        os.path.join(
                            self.exp_dir,
                            "revision_dcl_full_angles_tmp",
                            str(self.global_step) + "_log_input.png",
                        ),
                        log_input * 255,
                    )

                self.global_step += 1
                self.scheduler.step()

    def save_model(self):
        torch.save(
            self.model.state_dict(),
            os.path.join(
                self.exp_dir,
                "revision_dcl_full_angles_weights",
                "model_" + str(self.global_step) + ".pt",
            ),
        )
        torch.jit.save(
            torch.jit.trace(self.model, torch.rand(1, 2, 256, 256).to(self.device)),
            os.path.join(
                self.exp_dir,
                "revision_dcl_full_angles_weights",
                "jit_model_" + str(self.global_step) + ".pt",
            ),
        )

    def validation(self):
        self.model.eval()
        print("Validation...")
        if not os.path.exists(
            os.path.join(
                self.exp_dir,
                "revision_dcl_full_angles_validation_results",
                str(self.global_step),
            )
        ):
            os.mkdir(
                os.path.join(
                    self.exp_dir,
                    "revision_dcl_full_angles_validation_results",
                    str(self.global_step),
                )
            )
        cnt = 0
        total_data_len = 0

        for each in self.validation_list.values():
            total_data_len += len(each)

        total_mean_validation_loss = 0

        for data_source in self.validation_list:
            for acq in self.validation_list[data_source]:
                cnt += 1
                print("Processing... {} / {}".format(cnt, total_data_len))

                if os.path.exists(
                    "./val_npy_wvln3/{}_{}_i_frames.npy".format(data_source, acq)
                ):

                    acq = str(acq)
                    data_source = str(data_source)

                    # check cachaed_dict
                    if data_source + "_" + acq in cached_dict:
                        i_frames = cached_dict[data_source + "_" + acq][0]
                        q_frames = cached_dict[data_source + "_" + acq][1]
                    else:
                        i_frames = np.load(
                            "./val_npy_wvln3/{}_{}_i_frames.npy".format(
                                data_source, acq
                            )
                        )
                        q_frames = np.load(
                            "./val_npy_wvln3/{}_{}_q_frames.npy".format(
                                data_source, acq
                            )
                        )
                        # cached_dict[data_source + '_' + acq] = [i_frames, q_frames]

                    if data_source != "PICMUS":
                        acq = int(acq)
                        P, xlims, zlims = load_data(data_source, acq)
                        zlims[0] = 2e-3
                        wvln = P.c / P.fc
                        dx = wvln / 3
                        dz = dx
                        grid = make_pixel_grid(xlims, zlims, dx, dz, 4)
                        fnum = 1
                    else:
                        type = acq.split("-")[0]
                        target = acq.split("-")[1]
                        if not type == "in_vivo":
                            P = PICMUSData(
                                self.picmus_database_path,
                                type,
                                target,
                                self.picmus_dtype,
                            )
                        else:
                            P = PICMUSInvivo(
                                self.picmus_database_path,
                                type,
                                target,
                                self.picmus_dtype,
                            )
                        # Define pixel grid limits (assume y == 0)
                        xlims = [P.ele_pos[0, 0], P.ele_pos[-1, 0]]
                        zlims = [0e-3, 50e-3]
                        if acq.startswith("simulation"):
                            zlims[0] = 5e-3
                            zlims[1] = 50e-3
                        if data_source == "PICMUS" and not acq.startswith("simulation"):
                            zlims = [0e-3, 42e-3]
                        wvln = P.c / P.fc
                        dx = wvln / 3
                        dz = dx  # Use square pixels
                        grid = make_pixel_grid(xlims, zlims, dx, dz, 4)

                else:
                    if data_source != "PICMUS":
                        acq = int(acq)
                        P, xlims, zlims = load_data(data_source, acq)
                        zlims[0] = 2e-3
                        wvln = P.c / P.fc
                        dx = wvln / 3
                        dz = dx
                        grid = make_pixel_grid(xlims, zlims, dx, dz, 4)

                    else:
                        type = acq.split("-")[0]
                        target = acq.split("-")[1]
                        if not type == "in_vivo":
                            P = PICMUSData(
                                self.picmus_database_path,
                                type,
                                target,
                                self.picmus_dtype,
                            )
                        else:
                            P = PICMUSInvivo(
                                self.picmus_database_path,
                                type,
                                target,
                                self.picmus_dtype,
                            )
                        # Define pixel grid limits (assume y == 0)
                        xlims = [P.ele_pos[0, 0], P.ele_pos[-1, 0]]
                        zlims = [0e-3, 50e-3]
                        if acq.startswith("simulation"):
                            zlims[0] = 5e-3
                            zlims[1] = 50e-3
                        if data_source == "PICMUS" and not acq.startswith("simulation"):
                            zlims = [0e-3, 42e-3]
                        wvln = P.c / P.fc
                        dx = wvln / 3
                        dz = dx  # Use square pixels
                        grid = make_pixel_grid(xlims, zlims, dx, dz, 4)

                    x = (P.idata, P.qdata)
                    dasN = DAS_PW_DL(P, grid)
                    idasN, qdasN, i_frames, q_frames, i_frames_apod, q_frames_apod = (
                        dasN.forward(x)
                    )

                    np.save(
                        "./val_npy_wvln3/{}_{}_i_frames.npy".format(data_source, acq),
                        i_frames,
                    )
                    np.save(
                        "./val_npy_wvln3/{}_{}_q_frames.npy".format(data_source, acq),
                        q_frames,
                    )

                print("--------------------")
                print("File name: {}_{}".format(data_source, acq))
                print("Xlims: {}, Zlims: {}".format(xlims, zlims))
                print("dx: {}, dz: {}".format(dx, dz))
                print("--------------------")

                self.validation_angles = [i_frames.shape[0] // 2]

                for angle in self.validation_angles:
                    each_input = np.zeros((1, 2, i_frames.shape[1], i_frames.shape[2]))
                    each_input[0, 0, :, :] = i_frames[angle, :, :]
                    each_input[0, 1, :, :] = q_frames[angle, :, :]

                    max = np.max(np.abs(each_input))
                    each_input = each_input / max

                    torch.cuda.empty_cache()
                    input = (
                        torch.from_numpy(each_input).float().to(torch.device("cuda"))
                    )
                    pad_x = 32 - input.shape[3] % 32
                    pad_y = 32 - input.shape[2] % 32
                    pad = (
                        pad_x // 2,
                        pad_x - pad_x // 2,
                        pad_y // 2,
                        pad_y - pad_y // 2,
                    )
                    input = F.pad(input, pad, "constant", 0)

                    mean_validation_loss = 0
                    with torch.no_grad():
                        pred = self.model(input)

                        for i in range(i_frames.shape[0]):
                            if i == angle:
                                continue

                            each_gt = np.zeros(
                                (1, 2, i_frames.shape[1], i_frames.shape[2])
                            )
                            each_gt[0, 0, :, :] = i_frames[i, :, :]
                            each_gt[0, 1, :, :] = q_frames[i, :, :]
                            each_gt = (
                                torch.from_numpy(each_gt)
                                .float()
                                .to(torch.device("cuda"))
                            )
                            each_gt = F.pad(each_gt, pad, "constant", 0)
                            mean_validation_loss += self.criterion(pred, each_gt)

                    mean_validation_loss /= i_frames.shape[0] - 1
                    total_mean_validation_loss += mean_validation_loss

                    pred = pred.detach().to(torch.device("cpu")).numpy()
                    pred = pred[
                        :,
                        :,
                        pad[2] : pred.shape[2] - pad[3],
                        pad[0] : pred.shape[3] - pad[1],
                    ]
                    i_pred = pred[0, 0, :, :]
                    q_pred = pred[0, 1, :, :]
                    pred = pred[0, 0, :, :] + 1j * pred[0, 1, :, :]
                    pred = np.abs(pred)
                    angle_frame_log = 20 * np.log10(pred)
                    angle_frame_log[angle_frame_log < np.max(angle_frame_log) - 60] = (
                        np.max(angle_frame_log) - 60
                    )
                    angle_frame_log = (
                        angle_frame_log + np.abs(np.min(angle_frame_log))
                    ) / 60

                    # if picmus, measure the evaluation
                    if (
                        data_source == "PICMUS"
                        and acq.startswith("simulation")
                        and "cont" in acq
                    ):
                        cnrs, gcnrs, circle_image = (
                            self.measure_picmus_simulation_contrast(i_pred, q_pred)
                        )
                        mean_cnr = np.mean(cnrs)
                        mean_gcnr = np.mean(gcnrs)
                        print("Mean CNR: {}, Mean GCNR: {}".format(mean_cnr, mean_gcnr))
                        self.tb_writer.add_scalar(
                            "Mean CNR of picmus contrast simulation",
                            mean_cnr,
                            self.global_step,
                        )
                        self.tb_writer.add_scalar(
                            "Mean GCNR of picmus contrast simulation",
                            mean_gcnr,
                            self.global_step,
                        )
                        cv2.imwrite(
                            os.path.join(
                                self.exp_dir,
                                "revision_dcl_full_angles_validation_results",
                                str(self.global_step),
                                data_source
                                + "_"
                                + str(acq)
                                + "_"
                                + str(angle)
                                + "_measured.png",
                            ),
                            circle_image,
                        )

                    if (
                        data_source == "PICMUS"
                        and acq.startswith("experiment")
                        and "cont" in acq
                    ):
                        cnrs, gcnrs, circle_image = (
                            self.measure_picmus_experiment_contrast(i_pred, q_pred)
                        )
                        mean_cnr = np.mean(cnrs)
                        mean_gcnr = np.mean(gcnrs)
                        print("Mean CNR: {}, Mean GCNR: {}".format(mean_cnr, mean_gcnr))
                        self.tb_writer.add_scalar(
                            "Mean CNR of picmus contrast experiment",
                            mean_cnr,
                            self.global_step,
                        )
                        self.tb_writer.add_scalar(
                            "Mean GCNR of picmus contrast experiment",
                            mean_gcnr,
                            self.global_step,
                        )
                        cv2.imwrite(
                            os.path.join(
                                self.exp_dir,
                                "revision_dcl_full_angles_validation_results",
                                str(self.global_step),
                                data_source
                                + "_"
                                + str(acq)
                                + "_"
                                + str(angle)
                                + "_measured.png",
                            ),
                            circle_image,
                        )

                    if (
                        data_source == "PICMUS"
                        and acq.startswith("in")
                        and "long" in acq
                    ):
                        cnrs, gcnrs, circle_image = self.measure_picmus_invivo_contrast(
                            i_pred, q_pred
                        )
                        mean_cnr = np.mean(cnrs)
                        mean_gcnr = np.mean(gcnrs)
                        print("Mean CNR: {}, Mean GCNR: {}".format(mean_cnr, mean_gcnr))
                        self.tb_writer.add_scalar(
                            "Mean CNR of picmus contrast invivo",
                            mean_cnr,
                            self.global_step,
                        )
                        self.tb_writer.add_scalar(
                            "Mean GCNR of picmus contrast invivo",
                            mean_gcnr,
                            self.global_step,
                        )
                        cv2.imwrite(
                            os.path.join(
                                self.exp_dir,
                                "revision_dcl_full_angles_validation_results",
                                str(self.global_step),
                                data_source
                                + "_"
                                + str(acq)
                                + "_"
                                + str(angle)
                                + "_measured.png",
                            ),
                            circle_image,
                        )

                    if (
                        data_source == "PICMUS"
                        and acq.startswith("simulation")
                        and "res" in acq
                    ):
                        fwhm, plot_image = self.measure_picmus_simulation_resolution(
                            i_pred, q_pred, np.array(xlims) * 1000
                        )
                        print("Mean FWHM: {}".format(fwhm))
                        self.tb_writer.add_scalar(
                            "Mean FWHM of picmus resolution simulation",
                            fwhm,
                            self.global_step,
                        )
                        cv2.imwrite(
                            os.path.join(
                                self.exp_dir,
                                "revision_dcl_full_angles_validation_results",
                                str(self.global_step),
                                data_source
                                + "_"
                                + str(acq)
                                + "_"
                                + str(angle)
                                + "_measured.png",
                            ),
                            plot_image,
                        )

                    if (
                        data_source == "PICMUS"
                        and acq.startswith("experiment")
                        and "res" in acq
                    ):
                        fwhm, plot_image = self.measure_picmus_experiment_resolution(
                            i_pred, q_pred, np.array(xlims) * 1000
                        )
                        print("Mean FWHM: {}".format(fwhm))
                        self.tb_writer.add_scalar(
                            "Mean FWHM of picmus resolution experiment",
                            fwhm,
                            self.global_step,
                        )
                        cv2.imwrite(
                            os.path.join(
                                self.exp_dir,
                                "revision_dcl_full_angles_validation_results",
                                str(self.global_step),
                                data_source
                                + "_"
                                + str(acq)
                                + "_"
                                + str(angle)
                                + "_measured.png",
                            ),
                            plot_image,
                        )

                    cv2.imwrite(
                        os.path.join(
                            self.exp_dir,
                            "revision_dcl_full_angles_validation_results",
                            str(self.global_step),
                            data_source + "_" + str(acq) + "_" + str(angle) + ".png",
                        ),
                        angle_frame_log * 255,
                    )

        total_mean_validation_loss /= cnt
        self.tb_writer.add_scalar(
            "Averaged validation loss", total_mean_validation_loss, self.global_step
        )
        print("Averaged validation loss: {}".format(total_mean_validation_loss))

    def prepare_data(self, P, grid, fnum, data_source, acq):
        load_dir = "./train_wvln3"
        if not os.path.exists(load_dir):
            os.mkdir(load_dir)

        acq = str(acq)

        if os.path.exists(
            os.path.join(load_dir, data_source + "_" + acq + "_i_frames.npy")
        ):
            if data_source + "_" + acq in cached_dict:
                i_frames = cached_dict[data_source + "_" + acq][0]
                q_frames = cached_dict[data_source + "_" + acq][1]
                i_frames_apod = cached_dict[data_source + "_" + acq][0]
                q_frames_apod = cached_dict[data_source + "_" + acq][1]

            else:
                i_frames = np.load(
                    os.path.join(load_dir, data_source + "_" + acq + "_i_frames.npy")
                )
                q_frames = np.load(
                    os.path.join(load_dir, data_source + "_" + acq + "_q_frames.npy")
                )
                i_frames_apod = np.load(
                    os.path.join(
                        load_dir, data_source + "_" + acq + "_i_frames_apod.npy"
                    )
                )
                q_frames_apod = np.load(
                    os.path.join(
                        load_dir, data_source + "_" + acq + "_q_frames_apod.npy"
                    )
                )
                # cached_dict[data_source + '_' + acq] = [i_frames, q_frames]

        else:
            # Make data torch tensors
            x = (P.idata, P.qdata)
            # Make 75-angle image*
            dasN = DAS_PW_DL(P, grid, rxfnum=fnum)

            idasN, qdasN, i_frames, q_frames, i_frames_apod, q_frames_apod = (
                dasN.forward(x)
            )
            np.save(
                os.path.join(load_dir, data_source + "_" + acq + "_i_frames.npy"),
                i_frames,
            )
            np.save(
                os.path.join(load_dir, data_source + "_" + acq + "_q_frames.npy"),
                q_frames,
            )
            np.save(
                os.path.join(load_dir, data_source + "_" + acq + "_i_frames_apod.npy"),
                i_frames_apod,
            )
            np.save(
                os.path.join(load_dir, data_source + "_" + acq + "_q_frames_apod.npy"),
                q_frames_apod,
            )

        self.i_frames = i_frames
        self.q_frames = q_frames

        self.i_frames_apod = i_frames_apod
        self.q_frames_apod = q_frames_apod

        self.i_cpwc = np.sum(i_frames, axis=0)
        self.q_cpwc = np.sum(q_frames, axis=0)

        self.data_source = data_source

    def get_data(self, batch_size, angle_num):

        save_path = "./debug_tmp/"
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        patches = []
        patch_num = batch_size
        gt = []
        cpwc = []

        x_locs = []
        y_locs = []

        if self.data_source != "JHU":
            self.validation_angles = [self.i_frames.shape[0] // 2]
        else:
            self.validation_angles = [0]

        self.validation_angle = self.validation_angles[0]

        for i in range(patch_num):
            each_patch = np.zeros((2, 256, 256))
            x = np.random.randint(0, self.i_frames.shape[1] - 256)
            y = np.random.randint(0, self.i_frames.shape[2] - 256)
            x_locs.append(x)
            y_locs.append(y)

            angle = np.random.randint(0, self.i_frames.shape[0])

            while angle in self.validation_angles:
                angle = np.random.randint(0, self.i_frames.shape[0])

            self.input_angle = angle

            each_patch[0, :, :] = self.i_frames[angle, x : x + 256, y : y + 256]
            each_patch[1, :, :] = self.q_frames[angle, x : x + 256, y : y + 256]

            each_gt = np.zeros((self.i_frames.shape[0] - 2, 2, 256, 256))

            cnt = 0
            for a in range(self.i_frames.shape[0]):
                if a == angle:
                    continue
                if a in self.validation_angles:
                    continue
                each_gt[cnt, 0, :, :] = self.i_frames[a, x : x + 256, y : y + 256]
                each_gt[cnt, 1, :, :] = self.q_frames[a, x : x + 256, y : y + 256]
                cnt += 1

            each_cpwc = np.zeros((2, 256, 256))
            each_cpwc[0, :, :] = self.i_cpwc[x : x + 256, y : y + 256]
            each_cpwc[1, :, :] = self.q_cpwc[x : x + 256, y : y + 256]

            patches.append(each_patch)
            gt.append(each_gt)
            cpwc.append(each_cpwc)

        patches = np.array(patches)
        gt = np.reshape(
            np.array(gt), (self.i_frames.shape[0] - 2, patch_num, 2, 256, 256)
        )
        cpwc = np.reshape(np.array(cpwc), (patch_num, 2, 256, 256))

        max_val = np.max(np.abs(gt))

        patches = patches / max_val
        gt = gt / max_val
        cpwc = cpwc / max_val

        return patches, gt, cpwc
    # metrics 정답 코드에서 가져온 함수
    def cnr(img1, img2):
        mean_1 = np.mean(img1[img1 > 0])
        mean_2 = np.mean(img2[img2 > 0])
        std_1 = np.std(img1[img1 > 0])
        std_2 = np.std(img2[img2 > 0])

        # check nan
        if np.isnan(mean_2):
            mean_2 = 0
        if np.isnan(std_2):
            std_2 = 0

        cnr_ratio = (mean_1 - mean_2) / np.sqrt((std_1**2 + std_2**2) / 2) + 1e-8
        cnr_ratio = 20 * np.log10(cnr_ratio)

    def measure_picmus_simulation_contrast(self, i_pred, q_pred):
        pred_frame = i_pred + 1j * q_pred
        pred_frame = np.abs(pred_frame)
        pred_frame = 20 * np.log10(pred_frame)
        pred_frame = pred_frame - np.max(pred_frame)

        cyst_11 = (70, 135)
        cyst_12 = (192, 135)
        cyst_13 = (315, 135)
        cyst_21 = (70, 260)
        cyst_22 = (192, 260)
        cyst_23 = (315, 260)
        cyst_31 = (70, 380)
        cyst_32 = (192, 380)
        cyst_33 = (315, 380)

        bg_11 = (132, 135)
        bg_12 = (255, 135)
        bg_21 = (132, 260)
        bg_22 = (255, 260)
        bg_31 = (132, 380)
        bg_32 = (255, 380)

        r = 17

        cysts = [
            cyst_11,
            cyst_12,
            cyst_13,
            cyst_21,
            cyst_22,
            cyst_23,
            cyst_31,
            cyst_32,
            cyst_33,
        ]
        bgs = [bg_11, bg_12, bg_12, bg_21, bg_22, bg_22, bg_31, bg_32, bg_32]

        cyst_masks = []
        bg_masks = []
        for i in range(len(cysts)):
            each_cyst = cysts[i]
            each_bg = bgs[i]
            cyst_mask = np.zeros_like(pred_frame, dtype=np.float32)
            cv2.circle(cyst_mask, each_cyst, r + 5, 1, -1)
            cyst_masks.append(cyst_mask)
            bg_mask = np.zeros_like(pred_frame, dtype=np.float32)
            cv2.circle(bg_mask, each_bg, r, 1, -1)
            bg_masks.append(bg_mask)

        cnrs, gcnrs = [], []

        pred_frame_intensity = (np.clip(pred_frame, -60, 0) + 60) / 60 * 255
        pred_frame_intensity = pred_frame_intensity.astype(np.float32)
        pred_frame_image = cv2.cvtColor(
            pred_frame_intensity.astype(np.uint8), cv2.COLOR_GRAY2RGB
        )

        for each_cyst, each_bg, each_cyst_mask, each_bg_mask in zip(
            cysts, bgs, cyst_masks, bg_masks
        ):

            cv2.circle(pred_frame_image, each_cyst, r, (0, 255, 0), 3)
            cv2.circle(pred_frame_image, each_bg, r, (0, 0, 255), 3)

            frame_cnr = cnr(
                pred_frame_intensity * each_bg_mask,
                pred_frame_intensity * each_cyst_mask,
            )
            frame_gcnr = gcnr(
                pred_frame_intensity * each_cyst_mask,
                pred_frame_intensity * each_bg_mask,
            )

            # check nan and if nan, return 0
            if np.isnan(frame_cnr):
                frame_cnr = 0
            if np.isnan(frame_gcnr):
                frame_gcnr = 0

            cnrs.append(frame_cnr)
            gcnrs.append(frame_gcnr)

        return cnrs, gcnrs, pred_frame_image
    # Compute contrast-to-noise ratio

    # metrics 정답 코드에서 가져온 함수
    def cnr(img1, img2):
        mean_1 = np.mean(img1[img1 > 0])
        mean_2 = np.mean(img2[img2 > 0])
        std_1 = np.std(img1[img1 > 0])
        std_2 = np.std(img2[img2 > 0])

        # check nan
        if np.isnan(mean_2):
            mean_2 = 0
        if np.isnan(std_2):
            std_2 = 0

        cnr_ratio = (mean_1 - mean_2) / np.sqrt((std_1**2 + std_2**2) / 2) + 1e-8
        cnr_ratio = 20 * np.log10(cnr_ratio)

        return cnr_ratio

    def measure_picmus_experiment_contrast(self, i_pred, q_pred):
        pred_frame = i_pred + 1j * q_pred
        pred_frame = np.abs(pred_frame)
        pred_frame = 20 * np.log10(pred_frame)
        pred_frame = pred_frame - np.max(pred_frame)

        roi_x = 191
        roi_y = 153
        roi_r = 15

        roi1 = (roi_x, roi_y, roi_r)
        roi2 = (roi_x + 45, roi_y, roi_r)

        cyst_mask = np.zeros(pred_frame.shape)
        cyst_mask = cv2.circle(cyst_mask, (roi1[0], roi1[1]), roi1[2], 1, -1)
        background_mask = np.zeros(pred_frame.shape)
        background_mask = cv2.circle(
            background_mask, (roi2[0], roi2[1]), roi2[2], 1, -1
        )

        pred_frame_intensity = (np.clip(pred_frame, -60, 0) + 60) / 60 * 255
        pred_frame_intensity = pred_frame_intensity.astype(np.float32)
        pred_frame_image = cv2.cvtColor(
            pred_frame_intensity.astype(np.uint8), cv2.COLOR_GRAY2RGB
        )

        cv2.circle(pred_frame_image, (roi1[0], roi1[1]), roi1[2], (0, 255, 0), 3)
        cv2.circle(pred_frame_image, (roi2[0], roi2[1]), roi2[2], (0, 0, 255), 3)

        frame_cnr = cnr(
            pred_frame_intensity * background_mask, pred_frame_intensity * cyst_mask
        )
        frame_gcnr = gcnr(
            pred_frame_intensity * cyst_mask, pred_frame_intensity * background_mask
        )
        # check nan and if nan, return 0
        if np.isnan(frame_cnr):
            frame_cnr = 0
        if np.isnan(frame_gcnr):
            frame_gcnr = 0
        return frame_cnr, frame_gcnr, pred_frame_image

    def measure_picmus_invivo_contrast(self, i_pred, q_pred):
        pred_frame = i_pred + 1j * q_pred
        pred_frame = np.abs(pred_frame)
        pred_frame = 20 * np.log10(pred_frame)
        pred_frame = pred_frame - np.max(pred_frame)

        mask = cv2.imread(
            "PICMUS_in_vivo-carotid_long_dcl_mask.png", cv2.IMREAD_GRAYSCALE
        )
        mask = cv2.resize(
            mask,
            (pred_frame.shape[1], pred_frame.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

        hypo_mask = mask == 32
        hyper_mask = mask == 33
        hypo_mask = hypo_mask.astype(np.float32)
        hyper_mask = hyper_mask.astype(np.float32)

        hypo_contour = np.zeros_like(hypo_mask)
        hyper_contour = np.zeros_like(hyper_mask)
        hypo_contours, _ = cv2.findContours(
            hypo_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        hyper_contours, _ = cv2.findContours(
            hyper_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )

        pred_frame_intensity = (np.clip(pred_frame, -60, 0) + 60) / 60 * 255
        pred_frame_intensity = pred_frame_intensity.astype(np.float32)
        pred_frame_image = cv2.cvtColor(
            pred_frame_intensity.astype(np.uint8), cv2.COLOR_GRAY2RGB
        )
        cv2.drawContours(pred_frame_image, hypo_contours, -1, (0, 255, 0), 2)
        cv2.drawContours(pred_frame_image, hyper_contours, -1, (0, 0, 255), 2)

        frame_cnr = cnr(
            pred_frame_intensity * hyper_mask, pred_frame_intensity * hypo_mask
        )
        frame_gcnr = gcnr(
            pred_frame_intensity * hypo_mask, pred_frame_intensity * hyper_mask
        )

        # check nan and if nan, return 0
        if np.isnan(frame_cnr):
            frame_cnr = 0
        if np.isnan(frame_gcnr):
            frame_gcnr = 0
        return frame_cnr, frame_gcnr, pred_frame_image

    def measure_picmus_simulation_resolution(self, i_pred, q_pred, xlim_mm):
        pred_frame = i_pred + 1j * q_pred
        pred_frame = np.abs(pred_frame)
        pred_frame = 20 * np.log10(pred_frame)
        pred_frame = pred_frame - np.max(pred_frame)

        pred_frame_intensity = (np.clip(pred_frame, -60, 0) + 60) / 60 * 255
        pred_frame_intensity = pred_frame_intensity.astype(np.float32)
        pred_frame_image = cv2.cvtColor(
            pred_frame_intensity.astype(np.uint8), cv2.COLOR_GRAY2RGB
        )

        near_y = 54
        center_y = 258
        far_y = 410
        x_center = pred_frame.shape[1] // 2

        near_axial_profile = pred_frame[near_y, x_center - 22 : x_center + 22]
        near_axial_profile = gaussian_filter(near_axial_profile, 1)
        near_axial_profile -= np.max(near_axial_profile)
        near_max_index = np.argmax(near_axial_profile)
        near_half_max = np.where(
            near_axial_profile > near_axial_profile[near_max_index] - 6
        )[0]
        near_fwhm = near_half_max[-1] - near_half_max[0]
        near_fwhm = near_fwhm * (xlim_mm[1] - xlim_mm[0]) / pred_frame.shape[1]

        center_axial_profile = pred_frame[center_y, x_center - 22 : x_center + 22]
        center_axial_profile = gaussian_filter(center_axial_profile, 1)
        center_axial_profile -= np.max(center_axial_profile)
        center_max_index = np.argmax(center_axial_profile)
        center_half_max = np.where(
            center_axial_profile > center_axial_profile[center_max_index] - 6
        )[0]
        center_fwhm = center_half_max[-1] - center_half_max[0]
        center_fwhm = center_fwhm * (xlim_mm[1] - xlim_mm[0]) / pred_frame.shape[1]

        far_axial_profile = pred_frame[far_y, x_center - 22 : x_center + 22]
        far_axial_profile = gaussian_filter(far_axial_profile, 1)
        far_axial_profile -= np.max(far_axial_profile)
        far_max_index = np.argmax(far_axial_profile)
        far_half_max = np.where(
            far_axial_profile > far_axial_profile[far_max_index] - 6
        )[0]
        far_fwhm = far_half_max[-1] - far_half_max[0]
        far_fwhm = far_fwhm * (xlim_mm[1] - xlim_mm[0]) / pred_frame.shape[1]

        # draw lines on the image
        cv2.line(
            pred_frame_image,
            (x_center - 22, near_y),
            (x_center + 22, near_y),
            (255, 0, 0),
            2,
        )
        cv2.line(
            pred_frame_image,
            (x_center - 22, center_y),
            (x_center + 22, center_y),
            (255, 0, 0),
            2,
        )
        cv2.line(
            pred_frame_image,
            (x_center - 22, far_y),
            (x_center + 22, far_y),
            (255, 0, 0),
            2,
        )
        # circle on the half max points
        cv2.circle(
            pred_frame_image,
            (x_center - 22 + near_half_max[0], near_y),
            2,
            (0, 255, 0),
            -1,
        )
        cv2.circle(
            pred_frame_image,
            (x_center - 22 + near_half_max[-1], near_y),
            2,
            (0, 255, 0),
            -1,
        )
        cv2.circle(
            pred_frame_image,
            (x_center - 22 + center_half_max[0], center_y),
            2,
            (0, 255, 0),
            -1,
        )
        cv2.circle(
            pred_frame_image,
            (x_center - 22 + center_half_max[-1], center_y),
            2,
            (0, 255, 0),
            -1,
        )
        cv2.circle(
            pred_frame_image,
            (x_center - 22 + far_half_max[0], far_y),
            2,
            (0, 255, 0),
            -1,
        )
        cv2.circle(
            pred_frame_image,
            (x_center - 22 + far_half_max[-1], far_y),
            2,
            (0, 255, 0),
            -1,
        )

        mean_fwhm = (near_fwhm + center_fwhm + far_fwhm) / 3

        return mean_fwhm, pred_frame_image

    def measure_picmus_experiment_resolution(self, i_pred, q_pred, xlim_mm):
        pred_frame = i_pred + 1j * q_pred
        pred_frame = np.abs(pred_frame)
        pred_frame = 20 * np.log10(pred_frame)
        pred_frame = pred_frame - np.max(pred_frame)

        pred_frame_intensity = (np.clip(pred_frame, -60, 0) + 60) / 60 * 255
        pred_frame_intensity = pred_frame_intensity.astype(np.float32)
        pred_frame_image = cv2.cvtColor(
            pred_frame_intensity.astype(np.uint8), cv2.COLOR_GRAY2RGB
        )

        x_center = pred_frame.shape[1] // 2 - 3
        y_rois = [97, 192, 285, 382]

        fwhms = []

        for y_center in y_rois:
            max_index = np.argmax(pred_frame[y_center, x_center - 28 : x_center + 28])
            max_index += x_center - 28
            beam_profile = gaussian_filter(
                pred_frame[y_center, max_index - 28 : max_index + 28]
                - np.max(pred_frame[y_center, max_index - 28 : max_index + 28]),
                1,
            )
            max_value = np.max(beam_profile)
            half_max = np.where(beam_profile > (max_value - 6))[0]
            fwhm = half_max[-1] - half_max[0]
            fwhm = fwhm * (xlim_mm[1] - xlim_mm[0]) / pred_frame.shape[1]
            fwhms.append(fwhm)

            cv2.line(
                pred_frame_image,
                (max_index - 28, y_center),
                (max_index + 28, y_center),
                (255, 0, 0),
                2,
            )
            cv2.circle(
                pred_frame_image,
                (max_index - 28 + half_max[0], y_center),
                2,
                (0, 255, 0),
                -1,
            )
            cv2.circle(
                pred_frame_image,
                (max_index - 28 + half_max[-1], y_center),
                2,
                (0, 255, 0),
                -1,
            )

        mean_fwhm = np.mean(fwhms)

        return mean_fwhm, pred_frame_image


if __name__ == "__main__":
    device = torch.device("cuda:0")

    config = {}
    config["step_per_file"] = 10
    config["num_epochs"] = 100
    config["angle_num"] = 1
    config["random_angle"] = True
    config["lr"] = 1e-4
    config["device"] = device
    config["batch_size"] = 1
    config["dr_coeff"] = 0.001
    config["patch_size"] = 256

    pw_trainer = PWTrainer(config)
    pw_trainer.model.eval()

    pw_trainer.train()
