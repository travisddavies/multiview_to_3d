import torch
import cv2

from models import bird_model
from .losses import (camera_fitting_loss, kpts_fitting_loss,
                     mask_fitting_loss, prior_loss)


class OptimiseSV:
    """
    Implementation of single view reconstruction optimiser
    """

    def __init__(self, prior_weight=1, mask_weight=1, step_size=1e-2,
                 num_iters=50, use_mask=False, renderer=None,
                 device=torch.device("cpu")):
        self.device = device
        self.step_size = step_size
        self.num_iters = num_iters
        self.prior_weight = prior_weight
        self.mask_weight = mask_weight
        self.use_mask = use_mask
        if use_mask:
            self.renderer = renderer

        # Load Bird Mesh Model and prior
        self.bird = bird_model(device=device)
        self.faces = torch.tensor(self.bird.dd["F"])
        self.p_m = self.bird.p_m
        self.b_m = self.bird.b_m
        self.p_cov_in = self.bird.p_cov.inverse()
        self.b_cov_in = self.bird.b_cov.inverse()

    def transform_t(self, tran):
        tran_xyz = tran.clone()
        tran_xyz[:, 1] = tran_xyz[:, 1] - 1
        tran_xyz[:, 2] = tran_xyz[:, 2]*18 + 180

        return tran_xyz

    def transform_p(self, pose):
        batch_size = len(pose)

        pose = pose.to("cpu")
        pose = pose.detach().clone()
        pose = pose.reshape(batch_size, -1, 3, 3)
        new_pose = torch.zeros([batch_size, pose.shape[1]*3]).float()

        for i in range(batch_size):
            for j in range(pose.shape[1]):
                R = pose[i, j]
                aa, _ = cv2.Rodrigues(R.numpy())
                new_pose[i, 3*j:3*(j+1)] = torch.tensor(aa).squeeze()

        return new_pose

    def render_silhouette(self, vertices):
        size = self.renderer.size
        faces = self.faces.clone().repeat(1, 1, 1)
        batch = vertices.shape[0]

        silhouette = torch.zeros([batch, size, size]).float().to(self.device)

        for i in range(batch):
            silhouette[i] = self.renderer(vertices[[i]], faces)[..., 3]

        return silhouette

    def __call__(self, init_pose, init_bone, init_t, focal_length=2167,
                 camera_center=128, keypoints=None, masks=None):
        """Perform single view reconstruction
        Input:
            init_pose: (BN, 25*9) initial pose estimate
            init_bone: (BN, 24) initial bone estimate
            init_t: (BN, 3) initial translation estimate
            scale: (1,) initial scale estimate
            keypoints: (BN, 12, 3) batch keypoints with confidence
            masks: (BN, 256, 256) batch silhouettes
        """
        # Unbind keypoint location and confidence
        keypoints_2d = keypoints[:, :, :2].to(self.device)
        keypoints_conf = keypoints[:, :, -1].to(self.device)

        # Copy all initialisation
        global_t = init_t.detach().clone().to(self.device)
        bone_length = init_pose.detach().clone()[:, :3].to(self.device)

        init_pose = self.transform_p(init_pose)
        global_orient = init_pose.detach().clone()[:, :3].to(self.device)
        body_pose = init_pose.detach().clone()[:, 3:].to(self.device)

        body_pose.requires_grad = False
        bone_length.requires_grad = False
        global_orient.requires_grad = True
        global_t.requires_grad = True

        body_opt_params = [global_orient, global_t]
        body_optimiser = torch.optim.Adam(body_opt_params, lr=self.step_size,
                                          betas=[0.9, 0.999])

        for i in range(self.num_iters):
            bird_output = self.bird(global_pose=global_orient,
                                    body_pose=body_pose,
                                    bone_length=bone_length,
                                    pose2rot=True)

            global_txyz = self.transform_t(global_t)
            model_keypoints = bird_output["keypoints"] + global_txyz.unsqueeze(1)
            loss = camera_fitting_loss(model_keypoints, None, None,
                                       focal_length, camera_center,
                                       keypoints_2d, keypoints_conf)

            body_optimiser.zero_grad()
            loss.backward()
            body_optimiser.step()

        # Step 2: Optimize all parameters with keypoints
        body_pose.requires_grad = True
        bone_length.requires_grad = True
        global_orient.requires_grad = True
        global_t.requires_grad = True

        body_opt_params = [body_pose, bone_length, global_orient, global_t]
        body_optimiser = torch.optim.Adam(body_opt_params, lr=self.step_size,
                                          betas=(0.9, 0.999))

        for i in range(self.num_iters):
            bird_output = self.bird(global_pose=global_orient,
                                    body_pose=body_pose,
                                    bone_length=bone_length, pose2rot=True)

            global_txyz = self.transform_t(global_t)
            model_keypoints = bird_output['keypoints'] + global_txyz.unsqueeze(1)

            loss = kpts_fitting_loss(model_keypoints, focal_length,
                                     camera_center, keypoints_2d,
                                     keypoints_conf, body_pose, bone_length,
                                     prior_weight=self.prior_weight)

            loss_p = prior_loss(body_pose, self.p_m, self.p_cov_in,
                                self.prior_weight)
            loss_b = prior_loss(bone_length, self.b_m, self.b_cov_in,
                                self.prior_weight)

            loss = loss + loss_p + loss_b

            body_optimiser.zero_grad()
            loss.backward()
            body_optimiser.step()

        # Step 3: Optimise all parameters with keypoints and silhouette
        if self.use_mask:
            masks = masks.detach().clone().to(self.device)
            pose_init = body_pose.detach().clone()
            bone_init = bone_length.detach().clone()

            for i in range(25):
                bird_output = self.bird(global_pose=global_orient,
                                        body_pose=body_pose,
                                        bone_length=bone_length,
                                        pose2rot=True)

                global_txyz = self.transform_t(global_t)
                model_keypoints = bird_output["keypoints"] + global_txyz.unsqueeze(1)
                model_mesh = bird_output["vertices"] + global_txyz.unsqueeze(1)

                loss = kpts_fitting_loss(model_keypoints, focal_length,
                                         camera_center, keypoints_2d,
                                         keypoints_conf, body_pose,
                                         bone_length, pose_init=pose_init,
                                         bone_init=bone_init)

                loss_p = prior_loss(body_pose, self.p_m, self.p_cov_in,
                                    self.prior_weight)
                loss_b = prior_loss(bone_length, self.b_m, self.b_cov_in,
                                    self.prior_weight)

                # Loss for equation 6 in paper
                loss_mask = mask_fitting_loss(
                    self.render_silhouette(model_mesh),
                    masks, self.mask_weight)

                loss = loss + loss_b + loss_p + loss_mask

                body_optimiser.zero_grad()
                loss.backward()
                body_optimiser.step()

        # Output
        bird_output = self.bird(global_pose=global_orient,
                                body_pose=body_pose,
                                bone_length=bone_length, pose2rot=True)
        global_txyz = self.transform_t(global_t)

        pose = torch.cat([global_orient, body_pose], dim=-1).detach().to("cpu")
        bone = bone_length.detach().to("cpu")
        global_t = global_t.detach().to("cpu")
        model_mesh = bird_output["vertices"] + global_txyz.unsqueeze(1)
        model_mesh = model_mesh.detach().to("cpu")

        return pose, bone, global_t, model_mesh
