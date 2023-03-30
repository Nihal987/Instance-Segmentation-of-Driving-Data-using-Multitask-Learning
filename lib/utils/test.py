import torch
import cv2

def scale(self, masks, colors, im_gpu=None, alpha=0.5):
    colors = torch.tensor(colors, device=im_gpu.device, dtype=torch.float32) / 255.0
    colors = colors[:, None, None]  # shape(n,1,1,3)
    masks = masks.unsqueeze(3)  # shape(n,h,w,1)
    masks_color = masks * (colors * alpha)  # shape(n,h,w,3)

    inv_alph_masks = (1 - masks * alpha).cumprod(0)  # shape(n,h,w,1)
    mcs = (masks_color * inv_alph_masks).sum(0) * 2  # mask color summand shape(n,h,w,3)

    im_gpu = im_gpu.flip(dims=[0])  # flip channel

    im_gpu = im_gpu.permute(1, 2, 0).contiguous()  # shape(h,w,3)
    im_gpu = im_gpu * inv_alph_masks[-1] + mcs
    # im_gpu = torch.clamp(im_gpu, min=0.0, max=1.0) 
    im_mask = (im_gpu * 255).byte().cpu().numpy()
    cv2.imwrite("inference/test/0.jpg",im_mask)
    cv2.imwrite("inference/test/1.jpg",self.im)
    self.im[:] = scale_image(im_gpu.shape, im_mask, self.im.shape)