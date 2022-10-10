import importlib
from omegaconf import OmegaConf
import json
import os
import sys
import numpy as np
from PIL import Image
from typing import Dict
import requests
from ldm.models.diffusion.plms import PLMSSampler
from torch import autocast
import hashlib
from tqdm import trange
from einops import rearrange

sys.path.append("./")
import torch

def instantiate_from_config(config):
    if not "target" in config:
        if config == '__is_first_stage__':
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)

def get_image(model, sample):
    x_samples_ddim = model.decode_first_stage(sample)
    x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
    x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()

    x_sample = torch.from_numpy(x_samples_ddim).permute(0, 3, 1, 2)[0]

    x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
    img = Image.fromarray(x_sample.astype(np.uint8))
    return img

def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model


class TogetherStableDiffusion():
    def __init__(self) -> None:
        config = OmegaConf.load(f"configs/stable-diffusion/v1-inference.yaml")
        model_ckpt = "models/ldm/stable-diffusion-v1/model.ckpt"
        self.model = load_model_from_config(config, model_ckpt)

    def infer(self, job_id, args) -> Dict:
        args = json.loads(args)
        # args: {"prompt": "a photo of a person", "ddim_steps": 50, "n_samples": 3, "scale": 7.5, "seed": 42, "viz_params": False}
        print(args)
        
        data = [1 * [args['prompt']]]
        base_count = 0
        batch_size = 1

        images = []
        paths = []
        sampler = PLMSSampler(self.model)
        precision_scope = autocast
        prompt_hash = hashlib.md5(args['prompt'].encode()).hexdigest()

        with torch.no_grad():
            with precision_scope("cuda"):
                with self.model.ema_scope():
                    for n in trange(args['n_samples'], desc="Sampling"):
                        for prompts in data:
                            uc = None
                            if args['scale'] != 1.0:
                                uc = self.model.get_learned_conditioning(batch_size * [""])
                            if isinstance(prompts, tuple):
                                prompts = list(prompts)
                            c = self.model.get_learned_conditioning(prompts)
                            shape = [4, 512 // 8, 512 // 8] # [C, H//f, W//f]

                            samples_ddim, intermediates = sampler.sample(S=args['ddim_steps'],
                                                                conditioning=c,
                                                                batch_size=1,
                                                                shape=shape,
                                                                verbose=False,
                                                                unconditional_guidance_scale=args['scale'],
                                                                log_every_t=1,
                                                                unconditional_conditioning=uc,
                                                                eta=0.0)

                            if args['viz_params']:
                                intermediates = intermediates['pred_x0']
                                for step_no, intermediate in enumerate(intermediates):
                                    img = get_image(self.model, intermediate)
                                    images.append(img)
                                    path = f"{prompt_hash}_{base_count:05}_{step_no:05}.png"
                                    paths.append(path)
                                    img.save(path)


                            else:
                                x_samples_ddim = self.model.decode_first_stage(samples_ddim)
                                x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                                x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()

                                x_checked_image_torch = torch.from_numpy(x_samples_ddim).permute(0, 3, 1, 2)

                                for x_sample in x_checked_image_torch:
                                    x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                                    img = Image.fromarray(x_sample.astype(np.uint8))
                                    images.append(img)

                            path = os.path.join(f"{prompt_hash}_{base_count:05}.png")
                            paths.append(path)
                            base_count += 1

        # save images to file
        filenames = []
        for path in paths:
            # upload images to s3
            with open(path, "rb") as fp:
                files = {"file": fp}
                res = requests.post("https://planetd.shift.ml/file", files=files).json()
                filenames.append(res["filename"])
            os.remove(f"path")
        # delete the file
        print("sending requests to global")
        # write results back
        coord_url = os.environ.get("COORDINATOR_URL", "localhost:8092/my_coord")
        worker_name = os.environ.get("WORKER_NAME", "planetv2")
        requests.patch(
            f"http://{coord_url}/api/v1/g/jobs/{job_id}",
            json={
                "status": "finished",
                "returned_payload": {"output": [filenames]},
                "source": "dalle",
                "type": "general",
                "processed_by": worker_name,
            },
        )


if __name__ == "__main__":
    fip = TogetherStableDiffusion()
    fip.start()