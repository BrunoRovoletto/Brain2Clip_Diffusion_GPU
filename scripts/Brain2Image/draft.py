def main(
        im_path="data/example_conditioning/superresolution/sample_0.jpg",
        ckpt="/workspace/Datasets/Models/sd-clip-vit-l14-img-embed_ema_only.ckpt",
        config="configs/stable-diffusion/sd-image-condition-finetune.yaml",
        outpath="im_variations",
        scale=1.0,
        h=200,
        w=200,
        n_samples=1,
        precision="fp32",
        plms=True,
        ddim_steps=10,
        ddim_eta=0.0,
        device_idx=0,
        save=True,
        eval=False,
        ):
        
        if isinstance(im_path, str):
            im_paths = glob.glob(im_path)
        im_paths = sorted(im_paths)




        all_similarities = []

        for im in im_paths:
            input_im = load_im(im).to(device)
            
            torch.save("shapes/aaa", im.shape)