# Download a Specific Subfolder from a Hugging Face Repository

This guide shows how to use the `huggingface-cli` command-line tool to download a subfolder from a public (or private) Hugging Face model repository.

## 1. Install the CLI

Make sure you have Python and `pip` installed, then run:

```bash
pip install --upgrade "huggingface_hub[cli]"
```

## 2. (Optional) Log In to Hugging Face

If the repository is private, log in with your Hugging Face access token:

```bash
huggingface-cli login
```

You will be prompted to paste your token.
If the repo is public, you can skip this step.

## 3. (Example) Run the Download Command

Use `huggingface-cli download` with `--allow-patterns` and `--ignore-patterns` to fetch only the subfolder you need. For example, to grab the `diffusion/` folder from the `TianyuLin/CARE` repo:

```bash
huggingface-cli download TianyuLin/CARE \
  --allow-patterns="diffusion/*" \
  --ignore-patterns="*" \
  --to-local-dir="./local_diffusion"
```

- `TianyuLin/CARE`  
  The full repo ID on Hugging Face (“username/repo_name”).  
- `--allow-patterns="diffusion/*"`  
  Only files whose paths start with `diffusion/` will be downloaded.  
- `--ignore-patterns="*"`  
  Ignore everything else that does not match `diffusion/*`.  
- `--to-local-dir="./local_diffusion"`  
  Place the downloaded files inside `./local_diffusion/`. If this option is omitted, files are cached under `~/.cache/huggingface/`.

After running the command, you should see progress output and end up with:

```bash
./local_diffusion/
└── diffusion/
    ├── config.json
    ├── model.ckpt
    └── ...
```
