# Checkpoint Placeholder

Pretrained model checkpoints are intentionally not stored in this GitHub repository. They are published on Zenodo:

- Pretrained models and generated samples: [`10.5281/zenodo.21635468`](https://doi.org/10.5281/zenodo.21635468)
- Checksums: `SHA256SUMS-models-samples.txt` inside `eta-models-samples-v1.0.0.zip`; verify with `shasum -a 256 -c`
- License: CC-BY-4.0

Stage downloaded or trained checkpoints under the repository checkout:

```text
/path/to/eta/models/
  precip-srcnn/
  fm/
/path/to/eta/samples/
```

Unpacking `eta-models-samples-v1.0.0.zip` at the repository root reproduces the `models/` and `samples/` layout above. Full ERA5-Land, GEVD, and Flow Matching notebook reruns also require the datasets from [`10.5281/zenodo.21635446`](https://doi.org/10.5281/zenodo.21635446).

Do not commit `.pt`, `.pth`, `.ckpt`, generated samples, or model archives here.
