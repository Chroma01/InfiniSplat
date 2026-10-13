# InfiniSplat

<p align="center">Feed-forward 3D Gaussian reconstruction from images.</p>

<p align="center">
  <strong><a href="#infinisplat-v2">v2 · Multi-view</a></strong>
  &nbsp;|&nbsp;
  <strong><a href="#infinisplat-v1">v1 · Single-image / LiDAR</a></strong>
</p>

---

<a name="infinisplat-v2"></a>

<div align="center">

<h2>🌌 InfiniSplat v2: Analyzing and Improving<br>Feed-Forward 3DGS from Any Views</h2>

<div align="center">
  <a href="https://zju3dv.github.io/InfiniSplat2/">
    <img src="https://img.shields.io/badge/Project-red?logo=googlechrome&logoColor=white" alt="InfiniSplat v2 project page">
  </a>
  <img src="https://img.shields.io/badge/Paper-Coming_Soon-lightgrey" alt="Paper: coming soon">
  <a href="https://huggingface.co/spaces/Edisoneh/InfiniSplat-v2-demo">
    <img src="https://img.shields.io/badge/Demo-yellow?logo=huggingface&logoColor=white" alt="Hugging Face Demo">
  </a>
  <a href="https://huggingface.co/Edisoneh/InfiniSplat-v2">
    <img src="https://img.shields.io/badge/Weights-yellow?logo=huggingface&logoColor=white" alt="InfiniSplat v2 weights">
  </a>
  <a href="v2/README.md">
    <img src="https://img.shields.io/badge/Code-blue?logo=github" alt="InfiniSplat v2 code">
  </a>
</div>

<p>
  <a href="https://github.com/Edisoneh">Yongzhen Hu<sup>*</sup></a> ·
  <a href="https://ritianyu.github.io/">Hao Yu<sup>*</sup></a> ·
  <a href="https://plus-wave.github.io/">Jiawei Wang</a> ·
  Xianming Zeng · Jianyun Xu<br>
  Xingliang Liu ·
  <a href="https://csse.szu.edu.cn/staff/ruizhenhu/">Ruizhen Hu</a> ·
  <a href="https://xzhou.me/">Xiaowei Zhou</a> ·
  <a href="https://person.zju.edu.cn/en/0093140">Hujun Bao</a> ·
  <a href="https://pengsida.net/">Sida Peng<sup>†</sup></a>
</p>

<p><sup>*</sup> Equal contribution. <sup>†</sup> Corresponding author.</p>

</div>

<div align="center">

<a href="https://zju3dv.github.io/InfiniSplat2/#demo"><img src="v2/assets/github_demo.gif" alt="InfiniSplat v2 reconstruction previews" width="90%" /></a>

</div>

<!-- Replace the paper badge with its public paper link when available; update the project-page URL if its hosting location changes. -->

InfiniSplat v2 reconstructs 3D Gaussian scenes from multiple images, with or without
camera poses, for novel-view rendering.

[Code and usage](v2/README.md) · [Installation](v2/INSTALL.md) ·
[Examples](v2/examples/README.md) · [COLMAP guide](v2/docs/colmap.md)

---

<a name="infinisplat-v1"></a>

<div align="center">

<h2>🌌 InfiniSplat: Implicit Gaussian Decoding<br>for Large-Baseline Monocular View Synthesis</h2>

<p align="center"><strong>SIGGRAPH Asia 2026 (ACM Transactions on Graphics)</strong></p>

<div align="center">
  <a href="https://zju3dv.github.io/InfiniSplat">
    <img src="https://img.shields.io/badge/Project-red?logo=googlechrome&logoColor=white">
  </a>
  <a href="https://arxiv.org/abs/2608.02437">
    <img src="https://img.shields.io/badge/Paper-blue?logo=arxiv&logoColor=white">
  </a>
  <a href="https://zju3dv.github.io/InfiniSplat/#visualization">
    <img src="https://img.shields.io/badge/Gallery-green?logo=googlephotos&logoColor=white">
  </a>
  <a href="https://huggingface.co/spaces/PLUS-WAVE/InfiniSplat">
    <img src="https://img.shields.io/badge/Demo-yellow?logo=huggingface&logoColor=white">
  </a>
</div>

<p align="center">
  <a href="https://plus-wave.github.io/">Jiawei Wang<sup>*</sup></a> ·
  <a href="https://ritianyu.github.io/">Hao Yu<sup>*</sup></a> ·
  <a href="https://github.com/Edisoneh">Yongzhen Hu</a> ·
  <a href="https://github.com/shmily768">Xinyi Yang</a> ·
  Tao Ni<br>
  Xin Zhan ·
  Junbo Chen<sup>†</sup> ·
  <a href="https://xzhou.me/">Xiaowei Zhou</a> ·
  <a href="https://csse.szu.edu.cn/staff/ruizhenhu/">Ruizhen Hu</a> ·
  <a href="https://pengsida.net/">Sida Peng<sup>†</sup></a>
</p>

<p align="center"><sup>*</sup> Equal contribution. <sup>†</sup> Corresponding authors.</p>

</div>

<div align="center">

<img src="v1/assets/github_demo.gif" alt="InfiniSplat v1 Demo" width="90%" />

</div>

InfiniSplat v1 reconstructs 3D Gaussian scenes from a single RGB image, optionally
guided by LiDAR depth, for novel-view rendering.

[Code and usage](v1/README.md) · [Installation](v1/INSTALL.md) ·
[Inference guide](v1/docs/inference.md)

---

## Getting started

From the repository root, run `cd v2` for multi-view reconstruction or `cd v1`
for single-image / LiDAR-guided reconstruction, then follow that version's installation guide.
Install each version in its own Python environment; their dependencies and weights differ.

## License and Project Use

License: [Project Registration License (PRL) v1.0](LICENSE)

Use solely for academic research, education, evaluation, or personal purposes is free and does not require registration. Research results and publications may be published without registration.

Project use by companies, universities, or other organizations is free but requires prior registration, including non-commercial projects, internal R&D, testing, deployment, and production. Complete and accurate registration automatically grants permission; no approval or fee is required. Register each materially distinct project once.

**Project use registration and Word template:** [Google Forms](https://docs.google.com/forms/d/e/1FAIpQLSff5QjjANlJ2hnXYqBSAzKWxEM58gIxPLh17-t3j3-MqO1a5w/viewform)


V1 and v2 share the project license above. Third-party components retain their
respective licenses and notices; see [v2 component provenance](v2/THIRD_PARTY.md).

## Acknowledgments

See the [v1 acknowledgments](v1/README.md#-acknowledgments) and
[v2 acknowledgments](v2/README.md#acknowledgments).
