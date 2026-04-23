 RI3D: Few-Shot Gaussian Splatting With Repair and Inpainting Diffusion Priors   

1.  [1 Introduction](https://arxiv.org/html/2503.10860v2#S1 "In RI3D: Few-Shot Gaussian Splatting With Repair and Inpainting Diffusion Priors")
2.  [2 Related Work](https://arxiv.org/html/2503.10860v2#S2 "In RI3D: Few-Shot Gaussian Splatting With Repair and Inpainting Diffusion Priors")
    1.  [2.1 Radiance Field and 3D Gaussian Splatting](https://arxiv.org/html/2503.10860v2#S2.SS1 "In 2 Related Work ‣ RI3D: Few-Shot Gaussian Splatting With Repair and Inpainting Diffusion Priors")
    2.  [2.2 Sparse Novel View Synthesis](https://arxiv.org/html/2503.10860v2#S2.SS2 "In 2 Related Work ‣ RI3D: Few-Shot Gaussian Splatting With Repair and Inpainting Diffusion Priors")
    3.  [2.3 Diffusion-based Methods](https://arxiv.org/html/2503.10860v2#S2.SS3 "In 2 Related Work ‣ RI3D: Few-Shot Gaussian Splatting With Repair and Inpainting Diffusion Priors")
3.  [3 Background](https://arxiv.org/html/2503.10860v2#S3 "In RI3D: Few-Shot Gaussian Splatting With Repair and Inpainting Diffusion Priors")
    1.  [3.1 3D Gaussian Splatting](https://arxiv.org/html/2503.10860v2#S3.SS1 "In 3 Background ‣ RI3D: Few-Shot Gaussian Splatting With Repair and Inpainting Diffusion Priors")
    2.  [3.2 Diffusion Models](https://arxiv.org/html/2503.10860v2#S3.SS2 "In 3 Background ‣ RI3D: Few-Shot Gaussian Splatting With Repair and Inpainting Diffusion Priors")
4.  [4 Algorithm](https://arxiv.org/html/2503.10860v2#S4 "In RI3D: Few-Shot Gaussian Splatting With Repair and Inpainting Diffusion Priors")
    1.  [4.1 3D Gaussian Initialization](https://arxiv.org/html/2503.10860v2#S4.SS1 "In 4 Algorithm ‣ RI3D: Few-Shot Gaussian Splatting With Repair and Inpainting Diffusion Priors")
    2.  [4.2 Repair and Inpainting Diffusion Models](https://arxiv.org/html/2503.10860v2#S4.SS2 "In 4 Algorithm ‣ RI3D: Few-Shot Gaussian Splatting With Repair and Inpainting Diffusion Priors")
    3.  [4.3 Optimization](https://arxiv.org/html/2503.10860v2#S4.SS3 "In 4 Algorithm ‣ RI3D: Few-Shot Gaussian Splatting With Repair and Inpainting Diffusion Priors")
        1.  [Stage 1:](https://arxiv.org/html/2503.10860v2#S4.SS3.SSS0.Px1 "In 4.3 Optimization ‣ 4 Algorithm ‣ RI3D: Few-Shot Gaussian Splatting With Repair and Inpainting Diffusion Priors")
        2.  [Stage 2:](https://arxiv.org/html/2503.10860v2#S4.SS3.SSS0.Px2 "In 4.3 Optimization ‣ 4 Algorithm ‣ RI3D: Few-Shot Gaussian Splatting With Repair and Inpainting Diffusion Priors")
5.  [5 Results](https://arxiv.org/html/2503.10860v2#S5 "In RI3D: Few-Shot Gaussian Splatting With Repair and Inpainting Diffusion Priors")
    1.  [5.1 Visual Results](https://arxiv.org/html/2503.10860v2#S5.SS1 "In 5 Results ‣ RI3D: Few-Shot Gaussian Splatting With Repair and Inpainting Diffusion Priors")
    2.  [5.2 Numerical Results](https://arxiv.org/html/2503.10860v2#S5.SS2 "In 5 Results ‣ RI3D: Few-Shot Gaussian Splatting With Repair and Inpainting Diffusion Priors")
    3.  [5.3 Ablations](https://arxiv.org/html/2503.10860v2#S5.SS3 "In 5 Results ‣ RI3D: Few-Shot Gaussian Splatting With Repair and Inpainting Diffusion Priors")
6.  [6 Limitations](https://arxiv.org/html/2503.10860v2#S6 "In RI3D: Few-Shot Gaussian Splatting With Repair and Inpainting Diffusion Priors")
7.  [7 Conclusion](https://arxiv.org/html/2503.10860v2#S7 "In RI3D: Few-Shot Gaussian Splatting With Repair and Inpainting Diffusion Priors")
8.  [8 Implementation Details](https://arxiv.org/html/2503.10860v2#S8 "In RI3D: Few-Shot Gaussian Splatting With Repair and Inpainting Diffusion Priors")
    1.  [8.1 Repair Model](https://arxiv.org/html/2503.10860v2#S8.SS1 "In 8 Implementation Details ‣ RI3D: Few-Shot Gaussian Splatting With Repair and Inpainting Diffusion Priors")
    2.  [8.2 Inpainting Model](https://arxiv.org/html/2503.10860v2#S8.SS2 "In 8 Implementation Details ‣ RI3D: Few-Shot Gaussian Splatting With Repair and Inpainting Diffusion Priors")
    3.  [8.3 Optimization](https://arxiv.org/html/2503.10860v2#S8.SS3 "In 8 Implementation Details ‣ RI3D: Few-Shot Gaussian Splatting With Repair and Inpainting Diffusion Priors")
9.  [9 Additional Results](https://arxiv.org/html/2503.10860v2#S9 "In RI3D: Few-Shot Gaussian Splatting With Repair and Inpainting Diffusion Priors")
10.  [10 Limitations](https://arxiv.org/html/2503.10860v2#S10 "In RI3D: Few-Shot Gaussian Splatting With Repair and Inpainting Diffusion Priors")

RI3D: Few-Shot Gaussian Splatting With Repair and Inpainting Diffusion Priors
=============================================================================

Avinash Paliwal1     Xilong Zhou1,3     Wei Ye2     Jinhui Xiong2  
Rakesh Ranjan2     Nima Khademi Kalantari1  
1Texas A&M University     2Meta Reality Labs     3 Max Planck Institute for Informatics  

###### Abstract

In this paper, we propose RI3D, a novel 3DGS-based approach that harnesses the power of diffusion models to reconstruct high-quality novel views given a sparse set of input images. Our key contribution is separating the view synthesis process into two tasks of reconstructing visible regions and hallucinating missing regions, and introducing two personalized diffusion models, each tailored to one of these tasks. Specifically, one model (’repair’) takes a rendered image as input and predicts the corresponding high-quality image, which in turn is used as a pseudo ground truth image to constrain the optimization. The other model (’inpainting’) primarily focuses on hallucinating details in unobserved areas. To integrate these models effectively, we introduce a two-stage optimization strategy: the first stage reconstructs visible areas using the repair model, and the second stage reconstructs missing regions with the inpainting model while ensuring coherence through further optimization. Moreover, we augment the optimization with a novel Gaussian initialization method that obtains per-image depth by combining 3D-consistent and smooth depth with highly detailed relative depth. We demonstrate that by separating the process into two tasks and addressing them with the repair and inpainting models, we produce results with detailed textures in both visible and missing regions that outperform state-of-the-art approaches on a diverse set of scenes with extremely sparse inputs111[https://people.engr.tamu.edu/nimak/Papers/RI3D](https://people.engr.tamu.edu/nimak/Papers/RI3D).

![[Uncaptioned image]](x1.png)

Figure 1: We introduce a novel sparse view synthesis method that employs two diffusion models, “repair” and “inpainting”, which are responsible for aiding in the reconstruction of visible regions and hallucinating missing regions, respectively. Our approach involves a two-stage optimization process. In the first stage, we use the repair model to constrain the 3DGS optimization and reconstruct the regions covered by the input images. As shown, the output of the first stage properly reconstructs the visible areas, but contains missing regions, which are marked in white. In the second stage, we utilize the inpainting model to fill in these missing areas and continue optimization using the repair model to seamlessly integrate the hallucinated regions with the rest of the scene. Here, we compare our method (“Stage 2”) against several state-of-the-art techniques on a 360° scene using only three input images.

1 Introduction
--------------

The introduction of novel 3D representations, such as neural radiance fields (NeRF) \[[18](https://arxiv.org/html/2503.10860v2#bib.bib18)\] and 3D Gaussian splatting (3DGS) \[[14](https://arxiv.org/html/2503.10860v2#bib.bib14)\], has revolutionized the field of novel view synthesis. While these techniques excel at reconstructing 3D scenes from a large number of images, view synthesis from a sparse set of images remains a challenging problem.

Most state-of-the-art sparse novel view synthesis approaches \[[49](https://arxiv.org/html/2503.10860v2#bib.bib49), [16](https://arxiv.org/html/2503.10860v2#bib.bib16), [33](https://arxiv.org/html/2503.10860v2#bib.bib33)\] introduce a series of regularizations to constrain the optimization process and avoid overfitting to the input images. While these approaches produce results with reasonable texture in areas visible in a few input images, they often struggle to hallucinate details in the occluded regions. This issue is especially pronounced in 360∘ scene reconstruction with extremely sparse input images where there are larger missing regions and areas covered by only a single image (see Fig. [1](https://arxiv.org/html/2503.10860v2#S0.F1 "Figure 1 ‣ RI3D: Few-Shot Gaussian Splatting With Repair and Inpainting Diffusion Priors")).

Recently, a couple of techniques \[[39](https://arxiv.org/html/2503.10860v2#bib.bib39), [8](https://arxiv.org/html/2503.10860v2#bib.bib8)\] propose to address this issue using a diffusion model as a prior during the optimization process. Specifically, they first train a diffusion model \[[26](https://arxiv.org/html/2503.10860v2#bib.bib26)\] on a large multiview image dataset to synthesize novel views given a set of input images. They then perform the optimization using a combination of the input images and novel views, synthesized using this _view synthesis diffusion model_. Although these methods produce significantly better results than the previously discussed techniques, particularly in 360∘ scenes, they often overblur details, especially in missing areas (see supplementary video). This is because although their view synthesis diffusion models produce visually pleasing novel views, the reconstructed images, particularly in occluded regions, are not 3D-consistent. Using such images during optimization thus leads to overblurring of details. Additionally, since these methods use NeRF as their 3D representation, they suffer from slow rendering times.

To address this issue, we propose to separate the view synthesis process into two tasks of reconstructing the visible and missing areas and introduce two diffusion models, each specialized to help with one of these two tasks. Specifically, inspired by Yang et al.’s approach \[[42](https://arxiv.org/html/2503.10860v2#bib.bib42)\], one diffusion model (repair model) is responsible for taking a rendered image as the input and producing a clean image. This model effectively suppresses the artifacts in the reconstructed images during the optimization process. The second diffusion model (inpainting model) is solely responsible for hallucinating details in the missing regions. To ensure these two diffusion models produce results consistent with the scene at hand, we personalize them by tuning the models on the input images.

To ensure fast inference speed, instead of using NeRF as done in the previous approaches \[[39](https://arxiv.org/html/2503.10860v2#bib.bib39), [8](https://arxiv.org/html/2503.10860v2#bib.bib8)\], we utilize 3DGS as our 3D representation. To initialize the Gaussians, we propose to obtain per image depth by combining the 3D-consistent, but smooth, depth estimates from DUSt3R \[[38](https://arxiv.org/html/2503.10860v2#bib.bib38)\] with the highly detailed relative depth from a monocular depth estimation approach through Poisson blending \[[23](https://arxiv.org/html/2503.10860v2#bib.bib23)\]. Using the estimated depth, we then assign one Gaussian to each pixel of every input image and project them into 3D space. Moreover, to effectively utilize the two diffusion models, we propose a two-stage optimization process. The goal of the first stage is to reconstruct the scene with detailed texture in the visible areas (see Fig. [1](https://arxiv.org/html/2503.10860v2#S0.F1 "Figure 1 ‣ RI3D: Few-Shot Gaussian Splatting With Repair and Inpainting Diffusion Priors") “Stage 1”) by utilizing the repair model to enhance the renderings and using them as pseudo ground truth. During the second stage, we hallucinate details in the missing areas using the inpainting model and continue the optimization by utilizing the repair model to seamlessly integrate the hallucinated details with the rest of the scene.

We show that our approach, dubbed RI3D, produces high-quality textures particularly in occluded areas, for challenging scenarios. We further demonstrate that our results outperform the state of the art, both numerically and visually. In summary, we make the following contributions:

*   •
    
    To improve the view synthesis process, we propose to utilize two personalized diffusion models: one for enhancing the rendered images and using them as pseudo ground truth during optimization and another for hallucinating details in the missing areas.
    
*   •
    
    We propose a novel approach to initialize Gaussians by combining 3D-consistent, smooth depth maps with highly detailed relative depth from monocular approaches.
    
*   •
    
    We introduce a two-stage optimization strategy that seamlessly incorporates the two diffusion models.
    

2 Related Work
--------------

### 2.1 Radiance Field and 3D Gaussian Splatting

Neural Radiance Field (NeRF) \[[18](https://arxiv.org/html/2503.10860v2#bib.bib18)\] is an optimization-based technique for reconstructing 3D scenes from dense input images. The key idea of NeRF is to encode a scene into an implicit neural network that takes a 3D position and view direction as inputs and outputs opacity and view-dependent color. By minimizing the loss between input images and renderings, the implicit neural network is optimized to represent real-world scenes. NeRF has gained considerable attention, inspiring a significant amount of follow-up work to improve rendering quality \[[1](https://arxiv.org/html/2503.10860v2#bib.bib1), [2](https://arxiv.org/html/2503.10860v2#bib.bib2), [4](https://arxiv.org/html/2503.10860v2#bib.bib4), [36](https://arxiv.org/html/2503.10860v2#bib.bib36)\] and efficiency \[[19](https://arxiv.org/html/2503.10860v2#bib.bib19), [5](https://arxiv.org/html/2503.10860v2#bib.bib5), [9](https://arxiv.org/html/2503.10860v2#bib.bib9), [24](https://arxiv.org/html/2503.10860v2#bib.bib24), [27](https://arxiv.org/html/2503.10860v2#bib.bib27)\].

The key limitation of NeRF, however, is its slow training and inference, as rendering requires evaluating the network multiple times along a ray. Kerbl et al. \[[14](https://arxiv.org/html/2503.10860v2#bib.bib14)\] address this problem by proposing 3D Gaussian Splatting (3DGS), which explicitly models the scene using a set of Gaussian primitives. We build our approach on 3DGS, as it produces results comparable to NeRF at a lower computational cost.

### 2.2 Sparse Novel View Synthesis

While both NeRF and 3DGS demonstrate high-quality rendering given dense input image sampling, they struggle with sparse input views. Several techniques have been proposed to address the problem of sparse-input novel view synthesis using NeRF \[[6](https://arxiv.org/html/2503.10860v2#bib.bib6), [31](https://arxiv.org/html/2503.10860v2#bib.bib31), [34](https://arxiv.org/html/2503.10860v2#bib.bib34), [43](https://arxiv.org/html/2503.10860v2#bib.bib43), [21](https://arxiv.org/html/2503.10860v2#bib.bib21), [45](https://arxiv.org/html/2503.10860v2#bib.bib45), [13](https://arxiv.org/html/2503.10860v2#bib.bib13)\] and 3DGS \[[22](https://arxiv.org/html/2503.10860v2#bib.bib22), [41](https://arxiv.org/html/2503.10860v2#bib.bib41), [49](https://arxiv.org/html/2503.10860v2#bib.bib49)\] representations.

More specifically, for NeRF-based methods, PixelNeRF \[[45](https://arxiv.org/html/2503.10860v2#bib.bib45)\] learns an image-based 3D feature extractor as a prior to optimize NeRF from sparse input views. DietNeRF \[[13](https://arxiv.org/html/2503.10860v2#bib.bib13)\] introduces an auxiliary semantic consistency loss to encourage realistic renderings from novel views. RegNeRF \[[21](https://arxiv.org/html/2503.10860v2#bib.bib21)\] proposes geometry and appearance regularization from novel viewpoints. DS-NeRF \[[6](https://arxiv.org/html/2503.10860v2#bib.bib6)\] incorporates depth supervision provided by structure-from-motion into the NeRF pipeline. SparseNeRF \[[41](https://arxiv.org/html/2503.10860v2#bib.bib41)\] introduces a local depth ranking method and spatial continuity constraints to regularize optimization. FreeNeRF \[[43](https://arxiv.org/html/2503.10860v2#bib.bib43)\] regularizes the optimization by reducing the frequency of positional encoding, while SimpleNeRF \[[31](https://arxiv.org/html/2503.10860v2#bib.bib31)\] provides additional supervision through point augmentation.

Among 3DGS-based methods, FSGS \[[49](https://arxiv.org/html/2503.10860v2#bib.bib49)\] introduces monocular depth supervision and proposes a specially designed densification strategy. SparseGS \[[41](https://arxiv.org/html/2503.10860v2#bib.bib41)\] presents a novel explicit operator for 3D representations to prune floating Gaussians. CoherentGS \[[22](https://arxiv.org/html/2503.10860v2#bib.bib22)\] enhances coherence in the Gaussian representation by constraining the movement of Gaussians and introducing single- and multi-view constraints. DNGaussian \[[16](https://arxiv.org/html/2503.10860v2#bib.bib16)\] employs both hard and soft depth regularization to improve sparse-view reconstruction by enforcing surface completeness.

While these methods significantly reduce the number of input images required for high-quality view synthesis, they still struggle to achieve robust results with extremely sparse inputs (e.g., three images). Additionally, they do not provide a reliable approach for reconstructing missing regions and typically fill these areas with overly smooth content.

### 2.3 Diffusion-based Methods

In recent years, Diffusion Models (DM) \[[12](https://arxiv.org/html/2503.10860v2#bib.bib12), [26](https://arxiv.org/html/2503.10860v2#bib.bib26)\] have stood out in image generation tasks due to stable training and high-quality results. Because of their success, these models have been used extensively as a prior for various tasks including view synthesis \[[17](https://arxiv.org/html/2503.10860v2#bib.bib17), [29](https://arxiv.org/html/2503.10860v2#bib.bib29), [11](https://arxiv.org/html/2503.10860v2#bib.bib11), [35](https://arxiv.org/html/2503.10860v2#bib.bib35), [10](https://arxiv.org/html/2503.10860v2#bib.bib10), [7](https://arxiv.org/html/2503.10860v2#bib.bib7), [42](https://arxiv.org/html/2503.10860v2#bib.bib42), [39](https://arxiv.org/html/2503.10860v2#bib.bib39), [28](https://arxiv.org/html/2503.10860v2#bib.bib28), [8](https://arxiv.org/html/2503.10860v2#bib.bib8)\].

The majority of these methods utilize diffusion models for synthesizing novel views of objects. Specifically, GaussianObject \[[42](https://arxiv.org/html/2503.10860v2#bib.bib42)\] leverages ControlNet \[[47](https://arxiv.org/html/2503.10860v2#bib.bib47)\] to repair the artifacts introduced by 3DGS optimization and synthesize 3D objects only from four input images. We follow a similar strategy to enhance the quality of rendered images, but focus on reconstructing scenes.

For general scenes, ReconFusion \[[39](https://arxiv.org/html/2503.10860v2#bib.bib39)\] and CAT3D \[[8](https://arxiv.org/html/2503.10860v2#bib.bib8)\] train a view synthesis diffusion model, which is further used to regularize NeRF optimization. The main issue with these approaches is that the images reconstructed using the diffusion model are not 3D-consistent, causing the optimized NeRF model to produce blurry results, particularly in the missing areas. In contrast, we introduce two diffusion models to enhance renderings and inpaint missing areas, and we propose a two-stage optimization process that enables the reconstruction of detailed textures in both visible and missing regions.

3 Background
------------

In this section, we review two fundamental techniques used in our algorithm: 3D Gaussian splatting and diffusion models. Specifically, we use 3DGS as our 3D representation due to its fast rendering pipeline and diffusion models are used as priors to regularize the optimization process for spare input views.

![Refer to caption](x2.png)

Figure 2: We provide an overview of the different stages of our approach. First, we initialize the Gaussians by generating high-quality per-view depth maps (Sec.[4.1](https://arxiv.org/html/2503.10860v2#S4.SS1 "4.1 3D Gaussian Initialization ‣ 4 Algorithm ‣ RI3D: Few-Shot Gaussian Splatting With Repair and Inpainting Diffusion Priors")). Next, we fine-tune the repair and inpainting diffusion models on the scene at hand (Sec.[4.2](https://arxiv.org/html/2503.10860v2#S4.SS2 "4.2 Repair and Inpainting Diffusion Models ‣ 4 Algorithm ‣ RI3D: Few-Shot Gaussian Splatting With Repair and Inpainting Diffusion Priors")). Finally, we use these models to optimize the 3DGS representation in two stages (Sec. [4.3](https://arxiv.org/html/2503.10860v2#S4.SS3 "4.3 Optimization ‣ 4 Algorithm ‣ RI3D: Few-Shot Gaussian Splatting With Repair and Inpainting Diffusion Priors")). In the first stage, we reconstruct the areas covered by the input images (blue), using the repair model to generate pseudo ground truth images at MM novel views (orange) to constrain the optimization. In the second stage, we first select a subset of novel views (green) to inpaint the missing regions (left) and continue the optimization using the repair model (right). This process of inpainting and optimization is repeated multiple times until all missing areas are reconstructed.

### 3.1 3D Gaussian Splatting

3DGS \[[14](https://arxiv.org/html/2503.10860v2#bib.bib14)\] is a point-based rendering technique representing a scene with a dense set of 3D Gaussians, which can achieve high-quality, fast and differentiable scene rendering. In 3DGS, each 3D Gaussian is defined using a set of optimizable parameters: 3D position 𝐱{\\bf x}, opacity σ\\sigma, anisotropic covariance matrix Σ\\Sigma, and color 𝐜{\\bf c} represented with spherical harmonics (SH) coefficients. Given the representation, point-based α\\alpha\-blending is used to render the color 𝐜{\\bf c} at each pixel 𝐩{\\bf p} as follows:

𝐜​(𝐩)\=∑i∈N𝐜i​αi​∏j\=1i−1(1−αj),{\\bf c}({\\bf p})=\\sum\_{i\\in N}{\\bf c}\_{i}\\alpha\_{i}\\prod\_{j=1}^{i-1}(1-\\alpha\_{j}),

(1)

where NN represents the number of Gaussians overlapping with pixel 𝐩{\\bf p}, 𝐜i{\\bf c}\_{i} is view-dependent color computed from SH coefficients of the ithi^{\\text{th}} Gaussian, and αi\\alpha\_{i} is the effective opacity, obtained by evaluating the Gaussian and multiplying it with the per-Gaussian opacity σi\\sigma\_{i}. During optimization, given a set of input images, the parameters of the Gaussian particles are optimized by minimizing the loss between the input and rendered images.

### 3.2 Diffusion Models

Diffusion models \[[26](https://arxiv.org/html/2503.10860v2#bib.bib26)\] are a class of generative models that create data matching the target distribution q​(𝐱0)q({\\bf x}\_{0}) by progressively denoising Gaussian noise ε\\varepsilon. Specifically, in the forward process, noise is added to clean data 𝐱0{\\bf x}\_{0} over TT steps, producing a sequence of increasingly noisy data 𝐱0,…,𝐱T{\\bf x}\_{0},\\dots,{\\bf x}\_{T}. The reverse process then uses the diffusion model to invert this sequence, iteratively denoising from 𝐱T{\\bf x}\_{T} back to reconstruct 𝐱0{\\bf x}\_{0}. Generating high-resolution images with diffusion models is computationally intensive and memory-demanding. To address this issue, latent diffusion models (LDMs) \[[26](https://arxiv.org/html/2503.10860v2#bib.bib26)\] perform the diffusion process in the latent space of a variational autoencoder (VAE) \[[15](https://arxiv.org/html/2503.10860v2#bib.bib15)\], reducing memory and computational requirements. Our repair and inpainting models are based on LDMs.

4 Algorithm
-----------

Given a sparse set of NN images 𝐈1ref,⋯,𝐈Nref{\\bf I}^{\\text{ref}}\_{1},\\cdots,{\\bf I}^{\\text{ref}}\_{N} with their corresponding camera poses, our goal is to reconstruct the scene using a 3D Gaussian representation. To do this, we start by initializing the Gaussians using a detailed and 3D-consistent depth estimate at each input image. We then perform a two-stage optimization by utilizing two diffusion models (“repair” and “inpainting”), personalized for the scene at hand, to enhance the rendered images and inpaint the missing areas. An overview of our approach is provided in Fig. [2](https://arxiv.org/html/2503.10860v2#S3.F2 "Figure 2 ‣ 3 Background ‣ RI3D: Few-Shot Gaussian Splatting With Repair and Inpainting Diffusion Priors"). In the following sections, we discuss our initialization process, the two diffusion models, and the two-stage optimization process. Additional implementation details, such as hyperparameters and learning schedules, can be found in the supplementary materials.

### 4.1 3D Gaussian Initialization

Initialization is one of the key factors in effectiveness of 3DGS optimization, especially in sparse input settings. Ideally, we would like to start with dense and 3D-consistent Gaussians in areas covered by the input images. A potential solution is to use the point cloud estimated by the multi-view stereo (MVS) technique by Wang et al. \[[38](https://arxiv.org/html/2503.10860v2#bib.bib38)\] (DUSt3R), which has been shown to perform significantly better than other MVS techniques, particularly with a small number of images. However, DUSt3R’s point cloud is usually only accurate in the high-confidence regions where there is overlapping content from multiple input images. For example, for a 360∘ scene with extremely sparse inputs (e.g., 3 images), the high confidence point cloud generally covers only a small foreground region and is usually sparse. The point cloud in the background areas, which are covered by only a single image, is often highly inaccurate.

To obtain a dense initialization, we instead propose generating per-image depth maps. We then assign a Gaussian to each pixel of every input image and project them into 3D space according to their depth. The key challenge here is obtaining 3D-consistent and detailed depth maps, even in regions covered by only a single image. Our main observation is that the depth estimated by DUSt3R (one of the byproducts of this approach) and monocular depth estimation methods, such as Depth Anything V2 \[[44](https://arxiv.org/html/2503.10860v2#bib.bib44)\], have complementary properties (see Fig. [3](https://arxiv.org/html/2503.10860v2#S4.F3 "Figure 3 ‣ 4.1 3D Gaussian Initialization ‣ 4 Algorithm ‣ RI3D: Few-Shot Gaussian Splatting With Repair and Inpainting Diffusion Priors")). Specifically, DUSt3R depth in high-confidence regions is 3D-consistent and accurate but smooth, as the estimation is performed on low-resolution images. Additionally, its depth is inaccurate in low-confidence regions. On the other hand, monocular depth is detailed but relative and not 3D-consistent, as the depth maps of each image are obtained independently.

We therefore propose combining the two depth maps to leverage the strengths of both approaches. Specifically, we aim for the combined depth to have two properties: 1) it should preserve the details (edges) from the monocular depth, and 2) it should adhere to the absolute depth values from DUSt3R in high-confidence regions. Combining these properties results in the following objective with two terms:

![Refer to caption](x3.png)

Figure 3: The depth estimated by DUSt3R is geometrically consistent in the high confidence regions (marked in yellow), but of poor quality in the remaining areas. Monocular depth is highly detailed, but is not 3D consistent. Our proposed method combines the two depth maps into a detailed and geometrically consistent depth. Applying bilateral filtering, further sharpens the boundaries.

𝐝∗\=arg⁡min𝐝⁡\[𝐌⊙‖𝐝−𝐝D‖2+λ​‖∇𝐝−∇𝐝M‖2\],{\\bf d}^{\*}=\\arg\\min\_{{\\bf d}}\\left\[{\\bf M}\\odot\\|{\\bf d}-{\\bf d}^{D}\\|\_{2}+\\lambda\\|\\nabla{\\bf d}-\\nabla{\\bf d}^{\\text{M}}\\|\_{2}\\right\],

(2)

where 𝐝D{\\bf d}^{D} and 𝐝M{\\bf d}^{\\text{M}} are the DUSt3R and monocular depth estimates, respectively, and MM is a mask indicating the regions where DUSt3R has high confidence. The first term enforces similarity between the combined and DUSt3R depth maps in high-confidence regions, while the second term ensures that the gradient of the combined depth matches that of the monocular depth everywhere. The weight λ\\lambda controls the contribution of the gradient term, which we set to 10 in our implementation.

Notably, this objective is similar to the Poisson blending equation \[[23](https://arxiv.org/html/2503.10860v2#bib.bib23)\], but with a key difference: we enforce the gradient loss everywhere, not just in the low-confidence areas (1−M1-M), since DUSt3R depth is smooth, and we aim to incorporate details from monocular depth even in high-confidence regions. As the objective is quadratic, we obtain the solution by solving a linear system of equations (derived by setting the gradient of the objective to zero), which can be efficiently computed using sparse matrix solvers.

Directly using the monocular depth as 𝐝M{\\bf d}^{M} in our objective is problematic, as DUSt3R and monocular depth can have widely different ranges. To address this, we globally align the monocular depth map to DUSt3R depth using a piecewise linear function. Since DUSt3R depth is inaccurate in low-confidence regions, we perform the alignment only in high-confidence regions and linearly extrapolate the function for other values. Once we obtain this piecewise linear function, we apply it to the monocular depth to align it with DUSt3R depth. We then use this aligned monocular depth as 𝐝M{\\bf d}^{M} in Eq. [2](https://arxiv.org/html/2503.10860v2#S4.E2 "Equation 2 ‣ 4.1 3D Gaussian Initialization ‣ 4 Algorithm ‣ RI3D: Few-Shot Gaussian Splatting With Repair and Inpainting Diffusion Priors") to compute the combined depth. Finally, we apply bilateral filtering, as in previous approaches \[[30](https://arxiv.org/html/2503.10860v2#bib.bib30), [37](https://arxiv.org/html/2503.10860v2#bib.bib37)\], to the combined depth to produce per-image depth maps with sharp edges (see Fig. [3](https://arxiv.org/html/2503.10860v2#S4.F3 "Figure 3 ‣ 4.1 3D Gaussian Initialization ‣ 4 Algorithm ‣ RI3D: Few-Shot Gaussian Splatting With Repair and Inpainting Diffusion Priors")).

Given the per-image depth maps, we assign a Gaussian to each pixel and project it into 3D space along the ray connecting the camera and pixel center \[[22](https://arxiv.org/html/2503.10860v2#bib.bib22)\]. We initialize each Gaussian’s color using the corresponding pixel color and set the rotation matrix to identity. Additionally, we use an isometric scale, setting it so the projected Gaussians cover 1.4 times the pixel size, and assign an initial opacity of 0.1 to all Gaussians. Figure [4](https://arxiv.org/html/2503.10860v2#S4.F4 "Figure 4 ‣ 4.2 Repair and Inpainting Diffusion Models ‣ 4 Algorithm ‣ RI3D: Few-Shot Gaussian Splatting With Repair and Inpainting Diffusion Priors") shows our output after initialization.

### 4.2 Repair and Inpainting Diffusion Models

In extremely sparse input settings, 3DGS optimization, even with highly accurate initialization, is brittle and unable to properly reconstruct the scene due to two issues (see Fig. [5](https://arxiv.org/html/2503.10860v2#S4.F5 "Figure 5 ‣ 4.2 Repair and Inpainting Diffusion Models ‣ 4 Algorithm ‣ RI3D: Few-Shot Gaussian Splatting With Repair and Inpainting Diffusion Priors")). First, the optimization can quickly overfit to the input views, resulting in a representation that produces poor-quality results when rendered from novel camera angles. Second, due to the lack of supervision in occluded regions, these areas are often not reconstructed properly, resulting in blurry or dark appearances in the rendered images.

To address these issues, we propose using two diffusion models. Specifically, the first diffusion model, referred to as the “repair” model, takes as input a rendered image from a view not covered by the input cameras and produces a clean version of the image. The key idea is that the repair model can generate clean images for a set of novel views, which can then be used as pseudo ground truth images during optimization to address the first problem. The second diffusion model, referred to as the “inpainting” model, tackles the second issue by filling in the missing areas.

![Refer to caption](x4.png)

Figure 4: We show the output at different stages of our approach. Our initialization strategy ensures that Gaussians from different input images are roughly aligned and cover the visible areas of the scene. During the first stage of optimization, we use the repair model to constrain the problem, which in turn helps reconstruct the visible regions with detailed texture. The missing areas are then hallucinated and seamlessly incorporated into the scene during the second and final stage of optimization.

![Refer to caption](x5.png)

Figure 5: We show the result of 3DGS optimization using our initialization. In the absence of any constraints, 3DGS optimization quickly overfits to the input images (compare rendered and input images) but produces distracting artifacts in the novel view image. Additionally, unobserved areas will not be reconstructed during optimization, resulting in a dark and blurry appearance. We address these issues using our repair and inpainting models to constrain the optimization and hallucinate missing areas.

Following Yang et al.’s approach \[[42](https://arxiv.org/html/2503.10860v2#bib.bib42)\], we use a pre-trained ControlNet \[[47](https://arxiv.org/html/2503.10860v2#bib.bib47)\] as our base repair model and fine-tune it on the target scene by generating pairs of corrupted and clean images through a leave-one-out strategy. Specifically, we create NN subsets of images, each containing N−1N-1 images by excluding one input image. We then optimize NN separate 3DGS representations on these subsets. After a set number of iterations, we reintroduce the excluded image to its subset and continue optimization for an additional fixed number of iterations. During this process, we render the left-out view at different stages of optimization, pairing these intermediate renderings (corrupted images) with the original left-out image (clean image) to form the training data. Using the corrupted image as the condition for ControlNet and the clean image as ground truth, we train the diffusion model using the standard loss \[[42](https://arxiv.org/html/2503.10860v2#bib.bib42)\]. Note that by initially excluding an image and later reintroducing it, we generate progressively refined corrupted images, improving the repair model’s ability to handle the final 3DGS optimization (discussed in Sec. [4.3](https://arxiv.org/html/2503.10860v2#S4.SS3 "4.3 Optimization ‣ 4 Algorithm ‣ RI3D: Few-Shot Gaussian Splatting With Repair and Inpainting Diffusion Priors")).

For inpainting, we use the Stable Diffusion inpainting model \[[26](https://arxiv.org/html/2503.10860v2#bib.bib26)\]. Although this model is powerful, the missing regions in our problem are often large, providing limited context for hallucinating consistent details. To address this, we follow Tang et al. \[[32](https://arxiv.org/html/2503.10860v2#bib.bib32)\]’s approach and fine-tune the diffusion model on the input images by simulating a large set of input-output pairs through random masking.

Since both diffusion models operate on 512×512512\\times 512 images, we resize the input images during fine-tuning so that their smallest dimension (typically height) is 512 pixels, followed by random 512×512512\\times 512 cropping to create training data. Resizing is applied first to preserve scene context, which is essential for both the repair and inpainting models, as directly cropping high-resolution images could remove important contextual information.

![Refer to caption](x6.png)

Figure 6: We compare our approach against the other state-of-the-art methods. Existing methods are not able to properly handle both the visible (green arrows) and missing (red arrows) regions. Through the use of repair and inpainting models, our approach is able to produce high-quality textures in all areas.

![Refer to caption](x7.png)

Figure 7: We show the rendered novel view image of an inset of the Garden scene and the enhanced version using our repaired model. The repaired image is used as the pseudo ground truth during optimization to help with reconstructing detailed textures.

### 4.3 Optimization

We now have all the elements to begin discussing our proposed optimization strategy, consisting of two stages. The goal of the first stage of the optimization is to obtain a reasonable 3D representation covering only the regions that are visible in the input images. This is a critical step for properly identifying the missing areas. In the second stage, the missing regions are inpainted and the representation is further optimized to obtain a complete scene (see Fig. [4](https://arxiv.org/html/2503.10860v2#S4.F4 "Figure 4 ‣ 4.2 Repair and Inpainting Diffusion Models ‣ 4 Algorithm ‣ RI3D: Few-Shot Gaussian Splatting With Repair and Inpainting Diffusion Priors")). Below, we discuss the two stages in detail.

#### Stage 1:

Optimizing solely by minimizing the loss between the rendered and input images can lead to overfitting. To address this, we introduce a set of MM novel cameras along an elliptical path aligned with the input cameras. To leverage these novel cameras during optimization, we generate pseudo ground truth images, 𝐆jnov\=R​(𝐈^jnov){\\bf G}^{\\text{nov}}\_{j}=R(\\hat{{\\bf I}}^{\\text{nov}}\_{j}), by applying our repair diffusion model RR to the rendered images 𝐈^jnov\\hat{{\\bf I}}^{\\text{nov}}\_{j}. In summary, we optimize the following objective:

ℒstage1\\displaystyle\\mathcal{L}\_{\\text{stage1}}

\=∑i\=1Nℒrec​(𝐈^iref,𝐈iref)+∑j\=1Mλj​𝐌jα​ℒrec​(𝐈^jnov,𝐆jnov)\\displaystyle=\\sum\_{i=1}^{N}\\mathcal{L}\_{\\text{rec}}(\\hat{{\\bf I}}^{\\text{ref}}\_{i},{\\bf I}^{\\text{ref}}\_{i})+\\sum\_{j=1}^{M}\\lambda\_{j}{\\bf M}^{\\alpha}\_{j}\\mathcal{L}\_{\\text{rec}}(\\hat{{\\bf I}}^{\\text{nov}}\_{j},{\\bf G}^{\\text{nov}}\_{j})

+∑j\=1M‖𝐀j⊙(1−𝐌jα)⊙𝐌jb‖1.\\displaystyle+\\sum\_{j=1}^{M}\\|{\\bf A}\_{j}\\odot(1-{\\bf M}^{\\alpha}\_{j})\\odot{\\bf M}^{b}\_{j}\\|\_{1}.

(3)

The first term enforces similarity between the rendered 𝐈^iref\\hat{{\\bf I}}^{\\text{ref}}\_{i} and the input images. The reconstruction loss comprises the original 3DGS losses (L1L\_{1} and SSIM), as well as LPIPS and depth correlation \[[49](https://arxiv.org/html/2503.10860v2#bib.bib49)\] losses. For the depth correlation loss, we compute the Pearson correlation between the rendered depth from 3DGS and the monocular depth estimated from the input image. The second term encourages similarity between the rendered images and their repaired counterparts from the MM novel cameras. The factor λj\\lambda\_{j} represents the camera distance weight \[[42](https://arxiv.org/html/2503.10860v2#bib.bib42)\], giving higher weight to novel cameras closer to the input cameras. Additionally, 𝐌jα{\\bf M}^{\\alpha}\_{j} is a binary mask obtained by thresholding the rendered opacity, ensuring that the loss is enforced only in regions covered by the input images. Finally, the third term encourages the rendered opacity 𝐀j{\\bf A}\_{j} to be low in the missing regions (1−𝐌jα1-{\\bf M}^{\\alpha}\_{j}), as we do not want 3DGS to place any visible Gaussians in these areas. Note that 𝐌jb{\\bf M}^{b}\_{j} is a mask identifying the background regions, which we use here because the occluded regions in the foreground are typically small and can be easily inpainted with our repair model. We obtain this background mask by applying agglomerative clustering \[[37](https://arxiv.org/html/2503.10860v2#bib.bib37)\] on the monocular depth estimated for repaired images.

To summarize, this objective provides additional supervision using the repair model for MM novel cameras to constrain the optimization. Moreover, we focus solely on reconstructing the visible regions by ensuring that the opacity of Gaussians in the missing areas is minimized through the third term. See our output at this stage in Fig. [4](https://arxiv.org/html/2503.10860v2#S4.F4 "Figure 4 ‣ 4.2 Repair and Inpainting Diffusion Models ‣ 4 Algorithm ‣ RI3D: Few-Shot Gaussian Splatting With Repair and Inpainting Diffusion Priors").

#### Stage 2:

At the end of stage 1, the 3DGS representation can reconstruct the visible areas from the input images but lacks information in the missing regions, (1−𝐌jα)⊙𝐌jb(1-{\\bf M}^{\\alpha}\_{j})\\odot{\\bf M}^{b}\_{j}. To address this, we first use our personalized inpainting model to fill in the missing areas in novel camera views and then integrate the hallucinated details into the scene through optimization. Specifically, we select a subset of K<MK<M novel view images with non-overlapping content and inpaint the missing areas using our inpainting diffusion model. We inpaint only a subset of images to prevent independently inpainting overlapping content, which could lead to inconsistent results. We then project the inpainted areas into the scene using monocular depth estimated from the inpainted images. Since monocular depth is relative and has a different range than the reconstructed 3D scene, it cannot be used directly. We address this by combining the monocular depth in the inpainted regions with the rendered depth using Eq. [2](https://arxiv.org/html/2503.10860v2#S4.E2 "Equation 2 ‣ 4.1 3D Gaussian Initialization ‣ 4 Algorithm ‣ RI3D: Few-Shot Gaussian Splatting With Repair and Inpainting Diffusion Priors").

This process results in KK inpainted images, 𝐋^jnov\\hat{{\\bf L}}^{\\text{nov}}\_{j}. We then apply our repair model to these images, as well as the remaining M−KM-K novel view images, to obtain the pseudo ground truth images, 𝐆jnov{\\bf G}^{\\text{nov}}\_{j}. These images are then used to optimize the 3DGS representation by minimizing the following loss:

ℒstage2\\displaystyle\\mathcal{L}\_{\\text{stage2}}

\=∑i\=1Nℒrec​(𝐈^iref,𝐈iref)+∑j\=1Mλj​ℒrec​(𝐈^jnov,𝐆jnov)\\displaystyle=\\sum\_{i=1}^{N}\\mathcal{L}\_{\\text{rec}}(\\hat{{\\bf I}}^{\\text{ref}}\_{i},{\\bf I}^{\\text{ref}}\_{i})+\\sum\_{j=1}^{M}\\lambda\_{j}\\mathcal{L}\_{\\text{rec}}(\\hat{{\\bf I}}^{\\text{nov}}\_{j},{\\bf G}^{\\text{nov}}\_{j})

+∑k\=1K(1−𝐌kα)⊙𝐌kb⊙Lp​(𝐈^knov,𝐋^knov).\\displaystyle+\\sum\_{k=1}^{K}(1-{\\bf M}^{\\alpha}\_{k})\\odot{\\bf M}^{b}\_{k}\\odot L\_{p}(\\hat{{\\bf I}}^{\\text{nov}}\_{k},\\hat{{\\bf L}}^{\\text{nov}}\_{k}).

(4)

The first and second terms are similar to those in Eq. [4.3](https://arxiv.org/html/2503.10860v2#S4.Ex1 "Stage 1: ‣ 4.3 Optimization ‣ 4 Algorithm ‣ RI3D: Few-Shot Gaussian Splatting With Repair and Inpainting Diffusion Priors"). The main difference is that, since most of the scene has now been inpainted, we enforce the loss in the second term across the entire image by removing the visibility mask 𝐌jα{\\bf M}^{\\alpha}\_{j}. The third term ensures that the rendered images 𝐈^jnov\\hat{{\\bf I}}^{\\text{nov}}\_{j} remain close to the inpainted images 𝐋^jnov\\hat{{\\bf L}}^{\\text{nov}}\_{j} in the inpainted areas, (1−𝐌jα)⊙𝐌jb(1-{\\bf M}^{\\alpha}\_{j})\\odot{\\bf M}^{b}\_{j}. We perform this optimization for a fixed number of iterations, then repeat the inpainting step with a different subset of KK images before optimizing again. This iterative process continues until the remaining missing areas in the 3D scene are progressively filled.

At the end of this stage, we obtain a complete 3D representation of the scene, which can be rendered from novel camera views, as shown in Fig. [4](https://arxiv.org/html/2503.10860v2#S4.F4 "Figure 4 ‣ 4.2 Repair and Inpainting Diffusion Models ‣ 4 Algorithm ‣ RI3D: Few-Shot Gaussian Splatting With Repair and Inpainting Diffusion Priors").

5 Results
---------

We compare our approach against several state-of-the-art methods both visually and numerically and evaluate the effect of various aspects of our proposed method. Here, we show the results on the Mip-NeRF 360 \[[3](https://arxiv.org/html/2503.10860v2#bib.bib3)\] dataset but provide additional results on CO3D \[[25](https://arxiv.org/html/2503.10860v2#bib.bib25)\] and video comparisons in the supplementary materials.

### 5.1 Visual Results

In Fig. [6](https://arxiv.org/html/2503.10860v2#S4.F6 "Figure 6 ‣ 4.2 Repair and Inpainting Diffusion Models ‣ 4 Algorithm ‣ RI3D: Few-Shot Gaussian Splatting With Repair and Inpainting Diffusion Priors"), we compare our approach against several 3DGS-based approaches: DNGaussian\[[16](https://arxiv.org/html/2503.10860v2#bib.bib16)\], CoR-GS \[[46](https://arxiv.org/html/2503.10860v2#bib.bib46)\], and FSGS \[[49](https://arxiv.org/html/2503.10860v2#bib.bib49)\]. We use the authors’ publicly available source code for these methods. As the code for Reconfusion \[[39](https://arxiv.org/html/2503.10860v2#bib.bib39)\] and CAT3D \[[8](https://arxiv.org/html/2503.10860v2#bib.bib8)\] is unavailable, we cannot include their results in this figure. However, we show a comparison against their publicly available video results in our supplementary video. As shown in Fig. [6](https://arxiv.org/html/2503.10860v2#S4.F6 "Figure 6 ‣ 4.2 Repair and Inpainting Diffusion Models ‣ 4 Algorithm ‣ RI3D: Few-Shot Gaussian Splatting With Repair and Inpainting Diffusion Priors"), existing approaches fill the missing areas (red arrows) using either a few large Gaussians or many small Gaussians, leading to a blurry or noisy appearance (the latter in the second and third rows of DNGaussian). In contrast, by utilizing our powerful personalized inpainting model, we are able to reconstruct these areas with detailed textures.

Moreover, existing approaches struggle with properly reconstructing the visible areas (indicated by the green arrows) in these extremely sparse input settings. On the other hand, by providing additional novel view supervision through the repair model, our method is able to reconstruct the visible areas reasonably well. For example, as shown in Fig. [7](https://arxiv.org/html/2503.10860v2#S4.F7 "Figure 7 ‣ 4.2 Repair and Inpainting Diffusion Models ‣ 4 Algorithm ‣ RI3D: Few-Shot Gaussian Splatting With Repair and Inpainting Diffusion Priors"), our repair model is able to produce high-quality texture on the table from the ghosted rendering. Using these repaired images as pseudo ground truth during optimization enhances texture detail in visible areas.

Table 1: We compare our approach against other sparse view synthesis methods. PSNR and SSIM measure pixel-wise error and as such ReconFusion and CAT3D score higher in most cases, since these metrics favor blurrier results. Our LPIPS scores, however, demonstrate that our approach is better at preserving texture details. The numbers for approaches marked by ∗ are directly grabbed from ReconFusion. CAT3D results are obtained from the original paper. The methods marked by † are initialized using DUSt3R to improve their results.

![Refer to caption](x8.png)

Figure 8: We show the effect of various components of our system. The differences can be best seen in the supplementary video.

### 5.2 Numerical Results

In Table [1](https://arxiv.org/html/2503.10860v2#S5.T1 "Table 1 ‣ 5.1 Visual Results ‣ 5 Results ‣ RI3D: Few-Shot Gaussian Splatting With Repair and Inpainting Diffusion Priors"), we numerically evaluate our approach by reporting the average error in PSNR, SSIM, and LPIPS \[[48](https://arxiv.org/html/2503.10860v2#bib.bib48)\] across all scenes in the Mip-NeRF 360 \[[3](https://arxiv.org/html/2503.10860v2#bib.bib3)\] dataset. Our method outperforms all other approaches in PSNR and SSIM, except for ReconFusion and CAT3D, where our results are slightly worse. However, this is primarily because ReconFusion and CAT3D generate smooth, blurry textures in the missing regions (see supplementary video), which are favored by PSNR and SSIM. In contrast, our method achieves the best performance in LPIPS, highlighting its ability to synthesize detailed textures.

### 5.3 Ablations

We evaluate the effect of the core components of our method both visually and numerically in Fig. [8](https://arxiv.org/html/2503.10860v2#S5.F8 "Figure 8 ‣ 5.1 Visual Results ‣ 5 Results ‣ RI3D: Few-Shot Gaussian Splatting With Repair and Inpainting Diffusion Priors") and Table [2](https://arxiv.org/html/2503.10860v2#S5.T2 "Table 2 ‣ 5.3 Ablations ‣ 5 Results ‣ RI3D: Few-Shot Gaussian Splatting With Repair and Inpainting Diffusion Priors"), respectively. Directly using the DUSt3R depth (w/o depth enhancement) results in floating artifacts, as indicated by the red arrow, since DUSt3R produces inaccurate depth in low-confidence areas. By combining DUSt3R with monocular depth, our approach largely mitigates this issue.

To evaluate the impact of two-stage optimization, we compare our approach against an alternative method where optimization is performed in a single stage using only the repair model. Note that in this case, the repair model is responsible for both enhancing the rendered images and hallucinating details in the missing areas. As seen, the single-stage approach produces floating artifacts due to insufficient constraints to place the inpainted content in the correct 3D location. Additionally, since the repair model is not specifically trained for inpainting, it sometimes hallucinates low-quality textures, as indicated by the blue arrow.

Finally, to assess the impact of personalizing the inpainting diffusion model, we compare our approach against a variant that directly uses our base inpainting model without fine-tuning (w/o ft inpainting). Although the baseline inpainting model generates visually pleasing textures, they sometimes do not perfectly match the surrounding textures. For example, the synthesized textures include a grayish plant and leaf-like texture on the ground (red arrows).

Table 2: We evaluate the effect of different components of our method on the 3-input Mip-NeRF 360 \[[3](https://arxiv.org/html/2503.10860v2#bib.bib3)\] dataset.

6 Limitations
-------------

Our dense initialization, and consequently, the final reconstruction quality, depend on the DUSt3R depth. When DUSt3R produces a highly inaccurate depth map, our optimization cannot fully correct the misaligned initialization, resulting in ghosting artifacts (see supplementary). However, the modular nature of our approach allows us to replace DUSt3R with potentially more accurate models in the future as better methods become available, mitigating this issue. Moreover, unlike approaches like CAT3D \[[8](https://arxiv.org/html/2503.10860v2#bib.bib8)\], our method is not able to handle a single input image, mainly because of the use of leave-one-out strategy for training our repair model. In the future, it would be interesting to avoid this issue by training the repair model on a large scale training data in an offline manner.

7 Conclusion
------------

In this paper, we have presented a novel technique based on 3DGS that utilizes diffusion models to reconstruct novel views from a sparse set of input images. Specifically, we initialize 3D Gaussians by proposing a strategy to obtain per-view depth maps by combining multiview stereo depth with monocular depth. We then perform 3DGS optimization using two personalized diffusion models through two stages. We demonstrate that our approach produces better results than the state of the art on challenging scenes.

Acknowledgements
----------------

The project was funded in part by a generous gift from Meta. Portions of this research were conducted with the advanced computing resources provided by Texas A&M High Performance Research Computing.

References
----------

*   Barron et al. \[2021\] Jonathan T Barron, Ben Mildenhall, Matthew Tancik, Peter Hedman, Ricardo Martin-Brualla, and Pratul P Srinivasan. Mip-nerf: A multiscale representation for anti-aliasing neural radiance fields. In _Proceedings of the IEEE/CVF International Conference on Computer Vision_, pages 5855–5864, 2021.
*   Barron et al. \[2022a\] Jonathan T Barron, Ben Mildenhall, Dor Verbin, Pratul P Srinivasan, and Peter Hedman. Mip-nerf 360: Unbounded anti-aliased neural radiance fields. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_, pages 5470–5479, 2022a.
*   Barron et al. \[2022b\] Jonathan T. Barron, Ben Mildenhall, Dor Verbin, Pratul P. Srinivasan, and Peter Hedman. Mip-nerf 360: Unbounded anti-aliased neural radiance fields. _CVPR_, 2022b.
*   Barron et al. \[2023\] Jonathan T Barron, Ben Mildenhall, Dor Verbin, Pratul P Srinivasan, and Peter Hedman. Zip-nerf: Anti-aliased grid-based neural radiance fields. In _Proceedings of the IEEE/CVF International Conference on Computer Vision_, pages 19697–19705, 2023.
*   Chen et al. \[2022\] Anpei Chen, Zexiang Xu, Andreas Geiger, Jingyi Yu, and Hao Su. Tensorf: Tensorial radiance fields. In _European Conference on Computer Vision (ECCV)_, 2022.
*   Deng et al. \[2022\] Kangle Deng, Andrew Liu, Jun-Yan Zhu, and Deva Ramanan. Depth-supervised nerf: Fewer views and faster training for free. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_, pages 12882–12891, 2022.
*   Fei et al. \[2023\] Ben Fei, Zhaoyang Lyu, Liang Pan, Junzhe Zhang, Weidong Yang, Tianyue Luo, Bo Zhang, and Bo Dai. Generative diffusion prior for unified image restoration and enhancement. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)_, pages 9935–9946, 2023.
*   Gao et al. \[2024\] Ruiqi Gao, Aleksander Holynski, Philipp Henzler, Arthur Brussee, Ricardo Martin-Brualla, Pratul Srinivasan, Jonathan T Barron, and Ben Poole. Cat3d: Create anything in 3d with multi-view diffusion models. _arXiv preprint arXiv:2405.10314_, 2024.
*   Garbin et al. \[2021\] Stephan J. Garbin, Marek Kowalski, Matthew Johnson, Jamie Shotton, and Julien Valentin. Fastnerf: High-fidelity neural rendering at 200fps. In _Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)_, pages 14346–14355, 2021.
*   Graikos et al. \[2022\] Alexandros Graikos, Nikolay Malkin, Nebojsa Jojic, and Dimitris Samaras. Diffusion models as plug-and-play priors. _Advances in Neural Information Processing Systems_, 35:14715–14728, 2022.
*   Güngör et al. \[2023\] Alper Güngör, Salman UH Dar, Şaban Öztürk, Yilmaz Korkmaz, Hasan A Bedel, Gokberk Elmas, Muzaffer Ozbey, and Tolga Çukur. Adaptive diffusion priors for accelerated mri reconstruction. _Medical image analysis_, 88:102872, 2023.
*   Ho et al. \[2020\] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. _Advances in neural information processing systems_, 33:6840–6851, 2020.
*   Jain et al. \[2021\] Ajay Jain, Matthew Tancik, and Pieter Abbeel. Putting nerf on a diet: Semantically consistent few-shot view synthesis. In _Proceedings of the IEEE/CVF International Conference on Computer Vision_, pages 5885–5894, 2021.
*   Kerbl et al. \[2023\] Bernhard Kerbl, Georgios Kopanas, Thomas Leimkühler, and George Drettakis. 3d gaussian splatting for real-time radiance field rendering. _ACM Transactions on Graphics_, 42(4):1–14, 2023.
*   Kingma and Welling \[2013\] Diederik P Kingma and Max Welling. Auto-encoding variational bayes. _arXiv preprint arXiv:1312.6114_, 2013.
*   Li et al. \[2024\] Jiahe Li, Jiawei Zhang, Xiao Bai, Jin Zheng, Xin Ning, Jun Zhou, and Lin Gu. Dngaussian: Optimizing sparse-view 3d gaussian radiance fields with global-local depth normalization. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_, pages 20775–20785, 2024.
*   Lin et al. \[2024\] Xinqi Lin, Jingwen He, Ziyan Chen, Zhaoyang Lyu, Bo Dai, Fanghua Yu, Yu Qiao, Wanli Ouyang, and Chao Dong. Diffbir: Toward blind image restoration with generative diffusion prior. In _European Conference on Computer Vision_, pages 430–448. Springer, 2024.
*   Mildenhall et al. \[2021\] Ben Mildenhall, Pratul P Srinivasan, Matthew Tancik, Jonathan T Barron, Ravi Ramamoorthi, and Ren Ng. Nerf: Representing scenes as neural radiance fields for view synthesis. _Communications of the ACM_, 65(1):99–106, 2021.
*   Müller et al. \[2022\] Thomas Müller, Alex Evans, Christoph Schied, and Alexander Keller. Instant neural graphics primitives with a multiresolution hash encoding. _ACM transactions on graphics (TOG)_, 41(4):1–15, 2022.
*   Nguyen \[2023\] Thuan H. Nguyen. Unofficial implementation of realfill. [https://github.com/thuanz123/realfill](https://github.com/thuanz123/realfill), 2023.
*   Niemeyer et al. \[2022\] Michael Niemeyer, Jonathan T Barron, Ben Mildenhall, Mehdi SM Sajjadi, Andreas Geiger, and Noha Radwan. Regnerf: Regularizing neural radiance fields for view synthesis from sparse inputs. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_, pages 5480–5490, 2022.
*   Paliwal et al. \[2024\] Avinash Paliwal, Wei Ye, Jinhui Xiong, Dmytro Kotovenko, Rakesh Ranjan, Vikas Chandra, and Nima Khademi Kalantari. Coherentgs: Sparse novel view synthesis with coherent 3d gaussians. _arXiv preprint arXiv:2403.19495_, 2024.
*   Pérez et al. \[2003\] Patrick Pérez, Michel Gangnet, and Andrew Blake. Poisson image editing. In _ACM SIGGRAPH 2003 Papers_, page 313–318, New York, NY, USA, 2003. Association for Computing Machinery.
*   Reiser et al. \[2021\] Christian Reiser, Songyou Peng, Yiyi Liao, and Andreas Geiger. Kilonerf: Speeding up neural radiance fields with thousands of tiny mlps. In _Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)_, pages 14335–14345, 2021.
*   Reizenstein et al. \[2021\] Jeremy Reizenstein, Roman Shapovalov, Philipp Henzler, Luca Sbordone, Patrick Labatut, and David Novotny. Common objects in 3d: Large-scale learning and evaluation of real-life 3d category reconstruction. In _Proceedings of the IEEE/CVF international conference on computer vision_, pages 10901–10911, 2021.
*   Rombach et al. \[2022\] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-resolution image synthesis with latent diffusion models. In _Proceedings of the IEEE/CVF conference on computer vision and pattern recognition_, pages 10684–10695, 2022.
*   Sara Fridovich-Keil and Alex Yu et al. \[2022\] Sara Fridovich-Keil and Alex Yu, Matthew Tancik, Qinhong Chen, Benjamin Recht, and Angjoo Kanazawa. Plenoxels: Radiance fields without neural networks. In _CVPR_, 2022.
*   Sargent et al. \[2023\] Kyle Sargent, Zizhang Li, Tanmay Shah, Charles Herrmann, Hong-Xing Yu, Yunzhi Zhang, Eric Ryan Chan, Dmitry Lagun, Li Fei-Fei, Deqing Sun, et al. Zeronvs: Zero-shot 360-degree view synthesis from a single real image. _arXiv preprint arXiv:2310.17994_, 2023.
*   Shafir et al. \[2023\] Yonatan Shafir, Guy Tevet, Roy Kapon, and Amit H Bermano. Human motion diffusion as a generative prior. _arXiv preprint arXiv:2303.01418_, 2023.
*   Shih et al. \[2020\] Meng-Li Shih, Shih-Yang Su, Johannes Kopf, and Jia-Bin Huang. 3d photography using context-aware layered depth inpainting. In _IEEE Conference on Computer Vision and Pattern Recognition (CVPR)_, 2020.
*   Somraj et al. \[2023\] Nagabhushan Somraj, Adithyan Karanayil, and Rajiv Soundararajan. Simplenerf: Regularizing sparse input neural radiance fields with simpler solutions. In _SIGGRAPH Asia 2023 Conference Papers_, pages 1–11, 2023.
*   Tang et al. \[2024\] Luming Tang, Nataniel Ruiz, Qinghao Chu, Yuanzhen Li, Aleksander Holynski, David E. Jacobs, Bharath Hariharan, Yael Pritch, Neal Wadhwa, Kfir Aberman, and Michael Rubinstein. Realfill: Reference-driven generation for authentic image completion. _ACM Trans. Graph._, 43(4), 2024.
*   Wang et al. \[2023a\] Guangcong Wang, Zhaoxi Chen, Chen Change Loy, and Ziwei Liu. Sparsenerf: Distilling depth ranking for few-shot novel view synthesis. In _IEEE/CVF International Conference on Computer Vision (ICCV)_, 2023a.
*   Wang et al. \[2023b\] Guangcong Wang, Zhaoxi Chen, Chen Change Loy, and Ziwei Liu. Sparsenerf: Distilling depth ranking for few-shot novel view synthesis. In _Proceedings of the IEEE/CVF International Conference on Computer Vision_, pages 9065–9076, 2023b.
*   Wang et al. \[2024a\] Jianyi Wang, Zongsheng Yue, Shangchen Zhou, Kelvin CK Chan, and Chen Change Loy. Exploiting diffusion prior for real-world image super-resolution. _International Journal of Computer Vision_, 132(12):5929–5949, 2024a.
*   Wang et al. \[2023c\] Peng Wang, Yuan Liu, Zhaoxi Chen, Lingjie Liu, Ziwei Liu, Taku Komura, Christian Theobalt, and Wenping Wang. F2-nerf: Fast neural radiance field training with free camera trajectories. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)_, pages 4150–4159, 2023c.
*   Wang et al. \[2022\] Qianqian Wang, Zhengqi Li, David Salesin, Noah Snavely, Brian Curless, and Janne Kontkanen. 3d moments from near-duplicate photos. In _CVPR_, 2022.
*   Wang et al. \[2024b\] Shuzhe Wang, Vincent Leroy, Yohann Cabon, Boris Chidlovskii, and Jerome Revaud. Dust3r: Geometric 3d vision made easy. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_, pages 20697–20709, 2024b.
*   Wu et al. \[2023\] Rundi Wu, Ben Mildenhall, Philipp Henzler, Keunhong Park, Ruiqi Gao, Daniel Watson, Pratul P Srinivasan, Dor Verbin, Jonathan T Barron, Ben Poole, et al. Reconfusion: 3d reconstruction with diffusion priors. _arXiv preprint arXiv:2312.02981_, 2023.
*   Wynn and Turmukhambetov \[2023\] Jamie Wynn and Daniyar Turmukhambetov. Diffusionerf: Regularizing neural radiance fields with denoising diffusion models. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)_, pages 4180–4189, 2023.
*   Xiong et al. \[2023\] Haolin Xiong, Sairisheek Muttukuru, Rishi Upadhyay, Pradyumna Chari, and Achuta Kadambi. Sparsegs: Real-time 360 {\\{\\\\backslashdeg}\\} sparse view synthesis using gaussian splatting. _arXiv preprint arXiv:2312.00206_, 2023.
*   Yang et al. \[2024a\] Chen Yang, Sikuang Li, Jiemin Fang, Ruofan Liang, Lingxi Xie, Xiaopeng Zhang, Wei Shen, and Qi Tian. Gaussianobject: Just taking four images to get a high-quality 3d object with gaussian splatting. _arXiv preprint arXiv:2402.10259_, 2024a.
*   Yang et al. \[2023\] Jiawei Yang, Marco Pavone, and Yue Wang. Freenerf: Improving few-shot neural rendering with free frequency regularization. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_, pages 8254–8263, 2023.
*   Yang et al. \[2024b\] Lihe Yang, Bingyi Kang, Zilong Huang, Zhen Zhao, Xiaogang Xu, Jiashi Feng, and Hengshuang Zhao. Depth anything v2. _arXiv preprint arXiv:2406.09414_, 2024b.
*   Yu et al. \[2021\] Alex Yu, Vickie Ye, Matthew Tancik, and Angjoo Kanazawa. pixelnerf: Neural radiance fields from one or few images. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_, pages 4578–4587, 2021.
*   Zhang et al. \[2025\] Jiawei Zhang, Jiahe Li, Xiaohan Yu, Lei Huang, Lin Gu, Jin Zheng, and Xiao Bai. Cor-gs: sparse-view 3d gaussian splatting via co-regularization. In _European Conference on Computer Vision_, pages 335–352. Springer, 2025.
*   Zhang et al. \[2023\] Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models. In _Proceedings of the IEEE/CVF International Conference on Computer Vision_, pages 3836–3847, 2023.
*   Zhang et al. \[2018\] Richard Zhang, Phillip Isola, Alexei A. Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In _Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)_, 2018.
*   Zhu et al. \[2023\] Zehao Zhu, Zhiwen Fan, Yifan Jiang, and Zhangyang Wang. Fsgs: Real-time few-shot view synthesis using gaussian splatting. _arXiv preprint arXiv:2312.00451_, 2023.

\\thetitle  

Supplementary Material  

Table 3: We quantitatively compare our approach against other sparse view synthesis methods on CO3D dataset. PSNR and SSIM measure pixel-wise error and as such ReconFusion and CAT3D score higher in most cases, since these metrics favor blurrier results. We outperform the previous approaches in terms of perceptual quality (LPIPS) while being competitive with CAT3D, a concurrent approach. The numbers for approaches marked by ∗ are directly grabbed from ReconFusion. CAT3D results are also obtained from the original paper. The methods marked by † are initialized using DUSt3R to improve their results.

![Refer to caption](x9.png)

Figure 9: Number of views ablation.

![Refer to caption](x10.png)

Figure 10: DUSt3R sometimes fails to generate a reasonable pointcloud on ambiguous scenes. This in turn affects our optimization quality.

8 Implementation Details
------------------------

We provide additional implementation details for some of the models and systems described in the main paper. All optimization and fine-tuning are performed on a single Nvidia RTX A5000 GPU, except for Stage 2, which is carried out using two A5000 GPUs.

### 8.1 Repair Model

Similar to GaussianObject \[[42](https://arxiv.org/html/2503.10860v2#bib.bib42)\], the leave-one-out strategy is applied by introducing the left-out view after 6000 iterations of optimization. The optimization continues until iteration 10000, thereby obtaining training pairs for the repair model. We then fine-tune the repair model using these training pairs for 1800 iterations.

### 8.2 Inpainting Model

We adapt the RealFill \[[32](https://arxiv.org/html/2503.10860v2#bib.bib32)\] methodology to fine-tune the Stable Diffusion inpainting model on the sparse input views. Due to unavailability of code by the original authors, we utilize a third-party implementation by Nguyen \[[20](https://arxiv.org/html/2503.10860v2#bib.bib20)\]. Consistent with the referenced GitHub repository, we fine-tune the model for 2000 iterations.

### 8.3 Optimization

We run the Stage 1 optimization for 4000 iterations, utilizing 8 evenly distributed novel repaired views in addition to the input training images. The repaired views are refreshed every 400 iterations.

Similarly, Stage 2 optimization runs for 4000 iterations. During this stage, we sample 10 evenly distributed views every 200 iterations. For each sampling cycle, we sequentially inpaint and project every other view (5 views) before rendering and repairing all 10 views. Inpainting is performed up to iteration 2800, after which only the repair process is carried out to address minor artifacts.

9 Additional Results
--------------------

We provide additional results on the Mip-NeRF 360 \[[3](https://arxiv.org/html/2503.10860v2#bib.bib3)\] and CO3D \[[25](https://arxiv.org/html/2503.10860v2#bib.bib25)\] dataset for the 3-, 6- and 9-input setting. For both datasets, we utilize the training cameras provided by ReconFusion \[[39](https://arxiv.org/html/2503.10860v2#bib.bib39)\] in our evaluation. In addition to the numerical results presented by ReconFusion and CAT3D \[[8](https://arxiv.org/html/2503.10860v2#bib.bib8)\], we compare against recent state-of-the-art 3D Gaussian based sparse novel view synthesis methods including FSGS \[[49](https://arxiv.org/html/2503.10860v2#bib.bib49)\], CoR-GS \[[46](https://arxiv.org/html/2503.10860v2#bib.bib46)\] and DNGaussian \[[16](https://arxiv.org/html/2503.10860v2#bib.bib16)\]. In contrast to MipNeRF 360 dataset \[[3](https://arxiv.org/html/2503.10860v2#bib.bib3)\], which contains long range general scenes with large missing regions, most CO3D scenes are close up photos of objects in front of a simple background, e.g., a wooden table or floor. This limits the need for and the performance improvement gained from high-quality inpainting. As shown in Table [3](https://arxiv.org/html/2503.10860v2#Sx1.T3 "Table 3 ‣ RI3D: Few-Shot Gaussian Splatting With Repair and Inpainting Diffusion Priors"), we outperform previous approaches including ReconFusion in terms of perceptual quality (LPIPS) while being competitive with CAT3D. We also provide visual comparisons in Fig. [12](https://arxiv.org/html/2503.10860v2#S10.F12 "Figure 12 ‣ 10 Limitations ‣ RI3D: Few-Shot Gaussian Splatting With Repair and Inpainting Diffusion Priors"). Our approach generates complete and consistent results compared to other 3DGS-based approaches. We provide video comparisons for both MipNeRF 360 and CO3D in the supplementary video.

We provide an additional ablation result (Fig. [9](https://arxiv.org/html/2503.10860v2#Sx1.F9 "Figure 9 ‣ RI3D: Few-Shot Gaussian Splatting With Repair and Inpainting Diffusion Priors")) illustrating how the number of training views affects the output quality. As expected, increasing the number of training views leads to better reconstruction quality

10 Limitations
--------------

As discussed in the main paper, the quality of our dense initialization and consequently the final reconstruction is influenced by the accuracy of the depth maps produced by DUSt3R. In cases where DUSt3R generates highly inaccurate depth estimates, our optimization procedure cannot fully compensate for the resulting misalignments, leading to noticeable ghosting artifacts, as shown in Fig. [10](https://arxiv.org/html/2503.10860v2#Sx1.F10 "Figure 10 ‣ RI3D: Few-Shot Gaussian Splatting With Repair and Inpainting Diffusion Priors"). However, due to the modular design of our method, DUSt3R can be readily replaced with more accurate depth estimation models as they become available which mitigates this limitation.

![Refer to caption](x11.png)

Figure 11: We compare our approach against the other state-of-the-art sparse view synthesis methods on a few scenes from the Mip-NeRF 360 dataset.

![Refer to caption](x12.png)

Figure 12: We compare our approach against the other state-of-the-art sparse view synthesis methods on a few scenes from the CO3D dataset.

