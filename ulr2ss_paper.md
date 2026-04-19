Improved Semantic Segmentation from Ultra-Low-Resolution
RGB Images Applied to Privacy-Preserving Object-Goal Navigation

Xuying Huang

Sicong Pan

Olga Zatsarynna

Juergen Gall

Maren Bennewitz

5
2
0
2

l
u
J

1
2

]

O
R
.
s
c
[

1
v
4
3
0
6
1
.
7
0
5
2
:
v
i
X
r
a

Fig. 1: Our key innovation lies in enabling object-goal navigation through improved semantic segmentation from ultra-low-resolution RGB
inputs, thus preserving privacy while maintaining functionality. This figure shows an example of privacy-preserving object-goal navigation
using our approach. The High-Res Views show full-resolution RGB images for visualization only, while the Robot Views display the actual
robot input: ultra-low-resolution (16 × 16) monocular RGB images. Cyan dashed lines indicate that sensitive information is effectively
concealed due to the ultra-low-resolution input. On the left, when the target object has not yet been found, the robot performs floor-based
navigation by following waypoints (blue dots) along the floor centerline (blue dashed line), turning at branch points that may lead to new
rooms. On the right, once the target object has been detected, the robot switches to object-goal navigation, gradually approaching the
object's center. As can be seen, our pipeline achieves improved semantic segmentation from ultra-low-resolution RGB images.

Abstract— User privacy in mobile robotics has become a
critical concern. Existing methods typically prioritize either the
performance of downstream robotic tasks or privacy protection,
with the latter often constraining the effectiveness of task
execution. To jointly address both objectives, we study semantic-
based robot navigation in an ultra-low-resolution setting to
preserve visual privacy. A key challenge in such scenarios is re-
covering semantic segmentation from ultra-low-resolution RGB
images. In this work, we introduce a novel fully joint-learning
method that integrates an agglomerative feature extractor and a
segmentation-aware discriminator to solve ultra-low-resolution
semantic segmentation, thereby enabling privacy-preserving,
semantic object-goal navigation. Our method outperforms dif-
ferent baselines on ultra-low-resolution semantic segmentation
and our improved segmentation results increase the success rate
of the semantic object-goal navigation in a real-world privacy-
constrained scenario.

X. Huang, S. Pan and M. Bennewitz are with the Humanoid Robots Lab,
O. Zatsarynna and J. Gall are with Computer Vision Group at the University
of Bonn, Germany. X. Huang, S. Pan, J. Gall and M. Bennewitz are
additionally with the Lamarr Institute for Machine Learning and Artificial
Intelligence and the Center for Robotics, Bonn, Germany. This work has
been partially funded by the BMBF, grant No. 16KIS1949 and within
the Robotics Institute Germany, grant No. 16ME0999. Corresponding:
huang@cs.uni-bonn.de.

I. INTRODUCTION

With the widespread deployment of mobile robots in daily
life, user privacy has become an increasing concern. When
operating in private environments, robots may inadvertently
capture sensitive user data through their onboard cameras,
such as identification numbers, human faces, or medical
records. To mitigate privacy concerns, prior work has ex-
plored the use of alternative image modalities, such as depth
and semantic segmentation images [11, 14],
to generate
privacy-preserving representations for diverse downstream
robotic applications.

To obtain semantic segmentation images for robotic tasks,
high-resolution (HR) RGB images are often indispensable [1,
14, 25]. However, HR images inherently contain rich se-
mantic details, which raise privacy risks [5]. To address
this, another line of work investigates directly using low-
resolution (LR) RGB images (15 × 15) [8]. These LR inputs
are then used exclusively for privacy-preserving sparse map-
ping. Nevertheless, LR images convey very limited semantic
information, which limits downstream robotic applications,
as many tasks require access to depth or semantic informa-
tion. Therefore, we propose a novel pipeline that uses LR

RGB images to generate improved semantic segmentation
maps,
thereby enabling privacy-preserving, segmentation-
based robotic tasks, such as object-goal navigation [26].

To investigate the relationship between the resolution
threshold of RGB images and privacy, we follow the results
of the user study by Huang et al. [7]. RGB images with
a resolution of 32 × 32 are generally considered accept-
able by users, while resolutions of 16 × 16 or lower are
deemed necessary to fully ensure privacy. In this work, we
define such extremely LR inputs (i.e., 16 × 16) as ultra-
low-resolution (ULR), and focus on only utilizing them
to generate improved semantic segmentation, as they are
considered sufficient to fully preserve user privacy.

In ULR RGB images, visual information is severely lim-
ited, which hinders the performance of semantic segmen-
tation. A natural solution is to leverage priors from super-
resolution (SR) methods to recover semantic segmentation
from ULR inputs. However, existing approaches consis-
tently fail to achieve satisfactory segmentation performance,
whether through directly applying pretrained SR models [3],
training only the SR [6] or segmentation branch, or even
employing end-to-end joint learning [13]. To address these
limitations, we introduce a novel joint-learning framework
that incorporates a segmentation-aware discriminator and an
agglomerative feature extractor, enabling effective recovery
of improved semantic segmentation from ULR RGB images.
To the best of our knowledge, we are the first to achieve
recovering improved semantic segmentation from ULR RGB
inputs for privacy-preserving, semantic-based robotic tasks.
Fig. 1 illustrates the application and functionality of our
system. For simplicity, we refer to our pipeline that recovers
high-quality semantic segmentation from ULR RGB images
as ULR semantic segmentation. To summarize, our contri-
butions are the following:

• We develop a novel

joint-learning architecture in-
and
tegrating an agglomerative
segmentation-aware discriminator, which outperforms
different types of strong baselines on the SUN RGB-D
dataset [20] as well as our real-world custom dataset on
ULR semantic segmentation.

extractor

feature

• We provide further analysis to explain why super-
resolution modules in existing baselines do not directly
improve ULR semantic segmentation, supported by
additional super-resolved evaluation metrics.

• We show that our improved ULR semantic segmentation
results increase the success rate of the semantic object-
goal navigation in a real-world privacy-constrained sce-
nario.

To support reproducibility and future research, our im-
plementation will be made open-source at: https://
github.com/hxy-0818/ULR2SS.

II. RELATED WORK

In this section, we first introduce relevant work on user
privacy in mobile robots and, afterwards, approaches to LR
semantic segmentation.

A. User Privacy in Mobile Robots

As mobile robots become growing prevalent in everyday
life, user privacy in robotic applications has received increas-
ingly attention [17]. Early work by Reinhardt et al. [16]
introduce a comprehensive taxonomy of privacy dimensions,
i.e., informational, physical, psychological, and social pri-
vacy to characterize the risks posed when robots collect and
process data during operation. Vision sensors, particularly
HR RGB cameras, have emerged as a primary vector for
informational privacy violations in robotic applications [23].
Such cameras capture fine-grained textures that enable in-
ference of facial identity, license plates, textual identifiers,
and even emotional states. To mitigate the informational
privacy risks inherent in HR robotic vision, several studies
have investigated leveraging alternative image modalities to
support privacy-preserving robotic downstream tasks. For
example, MOSAIC [11] introduces a diffusion-based frame-
work that entirely removes RGB inputs, generating con-
sistent and privacy-preserving indoor scenes from multiple
depth views. SegLoc [14] constructs compact 3D maps from
semantic segmentation labels to enable privacy-preserving
visual localization. However, this method depends on HR
RGB inputs to generate improved semantic segmentation,
thereby necessitating the capture of detailed visual data and
perpetuating user privacy concerns [7].

Another line of work is to enforce privacy at capture time
by directly reducing the resolution of RGB images from the
source. [8] captures RGB images of 15 × 15 resolution and
achieves privacy-preserving sparse mapping solely from such
ULR RGB inputs. While this approach relies solely on ULR
RGB inputs to protect privacy, its applicability in robotics is
limited, as most tasks demand depth or semantic information.
To address the limitations of semantic representations
and ULR, we introduce a novel pipeline that reconstructs
improved semantic segmentation from ULR RGB images and
subsequently drives privacy-preserving, segmentation-based
robotic tasks.

B. Low-Resolution Semantic Segmentation

The problem of recovering semantic segmentation from
LR RGB inputs has been studied before. For example a com-
mon paradigm pretrained pipeline [3] leverages a pretrained
SR model to first enlarge the LR images and then applies a
pretrained segmentation network.

Beyond the direct use of pre-trained networks, some work
has explored monomodule training methods, which train
the SR or segmentation network to improve segmentation
performance in remote sensing imagery. [6] freeze a pre-
to serve purely as a feature
trained segmentation model
extractor while training only the SR network, whereas an-
other alternative [13] is to first upscale the LR RGB input,
e.g., using bicubic interpolation and then train solely the
segmentation model.

As an alternative, [13] also investigates adopting a fully
joint-learning strategy, in which a single end-to-end pipeline
simultaneously optimizes SR and semantic segmentation
objectives. Although fully joint-learning approaches achieve

Fig. 2: Overview of our proposed joint-learning framework for
ultra-low-resolution semantic segmentation. SS represents semantic
segmentation. Given an ULR RGB image, we leverage a GAN-
based SR model to generate HR RGB. The super-resolved image
is then fed into a semantic segmentation network, which produces
a predicted semantic map. We integrate a AFE module to extract
high-level semantic information and a SAD module to assess the
realism of concatenated super-resolved RGB image and predicted
segmentation map. The AFE module takes the SR output and the
GT HR as input to compute the feature loss Lfea. The SAD module
receives the concatenated pair to compute the segmentation-aware
discrimination loss LD and the adversarial loss Ladv. The SAD
module is trained on the segmentation-aware discrimination loss
LD. The entire network with SR and SS branches is jointly trained
using the pixel-wise loss L2, feature loss Lfea, adversarial loss Ladv
and cross-entropy loss Lce.

improved accuracy compared to others at LR conditions,
their gains remain marginal.

All of the aforementioned methods rely on inputs that are
not extremely low resolution from the source and therefore
retain substantial semantic content. Under our ULR setup,
where available semantic information is severely constrained,
these methods fail
to produce satisfactory segmentation
results as we show in our experiments. In this work, we
aim to solve improved semantic segmentation generation
from ULR RGB images, enabling semantic-based privacy-
preserving downstream robotic tasks.

III. ULTRA-LOW-RESOLUTION SEMANTIC
SEGMENTATION

We propose a novel joint-learning framework for ULR
semantic segmentation. An overview of our framework is
shown in Fig. 2. Given an ULR RGB input, we first employ
a Generative Adversarial Network (GAN)-based SR network
to reconstruct a high-resolution image. This super-resolution
output is then passed through an encoder-decoder semantic
segmentation network to produce improved segmentation
map. We incorporate a agglomerative feature extractor (AFE)
to extract high-level semantic features from the RGB repre-
sentation. We concatenate the predicted segmentation map
with the super-resolved RGB output, the GT segmentation
with the HR RGB, and design a segmentation-aware discrim-
inator (SAD) to assess the realism of both the generated HR
RGB image and their predicted semantic segmentation map.
The SAD module is trained separately with the segmentation-
aware discrimination loss LD of the concatenated pair. Our

Fig. 3: Proposed segmentation-aware discriminator network archi-
tecture. The SAD module receives the concatenated pair of RGB
image and semantic label and outputs the possibility of the gener-
ated RGB and its predicted label being judged as "real". The conv
block consists of two sets of stacked 3 × 3 convolutional layers,
followed by a spectral normalization layer (purple). The block
output is then passed through an AdaptiveAvgPool2d and flattened
into a one-dimensional representation. This vector is projected to a
linear layer and subsequently regularized by spectral normalization.

proposed joint-learning method is trained on four losses: a
pixel-wise difference between HR and super-resolved signal
L2, a feature loss Lfea, an adversarial loss on the concate-
nated pair Ladv and a pixel-wise cross-entropy loss Lce
between the predicted label and the ground-truth (GT) label.

A. Joint-Learning Architecture

Considering the real-time constraints of robotic appli-
cations and the quality of super-resolution for segmenta-
tion, we adopt the standard ESRGAN generator built upon
Residual-in-Residual Dense Network (RRDN) architecture
proposed by Wang et al. [24]. Our implementation includes
23 residual-in-residual dense blocks (RRDBs), each RRDB
encapsulating three dense blocks, and each dense block in
turn consisting of five sub-layers (ConvBlocks).

We employ DeepLabV3+ [4] as the semantic segmenta-
tion module of our proposed end-to-end fully joint-learning
framework. This method follows an encoder–decoder design.
For the encoder, we use a pretrained ResNet-101 backbone to
extract rich, multi-level feature maps, which are then passed
to an Atrous Spatial Pyramid Pooling (ASPP) module to
aggregate contextual information at multiple scales without
reducing spatial resolution. The decoder upsamples the ASPP
output, merges it with low-level encoder features to recover
fine edges, applies a 3×3 refinement convolution, and restores
full-resolution predictions through a second upsample.

The total loss is the balanced combination of SR and
semantic segmentation, with α controlling the influence of
each loss function.

Ltot = (1 − α) (λ1 L2 + λ2 Lfea + λ3 Ladv) + α Lce (1)

For the super-generation part, we utilize a L2 pixel-wise
reconstruction term as shown in Eq. (2), combined with a
feature loss and a adversarial loss. The scalar coefficients λ
controls the impact of each term.

L2 =

1
H W C

H
(cid:88)

W
(cid:88)

C
(cid:88)

h=1

w=1

c=1

(cid:0)Igt(h, w, c)−Isr(h, w, c)(cid:1)2

(2)

where H, W, C are the height, width and channel dimensions
of the image. Igt and Isr represent the GT and super-resolved

RGB image respectively. The details of feature loss Lf ea and
extended adversarial loss Ladv are provided in Sec. III-B and
Sec. III-C respectively.

We employ the pixel-wise cross-entropy loss for semantic

segmentation, which shows in Eq. (3):

Lce(x, y) = −xy + log

(cid:88)

exj

j

(3)

where x represents the raw logits and y is the ground-truth
class index. The term xy refers to the logit corresponding to
the true class, while xj denotes the logits for class j. This
loss computes the negative log-likelihood of the correct class
by implicitly applying a log-softmax operation.

B. Agglomerative Feature Extractor

Conventional GAN-based super-resolution networks cal-
culate a perceptual loss by feeding the generated and ground
truth RGB images to a pretrained network (VGG-16/19) [19].
However, when operating on the ULR inputs, the severely
degraded textures and details render VGG-based perceptual
guidance unsatisfactory. To better extract the semantic fea-
tures for segmentation under ULR settings, we utilize the
priors from large pretrained feature extraction model. In par-
ticular, we use RADIOv2.5-g version of AM-RADIO [15],
which consolidates multiple large vision foundation models
into a single, efficient student network as our feature ex-
tractor. As the module is required only during the training
phase, it does not influence the real-time requirements of
robotic applications.

We propose a feature loss, which combines a L1 loss and

cosine-similarity Lcos term:

Lfea = L1 + Lcos
(cid:13)
L1 = (cid:13)
(cid:13) ˆFreal − ˆFfake
(cid:13)1
Lcos = 1 − cos(cid:0) ˆFreal, ˆFfake)

(4)

(5)

(6)

The L1 component penalizes element-wise deviations to
reinforce local detail consistency, while the cosine compo-
nent aligns feature directions to improve global semantic co-
herence. By uniting these complementary objectives, our loss
achieves more holistic feature alignment, thereby facilitating
downstream semantic segmentation and improving the model
generalization ability.

ˆF denotes the feature representation after L2 normaliza-

tion, which is used to compute the angle.

C. Segmentation-Aware Discriminator

The key component of our approach is the Segmentation-
Aware Discriminator, shown in Fig. 3 which extends the
conventional GAN discriminator. A standard discriminator
evaluates the visual fidelity between generated and ground-
truth RGB images and primarily encourages the generator
to produce images closely resembling the GT RGB, rather
than images optimized for downstream task of semantic
segmentation task. Under ULR conditions, where semantic
information is severely limited, this exclusive focus on RGB

(7)

(8)

(9)

reconstruction does not lead to improved segmentation per-
formance. In contrast, inspired by [22], which introduces seg-
mentation cues to discriminator, our SAD module overcomes
this limitation by jointly assessing both RGB reconstruction
quality and segmentation accuracy, thereby balancing the
adversarial reconstruction loss with a segmentation-driven
objective to encourage the generator to produce outputs that
are beneficial for semantic segmentation.

We introduce a segmentation-aware discrimination loss to

train SAD:

LD = LBCE(D(zreal), 1) + LBCE(D(zfake), 0)
y log(cid:0)σ(u)(cid:1) + (1 − y) log(cid:0)1 − σ(u)(cid:1)(cid:105)

LBCE(x, y) = −

(cid:104)

σ(u) =

1
1 + e−u ,
zreal = concat(cid:0)Igt, Sgt

u ∈ { D(zreal), D(zfake) }

(cid:1),

zfake = concat(cid:0)Isr, Spred

(cid:1)

(10)

Instead of using the relativistic discrimination loss from
ESRGAN, we employ the binary cross-entropy (BCE) loss
from SRGAN [10] for the discriminator, which estimates the
probability that one input image comes from the true HR data
distribution. Here D(·) outputs the discriminator logits. zreal
denotes the concatenated GT image Igt with its segmentation
mask Sgt while zfake is the concatenated SR image Isr with
corresponding predicted semantic label Spred.

The adversarial loss Ladv derived from SAD in the total

loss Eq. 1 is:

Ladv = LBCE

(cid:0)D(zfake), 1(cid:1)

(11)

Moreover, we employ spectral normalization [12] to scale
the weights of SAD and guarantee Lipschitz-continuity. We
discover this to be essential in improving GAN training
stability. By bounding the gradients of SAD, it prevents large
spikes in the backpropagation signal of generator and reduces
the risk of instability.

IV. EXPERIMENTAL RESULTS

In this section, we validate the performance of our pro-
posed method. Firstly, the experimental setup and imple-
mentation details are described. Then, the experimental re-
sults on SUN RGB-D dataset [20] are analyzed, followed
by an ablation study. Subsequently, we investigate the re-
lationship between the super-resolution module and ULR
semantic segmentation. Finally, we analyze the zero-shot
domain generalization capability of our proposed method
on a custom real-world indoor dataset and evaluate its
success rate in semantic object-goal navigation tasks under
privacy-constrained conditions.

A. Setup

1) Dataset: We trained our network on the SUN RGB-D
dataset [20], a widely used indoor semantic segmentation
benchmark comprising 10,335 images across diverse indoor
environments (offices, bedrooms, kitchens, etc.). Each image
was cropped to 384 × 384 pixels to serve as our high-
resolution ground truth. We randomly split the dataset into

Type

Method-label

Network-Architecture

SR-Trained

Pretrained
Pipeline

Mono-Module
Training

Fully Joint-Learning

PP-1 [4]

PP-2 [3]

MMT-1 [13]

MMT-2 [6]

FJL-1 [13]

Deeplabv3+

ESRGAN + Deeplabv3+

Ours

ESRGAN + Deeplabv3+ + SAD + AFE

–

×

×
✓

✓

✓

Training Dataset

384→384

16→384

Seg-Trained
✓

✓

✓

×
✓

✓

TABLE I: Detailed configuration of each approach. Three types, in total five baselines and our proposed method are illustrated. Each
method is represented by a concise method-label, which distinguishes differences in network-architecture, module training, and the training
dataset resolutions. 384 → 384 and 16 → 384 represent that only the RGB inputs are processed to resolution of 384 and 16, but the
segmentation outputs are always generated with 384.

Method
mIoU

PP-1
0.0716

PP-2
0.1241

MMT-1 MMT-2
0.1640
0.2652

FJL-1
0.2654

Ours
0.3101

Method

mIoU

Ours w/o SAD &
w/o AFE (VGG-19)
0.2713

Ours w/o SAD

Ours w/o AFE

Ours

0.2824

0.3021

0.3101

TABLE II: Semantic segmentation test on 16 × 16 images of the
SUN RGB-D dataset. Our proposed method achieves the highest
semantic segmentation performance among all baselines.

a training set (9,000 images), a validation set (668 images),
and a test set (667 images). To better train different methods,
we processed the dataset into two groups of resolutions:
384 → 384 and 16 → 384.

2) Training Setup: Training was conducted on two
NVIDIA A100 GPUs with a batch size of 16, and we
evaluated our approach on a laptop equipped with an Intel
i7-14650HX CPU. We set the training hyperparameters as
follows: λ1 = 0.5, λ2 = 0.01, λ3 = 0.01 and α = 0.3.
We adopted a two-stage training procedure, where we first
pretrained the SR model using the original ESRGAN [24]
losses: a pixel-wise MAE loss, a vanilla adversarial loss,
and a VGG-based perceptual loss. This strategy accelerated
convergence and mitigated the early-stage instabilities. In the
second stage, we integrated the DeepLabv3+ segmentation
network [4] and jointly trained both modules. Both stages
used the same learning rate of 1e-4 and employed ADAM
optimizer with β1 = 0.9 and β2 = 0.999. We trained the model
for 100 epochs and selected the best model according to the
highest mean Intersection-over-Union on validation set.

3) Evaluation Metrics: We
the mean
Intersection-over-Union
[4], which measures
(mIoU)
segmentation accuracy by computing the intersection divided
by the union between predicted and GT segmentation regions
across all classes.

evaluate

4) Baselines: To assess the segmentation performance of
our approach, we compare our proposed method against three
categories of baselines: pretrained pipelines (PP), mono-
module training (MMT), and fully joint-learning (FJL).
Specifically, the following baselines are used:

• PP-1 [4] uses a DeepLabv3+ segmentation network
trained on the SUN RGB-D dataset at 384 × 384 reso-
lution.

• PP-2 [3] leverages a pretrained ESRGAN SR model

combined with PP-1.

• MMT-1 [13] trains exclusively segmentation network at
ULR without training the SR module, which upsamples
the ULR RGB images from 16 × 16 to 384 × 384 using
bicubic interpolation.

• MMT-2 [6] trains only the super-resolution network on

TABLE III: Module ablation on SUN RGB-D dataset at 16 × 16
resolution. Our proposed method with complete modules attains the
highest mIoU, whereas removing either the SAD, or the AFE, or
both modules (default VGG-19 loss) leads to a performance drop.

ULR inputs, with the segmentation network remaining
frozen.

• FJL-1 [13] jointly trains super-resolution and segmen-

tation model on ULR inputs.

Further detailed information about the different baselines and
our method is provided in Table I.

B. Ultra-Low-Resolution Semantic Segmentation Results

For comparison, we conducted an additional experiment
using full-resolution (384×384) RGB inputs. In this setting,
DeepLabv3+ achieves an mIoU of 0.4194, representing the
upper-bound performance.

As demonstrated in Table II, at the ULR of 16 × 16, our
proposed method attains the highest mIoU, outperforming
all baseline approaches. Applying a segmentation network
pretrained on full-resolution inputs directly to 16×16 images
(PP-1) causes the mIoU decrease dramatically from 0.4194
to 0.0716, highlighting the severe resolution sensitivity of
this task. Although other baselines incorporate SR priors
to enhance semantic segmentation performance, the highest
mIoU achieved is 0.2654, still far from the full-resolution
result (0.4194), indicating that conventional SR networks
cannot directly benefit semantic segmentation. As can be
seen, our proposed method surpasses the best-performing
baseline (FJL-1) by 16.84%, demonstrating that our adjusted
SR effectively facilitates ULR semantic segmentation.

C. Ablation Study

In this section, we conduct an ablation study of our method
on ULR 16 × 16 setting. We test semantic segmentation
performance of our proposed method on the SUN RGB-D
test set after separately removing segmentation-aware dis-
criminator and agglomerative feature extractor as demon-
strated in Table III. Regardless of which core module is
removed, the mIoU score of our proposed method drops.
Removing both modules and relying solely on the default
perceptual loss from VGG-19 results in a mIoU drop to
0.2713, highlighting that both modules are important for
ULR semantic segmentation. Specifically, removing the SAD

Method

PSNR↑

SSIM↑

LPIPS↓

FID↓

ARI↑

Covering↑

BF↑

mIoU↑

PP-1
PP-2
MMT-1
MMT-2
FJL-1
Ours

20.03
20.02
20.03
15.91
12.22
15.07

0.6124
0.5598
0.6124
0.4382
0.3218
0.4059

0.7088
0.5631
0.7088
0.5151
0.5839
0.4892

263.78
218.27
263.78
104.43
100.94
242.73

0.1220
0.1477
0.3462
0.2885
0.3524
0.4010

0.3666
0.3831
0.4934
0.4540
0.4917
0.5271

0.0514
0.0906
0.1374
0.1303
0.1452
0.1359

0.0716
0.1241
0.2652
0.1640
0.2654
0.3101

TABLE IV: Qualitative comparison of super-resolution outputs, segmentation and semantic segmentation results. Given the ultra-low-
resolution 16 × 16 inputs, we compare five baselines and our proposed approach. As shown in the table, our method achieves the
highest values in semantic segmentation (mIoU) as well as segmentation ARI and Covering, demonstrating the best overall segmentation
performance, despite not obtaining the strongest RGB reconstruction metrics. This indicates that segmentation accuracy is not directly
positively correlated with RGB image quality.

Fig. 4: Visual comparison of super-resolution RGB images and semantic segmentation maps. We compare the visualization results of
super-resolved outputs and predicted semantic-segmentation produced by five baselines and our proposed method against the GT, under
setting of 16×16 input. Our method produces the most accurate semantic segmentation compared to all baselines, despite generating RGB
images of lower visually quality. Our method amplifies object-boundary contrast to favor region delineation, and achieves best semantic
segmentation accuracy at the expense of boundary segmentation.

module reduces mIoU to 0.2824, with 9.8% less, indicating
that SAD is essential for guiding the generator to produce SR
images that are beneficial for semantic segmentation. While
the AFE underscores the importance of extracted high-level
features for robust segmentation when input information is
severely limited.

ceptual Image Patch Similarity (LPIPS) and Fréchet Incep-
tion Distance (FID) [9]. For the evaluation of segmentation
accuracy, we report two region-based indices, Adjusted Rand
Index (ARI), Segmentation Covering (Covering), and one
boundary-based metric, Boundary F-measure (BF) [2], and
one overall semantic segmentation metric mIoU.

D. Analysis Between Super-Resolution and Ultra-Low-
Resolution Semantic Segmentation

Building on our findings in Sec. IV-B, a natural ques-
tion arises: Does super-resolution really help for ultra-low-
resolution semantic segmentation? To better understand why
our proposed method helps ULR semantic segmentation
effectively, we calculated the metrics for the intermediate
super-resolved RGB images and their corresponding seg-
mentation outputs in Table IV, and visualized the results
in Fig. 4. Our proposed pipeline outputs only semantic
segmentation maps, without generating or storing intermedi-
ate super-resolved images. The intermediate super-resolved
results shown here in Fig. 4 are solely for visualization
purposes to facilitate analysis of the relationship between
super-resolution and ULR-based semantic segmentation, and
thus do not raise privacy concerns. Specifically, to assess the
RGB reconstruction fidelity, we employ four widely used
image-quality metrics: Peak Signal-to-Noise Ratio (PSNR),
Structural Similarity Index Measure (SSIM), Learned Per-

As shown in Table IV, we observed three interesting

findings:

• Under ULR conditions, standard super-resolution mod-
els do not directly improve downstream semantic seg-
mentation.

• Under ULR conditions, higher PSNR and SSIM do
not necessarily lead to better segmentation performance.
Both PP-1 and MMT-1 upscale 16 × 16 inputs to the
full-resolution size (384×384) via bicubic interpolation,
achieving the highest PSNR and SSIM among all meth-
ods. However, their segmentation results remain unsat-
isfactory. In contrast, FJL-1 generates RGB images with
lower PSNR and SSIM, yet its semantic segmentation
results are quite close to MMT-1.

• Under ULR conditions, lower LPIPS and FID do not
guarantee better segmentation performance. Although
MMT-2 produces outputs that are perceptually (LPIPS)
and distributionally (FID) closer to the ground truth, its
segmentation accuracy lags behind to that of MMT-1.

Method
mIoU

PP-1
0.0576

PP-2
0.1330

MMT-1 MMT-2
0.1650
0.2508

FJL-1
0.2391

Ours
0.3177

TABLE V: Semantic segmentation test on 16 × 16 images of our
custom real-world dataset. Our method achieves the best zero-shot
domain generalization semantic segmentation performance among
all baselines.

Our proposed method generates

intermediate super-
resolved RGB images with lower image quality and color
discrepancies compared to the ground truth. However, by
compromising RGB reconstruction fidelity, our approach
prioritizes generating images that are optimally tailored for
semantic segmentation. As a result, our method attains the
best performance in region-based segmentation. and delivers
the highest overall semantic segmentation performance. As
illustrated in Fig 4, despite the visual differences between our
generated RGB images and the ground truth, our approach
enhances inter-object region distinctions, thus effectively fa-
cilitating semantic segmentation. Nevertheless, our approach
yields a lower BF score, this might be due to color bleeding
at object boundaries, which broadens the edge regions. We
leave this as an open question and our future work.

We validate that a standard super-resolution module alone
does not directly help segmentation performance. While our
proposed approach, incorporating segmentation cues into the
discriminator and leveraging priors from a large pretrained
feature extractor, effectively enhances accuracy.

E. Zero-shot Domain Generalization

To assess the zero-shot domain generalization capability of
our method, i.e., without any fine-tuning on the new dataset,
we further created a custom small-scale indoor dataset com-
prising 90 annotated RGB and semantic segmentation pairs,
following the data collection scheme of the NYU Depth
V2 [18] dataset. We conducted this domain generalization
test before deploying it in a real-world navigation task, to
preliminarily evaluate our method's applicability and gener-
alization, thereby reducing the resources expenditure.

Table V shows the semantic segmentation results on our
custom indoor-scene dataset under the ULR setting (16×16).
Our method achieves the highest mIoU among all the ap-
proaches, representing a substantial improvement of 26.67%
over the strongest baseline (MMT-1). This result closely
matches our performance on the SUN RGB-D test set,
demonstrating the robustness of our approach and its strong
zero-shot domain generalization ability at ULR conditions.

F. Privacy-Preserving Semantic Object-Goal Navigation

Object-goal navigation, the task of enabling a robot to
locate and approach a specified object within unknown en-
vironments, has become essential in robotics [21]. A widely
adopted strategy leverages semantic information [26] to facil-
itate object-level navigation. We applied semantic object-goal
navigation to showcase that it preserves privacy using only
ULR RGBs as inputs and enables navigation functionality
through improved semantic segmentation outputs.

Our system comprises three modules. Given a target
object, the robot first starts the Semantic Coverage module.

Method
Success Rate (%)

FJL-1 MMT-1
62.5

65.0

Ours
85.0

TABLE VI: Privacy-preserving semantic object-goal navigation
success rate. We conducted a total of 40 trials, testing four target
objects from two distinct starting locations. Our proposed method
achieves the highest navigation success rate, surpassing MMT-1
and FJL-1. Thus, our improved ULR semantic segmentation results
increase the success rate of object-goal navigation in real-world
scenarios with privacy constraints.

is found,

the robot switches to the
If the target object
Object Goal Navigation module to approach it. If the target
object is not segmented, our system calls the Floor Based
Navigation module to determine the next best waypoint for
exploration. We randomly sample midpoints at fixed intervals
to define the robot's movement waypoints. Foreground pixels
deviating horizontally from fitted boundary lines beyond a
threshold are identified as branch points, indicating potential
new area entrances. If no further waypoint
is available,
our system concludes that the target object is not found.
Otherwise, the robot moves to the new waypoint, and the
loop continues.

We deployed the approaches in a real-world indoor envi-
ronment as shown in Fig. 1. The experiments were conducted
using the OrionStar GreetingBot Mini1 mobile robot, with
only the onboard RGB camera activated for perception.
ULR images are simulated by downsampling full-resolution
images to 16 × 16 using bicubic interpolation. To further
validate our findings in Sec. IV-B and Sec. IV-E, we compare
our proposed method against the best baselines on SUNRGB-
D and custom dataset
in the real world. We conducted
experiments using four target objects: sofa, chair, table, and
painting. For each object, the robot is initialized from two
starting locations: one in the corridor and the other inside
the room. Each object-starting point pair is tested five times,
resulting in a total of 40 trials. The success rate is used for
evaluation and computed as the proportion of trials in which
the robot successfully navigates from the starting location
to the target object. A success trail is defined when the
segmented pixels of the target object occupy more than 40%
of the image.

From the results shown in Table VI, our method achieves
the highest navigation success rate compared to other base-
lines, outperforming MMT-1 and FJL-1 by 20 and 22.5 per-
centage points respectively. Under a privacy-constrained sce-
nario using only ULR RGB images as inputs, our improved
ULR semantic segmentation results increase the success rate
of the semantic object-goal navigation. To further illustrate
how improved mIoU contributes to improved navigation, we
present semantic segmentation results across two consecutive
frames in Fig. 5. These results demonstrate that successful
navigation toward a target object requires consistently ac-
curate segmentation across frames. Our method yields more
stable and coherent segmentation results, whereas MMT-1
and FJL-1 fail to maintain correct segmentation in consecu-

1https://en.orionstar.com/mini.html

mIoU
Value

PP-1
0.1514

PP-2
0.2389

MMT-1 MMT-2
0.2951
0.3580

FJL-1
0.3832

Ours-32
0.3991

TABLE VII: Semantic segmentation test on 32 × 32 images of
the SUN RGB-D dataset. MMT-1, MMT-2, FJL-1 and Ours-32 are
trained on 32 → 384 dataset. Our proposed method outperforms all
baselines in semantic segmentation at 32 × 32, attaining the highest
mIoU.

gap between our proposed method at 32 × 32 and the full-
resolution result (0.4194) is minimal, we focus our work
on the more challenging 16 × 16 ULR setting, which fully
guarantees the user privacy.

REFERENCES
[1] G. K. Alshammari, A. Abubakar, N. M. Ahmed, and N. K. Al-
shammari, "Federated Learning-based Semantic Segmentation for
Lane and Object Detection in Autonomous Driving," arXiv preprint
arXiv:2504.18939, 2025.

[2] P. Arbelaez, M. Maire, C. Fowlkes, and J. Malik, "Contour detection
and hierarchical image segmentation," IEEE Trans. on Pattern Analysis
and Machine Intelligence (PAMI), 2010.

[3] J. Caputa, M. Wielgosz, D. Łukasik, P. Russek, J. Grzeszczyk, M. Kar-
watowski, S. Mazurek, R. Fr ˛aczek, A. ´Smiech, E. Jamro et al., "Using
super-resolution for enhancing visual perception and segmentation
performance in veterinary cytology," Journal of Life (Life), 2024.
[4] L.-C. Chen, Y. Zhu, G. Papandreou, F. Schroff, and H. Adam,
"Encoder-decoder with atrous separable convolution for semantic
image segmentation," in Proc. of the Europ. Conf. on Computer Vision
(ECCV), 2018.

[5] M. M. Coffer, "Balancing privacy rights and the production of high-

quality satellite imagery," 2020.

[6] T. Frizza, D. G. Dansereau, N. M. Seresht, and M. Bewley, "Se-
mantically accurate super-resolution generative adversarial networks,"
Journal of Computer Vision and Image Understanding (CVIU), vol.
221, 2022.

[7] X. Huang, S. Pan, and M. Bennewitz, "Privacy risks of robot vision:
A user study on image modalities and resolution," arXiv preprint
arXiv:2505.07766, 2025.

[8] M. U. Kim, H. Lee, H. J. Yang, and M. S. Ryoo, "Privacy-preserving
robot vision with anonymized faces by extreme low resolution," in
Proc. of the IEEE/RSJ Intl. Conf. on Intelligent Robots and Systems
(IROS), 2019.

[9] Y. Kong and S. Liu, "Dmsc-gan: A c-gan-based framework for super-
resolution reconstruction of sar images," Remote Sensing, 2023.
[10] C. Ledig, L. Theis, F. Huszár, J. Caballero, A. Cunningham, A. Acosta,
A. Aitken, A. Tejani, J. Totz, Z. Wang et al., "Photo-realistic sin-
gle image super-resolution using a generative adversarial network,"
in Proc. of the IEEE/CVF Conf. on Computer Vision and Pattern
Recognition (CVPR), 2017.

[11] Z. Liu, H. Zhu, R. Chen, J. Francis, S. Hwang, J. Zhang, and
J. Oh, "Mosaic: Generating consistent, privacy-preserving scenes from
multiple depth views in multi-room environments," arXiv preprint
arXiv:2503.13816, 2025.

[12] T. Miyato, T. Kataoka, M. Koyama, and Y. Yoshida, "Spectral
normalization for generative adversarial networks," arXiv preprint
arXiv:1802.05957, 2018.

[13] M. B. Pereira and J. A. dos Santos, "An end-to-end framework for
low-resolution remote sensing semantic segmentation," in IEEE Latin
American GRSS & ISPRS Remote Sensing Conference, 2020.
[14] M. Pietrantoni, M. Humenberger, T. Sattler, and G. Csurka, "Segloc:
Learning segmentation-based representations for privacy-preserving
visual localization," in Proc. of the IEEE/CVF Conf. on Computer
Vision and Pattern Recognition (CVPR), 2023.

[15] M. Ranzinger, G. Heinrich, J. Kautz, and P. Molchanov, "Am-radio:
Agglomerative vision foundation model reduce all domains into one,"
in Proc. of the IEEE/CVF Conf. on Computer Vision and Pattern
Recognition (CVPR), 2024.

[16] D. Reinhardt, M. Khurana, and L. H. Acosta, ""i still need my
privacy": Exploring the level of comfort and privacy preferences of
german-speaking older adults in the case of mobile assistant robots,"
Journal of Pervasive and Mobile Computing (PMC), vol. 74, 2021.

[17] M. Rueben and W. D. Smart, "Privacy in human-robot interaction:
Survey and future work," Proc. of the Intl. Conf. on We robot, 2016.

Fig. 5: Semantic segmentation results during navigation across two
consecutive frames. The first row shows the high-resolution image
for visualization at time step t and its corresponding segmentation
outputs from FJL-1, MMT-1, and our method. The second row
presents the results for the subsequent frame (t+1). Compared to
baselines, our method yields more stable and consistent segmenta-
tion across frames, enabling successful navigation by continuously
segmenting the target object. In contrast, MMT-1 and FJL-1 exhibit
segmentation failures in the subsequent frame, which leads to
navigation failure to the target object.

tive frames, ultimately causing the robot to miss the target.

V. CONCLUSION

In this paper, we present a novel joint-learning frame-
work that recovers HR semantic segmentation maps from
only ULR RGB inputs. The proposed method combines
a segmentation-aware discriminator and an agglomerative
feature extractor to address loss imbalance and semantic
feature extraction limitation at ULR conditions. Our experi-
ments demonstrate that, under ULR scenarios, our approach
outperforms different types of strong baselines in semantic
segmentation accuracy. We validate that conventional super-
resolution network in existing baselines fails to directly
improve segmentation in ULR setup, whereas our integration
of segmentation guidance and large pretrained feature extrac-
tion priors can improve semantic segmentation performance
effectively under ULR settings. Moreover, our proposed
method shows zero-shot domain generalization and enables
privacy-preserving robotic applications such as semantic
object-goal navigation. Our experiments demonstrate that
improving ULR semantic segmentation leads to a higher
success rate in semantic object-goal navigation under real-
world privacy-constrained conditions.

ACKNOWLEDGMENT

We gratefully acknowledge Liren Jin for the insightful
discussions and invaluable assistance throughout this work.

APPENDIX

As discussed earlier, some users consider that RGB images
with 32×32 resolution can meet their basic privacy needs [7].
Here we also evaluated the performance of our proposed
method at a resolution of 32 × 32. Table VII shows the
semantic segmentation results of our method and different
baselines. Our proposed method still achieves the highest
mIoU compared to other baselines. Since the performance

[18] N. Silberman, D. Hoiem, P. Kohli, and R. Fergus, "Indoor segmen-
the

inference from rgbd images," in Proc. of

tation and support
Europ. Conf. on Computer Vision (ECCV), 2012.

[19] K. Simonyan and A. Zisserman, "Very deep convolutional networks
for large-scale image recognition," arXiv preprint arXiv:1409.1556,
2014.

[20] S. Song, S. P. Lichtenberg, and J. Xiao, "Sun rgb-d: A rgb-d scene
understanding benchmark suite," in Proc. of the IEEE/CVF Conf. on
Computer Vision and Pattern Recognition (CVPR), 2015.

[21] J. Sun, J. Wu, Z. Ji, and Y.-K. Lai, "A survey of object goal
navigation," IEEE Trans. on Automation Science and Engineering
(TASE), 2024.

[22] V. Sushko, E. Schönfeld, D. Zhang, J. Gall, B. Schiele, and
A. Khoreva, "You only need adversarial supervision for semantic im-
age synthesis," in Proc. of the Intl. Conf. on Learning Representations

(ICLR), 2021.

[23] A. K. Taras, N. Suenderhauf, P. Corke, and D. G. Dansereau,
"The need for inherently privacy-preserving vision in trustworthy
autonomous systems," arXiv preprint arXiv:2303.16408, 2023.
[24] X. Wang, K. Yu, S. Wu, J. Gu, Y. Liu, C. Dong, Y. Qiao, and
C. Change Loy, "Esrgan: Enhanced super-resolution generative ad-
versarial networks," in Proceedings of the European conference on
computer vision (ECCV) workshops, 2018.

[25] M. Xu, M. Islam, L. Bai, and H. Ren, "Privacy-preserving synthetic
continual semantic segmentation for robotic surgery," IEEE Trans. on
medical imaging (TMI), 2024.

[26] N. Yokoyama, S. Ha, D. Batra, J. Wang, and B. Bucher, "Vlfm: Vision-
language frontier maps for zero-shot semantic navigation," in Proc. of
the IEEE Intl. Conf. on Robotics & Automation (ICRA), 2024.
