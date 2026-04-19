Active Boundary Loss for Semantic Segmentation

Chi Wang1,2, Yunke Zhang1,2, Miaomiao Cui2, Peiran Ren2
Yin Yang3, Xuansong Xie2, Xian-Sheng Hua2, Hujun Bao1, Weiwei Xu1*
1 State Key Lab of CAD&CG, Zhejiang University
3 Clemson University
{wangchi1995, yunkezhang}@zju.edu.cn, {miaomiao.cmm, peiran.rpr}@alibaba-inc.com
yin5@clemson.edu, xingtong.xxs@taobao.com, xiansheng.hxs@alibaba-inc.com, {bao, xww}@cad.zju.edu.cn

2 Alibaba Inc

2
2
0
2

b
e
F
3

]

V
C
.
s
c
[

2
v
6
9
6
2
0
.
2
0
1
2
:
v
i
X
r
a

RGB

CE

CIBL

DenseCRF

Segﬁx

Ours

Figure 1: Segmentation results of an internet image. CE: Produced by DeeplabV3 (Chen et al. 2017a) trained with cross-entropy loss on
Cityscapes (Cordts et al. 2016) dataset. CIBL: Produced by DeeplabV3 trained with cross-entry loss + lov´asz-softmax (Berman et al. 2018)
+ Boundary Loss (Kervadec et al. 2019) on Cityscapes dataset. DenseCRF: Reﬁned results of column 'CE' by DenseCRF (Kr¨ahenb¨uhl et al.
2011). Segﬁx: Reﬁned results of column 'CE' by Segﬁx (Yuan et al. 2020b). Ours: Re-trained by adding our loss.

Abstract

This paper proposes a novel active boundary loss for seman-
tic segmentation. It can progressively encourage the align-
ment between predicted boundaries and ground-truth bound-
aries during end-to-end training, which is not explicitly en-
forced in commonly used cross-entropy loss. Based on the
predicted boundaries detected from the segmentation results
using current network parameters, we formulate the boundary
alignment problem as a differentiable direction vector pre-
diction problem to guide the movement of predicted bound-
aries in each iteration. Our loss is model-agnostic and can be
plugged in to the training of segmentation networks to im-
prove the boundary details. Experimental results show that
training with the active boundary loss can effectively improve
the boundary F-score and mean Intersection-over-Union on
challenging image and video object segmentation datasets.

Introduction
Semantic segmentation is a ﬁne-grained, pixel-wise clas-
siﬁcation task that assigns each pixel a semantic class la-
bel to facilitate high-level image analysis and processing.
Recently, the accuracy of semantic segmentation has been

*Corresponding author.

Copyright © 2022, Association for the Advancement of Artiﬁcial
Intelligence (www.aaai.org). All rights reserved.

substantially improved with the introduction of fully con-
volutional networks (FCNs) (Long et al. 2015; Minaee
et al. 2021). FCNs leverage convolutional layers and down-
sampling operations to achieve a large receptive ﬁeld. Al-
though these operations can encode context information sur-
rounding a pixel, they tend to propagate feature informa-
tion throughout the image, leading to undesirable feature
smoothing across object boundaries. Thus, the segmentation
results might be blurred and lack ﬁne object boundary de-
tails. To address this issue, boundary-aware information ﬂow
control and multi-task training methods have been proposed
to improve the discriminative power of features belonging
to different objects (Bertasius et al. 2016; Takikawa et al.
2019; Zhu et al. 2019). Alternatively, the segmentation er-
rors at boundaries can be remedied by learning the corre-
spondence between a boundary pixel and its corresponding
interior pixel (Yuan et al. 2020b). Despite the empirical suc-
cess of boundary-aware methods in improving the segmen-
tation accuracy, there still exist a signiﬁcant amount of seg-
mentation errors at object boundaries, especially for small
and thin objects. The mutual dependence between seman-
tic segmentation and boundary detection should be further
studied to improve the quality of segmentation results.

In this paper, we propose a novel active boundary loss
(ABL) to progressively encourage the alignment between
predicted boundaries (PDBs) and ground-truth boundaries

(GTBs) during end-to-end training, in which the PDBs are
semantic boundaries detected in the segmentation results of
the current network. To facilitate end-to-end training, the
loss is formulated as a differentiable direction vector pre-
diction problem. Speciﬁcally, for a pixel on the PDBs, we
ﬁrst determine a direction pointing to the closest GTB pixel,
and then move the PDB at this pixel towards the direction in
a probabilistic manner. Moreover, we also propose to detach
the gradient ﬂow to suppress possible conﬂicts. Overall, the
behavior of ABL is dynamic because the PDBs are changing
with the updated network parameters during training. It can
be viewed as a variant of classical active contour methods
(Kass et al. 1988), since our method ﬁrst determines the di-
rection vectors in accordance with the PDBs in the current
iteration and lets the PDBs move along the direction vectors
to reach the GTBs.

Unlike the cross-entropy loss that only supervises pixel-
level classiﬁcation accuracy, ABL supervises the relation-
ship between PDB and GTB pixels. It embeds boundary in-
formation such that the network can pay attention to bound-
ary pixels to improve the segmentation results. Moreover,
Intersection-over-Union (IoU) loss pays more attention to
overall regions of semantic classes but does not focus on the
boundary matching. Thus, ABL can provide complementary
information during the training of the network. As a result, it
can be combined with other loss terms to further improve se-
mantic boundary quality. In our work, we let the ABL work
with the most commonly used cross-entropy loss and the
lov´asz-softmax loss (Berman et al. 2018), a surrogate IoU
loss, to signiﬁcantly improve the boundary details in image
segmentation. The lov´asz-softmax loss is introduced to regu-
larize the training so that the ABL can be used even when the
PDBs might be noisy and far from the GTBs. The advantage
of ABL is that it is model-agnostic and can be plugged into
the training of image segmentation networks to improve the
boundary details. As illustrated in Fig. 1, it is beneﬁcial to
preserve the boundaries of thin objects that contain a small
number of interior pixels.

We tested the ABL with state-of-the-art image segmenta-
tion networks, including CNN-based networks DeepLabV3
(Chen et al. 2017a), OCR network (Yuan et al. 2020a)
and Transformer-based network SwinTransformer (Liu et al.
2021). We have also tested the ABL with STM (Oh et al.
2019), a video object segmentation (VOS) network, to show
that our loss can be applied to improve VOS results as
well. The forward inference stage of these networks remains
the same during testing. The experimental results show that
training with the ABL can effectively improve the boundary
F-score and mean Intersection-over-Union (mIoU) on chal-
lenging segmentation datasets.

Related Work
FCN-based semantic segmentation. FCNs (Long et al.
2015) for semantic segmentation frequently utilize encoder-
decoder structure to generate pixel-wise labelling results for
high-resolution images. Successor methods (Ronneberger
et al. 2015; Ding et al. 2018; Minaee et al. 2021) are ded-
icated to a better fusion of multi-scale features to enhance
the accuracy of localization and handle small objects.

FCN-based methods have also been widely used in VOS,
including propagation-based methods (Hu et al. 2017; Oh
et al. 2019; Voigtlaender et al. 2019) and detection-based
methods (Caelles et al. 2017; Li et al. 2017; Shin Yoon et al.
2017). The key challenge is how to leverage temporal co-
herence and learn discriminative features of target objects to
handle occlusion, appearance change, and fast motion. Since
our loss is model-agnostic, it can also be applied to VOS for
the purpose of boundary reﬁnement.
Boundary-aware semantic segmentation. One way to ex-
ploit boundary information in deep learning-based semantic
segmentation is through multi-task training, in which addi-
tional branches are often inserted to detect semantic bound-
aries (Chen et al. 2020; Gong et al. 2018; Ruan et al. 2019;
Su et al. 2019; Xu et al. 2018a; Takikawa et al. 2019; Zhu
et al. 2021). A key challenge in these methods is how to ef-
ﬁciently fuse features from a boundary detection branch to
improve semantic segmentation.

There are also works focusing on the control of infor-
mation ﬂow through boundaries (Bertasius et al. 2016; Ke
et al. 2018; Bertasius et al. 2017; Ding et al. 2019; Chen
et al. 2016). These methods usually learn pairwise pixel-
level afﬁnity to maintain the feature difference for pixels
near semantic boundaries, while enhancing the similarity of
features for interior pixels simultaneously. Assuming that
boundaries can be correlated through a homography trans-
formation, Borse et al. (2021) proposed a frozen inverse
transformation network as a boundary-aware loss for bound-
ary distance measurement.

The boundary details of the segmentation results can also
be improved in post-reﬁnement. DenseCRF (Kr¨ahenb¨uhl
et al. 2011) is often used to reﬁne the segmentation results
around boundaries. Segﬁx (Yuan et al. 2020b) trains a sepa-
rate network to predict the correspondence between bound-
ary and interior pixels. Thus, labels of interior pixels can be
transferred to boundary pixels. Although these methods can
efﬁciently reﬁne most boundaries, they fail to model the re-
lationship of pixels inside thin objects that contain a small
number of interior pixels, which may downgrade the quality
of slender object boundaries, as shown in Fig. 1. In contrast,
the ABL encourages the alignment of PDBs and GTBs. Our
experiment shows that it can handle such boundaries well.

The uniqueness of our ABL is that it allows propagating
the GTB information with a distance transform for regulat-
ing the network behavior at the PDBs, while the network
structure can remain the same. As a loss, ABL can save
efforts in network design. Kervadec et al. (2019) proposed
Boundary Loss (BL) for image segmentation, which is most
related to our work. However, this loss is designed for unbal-
anced binary segmentation and actually a regional IoU loss.
In our implementation, the ABL is coupled with an IoU loss
in (Berman et al. 2018) to further reﬁne the boundary details.

Active Boundary Loss
The ABL continuously monitors the changes on the PDBs
in segmentation results to determine the plausible moving
directions. Its computation is divided into two phases. First,
for each pixel i on the PDBs, we determine its next candidate
boundary pixel j closer to the GTBs in accordance with the

Figure 2: Pipeline of ABL. Boundary distance map is obtained via the distance transform of GTBs, taken an image in the ADE20K (Zhou
et al. 2017) dataset as an example. The overlayed white and red lines in the boundary distance map indicate the GTBs and PDBs, respectively.
Local distance map: the number indicates the closest distance to the GTBs. Local probability map: X and Yi, i ∈ {0, 1, ..., 7} denote the
class probability distribution for these pixels.

relative location between the PDBs and GTBs. Second, we
use the KL divergence as logits to encourage the increase in
KL divergence between the class probability distribution of
i and j. Meanwhile, this process reduces the KL divergence
between i and the rest of its neighboring pixels. In this way,
the PDBs can be gradually pushed towards the GTBs. Un-
fortunately, candidate boundary pixel conﬂicts might occur,
severely degrading the performance of the ABL. Thus, we
carefully reduce the conﬂicts through gradient ﬂow control
in the computation of ABL, which is crucial to its success.

The overall pipeline of ABL is illustrated in Fig. 2. Each
phase and how to suppress the conﬂicts are detailed as fol-
lows. Hereafter, we use Ai to denote the value stored at pixel
i of a map A.
Phase I. This phase starts with detecting the PDBs using the
class probability map P ∈ RC×H×W output by the current
network, where C denotes the number of semantic classes,
and the image resolution is H × W . Speciﬁcally, we com-
pute a boundary map B through the computation of KL-
divergence to indicate the locations of PDBs. For a pixel i
in B, its value Bi is computed as follows:

Bi =

(cid:110) 1

if ∃KL(Pi, Pj) > (cid:15), j ∈ N2(i) ;

0 Otherwise ,

(1)

where 1 indicates the existence of PDBs, and Pi is the
C-dimensional vector extracted from the probability map
at pixel i. N2(i) indicates the 2-neighborhood of pixel i.
Speciﬁcally, the offsets of pixels in N2(i) to pixel i are
{{1, 0}, {0, 1}}. Since it's difﬁcult to deﬁne a perfect ﬁxed
threshold (cid:15) to detect PDBs, we choose an adaptive thresh-
old to ensure that the number of boundary pixels in B is
less than 1% of the total pixels of the input image, where
1% is a ratio to approximate the number of boundary pix-
els in an image. Empirically, we observe that setting (cid:15) in
this adaptive way can largely avoid the emergence of exces-
sive misleading pixels in B far from the GTBs, especially in
the early training period. Controlling boundary pixel number
also helps to save the computational cost of ABL.

Subsequently, for a pixel i on PDBs, its next candidate
boundary pixel j is selected as its neighboring pixel with the
smallest distance value computed by the distance transform1
of the GTBs. The GTBs are also determined using Eq. 1,
but the KL divergence is replaced by checking whether
the ground-truth class labels are equal between pixel i and
j ∈ N2(i). To represent the coordinate of pixel j in the com-
putation of ABL, we convert it into an offset to pixel i and
then encode it as a one-hot vector. Speciﬁcally, we compute

a target direction map Dg ∈ {0, 1}8×H×W , where the one-
hot vector for a pixel i stored at Dg
i is 8D, because we use
8-neighborhood in this operation. The formula to compute
Dg

i can be written as:

Dg

(2)

i = Φ(arg minj Mi+∆j ) , j ∈ {0, 1, ..., 7},
where ∆j represents the jth element in the set of directions
∆ = {{1, 0}, {−1, 0}, {0, −1}, {0, 1}, {−1, 1}, {1, 1},
{−1, −1}, {1, −1}}, and M is the result of distance trans-
form of GTBs. The pixel i + ∆j with the smallest distance
is selected as the next candidate boundary pixel. The func-
tion Φ converts index j into a one-hot vector. For instance, if
j = 1, Φ(j) should be {0, 1, 0, 0, 0, 0, 0, 0}, which is similar
to the direction representation used in Segﬁx. In implemen-
tation, we dilate B with 1 pixel and perform this operation
for all the pixels in dilated B to accelerate the movement of
the PDBs, since more pixels are covered.
Phase II. The 8D vector Dg
i computed in Eq. 2 is set to
be the target distribution in the cross-entropy loss. We aim
to increase the KL divergence between the class probability
distribution of i and j, and simultaneously reduce the KL di-
vergence between i and the rest of its neighboring pixels. An
8D vector using the KL divergence between pixel i and its
neighboring pixel j as logits, denoted by Dp
i , is then com-
puted as follows:

(cid:40)

Dp

i =

eKL(Pi,Pi+∆k )
m=0 eKL(Pi,Pi+∆m )

(cid:80)7

(cid:41)

, k ∈ {0, 1, ..., 7}

,

(3)

where KL indicates the function to compute the KL diver-
gence using Pi and Pi+∆k .

For those pixels on the PDBs, the ABL is computed as the

weighted cross-entropy loss:
Nb(cid:88)

ABL =

1
Nb

i

Λ(Mi)CE (Dp

i , Dg

i ) .

(4)

The weight function Λ is computed as Λ(x) = min(x,θ)
,
where Nb is the number of pixels on the PDBs and θ is a
hyper-parameter set to 20. The closest distance to the GTBs
at pixel i is used as a weight to penalize its deviation from
the GTBs. If Mi is 0, indicating that the pixel is already on
the GTBs, this pixel will be discarded in the ABL.
Conﬂict suppression. Determining pixels on the PDBs us-
ing KL divergence might lead to the conﬂict case, as shown

θ

1scipy.ndimage.morphology.distance transf orm edt is

used in the implementation of distance transform.

1232334450.000.050.100.150.200.00.20.40.60.81.0XY0Y3Y4Y5Y6Y1Y2Y7ArgminSoftmax(KL(X||Yi))CrossEntropyLocal Distance MapLocal Probability MapDistributionBoundary Distance MappixelFigure 3: An example of conﬂict. V4: pixel on a GTB. ↑:
increase. ↓: decrease. The KL divergence between V1 and V2
is required to increase for V1 but to decrease for V2, resulting
in contradictory gradients for V1 and V2.

in Fig. 3. In this case, pixels V1 and V2 are deemed to be on a
PDB (indicated by the red curve) because the KL divergence
values computed for (V1, W1) and (V2, V3) are larger than
the threshold. However, the GTB (indicated by the green
curve) leads to the conﬂict when computing the ABL for V1
and V2 because the GTB is to the right of V1 and V2. Thus,
for pixel V1, we need to increase KL(PV1, PV2 ) because V2
is the closest to the GTB and it should be the next candidate
pixel in the neighborhood of V1. In contrast, for pixel V2,
we need to decrease KL(PV2 , PV1) because pixel V3 is the
next candidate boundary pixel for V2 rather than V1. Thus,
the gradients of the ABL computed for PV1 and PV2 might
contradict with each other.

While it might be possible to design a global search al-
gorithm to remove such kind of conﬂicts, it will signiﬁ-
cantly slow down the training. Thus, we choose to suppress
the conﬂicts through the easy-to-implement detaching oper-
ation in Pytorch. Speciﬁcally, through the detaching opera-
tion, the gradient of ABL is computed only for the pixels on
the PDBs, but not for its neighboring pixels. This process
indicates that, for a 3 × 3 patch, we focus on the adjustment
of the class probability distribution of pixels on the PDBs
only so as to move the PDBs towards the GTBs. As a result,
the conﬂicting gradient ﬂow from KL(PV2 , PV1) to PV1 is
blocked in this case, and vice versa. Empirically, we found
the mIoU drops around 3% without the detaching operation.
Furthermore, we use label smoothing (Szegedy et al.
2016) to regularize the ABL by setting the largest probabil-
ity of the one-hot target probability distribution to 0.8 and
the rest to 0.2/7 (the parameters, 0.8 and 0.2/7 are deter-
mined through experiments). This process can avoid over-
conﬁdent decisions of network parameter updating, espe-
cially when there exist several pixels with the same distance
value in the neighborhood of pixels on the PDBs. The de-
taching operation is also beneﬁcial in this case to avoid con-
ﬂicts in the gradient ﬂow.

Training Loss
The training loss Lt we mainly used to train a semantic seg-
mentation network consists of three terms:

Lt = CE + IoU + waABL,
(5)
where CE is the most commonly used cross-entropy (CE)
loss, which focuses on the per-pixel classiﬁcation. The com-
bination of lov´asz-softmax loss, namely IoU, and our ABL
are two loss terms that are added to improve the boundary
details, and wa is a weight. The lov´asz-softmax loss is ex-
pressed as follows (Berman et al. 2018):

IoU =

1
|C|

(cid:88)

∆Jc(m(c)),

(6)

c∈C
where C is the number of classes, and m(c) is the vector of
prediction errors for class c ∈ C. ∆Jc indicates the lov´asz
extension of the Jaccard loss ∆Jc .

The reason for introducing the lov´asz-softmax loss is
twofold: 1) This loss tends to prevent small objects from be-
ing ignored in segmentation such that the ABL can be used
to improve their boundary details, since the ABL relies on
the existence of predicted boundaries as the beginning step
of its computation. 2) It can balance with the noisy predicted
boundary pixels, especially at the early training period. The
improvement of ABL over CE plus IoU is veriﬁed in the
Experiments section.

Experiments
We implemented the ABL on a GPU server (2 Intel Xeon
Gold 6148 CPUs, 512GB memory) with 4 Nvidia Tesla
V100 GPUs. In this section, we report ablation studies,
quantitative and qualitative results obtained from the eval-
uation of the ABL in image segmentation experiments and a
test of ﬁne-tuning VOS network.
Baselines. We use the OCR network (Yuan et al. 2020a)
[backbone: HRNetV2-W48 (Wang et al. 2019)], DeeplabV3
(Chen et al. 2017a) [backbone: ResNet-50 (He et al. 2016)],
and UperNet (Xiao et al. 2018) [backbone: SwinTrans-
former (Liu et al. 2021)] as the baseline models for the task
of semantic image segmentation. To verify that our ABL can
be applied to the task of video object segmentation, we use
STM (Oh et al. 2019) as the baseline, since its pre-trained
model is publicly available.
Dataset. We evaluate our loss mainly on the image segmen-
tation dataset Cityscapes (Cordts et al. 2016) and ADE20K
(Zhou et al. 2017). These two datasets provide densely
annotated images that are important for the training of our
method to align semantic boundaries. Cityscapes dataset
contains high-quality dense annotations of 5000 images
with 19 object classes, and ADE20K is a more challenging
dataset with 150 object classes. There are 20210/2000/3000
images for the training/validation/testing set in ADE20K,
respectively. Following the training protocol of
(Yuan
et al. 2020a), we use random crop, scaling (from 0.5 to 2),
left-right ﬂipping and brightness jittering between −10 and
10 degrees in data augmentation. In multi-scale inference,
we apply scales {0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0} and
{0.5, 0.75, 1.0, 1.25, 1.5, 1.75} as well as their mirrors.
Training parameters. We use stochastic gradient descent
as the optimizer and utilize a "ploy" learning rate policy
similar to Chen et al. (2017b) in the training. Hence, the
initial learning rate is multiplied by (1 − iter
maxiter )power
with power = 0.9. Sync Batch Normalization (Zhang et al.
2018) is used in all our experiments to improve stability.
The detailed training and testing settings for ADE20K and
Cityscapes are as follows:
• ADE20K:
initial
learning rate = 0.02, weight decay = 0.0001, crop size =
520 × 520, batch size = 16, and 150k training iterations,
which are the same as the setting in Yuan et al. (2020a).

the parameters are set as follows:

V0W0W1W2W3W4U0U1U2U3U4V1V2V3V4U0V0W0W1U1U2W2V1V2KL(P(V1) ,P(V2))KL(P(V1) , P(other)KL(P(V2) ,P(V3))KL(P(V2), P(other))PDBGTBDistance123450W1U1U2W2U3W3V1V2V3loss

CE
CE + ABL(20%)
CE + IoU
CE + IoU + BL
CE + IABL
CE + IABL w/o detach

mIoU

79.5
79.6
80.2
80.2
80.5
76.0

pixAcc

96.3
96.3
96.3
96.4
96.4
95.6

Table 1: The inﬂuence of loss terms on Cityscapes dataset.
These experiments are conducted using DeeplabV3 net-
work and single-scale inference. ABL(20%): addin addi-
tional ABL in the last 20% epochs.

Method

loss

OCR (2020a)
[HRNetV2
-W48 (2019)]

CE
CE + IoU
CE + IABL
CE + IFKL

mIoU

pixAcc

SS

MS

SS

MS

44.51 45.66 81.66 82.20
44.73 46.54 81.52 82.29
45.38 46.88 81.63 82.43
45.11 45.96 81.61 82.28

UperNet (2018)
[Swin-T (2021)]

CE
44.51 45.81 81.09 81.96
CE + IoU
45.39 47.15 81.20 82.22
45.98 47.58 81.34 82.39
CE + IABL
CE + IoU + BL 45.72 47.50 81.33 82.39

Table 2: The inﬂuence of loss terms on ADE20K dataset.
SS: single-scale inference. MS: multi-scale inference.

the parameters are set as follows:

• Cityscapes:
initial
learning rate = 0.01 or 0.04, crop size = 512 × 1024 (used
in OCR) or 769 × 769 (used in DeeplabV3), weight decay
= 0.0005, batch size = 8, and 80K training iterations. The
parameter setting is the same as the setting in Yuan et al.
(2020b), but we do not use coarse data in experiments.

Evaluation Metrics. Three metrics, i.e. pixel accuracy (pix-
Acc), mean Intersection-over-Union (mIoU), and boundary
F-score (Perazzi et al. 2016a; Yuan et al. 2020b), are used
to demonstrate the performance of the ABL. The ﬁrst two
metrics are used to evaluate the pixel-level and region-level
accuracy of a segmentation result, respectively. Boundary
F-scores are used to measure the quality of boundary align-
ment and computed within the area of the dilated GTBs. The
dilation parameters are set to 1, 3, 5 pixels in our implemen-
tation. To better preserve boundary details in the evaluation,
we do not use resize operation in the testing.
Combination of loss terms. To ease the description of the
ablation study, we denote different combinations of loss
terms used in the training as follows: CE = cross-entropy;
CE+IoU = cross-entropy + lov´asz-softmax; CE+IABL =
cross-entropy + lov´asz-softmax + ABL. wa is set to 1.0 for
ADE20K dataset but 1.5 for Cityscapes dataset, since train-
ing images' resolution is much larger for Cityscapes.

In addition, we rely on the KL divergence of the class
probability distributions of adjacent pixels, which can be
viewed as the pair-wise term used in condition random
ﬁeld (Lafferty et al. 2001). Hence, it is necessary to verify
how simply enforcing the KL divergence loss at each edge
of an image works in the image segmentation, i.e. enforcing
the loss for each edge between a pair of adjacent pixels, not
only at semantic boundaries. To this end, we deﬁne a full
KL-divergence (FKL) loss as follows:

FKL = 1
Ne

(cid:80)

e BCE(

1
KL(Pei

,Pej

1+e

) , (Gei (cid:54)= Gej )),

(7)

where e denotes an image edge that connects a pair of pixels
ei and ej, Ne is the total number of edges in an image, and

Method Backbone

OS mIoU pixAcc

CPN (2020) ResNet-101+CPL

PyConv (2020)
ACNet(2019b) ResNet-101+MG
DNL (2020) HRNetV2-W48

PyConvResNet-101

OCR (2020a) HRNetV2-W48
OCR+IABL HRNetV2-W48

UperNet (2018)
UperNet+IABL

Swin-B (2021)
Swin-B

8× 46.27 81.85
8× 44.58 81.77
4× 45.90 81.96
4× 45.82

-

4× 45.66 82.20
4× 46.88 82.43

4× 51.66 84.06
4× 52.40 84.11

Table 3: Results on ADE20K validation set. OS: Output
stride. All results are obtained using multi-scale inference.

Method Backbone

GSCNN (2019) WideResNet-38+ASPP
DANet (2019a) ResNet-101+MG
ACNet (2019b) ResNet-101+MG

OCR (2020a) HRNetV2-W48
OCR+IABL HRNetV2-W48

OS

8×
8×
4×

4×
4×

mIoU

80.8
81.5
82.0

82.2
82.9

Table 4: Results on Cityscapes validation set. OS: Output
stride. All results are obtained using multi-scale inference.

(Gei
(cid:54)= Gej ) returns 1 if the ground-truth label of pixel
ei is not equal to the label of ej, otherwise 0. If the FKL
loss is used with cross-entropy and lov´asz-softmax loss in
the training, we denote this combination as CE+IFKL.

Ablation Studies

Loss terms. We ﬁrst test the inﬂuence of loss terms on the
Cityscapes validation dataset by re-training the DeepLabV3
network and show the results in Tab. 1. Since the gradient
of ABL is not useful when PDBs are far from the GTBs,
adding ABL at the beginning of the training does not im-
prove network performance. Thus, we start to add ABL at
the last 20% epochs to verify its effect, but only obtain a
0.1% improvement over mIoU. Then, we re-train the net-
work with CE+IoU and CE+IABL. It shows that adding
ABL to CE+IoU in training can increase the mIoU by 0.3%,
and the combination of IoU loss and ABL, i.e. CE+IABL
contribute 1% improvement on mIoU in this study. Although
the ABL does not contribute most to mIoU in this case, we
do see the obvious improvement of boundary alignment in
qualitative comparisons. In Tab. 2, we test the contribution
of each loss term on ADE20K dataset by re-training the
OCR network. While the mIoU and pixel accuracy can both
be improved after adding IoU loss and ABL, CE+IABL con-
tributes most of the improvement to mIoU by around 0.65%
over CE+IoU in the single-scale inference, which veriﬁes
the contribution of the proposed ABL in this experiment. We
argue that the ABL can contribute more to a dataset with a
large number of semantic classes and hence more GTBs. For
instance, ADE20K has 150 classes, while Cityscapes only
has 19 classes. More GTBs give the ABL more space to ad-
just the network's behavior.
Detaching operation. We verify the effectiveness of the de-
taching operation in Tab. 1. Signiﬁcant drops of pixel accu-
racy and mIoU can be observed when training without the
aforementioned detaching operation to suppress conﬂicts.
Hence, it is important to control the gradient ﬂow when there
exist contradictory targets for KL divergence between two
neighboring pixels.
FKL loss. In Tabs. 2 and 7, it can be seen that the com-
bination of IoU loss and FKL, denoted by IFKL in the 3rd

Figure 4: Progressive reﬁnement of boundary details in the training. Dataset: Cityscapes. Network: DeepLabV3. The input
image is taken from the Cityscapes training set as an example. The GTBs are in blue and PDBs in red.

s
e
p
a
c
s
y
t
i

C

K
0
2
E
D
A

5
2
#

6
2
#

7
2
#

RGB

GT

CE

CE+IoU

CE+IABL

Figure 5: Qualitative results taken from Cityscapes and
ADE20K validation set.

row, can also improve the pixel accuracy and mIoU quan-
titatively. However, CE+IFKL does not perform as well as
CE+IABL. We speculate that it is because FKL treats every
pixel equally, while the ABL pays more attention to the pix-
els on the PDB. Such design allows the network to adjust
its behavior in a progressive way, avoiding over-conﬁdent
decisions when updating the network parameters.
Boundary pixels number threshold. We evaluate the inﬂu-
ence of different thresholds on the Cityscapes validation set
with the FCN [backbone: HRNetV2-W18s]. The results are
as follows: mIoU 75.59% using 1% threshold, 75.46% using
2%, and 75.41% using 0.5%. This empirically veriﬁes that
our choice of 1% threshold is reasonable.
The degree of ABL's dependence on IoU loss. To evaluate
ABL's contribution further, we design an IoU weight decay
experiment, which linearly decreases the weight of IoU loss
from 1 to 0 during training but increase the weight of ABL
from 0 to 1. It achieves mIoU 75.65% on the Cityscapes
validation set with the FCN [backbone: HRNetV2-W18s],
comparable to the mIoU 75.59% trained with CE+IoU+ABL
without weight decay. It can be seen that the decreased IoU
weight does not lead to the downgrade of segmentation per-
formance. Moreover, we do observe that ABL can reﬁne the
semantic boundary for thin structures and complex bound-
aries, as shown in Tab. 6 and Figs. 4–6.

Quantitative Evaluation
Results on ADE20K and Cityscapes validation sets. In
Tabs. 3 and 4, we show that training with IABL along with
cross-entropy loss can improve the pixel accuracy and mIoU
over state-of-the-art image segmentation networks. As for
the ADE20K dataset, training the OCR network (Cordts
et al. 2016) with additional IABL improves the mIoU
and pixel accuracy by 1.22% and 0.23% over that trained
with cross-entropy only on the validation set. Not only ef-

GT

STM

FT CE+IoU

FT CE+IABL

scooter-black (43 frames in total)
Figure 6: Qualitative results taken from DAVIS-2016 valida-
tion set. VOS Network: STM (Oh et al. 2019). FT: STM ﬁne-
tuning. #N: frame number. The boundary details are more
accurate after ﬁne-tuning with additional loss IABL.

fective on CNN-based network, additional IABL supervi-
sion on Transformer-based network UperNet (Xiao et al.
2018) [backbone: SwinTransformer-B (Liu et al. 2021)] also
brings an improvement in the mIoU by 0.74% on the vali-
dation set, which ranks the ﬁrst place in this table (Tab. 3,
last row). Similarly, for the Cityscape dataset, training the
OCR network with IABL improves the mIoU by 0.7% on
the validation set.
Comparison with Segﬁx. We compare our method with
Segﬁx (Yuan et al. 2020b) on the Cityscapes validation set
by using mIoU and boundary F-score metrics, since both
methods focus on improving boundary details in semantic
segmentation. In Tab. 5, our method achieves comparable
performance when using DeepLabV3 as the segmentation
network, but improves the mIoU over Segﬁx by 0.3% when
using the OCR network. In Tab. 6, we show the class-wise
boundary F-scores of Segﬁx and our method. The scores are
computed using the GTB dilation parameters 1, 3, and 5 pix-
els. While Segﬁx outperforms in the cases of 1 pixel, our
method achieves a higher score in the cases of 3, 5 pixels.

Segﬁx is an elegant boundary reﬁnement solution that
propagates the interior labels to class boundaries. However,
the propagation operation might downgrade the segmenta-
tion performance for thin objects that contain a small num-
ber of interior pixels. In contrast, the ABL is an end-to-end
training loss that encourages the alignment of PDBs and
GTBs, which achieves better mIoU and boundary F-scores,
even for thin structures. Taking the class of trafﬁc light as an
example (Tab 6, 9th column), our method achieves a consis-
tent improvement of the boundary F-score over Segﬁx in all
parameter settings, which shows that our method can han-
dle boundaries of thin objects well. Since Segﬁx is a post-
processing method, it can also be used to improve the seg-

Network With IABLRGBGTCE  80k iterationsCE+IABL  24k iterationsCE+IABL  40k iterationsCE+IABL  80k iterationsCE + IABLCE  80k iterations [Boundary]CE+IABL  80k iterations [Boundary]road

method

fence pole

terr-
ian
66.3
66.7
64.6
67.2
67.6
69.0
Table 5: Class-wise mIoU results obtained using single-scale inference on the Cityscapes validation set. +Segifx indicates that
Segﬁx is use to reﬁne the baseline output. +IABL indicates that the network is trained with additional loss IABL.

train moto-
rcycle
67.4
80.0
68.4
80.7
70.6
83.3
67.7
85.0
68.6
85.5
69.3
86.6

buil-
ding wall
63.9
93.1
64.6
93.5
59.9
93.1
62.2
93.3
62.6
93.5
63.9
93.6

DeepLabV3
+Segﬁx
+IABL
OCR
+Segﬁx
+IABL

79.5
80.5 (↑1.0)
80.5 (↑1.0)
81.1
81.7 (↑0.6)
82.0 (↑0.9)

trafﬁc
sign
80.0
82.4
82.3
82.0
83.3
83.2

vege-
tation
92.8
93.2
92.7
93.0
93.3
93.2

trafﬁc
light
72.2
74.9
75.4
73.9
75.7
76.9

bicy-
cle
78.4
79.7
80.2
79.2
80.0
80.8

side-
walk
86.5
87.1
84.9
86.8
87.3
86.7

per-
son
83.3
84.9
84.6
84.3
85.2
85.6

74.5
75.0
79.9
82.9
83.3
80.0

66.1
69.0
68.3
70.0
71.4
70.2

62.6
63.1
63.1
66.0
66.4
64.9

98.4
98.5
98.1
98.4
98.5
98.3

65.5
66.9
69.4
66.3
67.2
70.7

95.3
95.8
95.6
95.6
96.0
96.0

95.0
95.3
95.0
95.0
95.3
95.3

89.0
89.6
87.8
91.7
92.2
93.3

truck bus

mean

rider

sky

car

1px

road

scale

fence pole

terr-
ian
51.3
52.9
52.3
58.6
60.0
60.1
62.1
63.1
63.3
Table 6: Class-wise Boundary F-score results obtained using multi-scale inference on the Cityscapes validation set.

train moto-
rcycle
77.3
96.8
97.0
78.4
79.5
96.4
80.4
97.4
97.5
81.1
83.1
97.0
97.7
81.9
97.7
82.3
84.6
97.2

buil-
ding wall
56.8
57.2
58.1
59.3
53.7
57.4
62.5
75.7
63.0
76.4
59.4
75.4
65.0
82.3
65.1
82.5
61.7
81.8

65.4
67.3 (↑1.9)
67.0 (↑1.6)
77.5
78.1 (↑0.6)
78.9 (↑1.4)
81.4
81.5 (↑0.1)
82.5 (↑1.1)

OCR
+Segﬁx
+IABL
OCR
+Segﬁx
+IABL
OCR
+Segﬁx
+IABL

trafﬁc
sign
60.6
64.1
65.1
78.9
79.3
81.9
82.6
82.3
84.9

vege-
tation
55.6
57.7
55.7
76.4
77.3
76.5
84.4
84.6
84.3

trafﬁc
light
64.8
67.8
74.7
77.8
78.5
86.9
81.7
81.8
89.9

bicy-
cle
56.1
58.6
61.5
71.3
72.3
76.9
77.7
77.8
82.5

side-
walk
50.2
52.6
51.0
70.1
71.0
70.1
76.4
76.9
76.4

per-
son
56.3
59.3
60.1
73.9
74.5
77.6
78.9
78.7
82.2

61.0
64.2
62.1
79.6
79.7
80.7
82.8
82.6
83.8

74.1
76.0
74.6
86.5
87.2
86.2
90.3
90.6
89.8

54.6
55.5
50.8
60.1
60.7
56.3
62.9
63.0
58.6

65.7
67.1
66.8
82.9
83.5
84.1
88.1
88.2
89.1

65.9
67.6
68.9
76.7
77.4
80.4
80.8
81.1
84.1

64.7
68.7
67.0
84.6
85.9
86.1
89.5
90.1
90.6

84.3
84.7
84.6
86.5
86.6
87.2
87.5
87.3
88.1

89.8
89.8
90.3
92.8
92.3
93.2
93.7
93.1
94.0

truck bus

mean

rider

3px

5px

sky

car

Fine-tuning No
CE
Loss

YES
CE

YES

YES
CE+IoU CE+IFKL CE+IABL

YES

J -mean
F-mean

88.67 88.81
89.86 90.25

89.08
90.66

89.08
90.63

89.29
90.82

Table 7: VOS results on DAVIS-2016 (Perazzi et al. 2016b)
validation set. VOS network: STM. The deﬁnition of region
similarity metric J -mean and contour accuracy F-mean can
be found in the DAVIS-2016 dataset (Perazzi et al. 2016b).

mentation results of our method.
Comparison with Boundary Loss. We train ABL + gen-
eralized Dice loss (GDL) (Sudre et al. 2017) on the white
matter hyperintensities (WMH) dataset with the same net-
work architecture and training parameters as (Kervadec et al.
2019) for a fair comparison. We also use the same learning
rate 0.001, batch size 8, training epochs 200, and loss con-
junction method: Loss = α ∗ GDL + (1 − α) ∗ ABL,
where α linearly decreases from 1 to 0.01. In Tab. 8, we
show that training with ABL + GDL achieves higher dice
similarity coefﬁcient (DSC) and smaller Hausdorff distance
(HD) than Boundary Loss (BL) + GDL. Moreover, we ex-
tend BL to a multiple-class loss and make a further compari-
son on Cityscapes validation set. In Tabs. 1 and 2, IoU+ABL
archieves higher mIoU than IoU+BL.

The motivation of BL is to minimize the distance between
GTBs and PDBs. With the geo-cuts optimization techniques
(Boykov et al. 2006), this problem is converted to minimize
the regional integral. This behavior will weaken the inﬂu-
ence of pixels near GTBs since the distance weights there are
much smaller, and the ratio of these pixels is small compared
to the image size. In contrast, ABL focuses on PDB pix-
els, which can achieve better alignment. BL needs to work
with a region-based IoU loss GDL, to avoid making the net-
work collapse quickly into empty foreground classiﬁcation
results. Similarly, we use ABL and IoU loss together.
VOS results. We ﬁne-tune the state-of-the-art VOS network
STM (Oh et al. 2019) with our loss to verify that our method
can also be applied to VOS. Speciﬁcally, the STM is ﬁne-
tuned for 1k iterations with batch size 4 on both DAVIS-
2016 (Perazzi et al. 2016b) and YouTube-VOS (Xu et al.
2018b) training data. The learning rate is set to 5e−8, and
the weight wa is set to 5.0. In Tab. 7, it can be seen that
ﬁne-tuning with addition IABL can improve region similar-

Loss

GDL GDL+BL GDL+ABL

DSC
HD(mm)

0.727
1.045

0.748
0.987

0.768
0.980

Table 8: BL vs. ABL on WMH validation set.

ity metric J -mean and contour accuracy F-mean by around
0.7% and 1%, respectively, when testing on Davis-2016 val-
idation set. Similar to image segmentation, training with
CE+IABL can improve over CE+IoU loss, which also veri-
ﬁes ABL's contribution in VOS. However, adding FKL loss
does not show superior performance, as shown in the 5th col-
ume of Tab. 7.

Qualitative Results
Fig. 4 illustrates the progressive reﬁnement of boundary
details when using IABL as the additional training loss.
This result is obtained when training DeepLabV3 on the
Cityscapes dataset. It can be seen that the PDBs (red lines)
of the trafﬁc light and other objects are pushed toward the
GTBs (blue lines). In Figs. 1 and 5, we show how adding
loss terms inﬂuences the quality of semantic boundaries.
The results show that the proposed ABL can greatly im-
prove the semantic boundary details. Fig. 6 illustrates the
improved boundary details when ﬁne-tuning STM with ad-
ditional IABL. It also shows that ﬁne-tuning with CE+IABL
can further improve the boundary details over CE+IoU, such
as the tail of motorcycle.

Conclusion
In this work, we proposed an active boundary loss to be used
in the end-to-end training of segmentation networks. Its ad-
vantage is that it allows the propagation of the ground-truth
boundary information using a distance transform so as to
regulate the network behavior at predicted boundaries. We
have demonstrated that integrating the ABL into the network
training can substantially improve the boundary details in
semantic segmentation.

In the future, it would be interesting to investigate how
to reduce conﬂicts in our loss to further control the network
behavior around boundaries efﬁciently. In addition, we plan
to explore how to design boundary-aware loss to improve
the boundary details in the task of depth prediction.

Acknowledgments
We thank the reviewers for their constructive comments.
Weiwei Xu is partially supported by NSFC (No. 61732016).

References
Berman, M.; Triki, A. R.; and Blaschko, M. B. 2018. The
Lov´asz-Softmax Loss: A Tractable Surrogate for the Opti-
mization of the Intersection-Over-Union Measure in Neural
Networks. In IEEE Conf. Comput. Vis. Pattern Recog., 4413–
4421.
Bertasius, G.; Shi, J.; and Torresani, L. 2016. Semantic seg-
mentation with boundary neural ﬁelds. In IEEE Conf. Com-
put. Vis. Pattern Recog., 3602–3610.
Bertasius, G.; Torresani, L.; Yu, S. X.; and Shi, J. 2017. Con-
volutional random walk networks for semantic image seg-
mentation. In IEEE Conf. Comput. Vis. Pattern Recog., 858–
866.
Borse, S.; Wang, Y.; Zhang, Y.; and Porikli, F. 2021. Inverse-
Form: A Loss Function for Structured Boundary-Aware Seg-
mentation. In CVPR, 5901–5911.
Boykov, Y.; Kolmogorov, V.; Cremers, D.; and Delong, A.
2006. An integral solution to surface evolution PDEs via geo-
cuts. In ECCV, 409–422. Springer.
Caelles, S.; Maninis, K.-K.; Pont-Tuset, J.; Leal-Taix´e, L.;
Cremers, D.; and Van Gool, L. 2017. One-shot video object
In IEEE Conf. Comput. Vis. Pattern Recog.,
segmentation.
221–230.
Chen, L.; Papandreou, G.; Schroff, F.; and Adam, H. 2017a.
Rethinking Atrous Convolution for Semantic Image Segmen-
tation. arXiv:1706.05587.
Chen, L.-C.; Barron, J. T.; Papandreou, G.; Murphy, K.; and
Yuille, A. L. 2016. Semantic image segmentation with task-
speciﬁc edge detection using cnns and a discriminatively
trained domain transform. In IEEE Conf. Comput. Vis. Pat-
tern Recog., 4545–4554.
Chen, L.-C.; Papandreou, G.; Kokkinos, I.; Murphy, K.; and
Yuille, A. L. 2017b. Deeplab: Semantic image segmenta-
tion with deep convolutional nets, atrous convolution, and
fully connected crfs. IEEE Trans. Pattern Anal. Mach. Intell.,
40(4): 834–848.
Chen, X.; Lian, Y.; Jiao, L.; Wang, H.; Gao, Y.; and Lingling,
S. 2020. Supervised Edge Attention Network for Accurate
Image Instance Segmentation.
Cordts, M.; Omran, M.; Ramos, S.; Rehfeld, T.; Enzweiler,
M.; Benenson, R.; Franke, U.; Roth, S.; and Schiele, B. 2016.
The cityscapes dataset for semantic urban scene understand-
ing. In IEEE Conf. Comput. Vis. Pattern Recog., 3213–3223.
Ding, H.; Jiang, X.; Liu, A. Q.; Thalmann, N. M.; and Wang,
G. 2019. Boundary-aware feature propagation for scene seg-
mentation. In Int. Conf. Comput. Vis., 6819–6829.
Ding, H.; Jiang, X.; Shuai, B.; Qun Liu, A.; and Wang, G.
2018. Context contrasted feature and gated multi-scale ag-
In IEEE Conf. Comput.
gregation for scene segmentation.
Vis. Pattern Recog., 2393–2402.
Duta, I. C.; Liu, L.; Zhu, F.; and Shao, L. 2020. Pyramidal
Convolution: Rethinking Convolutional Neural Networks for
Visual Recognition. arXiv:2006.11538.

Fu, J.; Liu, J.; Tian, H.; Li, Y.; Bao, Y.; Fang, Z.; and Lu, H.
2019a. Dual attention network for scene segmentation.
In
IEEE Conf. Comput. Vis. Pattern Recog., 3146–3154.
Fu, J.; Liu, J.; Wang, Y.; Li, Y.; Bao, Y.; Tang, J.; and Lu, H.
2019b. Adaptive context network for scene parsing. In Int.
Conf. Comput. Vis., 6748–6757.
Gong, K.; Liang, X.; Li, Y.; Chen, Y.; Yang, M.; and Lin, L.
2018.
Instance-level human parsing via part grouping net-
work. In Eur. Conf. Comput. Vis., 770–785.
He, K.; Zhang, X.; Ren, S.; and Sun, J. 2016. Deep residual
learning for image recognition. In IEEE Conf. Comput. Vis.
Pattern Recog., 770–778.
Hu, Y.-T.; Huang, J.-B.; and Schwing, A. 2017. Maskrnn:
Instance level video object segmentation. In Adv. Neural In-
form. Process. Syst., 325–334.
Kass, M.; and Witkin, A. 1988. Snakes: Active contour mod-
els. Int. J. Comput. Vis., 1: 321–331.
Ke, T.-W.; Hwang, J.-J.; Liu, Z.; and Yu, S. X. 2018. Adap-
tive afﬁnity ﬁelds for semantic segmentation. In Eur. Conf.
Comput. Vis., 587–602.
Kervadec, H.; Bouchtiba, J.; Desrosiers, C.; Granger, E.;
Dolz, J.; and Ayed, I. B. 2019. Boundary loss for highly un-
balanced segmentation. In International conference on medi-
cal imaging with deep learning, 285–296.
Kr¨ahenb¨uhl, P.; and Koltun, V. 2011. Efﬁcient Inference in
Fully Connected CRFs with Gaussian Edge Potentials.
In
Advances in Neural Information Processing Systems 24: 25th
Annual Conference on Neural Information Processing Sys-
tems 2011. Proceedings of a meeting held 12-14 December
2011, Granada, Spain, 109–117.
Lafferty, J. D.; McCallum, A.; and Pereira, F. C. 2001. Con-
ditional Random Fields: Probabilistic Models for Segmenting
and Labeling Sequence Data. In ICML, 282–289.
Li, Y.; Qi, H.; Dai, J.; Ji, X.; and Wei, Y. 2017. Fully convolu-
tional instance-aware semantic segmentation. In IEEE Conf.
Comput. Vis. Pattern Recog., 2359–2367.
Liu, Z.; Lin, Y.; Cao, Y.; Hu, H.; Wei, Y.; Zhang, Z.; Lin,
S.; and Guo, B. 2021. Swin Transformer: Hierarchical Vision
Transformer using Shifted Windows. CoRR, abs/2103.14030.
Long, J.; Shelhamer, E.; and Darrell, T. 2015. Fully convo-
lutional networks for semantic segmentation. In IEEE Conf.
Comput. Vis. Pattern Recog., 3431–3440.
Minaee, S.; Boykov, Y. Y.; Porikli, F.; Plaza, A. J.; Kehtar-
navaz, N.; and Terzopoulos, D. 2021.
Image segmentation
using deep learning: A survey. PAMI.
Oh, S. W.; Lee, J.-Y.; Xu, N.; and Kim, S. J. 2019. Video
object segmentation using space-time memory networks. In
Int. Conf. Comput. Vis., 9226–9235.
Perazzi, F.; Pont-Tuset, J.; McWilliams, B.; Van Gool, L.;
Gross, M.; and Sorkine-Hornung, A. 2016a. A benchmark
dataset and evaluation methodology for video object segmen-
tation. In IEEE Conf. Comput. Vis. Pattern Recog., 724–732.
Perazzi, F.; Pont-Tuset, J.; McWilliams, B.; Van Gool, L.;
Gross, M.; and Sorkine-Hornung, A. 2016b. A Benchmark
Dataset and Evaluation Methodology for Video Object Seg-
mentation. In IEEE Conf. Comput. Vis. Pattern Recog.

Yuan, Y.; Chen, X.; and Wang, J. 2020a. Object-Contextual
Representations for Semantic Segmentation. In ECCV, vol-
ume 12351 of Lecture Notes in Computer Science, 173–190.
Springer.
Yuan, Y.; Xie, J.; Chen, X.; and Wang, J. 2020b. Segﬁx:
Model-agnostic boundary reﬁnement for segmentation.
In
Eur. Conf. Comput. Vis., 489–506. Springer.
Zhang, H.; Dana, K.; Shi, J.; Zhang, Z.; Wang, X.; Tyagi,
A.; and Agrawal, A. 2018. Context encoding for semantic
In IEEE Conf. Comput. Vis. Pattern Recog.,
segmentation.
7151–7160.
Zhou, B.; Zhao, H.; Puig, X.; Fidler, S.; Barriuso, A.; and
Torralba, A. 2017. Scene parsing through ade20k dataset. In
IEEE Conf. Comput. Vis. Pattern Recog., 633–641.
Zhu, F.; Zhu, Y.; Zhang, L.; Wu, C.; Fu, Y.; and Li, M. 2021.
A Uniﬁed Efﬁcient Pyramid Transformer for Semantic Seg-
mentation. In Int. Conf. Comput. Vis., 2667–2677.
Zhu, Y.; Sapra, K.; Reda, F. A.; Shih, K. J.; Newsam, S.; Tao,
A.; and Catanzaro, B. 2019. Improving semantic segmenta-
tion via video propagation and label relaxation. In IEEE Conf.
Comput. Vis. Pattern Recog., 8856–8865.

Ronneberger, O.; Fischer, P.; and Brox, T. 2015. U-net: Con-
volutional networks for biomedical image segmentation. In
International Conference on Medical image computing and
computer-assisted intervention, 234–241. Springer.
Ruan, T.; Liu, T.; Huang, Z.; Wei, Y.; Wei, S.; and Zhao, Y.
2019. Devil in the details: Towards accurate single and mul-
tiple human parsing. In AAAI, volume 33, 4814–4821.
Shin Yoon, J.; Rameau, F.; Kim, J.; Lee, S.; Shin, S.; and
So Kweon, I. 2017. Pixel-level matching for video object seg-
mentation using convolutional neural networks. In Int. Conf.
Comput. Vis., 2167–2176.
Su, J.; Li, J.; Zhang, Y.; Xia, C.; and Tian, Y. 2019. Selectivity
or invariance: Boundary-aware salient object detection. In Int.
Conf. Comput. Vis., 3799–3808.
Sudre, C. H.; Li, W.; Vercauteren, T.; Ourselin, S.; and Car-
doso, M. J. 2017. Generalised Dice Overlap as a Deep Learn-
ing Loss Function for Highly Unbalanced Segmentations. In
Deep learning in medical image analysis and multimodal
learning for clinical decision support., volume 10553, 240–
248. Springer, Cham.
Szegedy, C.; Vanhoucke, V.; Ioffe, S.; Shlens, J.; and Wojna,
Z. 2016. Rethinking the inception architecture for computer
In IEEE Conf. Comput. Vis. Pattern Recog., 2818–
vision.
2826.
Takikawa, T.; Acuna, D.; Jampani, V.; and Fidler, S. 2019.
Gated-scnn: Gated shape cnns for semantic segmentation. In
Int. Conf. Comput. Vis., 5229–5238.
Voigtlaender, P.; Chai, Y.; Schroff, F.; Adam, H.; Leibe, B.;
and Chen, L.-C. 2019. Feelvos: Fast end-to-end embedding
learning for video object segmentation. In IEEE Conf. Com-
put. Vis. Pattern Recog., 9481–9490.
Wang, J.; Sun, K.; Cheng, T.; Jiang, B.; Deng, C.; Zhao, Y.;
Liu, D.; Mu, Y.; Tan, M.; Wang, X.; Liu, W.; and Xiao, B.
2019. Deep High-Resolution Representation Learning for Vi-
sual Recognition. TPAMI.
Xiao, T.; Liu, Y.; Zhou, B.; Jiang, Y.; and Sun, J. 2018. Uni-
ﬁed Perceptual Parsing for Scene Understanding. In ECCV.
Springer.
Xu, D.; Ouyang, W.; Wang, X.; and Sebe, N. 2018a. Pad-net:
Multi-tasks guided prediction-and-distillation network for si-
In IEEE
multaneous depth estimation and scene parsing.
Conf. Comput. Vis. Pattern Recog., 675–684.
Xu, N.; Yang, L.; Fan, Y.; Yang, J.; Yue, D.; Liang, Y.; Price,
B.; Cohen, S.; and Huang, T. 2018b. Youtube-vos: Sequence-
to-sequence video object segmentation. In Eur. Conf. Com-
put. Vis., 585–601.
Yin, M.; Yao, Z.; Cao, Y.; Li, X.; Zhang, Z.; Lin, S.; and
Hu, H. 2020. Disentangled Non-local Neural Networks. In
Vedaldi, A.; Bischof, H.; Brox, T.; and Frahm, J., eds., Com-
puter Vision - ECCV 2020 - 16th European Conference, Glas-
gow, UK, August 23-28, 2020, Proceedings, Part XV, vol-
ume 12360 of Lecture Notes in Computer Science, 191–207.
Springer.
Yu, C.; Wang, J.; Gao, C.; Yu, G.; Shen, C.; and Sang, N.
2020. Context Prior for Scene Segmentation. In IEEE Conf.
Comput. Vis. Pattern Recog., 12416–12425.
