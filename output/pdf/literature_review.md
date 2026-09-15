# Single-Pixel Imaging: A Comparative Reading<br/>of Compressive Sampling and Proximal Unrolling

Zicong Luo (522026230043), Shuhao Yang (502026230083), Yubowen Li (502026230092)

## 1. Reading Scope

The selected topic is single-pixel imaging and compressed sensing. The paired readings are Duarte et al., Single-Pixel Imaging via Compressive Sampling (IEEE Signal Processing Magazine, 2008), and Wang et al., Proximal Algorithm Unrolling: Flexible and Efficient Reconstruction Networks for Single-Pixel Imaging (CVPR, 2025). This reading report compares their questions and evidence, then derives a feasible course experiment. Numerical findings from our implementation are discussed separately from the papers' claims.

## 2. The Foundational Question

Duarte et al. describe a camera that obtains spatially coded integrated measurements with a digital micromirror device and a single detector. The compressed-sensing formulation links those measurements to a sparse or compressible image representation. The key design decision moves spatial information into the sequence of known patterns, so reconstruction becomes an inverse problem [1].

This changes how the camera should be understood. One detector does not imply one observation, and one observation does not reveal the whole image. The detector count describes hardware parallelism. The measurement count describes the amount of acquired information and contributes to acquisition time. A fair project presentation should keep these quantities distinct. Otherwise, the phrase single-pixel camera can incorrectly suggest that a single scalar determines an arbitrary image.

Our mathematical interpretation starts from y = A x + e. When there are fewer rows than columns, the linear system is underdetermined. If x = D<super>T</super> alpha has a compressible coefficient vector alpha, a reconstruction method can favor solutions with small ||alpha||<sub>1</sub>. This preference supplements the measurements. It is not evidence that every possible image is recoverable at a fixed small sampling rate.

The practical distinction between sensing and compression is also useful. A software simulation computes y from an already available image. It reproduces the inverse-problem arithmetic, but it does not demonstrate the physical benefit of acquiring compressed data optically. For our coursework, that distinction defines an honest scope: compare reconstruction algorithms in simulation and discuss the hardware motivation without claiming an optical prototype.

## 3. The Modern Question

Wang et al. study reconstruction using a learned image restorer shared across a short proximal unrolling trajectory. The method includes HQS and ADMM versions and uses proximal trajectory supervision to guide intermediate outputs. Its proposed restorer combines convolution, attention, and feature memory [2].

The motivation is relevant to a camera with a changing measurement budget. A reconstruction system that works at several ratios with one set of weights is operationally simpler than keeping a different network for each ratio. However, a numerical experiment must still record which ratios appear during training and which appear only at evaluation. Testing several ratios alone does not establish generalization to every unseen sensing operator.

The comparison with classical sparsity is therefore about where the prior comes from. A transform-based prior is explicit and understandable through its penalty. A learned prior acquires image statistics from training data and may represent patterns that a fixed transform handles poorly. Its dependence on training data and the consequences of a distribution shift become part of the evaluation problem.

## 3.1. A Shared Mathematical View

Both reconstruction families can be expressed as a competition between measurement consistency and a preference over images. The measurements constrain the observable component, while the prior resolves the remaining ambiguity. This perspective helps explain why comparing algorithms with different matrices can be misleading: the measurement operator itself may reveal different information before the reconstruction algorithm has any role.

## 4. Code-Informed Critical Reading

The released code provides a concrete interpretation of the modern method. It stores separate height and width sensing factors and applies the operator as H X W<super>T</super>. This avoids materializing a very large dense matrix. The apparent two-dimensional measurement array is a computational representation of the scalar measurement sequence, not a second spatial detector array.

In the HQS inference loop, the implementation first corrects measurement residuals and then calls the restorer. In ADMM, a dual state additionally carries the disagreement between the split variables. The same restorer is reused at successive stages, and memory passes feature information between calls. These observations concern the released implementation used in our experiments.

A potential source of confusion is the original forward function accepting the clean image. Inspection shows that it creates synthetic measurements and also computes target-related training outputs. A deployment API should accept measurements directly. We therefore introduce a separate reconstruction function and compare its output against the original clean path. This check addresses information leakage more directly than assuming an inference routine is correct because it runs.

## 4.1. What a Trajectory Claim Requires

An ideal proximal target can depend on a known clean training image. A network can learn to approximate that target from degraded inputs. The target and the learned approximation must nevertheless remain conceptually separate. Success on a finite test set does not establish an exact proximal representation for arbitrary inputs or a global convergence theorem for an unconstrained neural network.

Our experimental response is to retain every intermediate reconstruction. We can inspect whether quality changes smoothly across the released six stages and whether an intervention disrupts that behavior. This is evidence about a finite computation. Proving asymptotic convergence would require additional assumptions and analysis. Retraining with and without trajectory loss would address a different, causal question that our pretrained evaluation does not answer.

## 4.2. Fairness and Missing Controls

A useful baseline should receive the identical measurement array. We choose an orthonormal DCT sparsity objective and solve it with FISTA, using a spectral step size and parameters selected on disjoint validation images. This provides a clear, reproducible classical comparator. It leaves open comparisons with total variation, wavelet penalties, stronger optimization schedules, and other learned methods.

Training exposure also matters. A pretrained neural network benefits from data and computation that the classical solver does not require. That is an intended difference in the comparison, but it prevents interpreting inference-time numbers as total system cost. The small public image suite may also overlap with training resources; we have not audited the original training set. This uncertainty should accompany the reported performance.

## 5. Experiment Derived from the Readings

The shared inverse model suggests four controlled axes: sampling budget, measurement noise, image content, and stage computation. Our project varies five sampling ratios without changing network weights, adds Gaussian noise directly to the shared measurements, includes periodic patterns and text, and records six stage outputs. Two validation images determine one DCT regularization weight before testing.

The stress targets are especially useful for discussion. A periodic pattern tests the preservation of repeated high-frequency structure. Small text tests whether thin strokes remain distinguishable. An improved global PSNR can coexist with an incorrect character or a missing line. These examples connect mathematical ambiguity to an observable application failure and motivate task-specific evaluation.

## 5.1. Metrics with Explicit Meanings

PSNR measures pixel fidelity under a chosen intensity range, and SSIM describes local structural similarity under a chosen implementation. Neither metric directly measures optical efficiency, detector cost, acquisition speed, or the correctness of a recognized symbol. Timing here measures the reconstruction routine after warmup. The metric definitions and aggregation rules must be specified before using them to support an engineering conclusion.

## 6. Connection to the Implemented Project

The accompanying technical report contains actual results from the released checkpoints and the new baseline. At nominal 10% sampling, HQS achieves 34.31 dB and 0.916 SSIM on the six regular evaluation images. DCT-FISTA achieves 29.10 dB and 0.694 SSIM under the same measurements. These are local experimental results, not transcribed values from either paper.

The run preserves 352 image-level metric records, including the stress targets and seeded noise repetitions. The technical report explains the sampling sweep and presents failure images. The code also records the difference between nominal and realized measurement ratios, since row counts are rounded upward. This bookkeeping prevents a small implementation detail from becoming an unreported mismatch between experimental conditions.

The evidence supports a limited conclusion: a comparison of these reconstruction methods, on these images, with this forward operator and these checkpoints. A stronger project could add independent photographs with documented acquisition conditions, a larger official benchmark, retraining across several random seeds, and an optical measurement calibration. Those extensions would increase external validity, but none is presented here as completed work.

## 7. Questions for the Presentation

<b>Why is the image not determined by one detector reading?</b> Each reading is an inner product with one mask. Multiple masks provide multiple constraints. The word single refers to the detector rather than the number of measurements.

<b>Does sparse recovery require the image itself to be mostly zero?</b> No. The intended assumption concerns a representation, such as transform coefficients or gradients. The chosen representation affects which image structures the prior favors.

<b>Why must measurement matrices be shared?</b> Changing the operator can change the information content of the data. Shared operators and identical noise realizations make reconstruction differences easier to attribute to the solver.

<b>Does feature-memory removal prove that memory is necessary?</b> It measures how a trained checkpoint reacts to removal. A controlled architecture ablation would retrain comparable models under the same training protocol.

<b>Why not call the results a complete reproduction of CVPR 2025?</b> We reproduce the released inference computation but use a smaller, explicitly named evaluation suite. We do not reproduce full training, the official dataset protocol, every competing method, or the physical-camera experiments.

<b>What explains a convincing-looking failure?</b> The measurement system leaves ambiguity, and the prior selects one possible image. Visual plausibility alone does not establish that a detail matches the unknown scene.

## 8. Reflection and AI Disclosure

The most useful connection between the readings is the separation of acquired information from prior knowledge. The foundational formulation explains why reconstruction is possible under structural assumptions. The modern method shows how to encode stronger image preferences in a finite learned computation. A careful experiment evaluates both the resulting quality and the conditions under which those preferences fail.

Codex assisted with source inspection, implementation, experiment execution, analysis, English writing, and presentation preparation. The pretrained network and weights remain the work of their cited authors. The named student team should review this report and the associated code, confirm the supplied identity information, and prepare to explain the forward model and experimental limitations. No unverified member-specific contribution is assigned.

## References and Primary Sources

[1] M. F. Duarte, M. A. Davenport, D. Takhar, J. N. Laska, T. Sun, K. F. Kelly, and R. G. Baraniuk. Single-Pixel Imaging via Compressive Sampling. IEEE Signal Processing Magazine 25(2):83-91, 2008. doi:10.1109/MSP.2007.914730. Author-hosted PDF: https://mdav.ece.gatech.edu/publications/ddtlskb-spm-2008.pdf.

[2] P. Wang, L. Wang, G. Qu, X. Wang, Y. Zhang, and X. Yuan. Proximal Algorithm Unrolling: Flexible and Efficient Reconstruction Networks for Single-Pixel Imaging. CVPR, 2025. https://arxiv.org/abs/2505.23180. Official source: https://github.com/pwangcs/ProxUnroll.

[3] A. Beck and M. Teboulle. A Fast Iterative Shrinkage-Thresholding Algorithm for Linear Inverse Problems. SIAM Journal on Imaging Sciences 2(1):183-202, 2009. doi:10.1137/080716542.
