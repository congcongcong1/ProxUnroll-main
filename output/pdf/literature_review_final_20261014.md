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

Our initial response retains every intermediate reconstruction and inspects the released six-stage computation. Later paired experiments compare weighted intermediate direct-ground-truth RMSE with final-only RMSE, using two training seeds and 800 updates per arm. This controls that loss factor under a fixed recipe. It does not reproduce or isolate the paper's ideal proximal-trajectory targets, prove feature-memory necessity, or establish asymptotic convergence.

## 4.2. Fairness and Missing Controls

A useful baseline should receive the identical measurement array. We choose an orthonormal DCT sparsity objective and solve it with FISTA, using a spectral step size and parameters selected on disjoint validation images. This provides a clear, reproducible classical comparator. It leaves open comparisons with total variation, wavelet penalties, stronger optimization schedules, and other learned methods.

Training exposure also matters. A pretrained neural network benefits from data and computation that the classical solver does not require. That is an intended difference in the comparison, but it prevents interpreting inference-time numbers as total system cost. The small public image suite may also overlap with training resources; we have not audited the original training set. This uncertainty should accompany the reported performance.

## 5. Experiment Derived from the Readings

The shared inverse model suggests four controlled axes: sampling budget, measurement noise, image content, and stage computation. Our project varies five sampling ratios without changing network weights, adds Gaussian noise directly to the shared measurements, includes periodic patterns and text, and records six stage outputs. Two validation images determine one DCT regularization weight before testing.

The stress targets are especially useful for discussion. A periodic pattern tests the preservation of repeated high-frequency structure. Small text tests whether thin strokes remain distinguishable. An improved global PSNR can coexist with an incorrect character or a missing line. These examples connect mathematical ambiguity to an observable application failure and motivate task-specific evaluation.

## 5.1. Metrics with Explicit Meanings

PSNR measures pixel fidelity under a chosen intensity range, and SSIM describes local structural similarity under a chosen implementation. Neither metric directly measures optical efficiency, detector cost, acquisition speed, or the correctness of a recognized symbol. Timing here measures the reconstruction routine after warmup. The metric definitions and aggregation rules must be specified before using them to support an engineering conclusion.

## 6. Connection to the Implemented Project

The original small-suite evaluation preserves 352 image-level metrics and 1056 stage metrics. On its six regular images at clean nominal 10% sampling, released HQS achieves 34.31 dB and 0.916 SSIM, while DCT-FISTA gives 29.10 dB and 0.694 SSIM under identical measurements. These historical software results are ours, not values copied from either paper. They cannot be compared directly with scores on a different image collection.

The updated project extends this evidence to Kodak, fixed spaced HEVC raw Y frames, DIV2K, and 30 attributed Nanjing web photographs. Fine-tuning uses 800 DIV2K training originals and 16 disjoint validation originals. The 52-image project holdout was locked before the preceding continuation; the later 20000-update extension reuses it and 110 previously tested regression originals. This later comparison therefore concerns observed inputs. Official pretraining exposure remains unknown, and the web images belong to external photographers.

Full-state HQS continuation reaches 20000 updates with all six sensing factors frozen. The unchanged validation rule selects step 19500, and the 20000 endpoint is retained separately. On the same primary52 inputs at 10% sampling, selected-19500 MSE increases by 0.83% for clean measurements and 0.41% at relative noise .05 versus selected step 11500, while SSIM increases slightly. More updates do not give a further universal 5-10% error reduction. The extension saves 3564 new final floating reconstructions and 21384 stage metrics, with independent metric verification and matched measurement hashes.

The separate two-by-two noise/stage-loss experiment holds initialization, data order, crops, sensing factors and 800-update budgets fixed within each training seed. Adding training noise under weighted loss reduces strong-noise MSE by 8.45% on average across the two seeds, while clean MSE increases by 0.74%. Intermediate supervision has much smaller, condition-dependent effects. These bounded controls support a limited recipe comparison rather than an explanation of all historical training gains. All negative cases remain in the technical report and raw tables.

The current four-page technical report presents these results separately from the papers' claims. Further work could test stronger classical priors, additional training replications and real optical calibration. We have completed software reconstruction and controlled adaptation, without claiming the official paper benchmark, physical acquisition or a temporal video model.

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
