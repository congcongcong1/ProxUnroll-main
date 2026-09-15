"""Build English conference-style reports from measured, completed experiments."""

import json
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfgen import canvas
from reportlab.platypus import Frame, Paragraph, Spacer, Table, TableStyle
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'output/pdf'
FIG = ROOT / 'coursework/figures'
STYLE = ParagraphStyle('body', fontName='Times-Roman', fontSize=10, leading=12.4,
                       spaceAfter=7, alignment=4)
HEAD = ParagraphStyle('heading', parent=STYLE, fontName='Times-Bold', fontSize=11.3,
                      leading=13.5, spaceBefore=6, spaceAfter=5, alignment=0)
SMALL = ParagraphStyle('small', parent=STYLE, fontSize=8.5, leading=10.3, alignment=0)


def p(text):
    return Paragraph(text, STYLE)


def h(text):
    item = Paragraph(text, HEAD)
    item.keep_with_next = True
    return item


def table(rows, widths):
    t = Table([[Paragraph(str(c), SMALL) for c in row] for row in rows], colWidths=widths, hAlign='LEFT')
    t.setStyle(TableStyle([('LINEABOVE', (0, 0), (-1, 0), .6, colors.black),
                          ('LINEBELOW', (0, 0), (-1, 0), .4, colors.black),
                          ('LINEBELOW', (0, -1), (-1, -1), .6, colors.black),
                          ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                          ('LEFTPADDING', (0, 0), (-1, -1), 3),
                          ('RIGHTPADDING', (0, 0), (-1, -1), 3),
                          ('TOPPADDING', (0, 0), (-1, -1), 4),
                          ('BOTTOMPADDING', (0, 0), (-1, -1), 4)]))
    return t


def draw_report(filename, title, pages):
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / filename
    c = canvas.Canvas(str(path), pagesize=(612, 792))
    c.setTitle(title)
    c.setAuthor('Zicong Luo; Shuhao Yang; Yubowen Li')
    markdown = [f'# {title}', 'Zicong Luo (522026230043), Shuhao Yang (502026230083), Yubowen Li (502026230092)']
    for index, page in enumerate(pages, 1):
        c.setFont('Times-Italic', 8)
        c.drawString(42, 767, 'Computational Imaging Course Project | Topic 7')
        c.drawRightString(570, 767, 'Reproducible software study')
        c.setStrokeColor(colors.HexColor('#666666'))
        c.line(42, 760, 570, 760)
        top = 748
        if index == 1:
            title_p = Paragraph(title, ParagraphStyle('title', fontName='Times-Bold', fontSize=17, leading=20, alignment=1))
            _, th = title_p.wrap(528, 100)
            title_p.drawOn(c, 42, top - th)
            top -= th + 12
            for line in ['Zicong Luo (522026230043)   Shuhao Yang (502026230083)',
                         'Yubowen Li (502026230092)']:
                c.setFont('Times-Roman', 9.5)
                c.drawCentredString(306, top, line)
                top -= 12
            top -= 6
        for image_file, caption in page.get('figures', []):
            image_path = FIG / image_file
            iw, ih = Image.open(image_path).size
            height = 528 * ih / iw
            c.drawImage(str(image_path), 42, top - height, 528, height, preserveAspectRatio=True)
            top -= height + 4
            cp = Paragraph(caption, SMALL)
            _, ch = cp.wrap(528, 70)
            cp.drawOn(c, 42, top - ch)
            top -= ch + 9
            markdown.append(f'![{caption}](../../coursework/figures/{image_file})')
        story = []
        for kind, text in page['blocks']:
            if kind == 'heading':
                story.append(h(text))
                markdown.append('## ' + text)
            elif kind == 'table':
                story.append(table(text, page.get('table_widths', [60, 50, 50, 91])))
                story.append(Spacer(1, 8))
                markdown.extend(' | '.join(str(x) for x in row) for row in text)
            else:
                item = p(text)
                item.keep_with_next = text.startswith('Table ')
                story.append(item)
                markdown.append(text)
        heights = [obj.wrap(251, 10000)[1] + obj.getSpaceBefore() + obj.getSpaceAfter() for obj in story]
        valid = [i for i in range(1, len(story)) if not getattr(story[i - 1], 'keep_with_next', False)]
        split = min(valid, key=lambda i: abs(sum(heights[:i]) - sum(heights[i:])))
        for left, column in [(42, story[:split]), (319, story[split:])]:
            frame = Frame(left, 45, 251, top - 45, leftPadding=0, rightPadding=0,
                          topPadding=0, bottomPadding=0, showBoundary=0)
            frame.addFromList(column, c)
            if column:
                raise RuntimeError(f'{filename}: page {index} overflows by {len(column)} blocks')
        c.setFont('Times-Roman', 9)
        c.drawCentredString(306, 27, str(index))
        c.showPage()
    c.save()
    (OUT / filename.replace('.pdf', '.md')).write_text('\n\n'.join(markdown) + '\n')
    print(path)


def main():
    result = ROOT / 'coursework/results'
    if not (result / 'COMPLETE').exists():
        raise RuntimeError('Results are incomplete')
    facts = json.loads((result / 'report_facts.json').read_text())
    meta = json.loads((result / 'provenance.json').read_text())
    def row(method, cr=.1, sigma=0, split='test'):
        return next(r for r in facts['summary'] if r['method'] == method and r['cr'] == cr and r['sigma'] == sigma and r['split'] == split)
    def case(name, method, cr):
        return next(r for r in facts['clean_cases'] if r['image'] == name and r['method'] == method and r['cr'] == cr)
    q = row('hqs')
    a = row('admm')
    f = row('fista_dct')
    ci = facts['paired_hqs_fista_bootstrap_95']
    lam = meta['selected_lambda']
    clean_table = [['CR', 'FISTA', 'HQS', 'ADMM']]
    for cr in [.01, .04, .1, .25, .5]:
        clean_table.append([f'{cr:.0%}', *[f'{row(m, cr)["psnr"]:.2f}' for m in ['fista_dct', 'hqs', 'admm']]])
    runtime_table = [['Method', 'PSNR', 'SSIM', 'Median seconds']]
    for method, name in [('adjoint', 'Adjoint'), ('fista_dct', 'FISTA'), ('hqs', 'HQS'), ('admm', 'ADMM')]:
        r = row(method)
        runtime_table.append([name, f'{r["psnr"]:.2f}', f'{r["ssim"]:.3f}', f'{r["seconds"]:.3f}'])
    pages = [
        dict(blocks=[
            ('heading', 'Abstract'),
            ('text', f'Single-pixel imaging reconstructs a spatial image from integrated optical measurements. This project studies how reconstruction priors affect quality when the number of measurements is limited. We reproduce the released HQS and ADMM variants of ProxUnroll and compare them with adjoint reconstruction and an independently implemented DCT-sparse FISTA solver. All methods receive the same measurements and sensing matrices. The evaluation uses six public grayscale test images, two independent validation images, and two synthetic stress targets at 256 x 256 pixels. Five clean sampling rates and two measurement-noise levels expose both reconstruction gains and failure cases. At nominal 10% sampling, HQS obtains {q["psnr"]:.2f} dB / {q["ssim"]:.3f} SSIM and DCT-FISTA obtains {f["psnr"]:.2f} dB / {f["ssim"]:.3f} SSIM. We additionally examine intermediate stages and disable feature memory at inference. All reported results in this report come from the included run, not from the original paper tables. This is a pretrained-model evaluation and software simulation, with no claim of optical acquisition or training from scratch.'),
            ('heading', '1. Introduction'),
            ('text', 'A conventional image sensor records spatially separated values in parallel. A single-pixel system instead applies a sequence of spatial masks and records one integrated response per mask. Reconstruction then uses the known masks to infer spatial structure. The attraction is particularly clear when a sensitive detector is available but an affordable detector array is not. The computational cost is paid after acquisition, while the required sequence of measurements imposes a temporal cost and a sensitivity to scene motion.'),
            ('text', 'Topic 7 connects the foundational camera and compressed-sensing formulation of Duarte et al. [1] with proximal algorithm unrolling in Wang et al. [2]. The classic formulation supplies the acquisition model and motivates sparsity as a prior. The modern approach introduces a learned image restorer into a short optimization trajectory. Our comparison focuses on the reconstruction component under a controlled acquisition model, rather than attempting to rebuild either paper\'s entire experimental apparatus.'),
            ('text', 'A successful demonstration on one image would leave several questions unresolved. Does the reconstruction advantage survive when the sampling rate changes? Does it depend on using different measurements? How much detail disappears under severe compression? Does a pretrained natural-image prior remain reliable for text and periodic patterns? We translate these questions into repeatable experiments, retaining failures as evidence.'),
            ('heading', '1.1. Contributions and Scope'),
            ('text', 'The project contribution is an audited evaluation package around the released network. We add a measurement-only reconstruction interface and verify its clean-output equivalence to the original implementation. We implement a classical sparse baseline using the identical separable operator, tune its regularization on disjoint validation images, and preserve individual results instead of reporting only averages. We include sampling sweeps, noisy measurements, stage trajectories, an inference-time memory intervention, and difficult image examples.'),
            ('text', 'We use the authors\' pretrained weights without fine-tuning. Reusing a pretrained network is therefore part of the method, and its training cost and training-data exposure remain relevant limitations. The study is neither a new reconstruction architecture nor a reproduction of the full CVPR benchmark. The classical baseline is inspired by the sparse inverse-problem formulation in [1], but it is not claimed to be the exact algorithm or hardware implementation of that paper.'),
            ('heading', '1.2. Evidence and Reproducibility'),
            ('text', 'The project stores image provenance, file hashes, checkpoint hashes, package versions, raw metrics, image reconstructions, and stage-level measurements. Every comparison in the report can be traced to a CSV row. Inputs used for parameter selection are excluded from reported test means. The test suite is small and intentionally diverse; the resulting numbers describe these images and should not be generalized to an entire dataset or application domain.'),
        ]),
        dict(blocks=[
            ('heading', '2. Forward Model and Reconstruction'),
            ('text', 'Let X be a normalized h x w grayscale image. With separable sensing factors H and W, the measurement model is Y = H X W<super>T</super> + E. Vectorization gives y = A x + e with A = W (Kronecker product) H, under column-major vectorization. Each row of A describes a spatial mask. We never construct this dense operator, since two smaller matrix products implement the same linear transformation.'),
            ('text', 'Define A(X) = H X W<super>T</super> and A*(Y) = H<super>T</super> Y W. The second operator is the adjoint, as verified by the inner-product identity &lt;A(X), Y&gt; = &lt;X, A*(Y)&gt;. It maps measurements back to image space but cannot restore unmeasured information on its own. For squared measurement error, the gradient is A*(A(X) - Y). This common expression underlies both our classical optimizer and the network\'s data-consistency update.'),
            ('text', 'The repository requests ceil(h sqrt(r)) rows from H and ceil(w sqrt(r)) rows from W for nominal ratio r, capped by stored factor sizes. The realized ratio is r_actual = m_h m_w / (hw), using the available selected rows. We record both ratios. At 256 x 256, nominal 1% contains 26 x 26 measurements (1.0315%). The released factors contain only 181 rows each, so nominal 50% uses 181 x 181 measurements (49.9893%), rather than the requested 182 x 182.'),
            ('heading', '2.1. Adjoint and DCT-FISTA Baselines'),
            ('text', 'Adjoint reconstruction returns X0 = A*(Y). It provides a useful reference for how much quality comes directly from the measurement model. The sparse baseline solves min_X 0.5 ||A(X) - Y||<super>2</super><sub>F</sub> + lambda ||D(X)||<sub>1</sub>, where D is an orthonormal two-dimensional discrete cosine transform. This objective encourages a compressible frequency representation while keeping the measurements consistent.'),
            ('text', 'Starting from X0, FISTA [3] alternates a gradient step and soft thresholding in the DCT domain, with its standard momentum update. The threshold is lambda/L, where L = ||H||<sub>2</sub><super>2</super> ||W||<sub>2</sub><super>2</super>. This spectral bound avoids assuming exact row orthogonality. The inverse orthonormal DCT converts thresholded coefficients back into an image. We run 200 iterations and select one global lambda from a small validation grid. Reconstruction remains unconstrained during optimization; clipping to [0,1] occurs only for evaluation and export.'),
            ('text', 'The transform and fixed iteration budget make this baseline interpretable and easy to reproduce. They also limit the comparison: DCT sparsity is only one classical prior, and a stronger TV or wavelet method could change the ranking. FISTA is a modern optimizer for a classical sparsity objective, rather than the precise reconstruction code used in the 2008 camera study.'),
            ('heading', '2.2. Released ProxUnroll Inference'),
            ('text', 'The released implementation uses six restorer stages with shared restorer parameters. Its HQS update has the form Z = X + rho_k A*(Y - A(X)), followed by X_next = R_theta(Z, memory). ADMM additionally maintains a scaled dual variable U: it applies the consistency update to X - U, feeds Z + U to the restorer, and updates U_next = U + Z - X. We preserve these operations and all checkpoint parameters.'),
            ('text', 'The restorer combines convolution and window attention with an encoder-decoder structure. Its stored feature memory supplies information from the previous stage. In the paper, proximal trajectory supervision encourages intermediate restoration steps to resemble ideal proximal targets [2]. The reference target depends on the clean image during training. It is not available during deployment. Our new inference function accepts only Y, the image shape, and the sampling ratio, so it cannot read the reference image.'),
            ('text', 'The original forward function also creates target-related outputs for training. Our inference path omits that branch, while a regression check compares every restorer-stage output with the original forward result. Strict checkpoint loading rejects missing or unexpected parameters. Inspection finds different sensing factors in the two checkpoints. The main experiment therefore supplies the same external factors from the released 256 x 256 MAT file to both networks. It measures reconstruction after transfer to a common operator. Separate native-operator checks at 10% preserve the original checkpoint factors.'),
            ('heading', '2.3. Interpretation of Interventions'),
            ('text', 'Intermediate-stage evaluation and feature-memory removal are inference interventions on a fixed checkpoint. They do not isolate the causal effect of a training loss or substitute for independently retrained ablations. Likewise, observing improved quality across six stages cannot prove convergence of an indefinitely repeated learned operator. We use these measurements only to characterize the released finite computation.'),
        ]),
        dict(figures=[('quality.png', 'Figure 1. Clean reconstruction on six evaluation images. All methods use identical measurements. Horizontal coordinates use realized sampling ratios.')], blocks=[
            ('heading', '3. Experimental Protocol'),
            ('text', 'The test split contains camera, astronaut, coins, moon, page, and clock from skimage.data. Coffee and chelsea are reserved for validation. We convert RGB inputs to the OpenCV Y channel and resize each image to 256 x 256 using area interpolation. The resize can alter aspect ratio; these results must not be presented as an official Set11 or CBSD68 evaluation. Checkerboard and small_text are procedural stress targets and are excluded from natural/test-image means.'),
            ('text', f'The baseline validation grid is lambda in {{0.001, 0.005, 0.02}}, evaluated at nominal ratios 1%, 10%, and 50% on the two validation images. The selected value is {lam:g}. This one value is then frozen across the test conditions, including noisy conditions. The network weights are also frozen. No test image selects a parameter.'),
            ('text', 'The clean sweep uses r in {0.01, 0.04, 0.10, 0.25, 0.50}. At r = 0.10, two noise levels use E = sigma RMS(Y_clean) N(0,1), with sigma in {0.01, 0.05}. Each image/noise condition uses three seeded realizations. Methods receive the same noisy measurement array. This is a signal-relative Gaussian perturbation model, not calibrated shot noise or a claim about a real photodetector.'),
            ('heading', '3.1. Metrics and Aggregation'),
            ('text', 'PSNR uses peak value 1 after output clipping. SSIM uses the skimage default spatial window with data_range=1. There is no border crop. We average per-image PSNR and SSIM, rather than pooling all pixel errors before calculating PSNR. For noisy conditions, we first average repeats within each image. Test means and stress-target results remain separate. Timing excludes acquisition and data transfer but includes reconstruction itself.'),
            ('heading', '4. Clean Reconstruction Results'),
            ('text', 'Table 1. Mean clean PSNR (dB), six images. CR labels are nominal; Figure 1 uses realized ratios.'),
            ('table', clean_table),
            ('text', f'At 10% nominal sampling, HQS averages {q["psnr"]:.2f} dB and ADMM averages {a["psnr"]:.2f} dB, compared with {f["psnr"]:.2f} dB for the tuned DCT baseline. The paired HQS-minus-FISTA difference is {facts["paired_hqs_fista_mean"]:.2f} dB. Resampling the six image-level differences 10,000 times gives a descriptive 95% percentile interval [{ci[0]:.2f}, {ci[1]:.2f}] dB. The small, selected image suite limits any population interpretation.'),
            ('text', 'Figure 1 describes quality as the measurement budget increases. More measurements reduce the dimension of the unobserved image space, but the trained restorer and the chosen regularization still influence the result. The comparison measures the benefit of these specific pretrained networks over this specific sparse baseline. It does not establish superiority to every classical reconstruction method.'),
            ('text', 'Content strongly affects the aggregate. The clock source is already motion blurred and therefore relatively smooth, whereas page contains fine printed strokes. A very high score on smooth content can raise the mean without resolving text failures. Per-image scores and the separate stress figures should accompany any interpretation of the averages.'),
        ]),
        dict(figures=[('comparison_3.png', 'Figure 2. Periodic checkerboard stress test. Equal display range preserves differences in contrast and residual structure.'),
                      ('comparison_4.png', 'Figure 3. Small-text stress test at 1% nominal sampling. Quantitative scores accompany the full reconstructions; legibility needs separate inspection.')], blocks=[
            ('heading', '5. Failure Cases and Interpretation'),
            ('text', 'The stress targets intentionally depart from smooth natural-image content. The checkerboard concentrates energy in repeated high-frequency transitions. Small text contains thin strokes whose semantic identity can change after modest image distortion. Both examples test details that a global pixel metric can hide. They are procedurally generated targets, not self-captured data.'),
            ('text', f'On the checkerboard at 10%, DCT-FISTA obtains {case("checkerboard", "fista_dct", .1)["psnr"]:.2f} dB, exceeding HQS at {case("checkerboard", "hqs", .1)["psnr"]:.2f} dB and ADMM at {case("checkerboard", "admm", .1)["psnr"]:.2f} dB. The learned outputs visibly contain incorrect coarse divisions and uneven pattern contrast. This reverses the average regular-image ranking and demonstrates a content-dependent failure of the pretrained prior under the common operator.'),
            ('text', 'Figures 2 and 3 are direct exports from the measured reconstruction run. Under severe undersampling, multiple images can fit nearly the same measurement values. A prior must fill that ambiguity, and its preferred texture or smoothness need not match the scene. The appropriate failure analysis considers lost boundaries, merged strokes, suppressed contrast, and repeated-pattern errors, rather than relying only on a high average PSNR.'),
            ('text', 'DCT sparsity can remove small coefficients that collectively encode thin edges. A learned restorer can also suppress structures that are unusual relative to its training distribution. Neither method receives an OCR objective or a character-level constraint. Consequently, this report does not equate improved PSNR with reliable text transcription or recognition.'),
            ('text', f'The lowest HQS score among the six regular evaluation images at 10% is associated with {escape(facts["hardest_at_10pct"])}. The additional comparison_2.png figure records its 1% reconstruction, enabling inspection of the same content under a more severe measurement shortage. The selection rule is explicit: the image is selected by its 10% HQS score, and the stress views have fixed rates rather than hand-selected favorable conditions.'),
            ('text', 'An operational response would be to acquire additional measurements, check reconstruction stability under measurement perturbation, or flag content where a task-specific requirement is not met. Such an acquisition policy is outside this implementation. Post-processing a plausible-looking image cannot certify that missing details were recovered correctly.'),
        ]),
        dict(figures=[('robustness.png', 'Figure 4. Left: signal-relative measurement noise at 10% nominal sampling. Right: clean stage trajectories. Curves aggregate the same six evaluation images.')], blocks=[
            ('heading', '6. Noise, Stages, and Runtime'),
            ('text', f'For HQS, mean PSNR changes from {q["psnr"]:.2f} dB in the clean condition to {row("hqs", sigma=.01)["psnr"]:.2f} dB at 1% relative RMS noise and {row("hqs", sigma=.05)["psnr"]:.2f} dB at 5%. The same noise levels give ADMM {row("admm", sigma=.01)["psnr"]:.2f} dB and {row("admm", sigma=.05)["psnr"]:.2f} dB. Noise is applied to measurements, not to reconstructed pixels. Since all methods share the perturbation, differences cannot be attributed to receiving cleaner inputs.'),
            ('text', f'Disabling HQS feature memory at 10% yields {facts["no_memory_psnr"]:.2f} dB versus {facts["hqs_full_psnr"]:.2f} dB for the unmodified model. The intervention changes the input distribution seen by the trained network. It measures sensitivity to that change, not the performance of a separately trained architecture without memory. Figure 4 also shows that evaluating intermediate outputs is necessary to assess a short trajectory; final-image quality alone cannot describe every stage.'),
            ('text', 'Table 2. Quality and median reconstruction time at nominal 10%. Timing is specific to this environment and implementation.'),
            ('table', runtime_table),
            ('text', f'Experiments use {escape(meta["platform"])} with PyTorch {meta["packages"]["torch"]}, device {meta["device"]}, and {meta["torch_threads"]} CPU threads. One full warmup precedes neural timing. GPU backends, when selected, synchronize around timed regions. Classical algorithms run on CPU. FISTA timing includes its spectral-norm calculation. These single-execution timings characterize the local workflow and do not reproduce the paper\'s hardware speed claims.'),
            ('heading', '6.1. Checkpoint-Native Operators'),
            ('text', f'At 10%, HQS with its own checkpoint factors achieves {facts["native_means"]["hqs"]["hqs"]:.2f} dB; its matched DCT-FISTA comparator achieves {facts["native_means"]["hqs"]["fista_dct"]:.2f} dB. ADMM with its own factors achieves {facts["native_means"]["admm"]["admm"]:.2f} dB; the corresponding FISTA score is {facts["native_means"]["admm"]["fista_dct"]:.2f} dB. These measurements supplement the common-operator experiment. Comparing HQS and ADMM across their native operators changes both sensing and reconstruction, so that difference cannot be assigned to the solver alone.'),
        ]),
        dict(blocks=[
            ('heading', '7. Validity and Reproduction'),
            ('text', f'The completed run contains {facts["metric_rows"]} metric records and {facts["stage_rows"]} stage records. The repository includes the dataset-generation command, pinned environment, run command, raw CSV files, source and checkpoint hashes, and report-generation scripts. A separate test checks the adjoint identity and an analytically solvable FISTA case. Clean neural outputs must match the original forward path. The report builder rejects an incomplete run.'),
            ('text', 'A fresh evaluation is launched with python -m coursework.run_experiments --device cpu --output coursework/results-repeat after installing requirements-coursework.txt and generating the data. The output directory must be new, so a repeat cannot silently overwrite the delivered measurements. The main metrics file contains an image identifier, split, nominal and actual ratios, noise level, seed, method, quality scores, elapsed seconds, and a relative measurement residual. The original floating-point metrics are retained even though visualization files are quantized.'),
            ('heading', '7.1. What the Checks Establish'),
            ('text', 'The adjoint test checks the linear-algebra contract independently of reconstruction quality. The FISTA test uses an identity sensing operator, for which orthonormal transform shrinkage has a known solution. The network check loads the actual released checkpoints strictly and compares every output stage with the original path, including the matrix-size cap at nominal 50%. These tests catch implementation mismatches, but cannot certify that a pretrained prior is appropriate for every scene.'),
            ('text', 'The learned data-consistency coefficients are unconstrained parameters in the released code. Consequently, a learned correction should not automatically be identified with the exact convex proximal formula for a strictly positive penalty. Our implementation preserves those coefficients and evaluates the resulting finite network. This distinction is another reason to separate empirical trajectory measurements from general convergence claims.'),
            ('text', 'Important limits remain. The common MAT-file operator differs from the checkpoint-native factors, so main results include operator transfer. There are only six regular evaluation images, and training-data overlap has not been audited. Measurements omit optical calibration, quantization, motion, and photon statistics. Validation covers only clean conditions. Neither architecture training nor the causal effect of trajectory loss is reproduced. No full-color reconstruction or physical camera experiment is claimed.'),
            ('heading', '8. Conclusion and AI Disclosure'),
            ('text', 'The project supplies a reproducible comparison of explicit sparsity and learned proximal unrolling under shared measurements. Sampling, noise, and content affect reconstruction reliability, and the stress cases show why an average metric is insufficient. A larger independent image set and matched retraining would be the next steps toward stronger conclusions.'),
            ('text', 'Codex assisted with implementation, experiment execution, analysis, report writing, and presentation preparation. Model architecture and weights originate from the cited authors. The student team must review the code, verify the results, and be able to explain the method and limitations before submission. No member-specific contribution or unperformed experiment is claimed.'),
            ('heading', 'References'),
            ('text', '[1] M. F. Duarte et al. Single-Pixel Imaging via Compressive Sampling. IEEE Signal Processing Magazine, 25(2):83-91, 2008. doi:10.1109/MSP.2007.914730.'),
            ('text', '[2] P. Wang et al. Proximal Algorithm Unrolling: Flexible and Efficient Reconstruction Networks for Single-Pixel Imaging. CVPR, 2025. arXiv:2505.23180. Code: github.com/pwangcs/ProxUnroll.'),
            ('text', '[3] A. Beck and M. Teboulle. A Fast Iterative Shrinkage-Thresholding Algorithm for Linear Inverse Problems. SIAM Journal on Imaging Sciences, 2(1):183-202, 2009. doi:10.1137/080716542.'),
            ('text', '[4] S. van der Walt et al. scikit-image: image processing in Python. PeerJ 2:e453, 2014. doi:10.7717/peerj.453. Image provenance: skimage.org/docs/stable/api/skimage.data.html.'),
        ]),
    ]
    draw_report('technical_report.pdf', 'Single-Pixel Imaging under Limited Measurements:<br/>A Reproducible Study of Sparse Recovery and ProxUnroll', pages)
    build_reading(facts, q, f)


def build_reading(facts, q, f):
    pages = [dict(blocks=[
        ('heading', '1. Reading Scope'),
        ('text', 'The selected topic is single-pixel imaging and compressed sensing. The paired readings are Duarte et al., Single-Pixel Imaging via Compressive Sampling (IEEE Signal Processing Magazine, 2008), and Wang et al., Proximal Algorithm Unrolling: Flexible and Efficient Reconstruction Networks for Single-Pixel Imaging (CVPR, 2025). This reading report compares their questions and evidence, then derives a feasible course experiment. Numerical findings from our implementation are discussed separately from the papers\' claims.'),
        ('heading', '2. The Foundational Question'),
        ('text', 'Duarte et al. describe a camera that obtains spatially coded integrated measurements with a digital micromirror device and a single detector. The compressed-sensing formulation links those measurements to a sparse or compressible image representation. The key design decision moves spatial information into the sequence of known patterns, so reconstruction becomes an inverse problem [1].'),
        ('text', 'This changes how the camera should be understood. One detector does not imply one observation, and one observation does not reveal the whole image. The detector count describes hardware parallelism. The measurement count describes the amount of acquired information and contributes to acquisition time. A fair project presentation should keep these quantities distinct. Otherwise, the phrase single-pixel camera can incorrectly suggest that a single scalar determines an arbitrary image.'),
        ('text', 'Our mathematical interpretation starts from y = A x + e. When there are fewer rows than columns, the linear system is underdetermined. If x = D<super>T</super> alpha has a compressible coefficient vector alpha, a reconstruction method can favor solutions with small ||alpha||<sub>1</sub>. This preference supplements the measurements. It is not evidence that every possible image is recoverable at a fixed small sampling rate.'),
        ('text', 'The practical distinction between sensing and compression is also useful. A software simulation computes y from an already available image. It reproduces the inverse-problem arithmetic, but it does not demonstrate the physical benefit of acquiring compressed data optically. For our coursework, that distinction defines an honest scope: compare reconstruction algorithms in simulation and discuss the hardware motivation without claiming an optical prototype.'),
        ('heading', '3. The Modern Question'),
        ('text', 'Wang et al. study reconstruction using a learned image restorer shared across a short proximal unrolling trajectory. The method includes HQS and ADMM versions and uses proximal trajectory supervision to guide intermediate outputs. Its proposed restorer combines convolution, attention, and feature memory [2].'),
        ('text', 'The motivation is relevant to a camera with a changing measurement budget. A reconstruction system that works at several ratios with one set of weights is operationally simpler than keeping a different network for each ratio. However, a numerical experiment must still record which ratios appear during training and which appear only at evaluation. Testing several ratios alone does not establish generalization to every unseen sensing operator.'),
        ('text', 'The comparison with classical sparsity is therefore about where the prior comes from. A transform-based prior is explicit and understandable through its penalty. A learned prior acquires image statistics from training data and may represent patterns that a fixed transform handles poorly. Its dependence on training data and the consequences of a distribution shift become part of the evaluation problem.'),
        ('heading', '3.1. A Shared Mathematical View'),
        ('text', 'Both reconstruction families can be expressed as a competition between measurement consistency and a preference over images. The measurements constrain the observable component, while the prior resolves the remaining ambiguity. This perspective helps explain why comparing algorithms with different matrices can be misleading: the measurement operator itself may reveal different information before the reconstruction algorithm has any role.'),
    ]), dict(blocks=[
        ('heading', '4. Code-Informed Critical Reading'),
        ('text', 'The released code provides a concrete interpretation of the modern method. It stores separate height and width sensing factors and applies the operator as H X W<super>T</super>. This avoids materializing a very large dense matrix. The apparent two-dimensional measurement array is a computational representation of the scalar measurement sequence, not a second spatial detector array.'),
        ('text', 'In the HQS inference loop, the implementation first corrects measurement residuals and then calls the restorer. In ADMM, a dual state additionally carries the disagreement between the split variables. The same restorer is reused at successive stages, and memory passes feature information between calls. These observations concern the released implementation used in our experiments.'),
        ('text', 'A potential source of confusion is the original forward function accepting the clean image. Inspection shows that it creates synthetic measurements and also computes target-related training outputs. A deployment API should accept measurements directly. We therefore introduce a separate reconstruction function and compare its output against the original clean path. This check addresses information leakage more directly than assuming an inference routine is correct because it runs.'),
        ('heading', '4.1. What a Trajectory Claim Requires'),
        ('text', 'An ideal proximal target can depend on a known clean training image. A network can learn to approximate that target from degraded inputs. The target and the learned approximation must nevertheless remain conceptually separate. Success on a finite test set does not establish an exact proximal representation for arbitrary inputs or a global convergence theorem for an unconstrained neural network.'),
        ('text', 'Our experimental response is to retain every intermediate reconstruction. We can inspect whether quality changes smoothly across the released six stages and whether an intervention disrupts that behavior. This is evidence about a finite computation. Proving asymptotic convergence would require additional assumptions and analysis. Retraining with and without trajectory loss would address a different, causal question that our pretrained evaluation does not answer.'),
        ('heading', '4.2. Fairness and Missing Controls'),
        ('text', 'A useful baseline should receive the identical measurement array. We choose an orthonormal DCT sparsity objective and solve it with FISTA, using a spectral step size and parameters selected on disjoint validation images. This provides a clear, reproducible classical comparator. It leaves open comparisons with total variation, wavelet penalties, stronger optimization schedules, and other learned methods.'),
        ('text', 'Training exposure also matters. A pretrained neural network benefits from data and computation that the classical solver does not require. That is an intended difference in the comparison, but it prevents interpreting inference-time numbers as total system cost. The small public image suite may also overlap with training resources; we have not audited the original training set. This uncertainty should accompany the reported performance.'),
        ('heading', '5. Experiment Derived from the Readings'),
        ('text', 'The shared inverse model suggests four controlled axes: sampling budget, measurement noise, image content, and stage computation. Our project varies five sampling ratios without changing network weights, adds Gaussian noise directly to the shared measurements, includes periodic patterns and text, and records six stage outputs. Two validation images determine one DCT regularization weight before testing.'),
        ('text', 'The stress targets are especially useful for discussion. A periodic pattern tests the preservation of repeated high-frequency structure. Small text tests whether thin strokes remain distinguishable. An improved global PSNR can coexist with an incorrect character or a missing line. These examples connect mathematical ambiguity to an observable application failure and motivate task-specific evaluation.'),
        ('heading', '5.1. Metrics with Explicit Meanings'),
        ('text', 'PSNR measures pixel fidelity under a chosen intensity range, and SSIM describes local structural similarity under a chosen implementation. Neither metric directly measures optical efficiency, detector cost, acquisition speed, or the correctness of a recognized symbol. Timing here measures the reconstruction routine after warmup. The metric definitions and aggregation rules must be specified before using them to support an engineering conclusion.'),
    ]), dict(blocks=[
        ('heading', '6. Connection to the Implemented Project'),
        ('text', f'The accompanying technical report contains actual results from the released checkpoints and the new baseline. At nominal 10% sampling, HQS achieves {q["psnr"]:.2f} dB and {q["ssim"]:.3f} SSIM on the six regular evaluation images. DCT-FISTA achieves {f["psnr"]:.2f} dB and {f["ssim"]:.3f} SSIM under the same measurements. These are local experimental results, not transcribed values from either paper.'),
        ('text', f'The run preserves {facts["metric_rows"]} image-level metric records, including the stress targets and seeded noise repetitions. The technical report explains the sampling sweep and presents failure images. The code also records the difference between nominal and realized measurement ratios, since row counts are rounded upward. This bookkeeping prevents a small implementation detail from becoming an unreported mismatch between experimental conditions.'),
        ('text', 'The evidence supports a limited conclusion: a comparison of these reconstruction methods, on these images, with this forward operator and these checkpoints. A stronger project could add independent photographs with documented acquisition conditions, a larger official benchmark, retraining across several random seeds, and an optical measurement calibration. Those extensions would increase external validity, but none is presented here as completed work.'),
        ('heading', '7. Questions for the Presentation'),
        ('text', '<b>Why is the image not determined by one detector reading?</b> Each reading is an inner product with one mask. Multiple masks provide multiple constraints. The word single refers to the detector rather than the number of measurements.'),
        ('text', '<b>Does sparse recovery require the image itself to be mostly zero?</b> No. The intended assumption concerns a representation, such as transform coefficients or gradients. The chosen representation affects which image structures the prior favors.'),
        ('text', '<b>Why must measurement matrices be shared?</b> Changing the operator can change the information content of the data. Shared operators and identical noise realizations make reconstruction differences easier to attribute to the solver.'),
        ('text', '<b>Does feature-memory removal prove that memory is necessary?</b> It measures how a trained checkpoint reacts to removal. A controlled architecture ablation would retrain comparable models under the same training protocol.'),
        ('text', '<b>Why not call the results a complete reproduction of CVPR 2025?</b> We reproduce the released inference computation but use a smaller, explicitly named evaluation suite. We do not reproduce full training, the official dataset protocol, every competing method, or the physical-camera experiments.'),
        ('text', '<b>What explains a convincing-looking failure?</b> The measurement system leaves ambiguity, and the prior selects one possible image. Visual plausibility alone does not establish that a detail matches the unknown scene.'),
        ('heading', '8. Reflection and AI Disclosure'),
        ('text', 'The most useful connection between the readings is the separation of acquired information from prior knowledge. The foundational formulation explains why reconstruction is possible under structural assumptions. The modern method shows how to encode stronger image preferences in a finite learned computation. A careful experiment evaluates both the resulting quality and the conditions under which those preferences fail.'),
        ('text', 'Codex assisted with source inspection, implementation, experiment execution, analysis, English writing, and presentation preparation. The pretrained network and weights remain the work of their cited authors. The named student team should review this report and the associated code, confirm the supplied identity information, and prepare to explain the forward model and experimental limitations. No unverified member-specific contribution is assigned.'),
        ('heading', 'References and Primary Sources'),
        ('text', '[1] M. F. Duarte, M. A. Davenport, D. Takhar, J. N. Laska, T. Sun, K. F. Kelly, and R. G. Baraniuk. Single-Pixel Imaging via Compressive Sampling. IEEE Signal Processing Magazine 25(2):83-91, 2008. doi:10.1109/MSP.2007.914730. Author-hosted PDF: https://mdav.ece.gatech.edu/publications/ddtlskb-spm-2008.pdf.'),
        ('text', '[2] P. Wang, L. Wang, G. Qu, X. Wang, Y. Zhang, and X. Yuan. Proximal Algorithm Unrolling: Flexible and Efficient Reconstruction Networks for Single-Pixel Imaging. CVPR, 2025. https://arxiv.org/abs/2505.23180. Official source: https://github.com/pwangcs/ProxUnroll.'),
        ('text', '[3] A. Beck and M. Teboulle. A Fast Iterative Shrinkage-Thresholding Algorithm for Linear Inverse Problems. SIAM Journal on Imaging Sciences 2(1):183-202, 2009. doi:10.1137/080716542.'),
    ])]
    draw_report('literature_review.pdf', 'Single-Pixel Imaging: A Comparative Reading<br/>of Compressive Sampling and Proximal Unrolling', pages)


if __name__ == '__main__':
    main()
