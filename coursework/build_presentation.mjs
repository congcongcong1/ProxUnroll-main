import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const runtime = process.env.CODEX_RUNTIME_ROOT ?? path.join(process.env.HOME, '.cache/codex-runtimes/codex-primary-runtime/dependencies');
process.env.RUNTIME_NODE_MODULES ??= path.join(runtime, 'node/node_modules');
const skill = process.env.PRESENTATIONS_SKILL_DIR ?? path.join(process.env.HOME, '.codex/plugins/cache/openai-primary-runtime/presentations/26.904.11930/skills/presentations');
const { Presentation, PresentationFile, FileBlob } = await import(pathToFileURL(path.join(runtime, 'node/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs')));
const { resolvePresentationFont, applyPresentationChartFont, finalizePresentation } = await import(pathToFileURL(path.join(skill, 'container_tools/artifact_tool_utils.mjs')));
const family = resolvePresentationFont();
const facts = JSON.parse(await fs.readFile(path.join(root, 'coursework/results/report_facts.json'), 'utf8'));
const provenance = JSON.parse(await fs.readFile(path.join(root, 'coursework/results/provenance.json'), 'utf8'));
await fs.access(path.join(root, 'coursework/results/COMPLETE'));
const buildRoot = path.join(root, 'tmp/presentation');
const out = path.join(root, 'output/presentation');
await fs.mkdir(buildRoot, { recursive: true });
const tmp = await fs.mkdtemp(path.join(buildRoot, 'build-'));
await fs.mkdir(out, { recursive: true });
const presentation = Presentation.create({ slideSize: { width: 1280, height: 720 } });
const ink = '#20272B', teal = '#147D78', muted = '#59666C';
const names = { adjoint: 'Adjoint', fista_dct: 'DCT-FISTA', hqs: 'HQS', admm: 'ADMM' };
const colors = { adjoint: '#777777', fista_dct: '#D78924', hqs: teal, admm: '#B74760' };
const times = [25, 45, 45, 55, 45, 55, 35, 45, 45, 55, 90, 60];
const guide = ['# Presentation Guide', 'Total planned speaking time: 600 seconds (10 minutes). Questions: 2 minutes.',
  'Suggested speaking split, to be confirmed by the team: slides 1-4, 5-8, and 9-12. This is a rehearsal suggestion, not a claim about completed individual contributions.'];
const get = (m, r=.1, sigma=0) => facts.summary.find(x => x.method===m && x.cr===r && x.sigma===sigma && x.split==='test');

function text(slide, value, x, y, w, ht, size=28, bold=false, color=ink) {
  const shape = slide.shapes.add({ geometry: 'textbox', position: { left:x, top:y, width:w, height:ht }, fill:'none', line:{fill:'none',width:0} });
  shape.text = value;
  shape.text.style = { typeface:family, fontSize:size, bold, color, autoFit:'none' };
  return shape;
}
function slide(title, notes) {
  const s = presentation.slides.add();
  s.background.fill = '#FFFFFF';
  const i = presentation.slides.items.length;
  text(s, title, 60, 36, 1155, 100, 44, true);
  text(s, String(i).padStart(2,'0'), 1180, 673, 50, 24, 17, false, muted);
  s.speakerNotes.textFrame.setText(`Planned speaking time: ${times[i-1]} seconds.\n\n${notes}`);
  guide.push(`## Slide ${i}: ${title} (${times[i-1]} seconds)`, notes);
  return s;
}
async function image(s, rel, x,y,w,ht,alt) {
  s.images.add({ blob:new Uint8Array(await fs.readFile(path.join(root,rel))), contentType:'image/png', alt, fit:'contain',
    position:{left:x,top:y,width:w,height:ht} });
}
function chart(s, type, categories, series, {x=65,y=155,w=1135,ht=465,xTitle='',yTitle='',legend=true}={}) {
  series = series.map(s => ({...s, values:s.values.map(v=>Number(v.toFixed(5)))}));
  const c = s.charts.add(type, { position:{left:x,top:y,width:w,height:ht}, categories, series,
    hasLegend:legend, legend:{position:'bottom', textStyle:{typeface:family,fontSize:21}},
    chartFill:'#FFFFFF', plotAreaFill:'#FFFFFF',
    xAxis:{title:xTitle, textStyle:{typeface:family,fontSize:20}, majorGridlines:null},
    yAxis:{title:yTitle, numberFormatCode:'0.0', textStyle:{typeface:family,fontSize:20}, majorGridlines:{fill:'#E3E7E7',width:1}},
    ...(type==='bar' ? {barOptions:{direction:'column',grouping:'clustered'},dataLabels:{showValue:true,position:'outEnd',textStyle:{typeface:family,fontSize:20}}} : {lineOptions:{smooth:false}})
  });
  applyPresentationChartFont(c,{fontFamily:family});
  return c;
}
function series(method, values) {
  return {name:names[method], values, fill:colors[method], line:{fill:colors[method],width:3}, marker:{symbol:'circle',size:7}};
}

let s = slide('Single-Pixel Imaging', 'We study reconstruction when only a fraction of image measurements is available. Our course project pairs the classic compressed-sensing formulation with the released ProxUnroll networks. This is a reproducible software evaluation using pretrained models. It includes controlled comparisons and deliberately difficult inputs. The report and code disclose AI assistance and retain the original model attribution.');
text(s,'Sparse recovery and proximal unrolling',65,148,710,100,38,false,teal);
text(s,'Zicong Luo    522026230043\nShuhao Yang    502026230083\nYubowen Li    502026230092',65,306,720,170,29);
text(s,'Topic 7  /  Computational Imaging',65,582,760,48,25,false,muted);
await image(s,'coursework/data/coins.png',880,192,300,300,'Grayscale coins evaluation image');

s=slide('The measurement model', 'A single detector produces one scalar for each spatial mask. An image requires a sequence of masks, so single pixel does not mean one observation. In our simulation, separable factors implement the masks efficiently. The adjoint maps measurements back to image space but cannot determine the unobserved component. A reconstruction prior resolves some of this ambiguity. We simulate this arithmetic and do not claim an optical prototype. Source: Duarte et al., https://doi.org/10.1109/MSP.2007.914730.');
text(s,'Y = H X W^T + E',90,165,1100,94,58,true,teal);
text(s,'X: image     H, W: known sensing factors     E: measurement noise',90,282,1100,95,28);
text(s,'One integrated measurement per mask\nA sequence of masks supplies spatial constraints\nFewer measurements leave more ambiguity',90,412,1110,180,33);

s=slide('Explicit sparsity and a learned prior', 'Our classical comparator minimizes squared measurement error plus an L1 penalty on an orthonormal DCT representation. FISTA combines a gradient step, transform-domain thresholding, and momentum. Its regularization weight is chosen only on validation images. The modern comparator replaces the hand-designed shrinkage step with a learned image restorer. Both use the same measurement residual in the main experiment. This comparison is not the exact reconstruction implementation of the 2008 paper. Sources: Beck and Teboulle, https://doi.org/10.1137/080716542; Wang et al., https://arxiv.org/abs/2505.23180.');
text(s,'DCT-FISTA',65,160,530,55,35,true,colors.fista_dct);
text(s,'Explicit transform penalty\n200 optimization iterations\nValidation-selected regularization\nNo network training',65,246,530,292,31);
text(s,'ProxUnroll',695,160,515,55,35,true,teal);
text(s,'Released pretrained restorer\nSix shared-weight stages\nMeasurement consistency updates\nHQS and ADMM variants',695,246,515,292,31);
text(s,'Main comparison: identical images, sensing factors, measurements, and noise',65,608,1110,47,25,false,muted);

s=slide('The shared image restorer', 'The released restorer has an encoder-decoder structure combining convolution and window attention. Feature memory passes information between stages. HQS applies a data-consistency correction and then the restorer. ADMM also updates a dual state that records the split-variable disagreement. The pretrained weights remain fixed in our experiments. The training-time target uses the clean image, but our reconstruction interface receives measurements only. We test equality with the original clean forward path. Figure from the official ProxUnroll repository, fig/network.png, https://github.com/pwangcs/ProxUnroll.');
await image(s,'coursework/figures/restorer_overview.png',65,165,1145,315,'ProxUnroll restorer architecture, complete overview panel (a)');
text(s,'Convolution and attention restore the image at multiple scales\nShared weights and feature memory link the six stages',65,530,1140,100,28);

s=slide('Evaluation design', 'We use six regular evaluation images and two independent validation images, all resized to 256 by 256 grayscale. Two procedural stress targets test periodic detail and small text. The clean experiment sweeps five sampling ratios. At ten percent sampling, two Gaussian noise levels each use three seeded realizations. We found that the checkpoints contain different sensing factors. The main comparison therefore supplies the same factors from the released MAT file to both networks, and separate diagnostics retain each native operator. The study includes operator transfer and is not an official Set11 or CBSD68 benchmark. Data source: https://scikit-image.org/docs/stable/api/skimage.data.html.');
text(s,'6 evaluation images + 2 validation images\n2 stress targets, reported separately\n5 clean rates: 1%, 4%, 10%, 25%, 50%\n2 noise levels at 10%, each with 3 seeds',65,158,1110,225,32);
for (const [i,n] of ['astronaut','moon','page','clock'].entries()) await image(s,`coursework/data/${n}.png`,70+i*285,422,220,180,n);
text(s,'Common MAT-file operator for the main comparison',65,627,1120,37,24,true,teal);

s=slide('Reconstruction quality across rates', `The chart shows mean PSNR on the same six evaluation images. The horizontal labels are nominal sampling ratios. The raw files record realized ratios after rounding and the stored-matrix row limit. Nominal 50% uses 49.9893%. HQS minus DCT-FISTA at ten percent is ${facts.paired_hqs_fista_mean.toFixed(2)} dB on average. These values describe this small image suite and this common operator. They do not establish superiority over all classical algorithms. Source: coursework/results/summary.csv; measurements generated by coursework/run_experiments.py.`);
chart(s,'line',['1%','4%','10%','25%','50%'],Object.keys(names).map(m=>series(m,[.01,.04,.1,.25,.5].map(r=>get(m,r).psnr))),{xTitle:'Nominal sampling ratio',yTitle:'Mean PSNR (dB)'});

s=slide('Local reconstruction time', `The chart uses median reconstruction time at nominal ten percent. All methods run locally, and this run uses ${provenance.device}. Timing excludes measurement generation and data transfer. Neural execution follows a full warmup. FISTA includes its spectral-norm calculation. One timed execution is recorded for each input, so these are workflow measurements rather than a dedicated hardware benchmark. The result should not be compared directly with GPU timings in the paper. Source: coursework/results/summary.csv and provenance.json.`);
chart(s,'bar',['Adjoint','DCT-FISTA','HQS','ADMM'],[{name:'Median time',values:Object.keys(names).map(m=>get(m).seconds),fill:teal}],{xTitle:'Method',yTitle:'Seconds per image',legend:false,ht:420});
text(s,`CPU evaluation at 256 x 256; ${provenance.torch_threads} threads; excludes acquisition`,65,620,1120,36,24,false,muted);

s=slide('Measurement noise sensitivity', 'We perturb the measurements before reconstruction. Noise standard deviation is a fixed fraction of the clean measurement RMS. Every solver receives exactly the same realization. Three seeds reduce dependence on a single random draw, but the six images remain the unit of aggregation. The noise model does not simulate photon counting, detector calibration, motion, or quantization. The chart therefore supports a claim about this perturbation model only. Source: coursework/results/summary.csv.');
chart(s,'line',['0','0.01','0.05'],['fista_dct','hqs','admm'].map(m=>series(m,[0,.01,.05].map(n=>get(m,.1,n).psnr))),{xTitle:'Noise standard deviation / clean measurement RMS',yTitle:'Mean PSNR (dB)'});

s=slide('Stages and feature memory', `We retain the output of every restorer stage. This reveals the finite optimization trajectory instead of looking only at the final image. At ten percent, disabling HQS memory gives ${facts.no_memory_psnr.toFixed(2)} dB compared with ${facts.hqs_full_psnr.toFixed(2)} dB for the intact checkpoint. This is an inference intervention. A causal architecture ablation would require retraining. Six-stage behavior also cannot prove convergence under infinitely repeated updates. Sources: coursework/results/stages.csv and interventions.csv.`);
chart(s,'line',['1','2','3','4','5','6'],['hqs','admm'].map(m=>series(m,facts.stage_means[m])),{xTitle:'Restorer stage',yTitle:'Mean PSNR (dB)',ht:405});
text(s,`HQS: ${facts.hqs_full_psnr.toFixed(2)} dB with memory / ${facts.no_memory_psnr.toFixed(2)} dB without memory`,65,625,1110,40,25,false,teal);

s=slide('A shared-input visual comparison', 'These images use the camera test image at ten percent nominal sampling, with no measurement noise. All methods receive the same data. Compare the edges and local textures, not just the overall silhouette. Images use a fixed display range and come directly from the saved reconstructions. Metrics are computed before export to eight-bit PNG. Source: coursework/data/camera.png and coursework/results/reconstructions/camera_cr0.1_noise0_*.png.');
for(const [i,m] of ['ground_truth','fista_dct','hqs','admm'].entries()) {
  const rel=m==='ground_truth'?'coursework/data/camera.png':`coursework/results/reconstructions/camera_cr0.1_noise0_${m}.png`;
  await image(s,rel,65+i*300,219,250,250,m);
  text(s,m==='ground_truth'?'Reference':names[m],65+i*300,496,260,50,28,true);
}
text(s,'Nominal sampling ratio: 10%     Common sensing operator',65,602,1120,44,28,false,muted);

s=slide('Failure cases: texture and small text', 'The top row uses a periodic checkerboard at ten percent. The bottom row uses small text at one percent. These targets are generated, not self-collected photographs. Under undersampling, image priors must resolve ambiguity, and the selected structure can be incorrect. Inspect contrast, repeated boundaries, missing strokes, and merged letters. Neither PSNR nor SSIM directly certifies legibility. These examples motivate additional measurements or task-specific validation before relying on fine details. Sources: coursework/data and coursework/results/reconstructions; fixed conditions and unretouched outputs.');
for(const [j,[target,cr]] of [['checkerboard','0.1'],['small_text','0.01']].entries()) {
  for(const [i,m] of ['ground_truth','fista_dct','hqs','admm'].entries()) {
    const rel=m==='ground_truth'?`coursework/data/${target}.png`:`coursework/results/reconstructions/${target}_cr${cr}_noise0_${m}.png`;
    await image(s,rel,85+i*300,172+j*242,205,205,`${target} ${m}`);
    if(j===0) text(s,m==='ground_truth'?'Reference':names[m],75+i*300,132,270,35,25,true);
  }
}
text(s,'Top: checkerboard at 10%     Bottom: small text at 1%',65,659,1120,30,23,false,muted);

s=slide('Conclusions and limits', 'The contribution is a reproducible controlled study around the released model, with a classical baseline, parameter validation, noise tests, stage records, and failure cases. The main comparison holds sensing fixed, but that requires transfer away from the checkpoint-native operators. The study has six regular evaluation images, no training-set overlap audit, no retraining, and no optical camera experiment. Our files include raw results and reproduction commands. Codex assisted with code, experiment execution, analysis, English writing, and presentation preparation. The original authors supplied the architecture and weights. The team must understand and review the work before submission.');
text(s,'Shared measurements make solver comparisons interpretable\nFine detail can fail despite plausible global appearance\nRaw results and reproduction commands accompany the reports',65,160,1130,195,32,true,teal);
text(s,'Limits: small image suite, common-operator transfer, pretrained models\nNo optical acquisition, retraining, or full-color reconstruction',65,414,1130,119,28);
text(s,'AI disclosure: Codex assisted with code, experiments, analysis, and writing.\nArchitecture and weights: Wang et al., CVPR 2025.',65,585,1110,71,23,false,muted);

const candidatePath=path.join(tmp,'candidate.pptx');
await (await PresentationFile.exportPptx(presentation)).save(candidatePath);
const finalPath=path.join(out,process.env.PRESENTATION_FILENAME??'course_presentation.pptx');
await finalizePresentation({workspaceDir:root,candidatePath,finalPath,
  pythonExecutable:path.join(runtime,'python/bin/python3'),
  integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),
  layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),
  layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit'],
  explicitTotalSlideCount:12,requiredNativeChartOwnerSlides:[6,7,8,9],
  materializeLiteralChartWorkbooks:true,fontPolicy:{basis:'design',families:[family]},
  verifyArtifactToolImport:true,receiptPath:path.join(tmp,'validation.json')});
const finalDeck=await PresentationFile.importPptx(await FileBlob.load(finalPath));
for (let i=0;i<finalDeck.slides.items.length;i++) {
  const sl=finalDeck.slides.items[i];
  const preview=await finalDeck.export({slide:sl,format:'png',scale:1});
  await fs.writeFile(path.join(tmp,`slide-${i+1}.png`),new Uint8Array(await preview.arrayBuffer()));
}
await fs.writeFile(path.join(root,'coursework/PRESENTATION_GUIDE.md'),guide.join('\n\n')+'\n');
console.log(JSON.stringify({finalPath,buildDirectory:tmp,font:family,slides:12}));
