import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath, pathToFileURL} from 'node:url';
import crypto from 'node:crypto';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const runtime = path.join(process.env.HOME, '.cache/codex-runtimes/codex-primary-runtime/dependencies');
const skill = path.join(process.env.HOME, '.codex/plugins/cache/openai-primary-runtime/presentations/26.1007.11041/skills/presentations');
process.env.RUNTIME_NODE_MODULES ??= path.join(runtime, 'node/node_modules');
const {PresentationFile, FileBlob} = await import(pathToFileURL(path.join(runtime, 'node/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs')));
const {finalizePresentation, applyPresentationChartFont} = await import(pathToFileURL(path.join(skill, 'container_tools/artifact_tool_utils.mjs')));
const base = path.join(root, 'runs/hqs_ablation_20261013');
const facts = JSON.parse(await fs.readFile(path.join(base, 'report_facts.json'), 'utf8'));
const presentationFacts = JSON.parse(await fs.readFile(path.join(base, 'presentation_facts.json'), 'utf8'));
const primary = facts.phases.primary.analysis;
const source = path.join(root, 'output/presentation/course_presentation_nanjing_20261009.pptx');
const deck = await PresentationFile.importPptx(await FileBlob.load(source));
if (deck.slides.items.length !== 12) throw new Error('Expected the existing twelve-slide reference');
const build = await fs.mkdtemp(path.join(root, 'tmp/presentation/ablation-'));
const output = path.join(root, 'output/presentation', process.env.PRESENTATION_FILENAME ?? 'course_presentation_ablation_20261013.pptx');
const font = 'Helvetica Neue', ink = '#20272B', teal = '#147D78', muted = '#59666C';
const times = [25,45,45,50,55,60,60,60,60,60,45,35];
if (times.reduce((a,b)=>a+b,0) !== 600) throw new Error('Speaking plan must be 600 seconds');
const guide = ['# Continuation and Factorial Ablation Presentation Guide', 'Planned speaking time: 600 seconds. Historical deck and reports remain preserved.'];
const notes = {};
const record = (index, body) => {
  notes[index] = body;
  deck.slides.items[index-1].speakerNotes.textFrame.setText(`Planned speaking time: ${times[index-1]} seconds.\n\n${body}`);
  guide.push(`## Slide ${index}`, body);
};
function text(slide, value, x, y, w, h, size=28, bold=false, color=ink) {
  const shape = slide.shapes.add({geometry:'textbox',position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});
  shape.text = value; shape.text.style = {typeface:font,fontSize:size,bold,color,autoFit:'none'};
  return shape;
}
function reset(index, title, body) {
  const slide=deck.slides.items[index-1];
  slide.shapes.deleteAll();
  for(const collection of [slide.images, slide.charts, slide.tables]) for(const item of [...collection.items]) collection.deleteById(item.id);
  slide.background.fill='#FFFFFF';
  text(slide,title,60,36,1155,100,44,true);
  text(slide,String(index).padStart(2,'0'),1180,673,50,24,17,false,muted);
  record(index,body);
  return slide;
}
function chart(slide,categories,series,yTitle) {
  const c=slide.charts.add('bar',{
    position:{left:65,top:160,width:1135,height:420},categories,
    series:series.map(r=>({...r,values:r.values.map(v=>Number(v.toFixed(7)))})),
    hasLegend:true,legend:{position:'bottom',textStyle:{typeface:font,fontSize:21}},
    chartFill:'#FFFFFF',plotAreaFill:'#FFFFFF',
    xAxis:{tickLabelPosition:'low',textStyle:{typeface:font,fontSize:20}},
    yAxis:{title:yTitle,numberFormatCode:'0.0%',textStyle:{typeface:font,fontSize:20},majorGridlines:{fill:'#E3E7E7',width:1}},
    barOptions:{direction:'column',grouping:'clustered'}
  });
  applyPresentationChartFont(c,{fontFamily:font});
  return c;
}
const pair=(candidate,reference,sigma=0)=>primary.paired_summary.find(r=>r.dataset==='DIV2K_holdout_10000'&&r.cr===.1&&r.sigma===sigma&&r.candidate===candidate&&r.reference===reference);
const factor=(comparison,seed,sigma,view='last')=>primary.paired_summary.find(r=>r.dataset==='DIV2K_holdout_10000'&&r.cr===.1&&r.sigma===sigma&&r.training_seed===seed&&r.checkpoint_view===view&&r.comparison===comparison);
record(1,'We simulate single-pixel grayscale reconstruction, compare four solvers, and extend HQS training to 12000 total round-2 updates. Two training seeds each run a matched 2x2 noise/stage-supervision ablation for 800 updates per arm. Architecture and official weights: Wang et al., CVPR 2025, github.com/pwangcs/ProxUnroll. The cover retains the standard scikit-image coins test image from the original project inputs. Codex assisted code, experiments, analysis and materials. Results are condition-specific; no optical camera or temporal video model is claimed. Source evidence: runs/hqs_ablation_20261013/report_facts.json.');
record(2,'Y = H X W^T + E simulates integrated measurements from existing grayscale images. Fewer measurements leave greater ambiguity. All methods share the same sensing operator and measured values. No physical acquisition is performed. Reference: Duarte et al., DOI 10.1109/MSP.2007.914730.');
record(3,'Adjoint and DCT-FISTA are classical baselines. FISTA uses 200 iterations and lambda .001 selected only on original independent validation. Released HQS/ADMM share the common MAT operator during evaluation. Architecture: Wang et al., arXiv:2505.23180; FISTA: Beck and Teboulle, DOI 10.1137/080716542. The newly implemented work is matched evaluation and controlled fine-tuning rather than a new network architecture.');
record(4,'The source diagram describes the supplied convolution/attention restorer and stage feature memory. Source: Wang et al., CVPR 2025, github.com/pwangcs/ProxUnroll. Memory connects stages within one image, not previous/future video frames.');
// Preserve the existing photographs and their exact source/license notes.
const photoManifest=JSON.parse(await fs.readFile(path.join(root,'runs/nanjing_web30_20261009/data/test.json'),'utf8'));
const photoCredits=['web_nju_gulou_04','web_nju_xianlin_05','web_qinhuai_fuzimiao_03'].map(name=>{
  const r=photoManifest.find(r=>r.name===name);
  return `${r.title}; ${r.author}; ${r.license}; source ${r.page_url}; license ${r.license_url}. Adapted by grayscale center crop and area resize. Full credits: runs/nanjing_web30_20261009/ATTRIBUTION.md.`;
}).join('\n');
const dataSlide=reset(5,'Training, independent tests and regression scenes',`Training uses 800 DIV2K originals and validation uses 16 distinct originals. The primary test is the previously locked, project-unused 52 DIV2K originals. Regression tests include 110 previously observed images: Kodak24, eight HEVC videos with three spaced frames each, DIV2K32, and 30 Nanjing web photographs. The photographs shown are external-author regression images, not team captures. Source manifests and original licenses are preserved. DIV2K: data.vision.ee.ethz.ch/cvl/DIV2K/.\n\nPhotograph attribution:\n${photoCredits}`);
text(dataSlide,'800 training / 16 validation / 52 primary test originals\nRegression scenes include 30 Nanjing web photographs',65,149,1120,88,28,true,teal);
for(const [i,name] of ['web_nju_gulou_04','web_nju_xianlin_05','web_qinhuai_fuzimiao_03'].entries()) {
  const row=photoManifest.find(r=>r.name===name);
  const file=path.join(root,'runs/nanjing_web30_20261009/data',row.path);
  dataSlide.images.add({blob:new Uint8Array(await fs.readFile(file)),contentType:'image/png',alt:row.title,fit:'contain',position:{left:65+i*390,top:257,width:340,height:300}});
  text(dataSlide,['NJU Gulou','NJU Xianlin','Qinhuai night'][i],65+i*390,567,350,38,26);
}
text(dataSlide,'External-author photographs; source and license retained',65,623,1120,35,23,false,muted);
record(6,'This preserved graph contains four-method results on all 18 campus originals, at five clean ratios, under identical measurements. At clean 10%, official HQS/ADMM have mean PSNR 28.35/28.38 dB, compared with 24.60 for DCT-FISTA and 23.92 for adjoint. These are previously observed regression results, not the new primary52 experiment. Source: runs/nanjing_web30_20261009/comparison/dataset_summary.csv.');
// Recreate the literal series from frozen CSV-derived JSON. Importing the old
// chart retains workbook references but loses that workbook in Artifact Tool;
// never fabricate its missing formulas or bypass the portability validator.
const baseline=JSON.parse(await fs.readFile(path.join(root,'runs/nanjing_web30_20261009/comparison/summary.json'),'utf8'));
const baselineSlide=reset(6,'Four-method reconstruction on campus scenes',notes[6]);
const methodLabels={adjoint:'Adjoint',fista_dct:'DCT-FISTA',hqs:'HQS pretrained',admm:'ADMM pretrained'};
const methodColors={adjoint:'#777777',fista_dct:'#D78924',hqs:teal,admm:'#B74760'};
const baselineChart=baselineSlide.charts.add('scatter',{
  position:{left:65,top:160,width:1135,height:420},categories:[],
  series:Object.entries(methodLabels).map(([method,name])=>({name,xValues:[1,4,10,25,50],
    values:[.01,.04,.1,.25,.5].map(cr=>Number(baseline.datasets.find(r=>r.dataset==='NJU_campus_web'&&r.method===method&&r.cr===cr&&r.sigma===0).psnr.toFixed(5))),
    fill:methodColors[method],line:{fill:methodColors[method],width:3},marker:{symbol:'circle',size:7}})),
  hasLegend:true,legend:{position:'bottom',textStyle:{typeface:font,fontSize:21}},
  chartFill:'#FFFFFF',plotAreaFill:'#FFFFFF',scatterOptions:{style:'lineWithMarkers'},
  xAxis:{title:'Nominal clean sampling (%)',min:0,max:50,majorUnit:10,textStyle:{typeface:font,fontSize:20}},
  yAxis:{title:'PSNR (dB)',min:0,max:45,numberFormatCode:'0.00',textStyle:{typeface:font,fontSize:20},majorGridlines:{fill:'#E3E7E7',width:1}}
});
applyPresentationChartFont(baselineChart,{fontFamily:font});
text(baselineSlide,'Every source photograph contributes equally',65,610,1120,45,25,false,muted);

let slide=reset(7,'Matched training ablations',
  'All four arms start from identical official HQS weights with fresh Adam 1e-5, batch one, FP32, frozen sensing factors, 800 DIV2K training and 16 validation originals. Seeds 20261013 and 20261014 each complete 800 steps per arm. Same-seed image order, crops, flips, sampling rates and potential noise draws are identical. Removing noise sets amplitude to zero. Removing intermediate loss keeps total weight one. Separate continuation restores full step-10000 state and adds 2000 updates under the unchanged update rule. Source: runs/hqs_ablation_20261013/lock.json and all training/pairing audits.');
const values=[['Training arm','Measurement noise','Stage supervision','Budget'],['On / weighted','0, 0, .01, .05','.01 x5 + .95 final','800 x2 seeds'],['Off / weighted','0 only','.01 x5 + .95 final','800 x2 seeds'],['On / final','0, 0, .01, .05','final weight 1','800 x2 seeds'],['Off / final','0 only','final weight 1','800 x2 seeds']];
const table=slide.tables.add({rows:values.length,columns:4,left:65,top:185,width:1135,height:290,columnWidths:[245,295,325,270],values});
for(let r=0;r<values.length;r++) for(let c=0;c<4;c++) {
  const cell=table.getCell(r,c); cell.text.style={typeface:font,fontSize:25,bold:r===0,color:ink};
  cell.fill=r===0?'#EDF1F1':'#FFFFFF';
}
text(slide,'Same weights, crops, sampling and validation rule\nSeparate continuation: 10000 to 12000 total updates',65,525,1120,91,29,true,teal);
text(slide,'All six sensing factors stay frozen',65,630,1120,34,23,false,muted);

slide=reset(8,'Continuation on the independent 52-image test',
  `The validation-selected continuation is step ${presentationFacts.selected_step}; all 12000 round-2 updates completed. Selection uses mixed validation PSNR with clean initialization minus .03 dB guard. No test score selects weights. Primary52 was locked before continuation and was not used in project training or selection; original official-pretraining exposure is unknown. Bars are equal-original mean-MSE reductions at 10% sampling; repeats are averaged inside each original. Sources: evaluation_primary/paired_summary.csv, continuation/audit.json and evaluation_lock.json under runs/hqs_ablation_20261013.`);
chart(slide,['Clean','Noise .01','Noise .05'],[
  {name:'vs official HQS',fill:'#D78924',values:[0,.01,.05].map(s=>pair('hqs_12000','hqs',s).mse_reduction_percent/100)},
  {name:'vs 10000-step HQS',fill:teal,values:[0,.01,.05].map(s=>pair('hqs_12000','hqs_10000',s).mse_reduction_percent/100)}
],'Selected continuation MSE reduction');
text(slide,`Validation selected step ${presentationFacts.selected_step}; test scores do not choose weights`,65,610,1120,45,25,false,muted);

slide=reset(9,'Effect of training measurement noise',
  'Fixed-budget endpoint comparison: noise-on versus noise-off with weighted stage supervision, same official starting state, seed, image/crop/sampling sequence and 800 updates. Each bar is one training seed on the primary52 set at 10% sampling. Noise levels are test measurement noise, not training-seed labels. Positive means adding training noise reduces MSE. All seeds and conditions, including negative values, are retained. Validation-selected comparisons are separately available. Source: evaluation_primary/paired_summary.csv and seed_effect_stability.csv.');
chart(slide,['Clean','Noise .01','Noise .05'],[20261013,20261014].map((seed,i)=>({
  name:`Training seed ${seed}`,fill:[teal,'#D78924'][i],
  values:[0,.01,.05].map(s=>factor('noise_effect_at_weighted',seed,s).mse_reduction_percent/100)
})),'MSE reduction: add training noise');
text(slide,'Same 800-step endpoint; weighted stage loss held fixed',65,610,1120,45,25,false,muted);

slide=reset(10,'Effect of intermediate-stage supervision',
  'Fixed-budget endpoint comparison: weighted stage loss versus final-only loss under noisy training. Five intermediate weights .01 plus final .95 are compared with final-only weight 1, preserving total loss weight. All other factors, source order, crops, sampling and measurement noise are paired within each training seed. Primary52 test, 10% sampling. Positive means adding intermediate supervision reduces MSE. The full 2x2 design also estimates effects under the other factor level and their interaction. Two seeds do not establish robust significance. Source: evaluation_primary/paired_summary.csv and factorial_effects.csv.');
chart(slide,['Clean','Noise .01','Noise .05'],[20261013,20261014].map((seed,i)=>({
  name:`Training seed ${seed}`,fill:[teal,'#D78924'][i],
  values:[0,.01,.05].map(s=>factor('stage_effect_at_noise_on',seed,s).mse_reduction_percent/100)
})),'MSE reduction: add stage supervision');
text(slide,'Same 800-step endpoint; noisy training held fixed',65,610,1120,45,25,false,muted);

const failure=facts.failure_case;
slide=reset(11,'A reconstruction that worsens',
  `Diagnostic image ${failure.image}; original source ${failure.source.source}; sampling ${failure.cr}, noise ${failure.sigma}, measurement seed ${failure.measurement_seed}. Reference ${failure.reference}; candidate ${failure.candidate}; MSE reduction ${Number(failure.mse_reduction_percent).toFixed(4)}%, PSNR delta ${Number(failure.delta_psnr).toFixed(4)} dB. Selected after testing for diagnosis only. The selected continuation is used unless it never regresses, in which case the endpoint is explicitly labeled. Absolute error from clipped floating predictions, scale 0 to .3. The error pattern locates mismatch, not a proven training cause. DIV2K source dataset: data.vision.ee.ethz.ch/cvl/DIV2K/. Full negative CSV remains preserved.`);
slide.images.add({blob:new Uint8Array(await fs.readFile(path.join(base,'figures/failure.png'))),contentType:'image/png',alt:'Reference, old HQS, continuation state and absolute error',fit:'contain',position:{left:65,top:176,width:1140,height:324}});
text(slide,`MSE reduction vs 10000-step model: ${Number(failure.mse_reduction_percent).toFixed(2)}%\nSampling ${(Number(failure.cr)*100).toFixed(0)}%; noise ${failure.sigma}`,65,523,1120,84,28,true,teal);
text(slide,`Candidate: ${failure.candidate}; diagnostic selection after testing`,65,622,1120,35,23,false,muted);

slide=reset(12,'Findings and limits',
  `New formal updates: 8400. Primary52 mean clean/strong-noise MSE changes versus prior selected HQS: ${presentationFacts.clean_mse_reduction_vs_10000.toFixed(4)}% / ${presentationFacts.strong_noise_mse_reduction_vs_10000.toFixed(4)}%. The selected and fixed-budget views are distinct. Four ablation arms, two training seeds and full paired provenance are complete; the factors concern 800-step adaptation from official weights and do not isolate the entire historical 10000-step recipe. Primary and regression evaluations retain ${presentationFacts.primary_rows} and ${presentationFacts.regression_rows} rows, with floating metric verification. Software grayscale simulation, no temporal video inputs, unknown upstream pretraining exposure. Nanjing photographs are external-author web images. Codex assisted execution, analysis and writing. Students must understand the code and rehearse; course-platform submission is not performed.`);
text(slide,`Primary52: selected continuation vs 10000-step HQS\nClean: ${presentationFacts.clean_mse_reduction_vs_10000.toFixed(2)}%   Noise .05: ${presentationFacts.strong_noise_mse_reduction_vs_10000.toFixed(2)}%`,65,168,1120,132,30,true,teal);
text(slide,'Two training seeds; all four ablation arms retained\nShort adaptation does not explain all earlier training gains\nAll regressions remain in the result tables',65,371,1120,155,29);
text(slide,'Software grayscale study; independent video frames\nAI disclosure: Codex assisted code, experiments, analysis and materials',65,606,1120,52,19,false,muted);

const candidate=path.join(build,'candidate.pptx');
await (await PresentationFile.exportPptx(deck)).save(candidate);
await finalizePresentation({
  workspaceDir:root,candidatePath:candidate,finalPath:output,
  pythonExecutable:path.join(runtime,'python/bin/python3'),
  integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),
  layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),
  layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit','--require-native-table-slide','7'],
  explicitTotalSlideCount:12,requiredNativeTableOwnerSlides:[7],requiredNativeChartOwnerSlides:[6,8,9,10],
  materializeLiteralChartWorkbooks:true,
  fontPolicy:{basis:'reference',families:[font],referencePath:source,referenceSha256:crypto.createHash('sha256').update(await fs.readFile(source)).digest('hex')},
  verifyArtifactToolImport:true,receiptPath:path.join(build,'validation.json')
});
const finalDeck=await PresentationFile.importPptx(await FileBlob.load(output));
for(let i=0;i<12;i++) {
  const blob=await finalDeck.export({slide:finalDeck.slides.items[i],format:'png',scale:1});
  await fs.writeFile(path.join(build,`slide-${i+1}.png`),new Uint8Array(await blob.arrayBuffer()));
}
await fs.writeFile(path.join(root,'coursework/PRESENTATION_GUIDE_ABLATION_20261013.md'),guide.join('\n\n')+'\n');
await fs.writeFile(path.join(base,'presentation_build.json'),JSON.stringify({output,previews:build,notes,slideTimes:times},null,2)+'\n');
console.log(JSON.stringify({output,previews:build}));
